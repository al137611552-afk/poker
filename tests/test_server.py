"""HTTP / WebSocket 接口测试。

用 TestClient 走完整的请求路径，确认适配层没有把牌桌逻辑弄拧，
以及错误情况返回的是可读的状态码而不是 500。
"""

import time

import pytest

# 服务端依赖是可选的：没装 fastapi 就整个文件跳过，而不是收集失败
pytest.importorskip("fastapi", reason="未安装 fastapi，跳过服务端测试")
pytest.importorskip("httpx", reason="未安装 httpx，TestClient 不可用")

from fastapi.testclient import TestClient  # noqa: E402

from holdem_server.app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    # bot_delay=0：测试里不需要为了「看清对手动作」而等待
    app = create_app(tmp_path / "hands.sqlite", bot_delay=0)
    with TestClient(app) as test_client:
        yield test_client


def advance(client, limit=200):
    """轮询直到轮到真人或牌局结束。"""
    view = client.get("/api/state").json()
    for _ in range(limit):
        if not view["inProgress"] or view["waitingForHuman"]:
            return view
        time.sleep(0.005)
        view = client.get("/api/state").json()
    raise AssertionError("牌局既没结束也没轮到真人")


def six_max_payload(**overrides):
    payload = {
        "seats": [{"name": "你", "isHuman": True, "style": "tag"}]
        + [{"name": f"bot{i}", "isHuman": False, "style": "tag"} for i in range(1, 6)],
        "startingStack": 1000,
        "bigBlind": 10,
        "smallBlind": 5,
        "ante": 0,
        "seed": 42,
    }
    payload.update(overrides)
    return payload


def test_styles_endpoint_lists_presets(client):
    body = client.get("/api/styles").json()
    ids = {s["id"] for s in body["styles"]}
    assert {"tag", "lag", "nit", "station", "maniac"} <= ids
    assert all(s["label"] for s in body["styles"])


def test_state_before_table_is_created(client):
    assert client.get("/api/state").status_code == 409
    assert client.post("/api/hand").status_code == 409


def test_create_table_and_play_a_hand(client):
    view = client.post("/api/table", json=six_max_payload()).json()
    assert view["handNo"] == 0
    assert len(view["seats"]) == 6
    assert view["heroSeat"] == 0

    view = client.post("/api/hand").json()
    assert view["handNo"] == 1
    assert view["inProgress"] is True
    assert view["pot"] >= 15

    view = advance(client)
    assert view["waitingForHuman"] or not view["inProgress"]


def test_hero_can_fold_and_start_next_hand(client):
    client.post("/api/table", json=six_max_payload())
    client.post("/api/hand")
    view = advance(client)
    if view["waitingForHuman"]:
        client.post("/api/action", json={"kind": "fold"})
        view = advance(client)
    assert view["inProgress"] is False
    assert view["result"] is not None
    assert sum(view["result"]["net"]) == 0

    view = client.post("/api/hand").json()
    assert view["handNo"] == 2


def test_acting_out_of_turn_returns_409(client):
    client.post("/api/table", json=six_max_payload())
    client.post("/api/hand")
    view = advance(client)
    if not view["waitingForHuman"]:
        pytest.skip("这一手没轮到真人")
    client.post("/api/action", json={"kind": "fold"})
    response = client.post("/api/action", json={"kind": "fold"})
    assert response.status_code == 409
    assert "轮到" in response.json()["detail"] or "没有进行中" in response.json()["detail"]


def test_illegal_action_returns_409(client):
    client.post("/api/table", json=six_max_payload())
    client.post("/api/hand")
    view = advance(client)
    if not view["waitingForHuman"]:
        pytest.skip("这一手没轮到真人")
    response = client.post("/api/action", json={"kind": "raise", "amount": 999999})
    assert response.status_code == 409


def test_unknown_action_kind_is_rejected(client):
    client.post("/api/table", json=six_max_payload())
    client.post("/api/hand")
    view = advance(client)
    if not view["waitingForHuman"]:
        pytest.skip("这一手没轮到真人")
    response = client.post("/api/action", json={"kind": "梭哈"})
    assert response.status_code == 409


def test_bad_table_config_is_rejected(client):
    assert client.post("/api/table", json=six_max_payload(seats=[])).status_code == 422
    bad_style = six_max_payload()
    bad_style["seats"][1]["style"] = "无敌"
    assert client.post("/api/table", json=bad_style).status_code == 422
    assert client.post("/api/table", json=six_max_payload(bigBlind=0)).status_code == 422


def test_two_human_seats_rejected(client):
    payload = six_max_payload()
    payload["seats"][1]["isHuman"] = True
    response = client.post("/api/table", json=payload)
    assert response.status_code == 422
    assert "真人" in response.json()["detail"]


def test_api_never_leaks_opponent_cards(client):
    client.post("/api/table", json=six_max_payload())
    for _ in range(5):
        client.post("/api/hand")
        for _ in range(80):
            view = client.get("/api/state").json()
            for seat in view["seats"]:
                if seat["seat"] == view["heroSeat"]:
                    continue
                if seat["cards"] is not None:
                    assert not view["inProgress"], "牌局进行中泄露了对手底牌"
            if not view["inProgress"]:
                break
            if view["waitingForHuman"]:
                client.post("/api/action", json={"kind": "fold"})


def test_websocket_pushes_state(client):
    client.post("/api/table", json=six_max_payload())
    with client.websocket_connect("/ws") as websocket:
        first = websocket.receive_json()
        assert first["heroSeat"] == 0
        assert first["seats"][1]["cards"] is None


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "德扑训练台" in response.text
    assert "actionbar" in response.text

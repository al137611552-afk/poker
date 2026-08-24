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


# ------------------------------------------------------------------ HUD（FR-8）


def _play_some_hands(client, count=4):
    """打若干手，人类一律弃牌——只要牌进库，统计就有东西可算。"""
    for _ in range(count):
        client.post("/api/hand")
        view = advance(client)
        while view["inProgress"] and view["waitingForHuman"]:
            client.post("/api/action", json={"kind": "fold"})
            view = advance(client)


def test_hud_needs_a_table_first(client):
    assert client.get("/api/hud").status_code == 409


def test_hud_reports_every_seat_even_before_any_hand(client):
    """一手没打时也要给出每个座位——**HUD 缺一行比给 0% 更让人困惑**。"""
    client.post("/api/table", json=six_max_payload())
    body = client.get("/api/hud").json()
    assert [s["seat"] for s in body["seats"]] == list(range(6))
    assert all(s["hands"] == 0 for s in body["seats"])


def test_every_metric_carries_its_sample_size(client):
    """**HUD 最大的坑是拿 5 手牌的 VPIP 当真。**

    够不够取决于看哪个指标（VPIP 几十手就稳，3bet 要几百手），所以服务端不替
    前端定阈值，而是把原始计数一并给出去，让它自己决定标不标灰。
    """
    client.post("/api/table", json=six_max_payload())
    _play_some_hands(client)
    body = client.get("/api/hud").json()

    for seat in body["seats"]:
        rates = [m for m in seat["stats"] if "rate" in m]
        assert len(rates) >= 8, "比率型指标不该少"
        for metric in rates:
            assert metric["hits"] <= metric["chances"], metric
            if metric["chances"] == 0:
                assert metric["rate"] is None, "没机会就不该给一个 0%"

        # AF 不是比率，它有自己的形状——**别拿通用逻辑套它**，
        # 那会画出一个 >100% 的百分比。
        af = next(m for m in seat["stats"] if m["key"] == "aggression_factor")
        assert "rate" not in af and "chances" not in af
        assert af["value"] is None or af["value"] >= 0


def test_a_metric_with_no_chances_is_none_not_zero(client):
    """「从没面对过 3bet」和「面对 3bet 从不弃牌」是两件事，压成 0% 就分不开了。"""
    client.post("/api/table", json=six_max_payload())
    body = client.get("/api/hud").json()
    rates = {m["key"]: m["rate"] for m in body["seats"][0]["stats"] if "rate" in m}
    assert rates and all(value is None for value in rates.values()), "一手没打，全都该是「不知道」"


def test_hud_scope_is_validated(client):
    client.post("/api/table", json=six_max_payload())
    assert client.get("/api/hud", params={"scope": "session"}).status_code == 200
    assert client.get("/api/hud", params={"scope": "all"}).status_code == 200
    assert client.get("/api/hud", params={"scope": "宇宙"}).status_code == 422


def test_hud_is_not_pushed_with_every_state_broadcast(client):
    """统计一手牌才变一次，跟着每次动作广播走等于把它推几十遍。"""
    client.post("/api/table", json=six_max_payload())
    view = client.get("/api/state").json()
    assert "hud" not in view and "stats" not in view


# ------------------------------------------------------------------ 场景训练（FR-12）


def test_training_scenarios_come_from_the_engine(client):
    """位置表不在前端写死——引擎改了座位命名，界面得跟着变。"""
    body = client.get("/api/training/scenarios").json()
    assert "UTG" in body["positions"] and "BTN" in body["positions"]
    assert {k["id"] for k in body["kinds"]} == {"开牌", "面对开牌", "面对再加注"}


def test_answering_before_dealing_is_refused(client):
    assert client.get("/api/training/current").status_code == 409
    assert client.post("/api/training/answer", json={"kind": "fold"}).status_code == 409


def test_a_dealt_spot_carries_everything_the_ui_needs(client):
    body = client.post("/api/training/deal", json={"kind": "开牌", "hero": "UTG"}).json()
    assert body["hero"]["position"] == "UTG"
    assert len(body["hero"]["cards"]) == 2
    assert body["pot"] > 0
    # **合法动作由引擎给，前端不该自己推**：猜一份出来会让界面允许打不出来的动作
    legal = body["legal"]
    assert legal["canRaise"] and legal["minRaiseTo"] < legal["maxRaiseTo"]


def test_an_illegal_raise_is_refused_with_a_reason(client):
    """判卷的前提是这确实是个可选项，否则评的不是决策、是笔误。"""
    client.post("/api/training/deal", json={"kind": "面对开牌", "hero": "BB",
                                            "villain": "BTN"})
    response = client.post("/api/training/answer", json={"kind": "raise", "to": 11})
    assert response.status_code == 422
    assert "不合法" in response.json()["detail"]


def test_a_raise_without_an_amount_is_refused(client):
    client.post("/api/training/deal", json={"kind": "开牌", "hero": "CO"})
    assert client.post("/api/training/answer", json={"kind": "raise"}).status_code == 422


def test_grading_returns_the_whole_distribution(client):
    client.post("/api/training/deal", json={"kind": "开牌", "hero": "BTN"})
    body = client.post("/api/training/answer", json={"kind": "fold"}).json()
    assert body["graded"] is True
    assert body["weights"] and body["best"] in body["weights"]
    assert 0.0 <= body["frequency"] <= 1.0


def test_an_unknown_scenario_or_position_is_refused(client):
    assert client.post("/api/training/deal",
                       json={"kind": "跳舞", "hero": "UTG"}).status_code == 422
    assert client.post("/api/training/deal",
                       json={"kind": "开牌", "hero": "楼上"}).status_code == 422


def test_dealing_again_replaces_the_current_spot(client):
    """一次一道题：发了新的，旧的就该判不了了（否则会对着上一题打分）。"""
    first = client.post("/api/training/deal", json={"kind": "开牌", "hero": "UTG"}).json()
    second = client.post("/api/training/deal", json={"kind": "开牌", "hero": "BTN"}).json()
    assert second["hero"]["position"] == "BTN" != first["hero"]["position"]
    assert client.get("/api/training/current").json()["hero"]["position"] == "BTN"


# ------------------------------------------------------------------ 复盘（FR-9）


def test_review_before_anything_is_played(client):
    assert client.get("/api/review").status_code == 409
    client.post("/api/table", json=six_max_payload())
    assert client.get("/api/review").status_code == 409, "建了桌但没打牌，照样没得复盘"


def test_review_reports_the_preflop_decisions(client):
    client.post("/api/table", json=six_max_payload())
    _play_some_hands(client, count=1)
    body = client.get("/api/review").json()

    assert len(body["heroCards"]) == 2
    assert isinstance(body["preflop"], list)
    for step in body["preflop"]:
        assert step["position"] and step["action"]
        # 底池与待跟额记的是**那一刻**的，不是终局的
        assert step["potBefore"] >= 0 and step["toCall"] >= 0
        if step["graded"]:
            assert 0.0 <= step["frequency"] <= 1.0
            assert step["best"] in step["weights"]


def test_review_says_plainly_when_there_is_no_solver(client):
    """**没装求解器就如实说**，不降级成一个看着差不多的数。

    PRD 的「不得冒充精确解」防的就是这个：翻后 EV 损失算不了时，
    给一句「没装」比给一个来路不明的数字有用得多。
    """
    client.post("/api/table", json=six_max_payload())
    _play_some_hands(client, count=1)
    postflop = client.get("/api/review").json()["postflop"]
    assert isinstance(postflop["available"], bool)
    if not postflop["available"]:
        assert "没装求解器" in postflop["why"]


def test_review_follows_the_latest_hand(client):
    """复盘的是**刚打完**那手，不是第一手——否则打了十手还在看第一手。"""
    client.post("/api/table", json=six_max_payload())
    _play_some_hands(client, count=1)
    first = client.get("/api/review").json()
    _play_some_hands(client, count=1)
    second = client.get("/api/review").json()
    assert second["handNo"] > first["handNo"]

"""SQLite 落库测试。

重点是「PHH 原文是唯一事实来源」这条约定能站得住：存进去的牌谱读出来后，
必须能原样重放出同一手牌。
"""

import sqlite3

import pytest

from holdem.actions import call, check, fold, raise_to
from holdem.deck import deck_from_seed, stacked_deck
from holdem.phh import loads, phh_player_order
from holdem.state import HandConfig, HandState
from holdem.store import SCHEMA_VERSION, HandStore

SIX_MAX = [f"bot{i}" for i in range(6)]


def _play(stacks, button, actions, deck=None, bb=10, sb=5, ante=0):
    config = HandConfig(
        stacks=tuple(stacks), button=button, big_blind=bb, small_blind=sb, ante=ante
    )
    hand = HandState(config, deck if deck is not None else deck_from_seed(11))
    for action in actions:
        hand.apply(action)
    return hand


@pytest.fixture()
def store():
    with HandStore(":memory:") as s:
        yield s


def test_saves_and_counts_hands(store):
    session = store.create_session("测试局", small_blind=5, big_blind=10)
    for _ in range(3):
        hand = _play([1000] * 6, button=0, actions=[fold()] * 5)
        store.save_hand(hand, session_id=session, players=SIX_MAX)
    assert store.count_hands() == 3
    assert store.count_hands(session) == 3


def test_hand_numbers_increment_per_session(store):
    a = store.create_session("A")
    b = store.create_session("B")
    for _ in range(2):
        store.save_hand(
            _play([1000] * 6, button=0, actions=[fold()] * 5), session_id=a, players=SIX_MAX
        )
    store.save_hand(
        _play([1000] * 6, button=0, actions=[fold()] * 5), session_id=b, players=SIX_MAX
    )
    rows = store.conn.execute(
        "SELECT session_id, hand_no FROM hands ORDER BY id"
    ).fetchall()
    assert [(r["session_id"], r["hand_no"]) for r in rows] == [(a, 1), (a, 2), (b, 1)]


def test_stored_phh_replays_to_the_same_hand(store):
    session = store.create_session("回放")
    deck = stacked_deck(
        hole={0: "AsAd", 1: "KsKd"}, board="2c3d4h5s7c", num_seats=2, button=0
    )
    hand = _play([500, 500], button=0, actions=[raise_to(500), call()], deck=deck)
    hand_id = store.save_hand(hand, session_id=session, players=["me", "villain"])

    replayed = loads(store.load_phh(hand_id))
    order = phh_player_order(2, 0)
    for phh_index, seat in enumerate(order):
        assert replayed.stacks[phh_index] == hand.stacks[seat]
        assert replayed.hole[phh_index] == hand.hole[seat]
    assert replayed.board == hand.board


def test_player_rows_capture_outcome(store):
    session = store.create_session("结算")
    hand = _play([1000] * 6, button=0, actions=[fold()] * 5)
    hand_id = store.save_hand(hand, session_id=session, players=SIX_MAX)

    rows = store.conn.execute(
        "SELECT * FROM hand_players WHERE hand_id = ? ORDER BY seat", (hand_id,)
    ).fetchall()
    assert len(rows) == 6
    by_seat = {r["seat"]: r for r in rows}
    assert by_seat[2]["position"] == "BB"
    assert by_seat[2]["net"] == 5, "大盲赢下小盲"
    assert by_seat[1]["net"] == -5
    assert by_seat[1]["folded"] == 1
    assert sum(r["net"] for r in rows) == 0
    assert all(len(r["hole"]) == 4 for r in rows)


def test_action_rows_record_decision_context(store):
    session = store.create_session("动作")
    hand = _play(
        [1000] * 6,
        button=0,
        # 翻前三家看翻牌，其后各街过牌到摊牌
        actions=[raise_to(30), fold(), fold(), call(), fold(), call()] + [check()] * 9,
    )
    assert hand.is_complete
    hand_id = store.save_hand(hand, session_id=session, players=SIX_MAX)

    rows = store.conn.execute(
        "SELECT * FROM hand_actions WHERE hand_id = ? ORDER BY seq", (hand_id,)
    ).fetchall()
    first = rows[0]
    assert first["kind"] == "raise"
    assert first["position"] == "UTG"
    assert first["player"] == "bot3"
    assert first["to_amount"] == 30
    assert first["pot_before"] == 15
    assert first["to_call"] == 10
    # 翻前每人一个动作，翻后三家继续
    assert sum(1 for r in rows if r["street"] == 0) == 6


def test_player_summary(store):
    session = store.create_session("汇总", small_blind=5, big_blind=10)
    for _ in range(4):
        hand = _play([1000] * 6, button=0, actions=[fold()] * 5)
        store.save_hand(hand, session_id=session, players=SIX_MAX)

    winner = store.player_summary("bot2")
    assert winner.hands == 4
    assert winner.net == 20
    assert winner.big_blind == 10
    assert winner.bb_per_100 == pytest.approx(50.0)

    loser = store.player_summary("bot1")
    assert loser.net == -20
    assert loser.bb_per_100 == pytest.approx(-50.0)

    unknown = store.player_summary("查无此人")
    assert unknown.hands == 0 and unknown.bb_per_100 == 0.0


def test_rejects_unfinished_hand(store):
    session = store.create_session("未结束")
    hand = _play([1000] * 6, button=0, actions=[fold()])
    with pytest.raises(ValueError, match="已结束"):
        store.save_hand(hand, session_id=session, players=SIX_MAX)


def test_rejects_wrong_player_count(store):
    session = store.create_session("人数")
    hand = _play([1000] * 6, button=0, actions=[fold()] * 5)
    with pytest.raises(ValueError, match="座位数"):
        store.save_hand(hand, session_id=session, players=["only", "two"])


def test_duplicate_hand_number_is_rejected(store):
    session = store.create_session("重号")
    hand = _play([1000] * 6, button=0, actions=[fold()] * 5)
    store.save_hand(hand, session_id=session, players=SIX_MAX, hand_no=1)
    with pytest.raises(sqlite3.IntegrityError):
        store.save_hand(hand, session_id=session, players=SIX_MAX, hand_no=1)


def test_persists_to_disk_and_reopens(tmp_path):
    path = tmp_path / "hands.sqlite"
    with HandStore(path) as store:
        session = store.create_session("落盘")
        hand = _play([1000] * 6, button=0, actions=[fold()] * 5)
        store.save_hand(hand, session_id=session, players=SIX_MAX)

    with HandStore(path) as reopened:
        assert reopened.count_hands() == 1
        phh_texts = list(reopened.iter_phh())
        assert len(phh_texts) == 1
        assert loads(phh_texts[0]).is_complete


def test_schema_version_mismatch_is_loud(tmp_path):
    path = tmp_path / "old.sqlite"
    HandStore(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE schema_info SET version = ?", (SCHEMA_VERSION + 1,))
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="版本不匹配"):
        HandStore(path)


def test_deleting_a_hand_cascades(store):
    session = store.create_session("级联")
    hand = _play([1000] * 6, button=0, actions=[fold()] * 5)
    hand_id = store.save_hand(hand, session_id=session, players=SIX_MAX)
    store.conn.execute("DELETE FROM hands WHERE id = ?", (hand_id,))
    store.conn.commit()
    for table in ("hand_players", "hand_actions"):
        row = store.conn.execute(
            f"SELECT COUNT(*) AS c FROM {table} WHERE hand_id = ?", (hand_id,)
        ).fetchone()
        assert row["c"] == 0, f"{table} 未随牌局一起删除"

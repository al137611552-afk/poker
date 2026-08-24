"""SQLite 落库测试。

重点是「PHH 原文是唯一事实来源」这条约定能站得住：存进去的牌谱读出来后，
必须能原样重放出同一手牌。
"""

import sqlite3

import pytest

from holdem.actions import bet, call, check, fold, raise_to
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


def test_a_newer_database_is_refused(tmp_path):
    """**比程序还新的库要拒绝打开**：字段可能已经改了语义，硬读会给出错的数。"""
    path = tmp_path / "future.sqlite"
    HandStore(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE schema_info SET version = ?", (SCHEMA_VERSION + 1,))
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="比程序还新"):
        HandStore(path)


def test_an_older_database_is_upgraded_in_place(tmp_path):
    """**旧库直接升级**，不报错也不丢数据。

    v1→v2 是纯增量（只加了 `quiz_answers` 一张表），所以能这么升；
    将来若有破坏性变更，不能照抄这条路。这条用例连带守住「老数据还在」。
    """
    path = tmp_path / "old.sqlite"
    with HandStore(path) as store:
        session = store.create_session("升级前", small_blind=5, big_blind=10)
        hand = _play([1000] * 6, button=0, actions=[fold()] * 5)
        store.save_hand(hand, session_id=session, players=SIX_MAX)

    conn = sqlite3.connect(path)
    conn.execute("UPDATE schema_info SET version = 1")
    conn.execute("DROP TABLE quiz_answers")
    conn.commit()
    conn.close()

    with HandStore(path) as upgraded:
        version = upgraded.conn.execute("SELECT version FROM schema_info").fetchone()
        assert version["version"] == SCHEMA_VERSION
        assert upgraded.count_hands() == 1, "老数据一个字节都不该动"
        assert upgraded.quiz_track() == (0, 0, 0), "新表建出来了，只是还没有记录"


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


# ------------------------------------------------------------------ 从库里算统计（FR-7）


def _finish(hand):
    while not hand.is_complete:
        hand.apply(check() if hand.legal_actions().can_check else fold())


def test_stats_from_the_database_match_the_ones_computed_in_memory(store):
    """**同一手牌，走内存和走数据库必须得出一模一样的统计。**

    这条是整次解耦的意义所在：口径只在 `stats.py` 定义一次，
    两条来源只负责把牌折成同样的 `(记录序列, HandFacts)`。
    一旦有人在 store 这边"顺手"重算某个指标，这条就会红。
    """
    from holdem.stats import accumulate

    session = store.create_session("统计", small_blind=5, big_blind=10)
    hands = []
    for actions in (
        [raise_to(30), fold(), fold(), fold(), fold(), fold()],
        [call(), raise_to(40), fold(), fold(), fold(), fold(), fold()],
        [raise_to(30), fold(), fold(), fold(), fold(), call(), check(), bet(40), fold()],
    ):
        hand = _play([1000] * 6, button=0, actions=actions)
        _finish(hand)
        store.save_hand(hand, session_id=session, players=SIX_MAX)
        hands.append(hand)

    in_memory = {}
    for hand in hands:
        accumulate(hand, in_memory, key_of=lambda seat: SIX_MAX[seat])
    from_db = store.player_stats(session_id=session)

    assert set(from_db) == set(in_memory), "两条路统计到的人得是同一批"
    for name, line in in_memory.items():
        other = from_db[name]
        assert other.hands == line.hands, name
        for field in ("vpip", "pfr", "rfi", "threebet", "fold_to_threebet",
                      "cbet_flop", "fold_to_cbet_flop", "wtsd", "wsd"):
            mine, theirs = getattr(line, field), getattr(other, field)
            assert (theirs.chances, theirs.hits) == (mine.chances, mine.hits), (name, field)
        assert other.postflop_aggressive == line.postflop_aggressive, name
        assert other.postflop_calls == line.postflop_calls, name


def test_stats_are_keyed_by_player_not_by_seat(store):
    """HUD 问的是「这个对手怎么打」，而同一个人跨局会换座位。"""
    session = store.create_session("换座", small_blind=5, big_blind=10)
    hand = _play([1000] * 6, button=0, actions=[raise_to(30)] + [fold()] * 5)
    _finish(hand)
    store.save_hand(hand, session_id=session, players=SIX_MAX)

    # 同一批人，按钮挪一位：座位与位置的对应关系整个变了
    hand2 = _play([1000] * 6, button=1, actions=[raise_to(30)] + [fold()] * 5)
    _finish(hand2)
    store.save_hand(hand2, session_id=session, players=SIX_MAX)

    stats = store.player_stats(session_id=session)
    assert set(stats) == set(SIX_MAX)
    assert all(line.hands == 2 for line in stats.values()), "每个人都打了两手"


def test_filtering_by_player_still_feeds_the_whole_hand_to_the_algorithm(store):
    """只要一个人的数，也得把整手牌喂进去——3bet 的**分母**得知道别人加没加注。"""
    session = store.create_session("过滤", small_blind=5, big_blind=10)
    hand = _play([1000] * 6, button=0,
                 actions=[raise_to(30), raise_to(90), fold(), fold(), fold(), fold(), fold()])
    _finish(hand)
    store.save_hand(hand, session_id=session, players=SIX_MAX)

    only = store.player_stats(session_id=session, players=("bot4",))
    assert set(only) == {"bot4"}
    assert only["bot4"].threebet.chances == 1 and only["bot4"].threebet.hits == 1


def test_stats_can_be_split_by_position(store):
    session = store.create_session("位置", small_blind=5, big_blind=10)
    hand = _play([1000] * 6, button=0, actions=[raise_to(30)] + [fold()] * 5)
    _finish(hand)
    store.save_hand(hand, session_id=session, players=SIX_MAX)

    stats = store.player_stats(session_id=session, by_position=True)
    assert ("bot3", "UTG") in stats
    assert stats[("bot3", "UTG")].rfi.hits == 1


# ------------------------------------------------------------------ 测验轨（FR-14）


def test_quiz_answers_are_stored_with_the_verdict_not_just_the_action(store):
    """**存判卷的结论，不只存动作。**

    判卷依赖范围表，表以后会重算；那时拿旧答题按新表重判会得出不同结论——
    而用户看到的「我当时答对了」应该是当时那张表说的。
    """
    store.record_answer(kind="开牌", hero="UTG", villain=None, hole="AhKh",
                        taken="加注到 2.5bb", best="加注到 2.5bb", frequency=0.92,
                        on_solution=True, blunder=False)
    store.record_answer(kind="开牌", hero="UTG", villain=None, hole="7c2d",
                        taken="加注到 2.5bb", best="弃牌", frequency=0.0,
                        on_solution=False, blunder=True)
    assert store.quiz_track() == (2, 1, 1)


def test_quiz_track_can_be_filtered_by_scenario(store):
    store.record_answer(kind="开牌", hero="UTG", villain=None, hole="AhKh",
                        taken="x", best="x", frequency=1.0, on_solution=True, blunder=False)
    store.record_answer(kind="面对开牌", hero="BB", villain="BTN", hole="7c2d",
                        taken="y", best="z", frequency=0.0, on_solution=False, blunder=True)
    assert store.quiz_track(kind="开牌") == (1, 1, 0)
    assert store.quiz_track(kind="面对开牌") == (1, 0, 1)


def test_replaying_stored_hands_gives_back_playable_states(store):
    session = store.create_session("重放", small_blind=5, big_blind=10)
    for _ in range(3):
        hand = _play([1000] * 6, button=0, actions=[fold()] * 5)
        store.save_hand(hand, session_id=session, players=SIX_MAX)
    replayed = list(store.replay_hands(session_id=session))
    assert len(replayed) == 3
    assert all(h.result is not None for h in replayed), "重放出来的必须是打完的牌"


def test_a_broken_row_does_not_take_down_the_whole_batch(store):
    """一手牌谱有问题不该让「看看我的评级」整个失败。"""
    session = store.create_session("坏行", small_blind=5, big_blind=10)
    hand = _play([1000] * 6, button=0, actions=[fold()] * 5)
    store.save_hand(hand, session_id=session, players=SIX_MAX)
    store.conn.execute(
        "INSERT INTO hands (session_id, hand_no, played_at, num_seats, button,"
        " small_blind, big_blind, ante, board, pot, went_to_showdown, phh)"
        " VALUES (?, 99, '2026-01-01', 6, 0, 5, 10, 0, '', 0, 0, '这不是牌谱')",
        (session,),
    )
    store.conn.commit()
    assert len(list(store.replay_hands(session_id=session))) == 1, "坏的跳过，好的照给"

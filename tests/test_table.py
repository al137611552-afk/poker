"""牌桌编排测试（不经过 HTTP）。

最要紧的一条是**不能泄露对手底牌**——这属于安全性质的缺陷，必须有测试守着。
"""

import pytest

from holdem.store import HandStore
from holdem_server.table import SeatConfig, TableConfig, TableSession


def make_session(num_seats=6, human_seat=0, seed=42, store=None, session_id=None, **kwargs):
    seats = [
        SeatConfig(name="你" if i == human_seat else f"bot{i}", is_human=(i == human_seat))
        for i in range(num_seats)
    ]
    config = TableConfig(seats=seats, seed=seed, **kwargs)
    return TableSession(config=config, store=store, store_session_id=session_id)


def drive_to_human(session, limit=200):
    """推进到轮到真人或牌局结束。"""
    for _ in range(limit):
        if not session.in_progress or session.waiting_for_human:
            return
        if not session.step_bot():
            return
    raise AssertionError("牌局没有停在真人的行动点上")


# ------------------------------------------------------------------ 配置


def test_rejects_multiple_humans():
    with pytest.raises(ValueError, match="一个真人"):
        TableConfig(
            seats=[SeatConfig("a", True), SeatConfig("b", True)],
        )


def test_rejects_out_of_range_seat_count():
    with pytest.raises(ValueError, match="座位数"):
        TableConfig(seats=[SeatConfig("solo", True)])
    with pytest.raises(ValueError, match="座位数"):
        TableConfig(seats=[SeatConfig(f"p{i}") for i in range(10)])


def test_rejects_unknown_style():
    with pytest.raises(ValueError, match="未知风格"):
        SeatConfig("bot", False, "超级高手")


# ------------------------------------------------------------------ 生命周期


def test_first_hand_puts_button_on_seat_zero():
    session = make_session()
    session.start_hand()
    assert session.button == 0
    assert session.hand_no == 1


def test_button_moves_each_hand():
    session = make_session()
    buttons = []
    for _ in range(4):
        session.start_hand()
        buttons.append(session.button)
        drive_to_human(session)
        while session.in_progress:
            if session.waiting_for_human:
                session.apply_human("fold")
            else:
                session.step_bot()
    assert buttons == [0, 1, 2, 3]


def test_cannot_start_hand_while_one_is_running():
    session = make_session()
    session.start_hand()
    with pytest.raises(RuntimeError, match="还没结束"):
        session.start_hand()


def test_human_cannot_act_out_of_turn():
    session = make_session()
    session.start_hand()
    if not session.waiting_for_human:
        with pytest.raises(RuntimeError, match="还没轮到"):
            session.apply_human("fold")


def test_stacks_carry_over_between_hands():
    session = make_session(num_seats=3)
    session.start_hand()
    while session.in_progress:
        if session.waiting_for_human:
            session.apply_human("fold")
        else:
            session.step_bot()
    assert sum(session.stacks) == 3000
    session.start_hand()
    assert list(session.hand.config.stacks) == session.stacks


def test_auto_rebuy_refills_busted_seats():
    session = make_session(num_seats=2, starting_stack=100)
    session.stacks = [0, 200]
    session.start_hand()
    assert session.hand.config.stacks[0] == 100, "筹码打光应自动补回"


def test_plays_many_hands_without_chip_leakage():
    session = make_session(num_seats=6, seed=5)
    for _ in range(25):
        session.start_hand()
        while session.in_progress:
            if session.waiting_for_human:
                legal = session.hand.legal_actions()
                session.apply_human("check" if legal.can_check else "call")
            else:
                session.step_bot()
        assert sum(session.stacks) % 1 == 0
        assert all(s >= 0 for s in session.stacks)


# ------------------------------------------------------------------ 视图


def test_view_hides_opponent_hole_cards():
    session = make_session()
    session.start_hand()
    drive_to_human(session)
    view = session.view()
    hero = view["heroSeat"]
    for seat in view["seats"]:
        if seat["seat"] == hero:
            assert seat["cards"] is not None and len(seat["cards"]) == 2
        else:
            assert seat["cards"] is None, "不能泄露对手底牌"


def test_view_reveals_only_showdown_hands():
    session = make_session(num_seats=2, seed=3)
    for _ in range(30):
        session.start_hand()
        while session.in_progress:
            if session.waiting_for_human:
                legal = session.hand.legal_actions()
                session.apply_human("check" if legal.can_check else "call")
            else:
                session.step_bot()
        view = session.view()
        shown = set(session.hand.result.showdown_scores)
        for seat in view["seats"]:
            if seat["seat"] == view["heroSeat"] or seat["seat"] in shown:
                continue
            assert seat["cards"] is None, "未摊牌的对手底牌不得公开"
        if view["result"]["wentToShowdown"]:
            revealed = [s for s in view["seats"] if s["cards"] is not None]
            assert len(revealed) >= 2
            return
    raise AssertionError("三十手内没有出现摊牌，样本不足")


def test_view_exposes_legal_actions_only_on_hero_turn():
    session = make_session()
    session.start_hand()
    drive_to_human(session)
    view = session.view()
    assert view["waitingForHuman"] is True
    legal = view["legal"]
    assert legal is not None
    assert legal["minRaiseTo"] <= legal["maxRaiseTo"] or not legal["canRaise"]
    assert legal["potSizedTo"] >= legal["minRaiseTo"] or not legal["canRaise"]

    session.apply_human("fold")
    while session.in_progress:
        session.step_bot()
    assert session.view()["legal"] is None


def test_log_records_the_action():
    session = make_session(num_seats=3)
    session.start_hand()
    drive_to_human(session)
    texts = [line["text"] for line in session.view()["log"]]
    assert any("盲注" in t for t in texts)


# ------------------------------------------------------------------ 落库


def test_finished_hands_are_persisted():
    with HandStore(":memory:") as store:
        session_id = store.create_session("测试", small_blind=5, big_blind=10)
        session = make_session(num_seats=4, store=store, session_id=session_id)
        for _ in range(3):
            session.start_hand()
            while session.in_progress:
                if session.waiting_for_human:
                    session.apply_human("fold")
                else:
                    session.step_bot()
        assert store.count_hands(session_id) == 3
        assert session.last_saved_hand_id is not None
        phh = store.load_phh(session.last_saved_hand_id)
        assert "variant" in phh and "actions" in phh

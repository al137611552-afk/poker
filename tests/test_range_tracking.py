"""从打完的牌回推翻牌范围的测试（FR-11 的第一块）。

翻后求解的第一个输入就是双方范围，**范围错了，后面每一个 EV 损失都是错的，而且看不出来**。
所以这里逐条钉住：认哪些线路、位置谁先说话、底池与有效筹码怎么算、
以及**认不出来的线路必须说清为什么**（拿一份不适用的范围去求解比不打分更糟）。
"""

import pytest

from holdem import preflop_ranges
from holdem.actions import call, check, fold, raise_to
from holdem.deck import deck_from_seed
from holdem.range_tracking import NotCovered, flop_ranges
from holdem.state import HandConfig, HandState

BIG_BLIND = 100

pytestmark = pytest.mark.skipif(
    not preflop_ranges.is_available(), reason="翻前范围表尚未生成"
)


def play(actions, *, seats=6, button=0, stack_bb=100.0):
    """六人桌打一手；按钮在 0 时座位号正好等于「相对按钮的偏移」。"""
    config = HandConfig(
        stacks=[int(stack_bb * BIG_BLIND)] * seats,
        button=button,
        big_blind=BIG_BLIND,
        small_blind=BIG_BLIND // 2,
    )
    hand = HandState(config, deck_from_seed(11))
    for action in actions:
        hand.apply(action)
    return hand


OPEN_FOLDS = [fold(), fold(), fold()]  # UTG / HJ / CO 依次弃牌，轮到按钮


# ------------------------------------------------------------------ 认得出的线路


def test_a_single_raised_pot_uses_the_open_and_call_ranges():
    """按钮开牌、大盲跟注：两边的范围就是表里那两格。"""
    hand = play([*OPEN_FOLDS, raise_to(250), fold(), call()])
    setup = flop_ranges(hand)

    table = preflop_ranges.load()
    assert setup.line == "单次加注底池"
    assert setup.ip == table.open_range("BTN")
    assert setup.oop == table.defense("BTN", "BB").action("跟注到2.5")


def test_the_big_blind_acts_first_after_the_flop():
    """位置不是按座位号定的，是按「小盲起数过来第一个还在牌里的人」。"""
    hand = play([*OPEN_FOLDS, raise_to(250), fold(), call()])
    setup = flop_ranges(hand)
    assert setup.oop_seat == 2 and setup.ip_seat == 0, "大盲没位置、按钮有位置"
    assert setup.player_index(2) == 0 and setup.player_index(0) == 1
    assert setup.range_of(0) == setup.ip
    with pytest.raises(KeyError, match="没看到翻牌"):
        setup.range_of(4)


def test_the_small_blind_is_out_of_position_against_the_button():
    """小盲跟注按钮：小盲翻后先说话（他在庄家左手第一个）。"""
    hand = play([*OPEN_FOLDS, raise_to(250), call(), fold()])
    setup = flop_ranges(hand)
    assert setup.oop_seat == 1 and setup.ip_seat == 0


def test_the_pot_and_stack_are_measured_at_the_flop():
    """底池是翻牌发出来那一刻的，有效筹码是**底池后面还剩多少**。"""
    hand = play([*OPEN_FOLDS, raise_to(250), fold(), call()])
    setup = flop_ranges(hand)
    assert setup.pot == pytest.approx(5.5), "小盲 0.5 + 双方各 2.5"
    assert setup.effective_stack == pytest.approx(97.5), "100 减去投进去的 2.5"


def test_a_three_bet_pot_uses_the_reraise_ranges():
    """按钮开牌、大盲 3bet、按钮跟：一边取 3bet 范围，另一边取「面对 3bet 的跟注」。"""
    hand = play([*OPEN_FOLDS, raise_to(250), fold(), raise_to(750), call()])
    setup = flop_ranges(hand)

    entry = preflop_ranges.load().defense("BTN", "BB")
    assert setup.line == "3bet 底池"
    assert setup.oop == entry.action("加注到7.5")
    assert setup.ip == entry.reraise_reply["跟注到7.5"]
    assert setup.pot == pytest.approx(15.5)
    assert setup.effective_stack == pytest.approx(92.5)


def test_heads_up_hands_use_the_heads_up_table():
    if not preflop_ranges.HEADSUP_PATH.exists():
        pytest.skip("单挑范围表尚未生成")
    hand = play([raise_to(250), call()], seats=2, stack_bb=200.0)
    setup = flop_ranges(hand)
    table = preflop_ranges.load(preflop_ranges.HEADSUP_PATH)
    assert setup.ip == table.open_range("BTN"), "单挑按钮翻后有位置"
    assert setup.oop == table.defense("BTN", "BB").action("跟注到2.5")


# ------------------------------------------------------------------ 认不出的要说清楚


def test_a_multiway_flop_is_not_scored():
    """求解器只解两人。三个人看翻牌就明说不打分，别拿两人解去凑。"""
    hand = play([*OPEN_FOLDS, raise_to(250), call(), call()])
    with pytest.raises(NotCovered, match="只解两人"):
        flop_ranges(hand)


def test_a_limped_pot_is_not_covered():
    hand = play([call(), fold(), fold(), fold(), fold(), check()])
    with pytest.raises(NotCovered, match="跛入"):
        flop_ranges(hand)


def test_a_cold_call_before_the_three_bet_is_not_covered():
    """开牌、冷跟、再 3bet：表是按「一个开牌者 + 一个防守者」解的，这条线路没有。"""
    hand = play([fold(), fold(), call(), raise_to(750), fold(), fold(), fold()])
    with pytest.raises(NotCovered):
        flop_ranges(hand)


def test_a_four_bet_pot_is_not_covered():
    hand = play([*OPEN_FOLDS, raise_to(250), fold(), raise_to(750), raise_to(1650), call()])
    with pytest.raises(NotCovered, match="4bet"):
        flop_ranges(hand)


def test_a_hand_that_never_saw_a_flop_is_not_scored():
    hand = play([*OPEN_FOLDS, raise_to(250), fold(), fold()])
    with pytest.raises(NotCovered, match="没走到翻牌|只解两人"):
        flop_ranges(hand)


def test_a_stack_depth_far_from_the_table_is_refused():
    """20bb 的桌子照 100bb 的表回推范围是错的——那个深度该走推/弃。"""
    hand = play([*OPEN_FOLDS, raise_to(250), fold(), call()], stack_bb=20.0)
    with pytest.raises(NotCovered, match="差得太多"):
        flop_ranges(hand)


def test_a_table_size_without_a_product_is_refused():
    hand = play([fold(), raise_to(250), call()], seats=4)
    with pytest.raises(NotCovered, match="4 人桌"):
        flop_ranges(hand)

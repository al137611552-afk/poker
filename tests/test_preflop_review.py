"""翻前复盘（FR-9 的翻前那半）。

重点测**重放**：重放对不上时如果只是跳过，后面的判卷全都在错误的局面上做，
而界面上还写着「这是你第三次决策」——那种错不会报错，只会给出可信的假话。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from holdem.actions import call, check, fold, raise_to  # noqa: E402
from holdem.deck import deck_from_seed  # noqa: E402
from holdem.preflop_policy import PreflopTablePolicy  # noqa: E402
from holdem.preflop_review import review_preflop  # noqa: E402
from holdem.state import HandConfig, HandState  # noqa: E402

pytestmark = pytest.mark.skipif(
    not PreflopTablePolicy.available(), reason="没有翻前范围表"
)


def _play(actions, *, button=0, seed=11):
    hand = HandState(
        HandConfig(stacks=(1000,) * 6, button=button, big_blind=10, small_blind=5),
        deck_from_seed(seed),
    )
    for action in actions:
        hand.apply(action)
    while not hand.is_complete:
        hand.apply(check() if hand.legal_actions().can_check else fold())
    return hand


def test_an_unfinished_hand_is_refused():
    """复盘的对象是**打完的**牌。"""
    hand = HandState(
        HandConfig(stacks=(1000,) * 6, button=0, big_blind=10, small_blind=5),
        deck_from_seed(11),
    )
    hand.apply(raise_to(30))
    with pytest.raises(ValueError, match="还没打完"):
        review_preflop(hand, hero_seat=3)


def test_only_the_heros_decisions_come_back():
    """别人的动作是背景，不是英雄的决策——混进来会让「第几次说话」全错位。"""
    hand = _play([raise_to(30), fold(), fold(), fold(), fold(), call()])
    steps = review_preflop(hand, hero_seat=3)
    assert len(steps) == 1
    assert steps[0].position == "UTG" and steps[0].index == 1


def test_multiple_decisions_are_numbered_in_order():
    """英雄开牌、被 3bet、再决定——三次说话里他做了两次决策。"""
    hand = _play([raise_to(30), fold(), fold(), raise_to(90), fold(), fold(), fold()])
    steps = review_preflop(hand, hero_seat=3)
    assert [s.index for s in steps] == [1, 2]
    assert steps[0].to_call == 10, "开牌时面对的是大盲"
    assert steps[1].to_call > 0, "第二次是面对 3bet"


def test_the_context_matches_the_moment_not_the_end():
    """底池与待跟额记的是**那一刻**的，不是终局的。

    这条盯的是重放：拿终局的数字填进去也不会报错，但每个决策点都会显示同一个底池。
    """
    hand = _play([raise_to(30), fold(), fold(), raise_to(90), fold(), fold(), fold()])
    steps = review_preflop(hand, hero_seat=3)
    assert steps[0].pot_before < steps[1].pot_before, "两次决策的底池必须不同"


def test_a_clear_mistake_is_called_out():
    # 五个人弃完牌局就结束了（大盲赢），别再多写一个动作
    hand = _play([fold()] * 5, seed=11)
    steps = review_preflop(hand, hero_seat=3)
    assert steps and steps[0].verdict is not None
    # UTG 拿到这副种子发的牌弃掉：判词要么是照解走要么是明显错误，但必须有判词
    assert steps[0].graded


def test_grading_stays_none_when_the_table_has_no_such_cell():
    """**说不了就说不了**：表里没这一格时 `verdict` 是 None，不编一个分数。"""
    class _Empty:
        def decide(self, hand, **kwargs):
            return None

    hand = _play([raise_to(30), fold(), fold(), fold(), fold(), call()])
    steps = review_preflop(hand, hero_seat=3, policy=_Empty())
    assert steps and all(step.verdict is None and not step.graded for step in steps)


def test_replay_covers_every_preflop_action_of_the_hand():
    """重放要走完整条翻前线路——中途断了后面的判卷就落在错误的局面上。"""
    hand = _play([call(), call(), fold(), fold(), fold(), check()])
    steps = review_preflop(hand, hero_seat=3)
    assert steps, "UTG 跛入，该有一次决策"
    assert steps[0].action.startswith("call") or "call" in steps[0].action.lower()

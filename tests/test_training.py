"""场景训练（FR-12）：造题与判卷。

测的重点是那几条**错了不会报错**的：造出来的是不是想要的场景、
判卷有没有把非法动作也评一遍、以及「说不了就说不了」有没有被将就掉。
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from holdem.actions import call, fold, raise_to  # noqa: E402
from holdem.preflop_policy import DEFEND, OPEN, VS_RERAISE, PreflopTablePolicy  # noqa: E402
from holdem.training import (  # noqa: E402
    deal_defend, deal_open, deal_threebet, grade,
)

pytestmark = pytest.mark.skipif(
    not PreflopTablePolicy.available(), reason="没有翻前范围表"
)


def _rng():
    return random.Random(20260822)


# ------------------------------------------------------------------ 造题


def test_each_maker_produces_the_scenario_it_says_it_does():
    """**造出来的必须正是想要的那个场景。**

    造题用 `identify` 反查一遍，对不上就抛错——将就的结果是练了半天练的是
    另一个场景，而界面上还写着原来那个名字。
    """
    rng = _rng()
    assert deal_open("UTG", rng=rng).spot.kind == OPEN
    assert deal_defend("BB", "BTN", rng=rng).spot.kind == DEFEND
    assert deal_threebet("UTG", "BTN", rng=rng).spot.kind == VS_RERAISE


def test_the_hero_is_the_one_to_act():
    """题目必须停在英雄该说话的那一刻，不能多走也不能少走。"""
    rng = _rng()
    for spot in (deal_open("CO", rng=rng), deal_defend("SB", "CO", rng=rng)):
        assert spot.hand.to_act == spot.hero_seat


def test_the_villain_actually_did_what_the_scenario_claims():
    """「面对 BTN 开牌」得真的是 BTN 加的注，不是随便谁。"""
    spot = deal_defend("BB", "BTN", rng=_rng())
    assert spot.spot.opener == "BTN"
    assert spot.hand.pot_size > 15, "有人加过注，底池不该只有两个盲注"


def test_repeated_deals_give_different_cards_but_the_same_scenario():
    """反复练的意义就在这儿：场景固定、牌面变。"""
    rng = _rng()
    spots = [deal_open("UTG", rng=rng) for _ in range(8)]
    holes = {tuple(sorted(s.hand.hole[s.hero_seat])) for s in spots}
    assert len(holes) > 1, "每次都发同一手牌就不是练习了"
    assert all(s.spot.kind == OPEN for s in spots)


def test_an_unknown_position_is_rejected():
    with pytest.raises(ValueError, match="没有这个位置"):
        deal_open("楼上", rng=_rng())


# ------------------------------------------------------------------ 判卷


def test_an_illegal_action_is_refused_not_graded():
    """**判卷的前提是这确实是个可选项**，否则评的不是决策、是笔误。

    面对 2.5bb 开牌时「加注到 2.5bb」根本打不出来，却很容易被写出来——
    不查的话用户会拿到一句像模像样的判词。
    """
    spot = deal_defend("BB", "BTN", rng=_rng())
    with pytest.raises(ValueError, match="不合法"):
        grade(spot.hand, raise_to(25))


def test_a_reasonable_open_is_on_the_solution():
    """UTG 拿到强牌开牌，应当判「照解走」；弃掉同一手牌应当判「明显错误」。"""
    rng = random.Random(7)
    spot = deal_open("UTG", rng=rng)          # 这个种子发的是 AQo
    opened = grade(spot.hand, raise_to(25))
    folded = grade(spot.hand, fold())
    assert opened.on_solution and not opened.blunder
    assert folded.blunder


def test_the_verdict_carries_the_whole_distribution():
    """界面要画条形图，判词之外得给完整分布。"""
    spot = deal_open("BTN", rng=_rng())
    verdict = grade(spot.hand, fold())
    assert verdict.weights and abs(sum(verdict.weights.values()) - 1.0) < 1e-6
    assert verdict.best in verdict.weights


def test_an_action_the_table_does_not_have_scores_zero_not_the_nearest_one():
    """**别把对不上的动作硬塞进最近的那一档。**

    塞了之后，一个解里根本没有的尺度会拿到别人的频率，判词就变成假的。
    """
    spot = deal_open("UTG", rng=_rng())
    legal = spot.hand.legal_actions()
    weird = grade(spot.hand, raise_to(min(legal.max_raise_to, legal.min_raise_to + 137)))
    assert weird.frequency == 0.0
    assert weird.blunder


def test_grading_says_nothing_when_the_table_has_no_such_cell():
    """表里没有这一格就返回 None——**说不了就说不了**，不编一个分数出来。"""
    spot = deal_open("UTG", rng=_rng())

    class _Empty:
        def decide(self, hand, **kwargs):
            return None

    assert grade(spot.hand, fold(), policy=_Empty()) is None


def test_frequency_and_verdict_agree():
    """三档判词必须与频率一致，不能各说各话。"""
    rng = _rng()
    for _ in range(12):
        spot = deal_defend("BB", "CO", rng=rng)
        for action in (fold(), call()):
            verdict = grade(spot.hand, action)
            if verdict is None:
                continue
            if verdict.blunder:
                assert verdict.frequency < 0.02 and "明显错误" in verdict.verdict
            elif verdict.on_solution:
                assert "照解走" in verdict.verdict
            else:
                assert "次优" in verdict.verdict

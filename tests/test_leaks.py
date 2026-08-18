"""漏洞报告的测试（FR-10）。

这一层不解牌也不算 EV，它只做一件事：**把逐点的 EV 损失归到对的格子里，再排对顺序**。
所以测试盯的是「归错格」与「排错序」——两者都会让人照着报告去改错东西：

1. **角色按翻前最后加注的人分**：进攻方的翻牌下注是持续下注，防守方的是领打，
   混进一格等于把信号平均掉。
2. **排序按总漏损，不按平均**：平均值最大的常常是一年遇不上几次的局面。
3. **覆盖率要自曝**：按 12 个决策算出来的报告和按 1200 个算出来的长得一模一样。
"""

import pytest

from holdem.cards import card_from_str, cards_from_str
from holdem.range_tracking import FlopRanges
from holdem.ranges import Range
from holdem.state import FLOP, RIVER, TURN
from holdem_solver.evaluate import DecisionScore
from holdem_solver.leaks import (
    AGGRESSOR,
    DEFENDER,
    FACING_BET,
    FACING_RAISE,
    IN_POSITION,
    NO_BET,
    OUT_OF_POSITION,
    build_report,
    scenario_of,
)
from holdem_solver.request import SolveRequest
from holdem_solver.review import DecisionPoint, ReviewPlan, ReviewResult, ScoredDecision, Step

HERO_SEAT, VILLAIN_SEAT = 0, 2
HERO_CARDS = (card_from_str("Kh"), card_from_str("Kd"))


def point(*, street=FLOP, prefix=(), kind="bet", amount=3.0, hero=1, seat=HERO_SEAT):
    return DecisionPoint(
        street=street,
        seat=seat,
        hero=hero,
        hero_cards=HERO_CARDS,
        prefix=tuple(prefix),
        taken=Step(kind=kind, seat=seat, amount=amount, street=street),
    )


def scored(decision_point, loss, *, in_range=True):
    """造一个「已经打上分」的决策：最优值 0，实际动作亏 `loss`。"""
    score = DecisionScore(
        evs={"CHECK": 0.0, "BET 3.0": -loss},
        strategy={"CHECK": 1.0, "BET 3.0": 0.0},
        taken="BET 3.0",
    )
    return ScoredDecision(
        point=decision_point, score=score, label="BET 3.0", in_range=in_range
    )


def result(decisions, *, aggressor_seat=HERO_SEAT):
    setup = FlopRanges(
        oop_seat=VILLAIN_SEAT,
        ip_seat=HERO_SEAT,
        oop=Range.parse("AA"),
        ip=Range.parse("KK"),
        pot=6.0,
        effective_stack=20.0,
        line="单次加注底池",
        aggressor_seat=aggressor_seat,
    )
    plan = ReviewPlan(
        request=SolveRequest(
            board=tuple(cards_from_str("Qs7h2c")),
            pot=6.0,
            effective_stack=20.0,
            oop_range=setup.oop,
            ip_range=setup.ip,
        ),
        setup=setup,
        hero_seat=HERO_SEAT,
        points=tuple(d.point for d in decisions),
    )
    return ReviewResult(plan=plan, decisions=tuple(decisions))


CHECKED = Step(kind="check", seat=VILLAIN_SEAT, street=FLOP)
BET = Step(kind="bet", seat=VILLAIN_SEAT, amount=3.0, street=FLOP)
RAISED = Step(kind="raise", seat=HERO_SEAT, amount=9.0, street=FLOP)


# ------------------------------------------------------------------ 归类


def test_the_classic_spot_gets_its_common_name():
    """翻前加注的人、有位置、翻牌无人下注——这就是持续下注。"""
    scenario = scenario_of(point(prefix=[CHECKED]), aggressor_seat=HERO_SEAT)
    assert (scenario.street, scenario.role) == ("翻牌", AGGRESSOR)
    assert (scenario.position, scenario.facing) == (IN_POSITION, NO_BET)
    assert scenario.alias == "持续下注"
    assert "持续下注" in scenario.title and "翻牌·进攻方" in scenario.title


def test_the_same_bet_is_a_different_scenario_for_the_defender():
    """同一个翻牌下注，防守方打出来是领打——**角色按翻前最后加注的人分**。"""
    scenario = scenario_of(point(prefix=[]), aggressor_seat=VILLAIN_SEAT)
    assert scenario.role == DEFENDER
    assert scenario.alias == "领打机会"


def test_facing_a_bet_and_facing_a_raise_are_told_apart():
    facing_bet = scenario_of(point(prefix=[BET]), aggressor_seat=VILLAIN_SEAT)
    facing_raise = scenario_of(
        point(prefix=[CHECKED, BET, RAISED]), aggressor_seat=VILLAIN_SEAT
    )
    assert facing_bet.facing == FACING_BET
    assert facing_raise.facing == FACING_RAISE


def test_last_street_action_does_not_leak_into_this_street():
    """上一街的下注不算数：转牌重新开始数，否则每个转牌决策都成了「面对下注」。"""
    prefix = [BET, Step(kind="call", seat=HERO_SEAT, street=FLOP),
              Step(kind="deal", card=card_from_str("9d"), street=TURN)]
    scenario = scenario_of(point(street=TURN, prefix=prefix), aggressor_seat=VILLAIN_SEAT)
    assert scenario.street == "转牌" and scenario.facing == NO_BET


def test_position_follows_the_solver_index():
    assert scenario_of(point(hero=0), aggressor_seat=HERO_SEAT).position == OUT_OF_POSITION
    assert scenario_of(point(hero=1), aggressor_seat=HERO_SEAT).position == IN_POSITION


# ------------------------------------------------------------------ 聚合与排序


def test_scenarios_are_ranked_by_total_loss_not_by_average():
    """一次亏 5bb 的罕见局面，排在十次各亏 1bb 的常见局面**后面**——钱是后者漏的。"""
    rare = scored(point(street=RIVER, prefix=[BET]), 5.0)
    common = [scored(point(prefix=[CHECKED]), 1.0) for _ in range(10)]
    report = build_report([result([rare, *common])], hands=100)

    assert report.leaks[0].scenario.alias == "持续下注"
    assert report.leaks[0].total_loss == pytest.approx(10.0)
    assert report.leaks[0].mean_loss == pytest.approx(1.0)
    assert report.leaks[1].total_loss == pytest.approx(5.0)
    assert report.total_loss == pytest.approx(15.0)


def test_the_leak_rate_is_measured_against_hands_looked_at():
    """「每 100 手漏多少」看的是**看过的手数**，不是打上分的决策数。"""
    report = build_report([result([scored(point(prefix=[CHECKED]), 2.0)])], hands=50)
    assert report.per_100_hands(report.total_loss) == pytest.approx(4.0)


def test_unscorable_spots_are_counted_not_swallowed():
    """打不了分的点不进任何场景，但必须逐条记下来——覆盖率就是报告的可信度。"""
    good = scored(point(prefix=[CHECKED]), 1.0)
    bad = ScoredDecision(point=point(prefix=[CHECKED]), skipped="尺度不在树里")
    report = build_report([result([good, bad, bad])], hands=10)

    assert report.scored_spots == 1
    assert report.skipped == (("尺度不在树里", 2),)
    assert report.coverage == pytest.approx(1 / 3)
    assert sum(leak.spots for leak in report.leaks) == 1


def test_hands_that_never_made_it_to_a_solve_are_reported_too():
    """连计划都做不出来的手牌（跛入底池、多人底池）也要写进报告。"""
    report = build_report(
        [result([scored(point(prefix=[CHECKED]), 1.0)])],
        hands=10,
        uncovered=["翻牌时还有 3 个人", "翻牌时还有 3 个人", "没人加注（跛入底池）"],
    )
    assert report.uncovered_hands[0] == ("翻牌时还有 3 个人", 2)
    assert report.reviewed_hands == 7


def test_hands_outside_the_assumed_range_are_counted_apart():
    """风格层打的表外牌照样算 EV，但要单独标出来，别跟正常点一起解读。"""
    inside = scored(point(prefix=[CHECKED]), 1.0)
    outside = scored(point(prefix=[CHECKED]), 3.0, in_range=False)
    report = build_report([result([inside, outside])], hands=10)

    (leak,) = report.leaks
    assert leak.spots == 2 and leak.off_range_spots == 1
    assert leak.total_loss == pytest.approx(4.0)


def test_a_negative_loss_does_not_create_money():
    """解没收敛干净时会冒出极小的负差。那不是「打得比解还好」，按 0 算。"""
    better_than_solved = ScoredDecision(
        point=point(prefix=[CHECKED]),
        score=DecisionScore(evs={"CHECK": 0.0, "BET 3.0": 1e-6}, strategy={}, taken="BET 3.0"),
        label="BET 3.0",
    )
    report = build_report([result([better_than_solved])], hands=10)
    assert report.total_loss == pytest.approx(0.0)


def test_an_empty_batch_is_a_valid_report():
    report = build_report([], hands=0)
    assert report.leaks == () and report.coverage == 0.0 and report.total_loss == 0.0
    assert report.per_100_hands(0.0) == 0.0

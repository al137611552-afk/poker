"""推弃纳什均衡的测试。

主验证不依赖任何外部图表：解出来之后直接算**可利用度**——双方各自最佳应对还能多赚多少。
可利用度趋近于零，就说明这确实是均衡。公开的纳什表只用来对个大数，防止整体跑偏。
"""

import pytest

from holdem import equity_table
from holdem.pushfold import solve_push_fold
from holdem.ranges import class_from_name

pytestmark = pytest.mark.skipif(
    not equity_table.is_available(),
    reason="翻前权益表尚未生成，先跑 scripts/build_preflop_equity.py",
)


@pytest.fixture(scope="module")
def solution_10bb():
    return solve_push_fold(10.0)


# ------------------------------------------------------------------ 自证


def test_solution_is_an_equilibrium(solution_10bb):
    """核心验证：可利用度接近零。"""
    assert solution_10bb.exploitability < 1e-3, (
        f"可利用度 {solution_10bb.exploitability:.5f} bb/手，未收敛到均衡"
    )


@pytest.mark.parametrize("stack", [3.0, 6.0, 15.0, 20.0])
def test_equilibrium_holds_across_stack_depths(stack):
    solution = solve_push_fold(stack)
    assert solution.exploitability < 2e-3, (
        f"{stack}bb 的可利用度 {solution.exploitability:.5f} bb/手过高"
    )


def test_zero_ev_hands_are_not_forced_either_way():
    """均衡里可以有混合频率，但权重必须落在 [0, 1]。"""
    solution = solve_push_fold(12.0)
    for weight in list(solution.push.weights.values()) + list(solution.call.weights.values()):
        assert 0.0 <= weight <= 1.0


# ------------------------------------------------------------------ 结构性质


def test_ranges_widen_as_stacks_shrink():
    deep = solve_push_fold(20.0)
    mid = solve_push_fold(10.0)
    shallow = solve_push_fold(4.0)
    assert deep.push_percent < mid.push_percent < shallow.push_percent, (
        f"全下范围应随筹码变浅而变宽："
        f"{deep.push_percent:.1%} / {mid.push_percent:.1%} / {shallow.push_percent:.1%}"
    )
    assert deep.call_percent < mid.call_percent < shallow.call_percent


def test_premium_hands_are_always_in(solution_10bb):
    for name in ["AA", "KK", "AKs"]:
        index = class_from_name(name)
        assert solution_10bb.push.weight(index) == pytest.approx(1.0, abs=0.02), name
        assert solution_10bb.call.weight(index) == pytest.approx(1.0, abs=0.02), name


def test_trash_is_out_of_the_calling_range_at_ten_bb(solution_10bb):
    for name in ["72o", "32o", "82o"]:
        index = class_from_name(name)
        assert solution_10bb.call.weight(index) < 0.05, (
            f"{name} 在 10bb 不该跟全下"
        )


def test_calling_range_is_tighter_than_pushing_range(solution_10bb):
    assert solution_10bb.call_percent < solution_10bb.push_percent, (
        "小盲有弃牌赢利，全下范围必然宽于大盲的跟注范围"
    )


def test_very_short_stacks_approach_any_two_cards():
    solution = solve_push_fold(1.5)
    assert solution.push_percent > 0.95, f"1.5bb 时几乎应全推，实测 {solution.push_percent:.1%}"
    assert solution.call_percent > 0.9, f"1.5bb 时几乎应全跟，实测 {solution.call_percent:.1%}"


def test_ante_widens_both_ranges():
    without = solve_push_fold(10.0)
    with_ante = solve_push_fold(10.0, ante=0.125)
    assert with_ante.push_percent > without.push_percent, "有前注时死钱更多，范围应更宽"
    assert with_ante.call_percent > without.call_percent


# ------------------------------------------------------------------ 与公开数值对照


def test_ten_bb_matches_published_nash_percentages(solution_10bb):
    """公开的单挑纳什表：10bb 时按钮全下约 54%、大盲跟注约 36%。

    容差放到 6 个百分点：我们的权益表是蒙特卡洛估计，而不同来源的公开值本身
    也有几个点的出入。
    """
    assert solution_10bb.push_percent == pytest.approx(0.54, abs=0.06), (
        f"10bb 全下范围 {solution_10bb.push_percent:.1%}，公开值约 54%"
    )
    assert solution_10bb.call_percent == pytest.approx(0.36, abs=0.06), (
        f"10bb 跟注范围 {solution_10bb.call_percent:.1%}，公开值约 36%"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "hand, published",
    [("72o", 2.6), ("87s", 6.5), ("K2s", 10.7), ("22", 15.0)],
)
def test_call_thresholds_match_published_per_hand_values(hand, published):
    """逐手门槛对照 HoldemResources 的公开纳什表——比总百分比严格得多。

    只选了四个能可靠读到的值。同一来源里 A2o / Q9s / J8s 的数字与直接算出的 EV 明显
    矛盾（例如 10bb 时 A2o 对上小盲的全下范围有约 50% 权益、跟注只需 45%，公开值却说
    8.1bb 就该弃），判定为抓取网页表格时读串了，不作为基准。
    """
    index = class_from_name(hand)
    low, high = 1.0, 20.0
    assert solve_push_fold(low).call.weight(index) >= 0.5, f"{hand} 在 1bb 就该跟注"
    for _ in range(8):
        middle = (low + high) / 2
        if solve_push_fold(round(middle, 2)).call.weight(index) >= 0.5:
            low = middle
        else:
            high = middle
    threshold = (low + high) / 2
    assert threshold == pytest.approx(published, abs=0.8), (
        f"{hand} 的跟注门槛实测 {threshold:.1f}bb，公开值 {published}bb"
    )


# ------------------------------------------------------------------ 入参


def test_input_validation():
    with pytest.raises(ValueError, match="有效筹码"):
        solve_push_fold(0)
    with pytest.raises(ValueError, match="前注"):
        solve_push_fold(10, ante=-1)

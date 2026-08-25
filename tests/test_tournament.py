"""锦标赛结构与 ICM（FR-13）。

ICM 的实现错了之后**每个数看着都还挺合理**，所以测试盯两样：
能手算的算例逐位对上，以及**守恒**（所有人的期望加起来必须等于奖池）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from holdem.tournament import (  # noqa: E402
    TURBO_SNG_9, BlindLevel, Structure, finish_distribution, icm_equity, icm_pressure,
)

PAY_3 = (0.50, 0.30, 0.20)


# ------------------------------------------------------------------ 手算对照


def test_heads_up_icm_is_just_the_chip_share_across_two_prizes():
    """两个人时可以直接手算：拿第一的概率就是筹码占比。"""
    equity = icm_equity({"A": 80, "B": 20}, (0.70, 0.30), prize_pool=100)
    assert equity["A"] == pytest.approx(0.8 * 70 + 0.2 * 30)
    assert equity["B"] == pytest.approx(0.2 * 70 + 0.8 * 30)


def test_three_way_matches_the_hand_computed_value():
    """筹码 60/30/10、奖金 50/30/20：

    P(A 第一) = 0.6
    P(A 第二) = 0.3×60/70 + 0.1×60/90 = 0.3238
    A 的 ICM = 0.6×50 + 0.3238×30 + 0.0762×20 = 41.24
    """
    stacks = {"A": 60, "B": 30, "C": 10}
    chances = finish_distribution(stacks, 3)["A"]
    assert chances[0] == pytest.approx(0.6)
    assert chances[1] == pytest.approx(0.3238, abs=1e-4)
    assert icm_equity(stacks, PAY_3, 100)["A"] == pytest.approx(41.24, abs=0.01)


def test_equal_stacks_split_the_pool_evenly():
    equity = icm_equity({"A": 100, "B": 100, "C": 100}, PAY_3, 300)
    assert all(value == pytest.approx(100.0) for value in equity.values())


# ------------------------------------------------------------------ 守恒


def test_the_whole_prize_pool_is_always_accounted_for():
    """**不守恒说明模型或实现错了**，而错了之后每个数看着都还挺合理。"""
    for stacks in (
        {"A": 60, "B": 30, "C": 10},
        {"A": 1, "B": 1, "C": 1, "D": 997},
        {"A": 500, "B": 300, "C": 150, "D": 50, "E": 1},
    ):
        equity = icm_equity(stacks, PAY_3, prize_pool=1000)
        assert sum(equity.values()) == pytest.approx(1000.0, abs=1e-6), stacks


def test_finish_probabilities_sum_to_one_for_each_place():
    """每个名次都必须正好被一个人拿走。"""
    distribution = finish_distribution({"A": 60, "B": 30, "C": 10, "D": 5}, 3)
    for place in range(3):
        total = sum(chances[place] for chances in distribution.values())
        assert total == pytest.approx(1.0, abs=1e-9), place


# ------------------------------------------------------------------ 出局的人


def test_busted_players_are_excluded_but_still_reported():
    """筹码 0 的人不该稀释别人的概率，但**键要在**——少一个键会让调用方 KeyError。

    **守恒的是「活人还能分的那部分」**：三人赛只剩两人时，第三名已经落定，
    剩下两人分的是前两名那 80%，不是全部 100%。第一版在这儿断错过。
    """
    equity = icm_equity({"A": 60, "B": 40, "C": 0}, PAY_3, 100)
    assert equity["C"] == 0.0, "0 表示不在此计算内，不是「他没拿到奖金」"
    assert sum(equity.values()) == pytest.approx(80.0), "两个活人分前两名的 50+30"
    assert "C" not in finish_distribution({"A": 60, "B": 40, "C": 0}, 3)


def test_everyone_busted_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError, match="没有还有筹码"):
        icm_equity({"A": 0, "B": 0}, PAY_3)


# ------------------------------------------------------------------ 泡沫圈的不对称


def test_the_short_stack_risks_far_more_than_it_gains():
    """**这就是 ICM 存在的理由。**

    泡沫圈上短筹码翻倍只多拿一点点，输光却直接归零。筹码 EV 看不到这个不对称，
    所以它在钱圈附近会给出错的建议。
    """
    gain, loss = icm_pressure({"A": 60, "B": 30, "C": 10}, PAY_3, "C", prize_pool=100)
    assert gain > 0 and loss > 0
    assert loss > gain * 3, "短筹码的下行远大于上行"


def test_doubling_up_is_worth_far_less_than_double():
    """**锦标赛的筹码不是钱**：筹码翻倍，奖金期望远远涨不到一倍。

    实测：A 从 60 到 120（+100%）ICM 只涨 8.5%；C 从 10 到 20 只涨 13.6%。
    """
    stacks = {"A": 60, "B": 30, "C": 10}
    now = icm_equity(stacks, PAY_3, 100)
    for who in ("A", "C"):
        doubled = dict(stacks)
        doubled[who] *= 2
        after = icm_equity(doubled, PAY_3, 100)[who]
        assert after > now[who], "多筹码总归是好事"
        assert after < now[who] * 1.5, f"{who} 的 ICM 涨幅远小于筹码涨幅"


def test_chips_are_worth_more_to_a_short_stack():
    """**边际价值递减**：同样多拿 10 个筹码，短筹码赚到的奖金期望多得多。

    第一版我在这儿断错过：以为「短筹码的损失/收益比更大」，
    实测反而是大筹码更大（11.8 vs 7.35）——因为大筹码本来 ICM 值就高，
    出局当然亏得多。**那个比值根本不是「压力」的正确度量。**
    """
    stacks = {"A": 60, "B": 30, "C": 10}
    now = icm_equity(stacks, PAY_3, 100)
    gains = {}
    for who in ("A", "C"):
        richer = dict(stacks)
        richer[who] += 10
        gains[who] = icm_equity(richer, PAY_3, 100)[who] - now[who]
    assert gains["C"] > gains["A"] * 2


def test_pressure_needs_a_live_player():
    with pytest.raises(ValueError, match="不在牌局里"):
        icm_pressure({"A": 60, "B": 40}, PAY_3, "C")


# ------------------------------------------------------------------ 结构


def test_payouts_must_add_up():
    with pytest.raises(ValueError, match="加起来应当是 1"):
        Structure(name="坏的", starting_stack=1500,
                  levels=(BlindLevel(10, 20),), payouts=(0.5, 0.3), entrants=9)


def test_cannot_pay_more_places_than_entrants():
    with pytest.raises(ValueError, match="比参赛人数还多"):
        Structure(name="坏的", starting_stack=1500, levels=(BlindLevel(10, 20),),
                  payouts=(0.5, 0.3, 0.2), entrants=2)


def test_the_level_at_a_given_minute():
    structure = TURBO_SNG_9
    assert structure.level_at(0).big_blind == 20
    assert structure.level_at(7).big_blind == 30, "第 6 分钟升级"
    assert structure.level_at(10_000) is structure.levels[-1], "超出盲注表就停在最后一级"


def test_stack_in_big_blinds_follows_the_clock():
    """同样的筹码，越往后越不值钱——这正是锦标赛的压力来源。"""
    structure = TURBO_SNG_9
    assert structure.stack_in_bb(1500, 0) == pytest.approx(75.0)
    assert structure.stack_in_bb(1500, 30) < 20.0


def test_cost_per_orbit_includes_the_ante():
    level = BlindLevel(75, 150, ante=15)
    assert level.cost_per_orbit == 240


# ------------------------------------------------------------------ ICM 口径的推弃


def _percent(hand_range) -> float:
    from holdem.ranges import TOTAL_COMBOS, class_combo_count

    return sum(
        hand_range.weight(i) * class_combo_count(i) for i in range(169)
    ) / TOTAL_COMBOS * 100


def _solve_both(stacks, hero="hero", villain="villain", big_blind=200):
    from holdem.pushfold import solve_push_fold
    from holdem.tournament import icm_push_fold_payoffs

    effective = min(stacks[hero], stacks[villain]) / big_blind
    chips = solve_push_fold(effective, iterations=300)
    payoffs = icm_push_fold_payoffs(
        stacks, PAY_3, hero=hero, villain=villain, big_blind=big_blind, prize_pool=100
    )
    return chips, solve_push_fold(effective, iterations=300, payoffs=payoffs)


def test_icm_pushes_and_calls_tighter_than_chip_ev():
    """**这就是接 ICM 的全部意义**：同一个局面，两个口径给出不同的范围。

    泡沫圈上输了直接归零、赢了只多拿一点，所以 ICM 口径必然更紧。
    """
    chips, icm = _solve_both({"hero": 1000, "villain": 4000, "big": 5000, "other": 5000})
    assert _percent(icm.push) < _percent(chips.push)
    assert _percent(icm.call) < _percent(chips.call)


def test_the_closer_to_the_money_the_tighter_icm_gets():
    """人越少、奖金越近，出局的代价越大。"""
    _, bubble = _solve_both({"hero": 1000, "villain": 4000, "big": 5000, "other": 5000})
    _, in_money = _solve_both({"hero": 1000, "villain": 4000, "big": 5000})
    assert _percent(in_money.push) < _percent(bubble.push)


def test_the_single_hand_icm_bias_is_documented_not_hidden():
    """**单手 ICM 会紧到离谱**（实测三人都有奖时推范围掉到 11.7%），

    因为它假定「不打就没事」——而真实锦标赛里不打也在被盲注吃。
    这条偏差必须写在模块文档里：照这个数直接当开牌表会输死。
    """
    from holdem import tournament

    doc = tournament.icm_push_fold_payoffs.__doc__
    assert "偏紧" in doc and "盲注" in doc
    assert "不适合直接当作开牌表照抄" in doc

    _, in_money = _solve_both({"hero": 1000, "villain": 4000, "big": 5000})
    assert _percent(in_money.push) < 25.0, "偏差确实存在，不是文档里的假设"


def test_icm_payoffs_are_asymmetric_where_chip_payoffs_are_not():
    """筹码口径下赢 +S / 输 −S 是对称的；ICM 下输得更多——不对称正是它的价值。"""
    from holdem.pushfold import chip_payoffs
    from holdem.tournament import icm_push_fold_payoffs

    chips = chip_payoffs(5.0, 0.0)
    assert chips.sb_win == pytest.approx(-chips.sb_lose)

    icm = icm_push_fold_payoffs(
        {"hero": 1000, "villain": 4000, "big": 5000, "other": 5000},
        PAY_3, hero="hero", villain="villain", big_blind=200, prize_pool=100,
    )
    assert abs(icm.sb_lose) > icm.sb_win, "输的代价大于赢的收益"


def test_both_sides_must_still_have_chips():
    from holdem.tournament import icm_push_fold_payoffs

    with pytest.raises(ValueError, match="都得还有筹码"):
        icm_push_fold_payoffs({"hero": 0, "villain": 4000}, PAY_3,
                              hero="hero", villain="villain", big_blind=200)

"""权益兑现模型的测试。

这个模型的参数是**假设**（见 `realization.py` 的 docstring），所以这里只验**性质**：
守恒、单调、退化。把某个具体数字钉成基准等于把假设当成事实。
"""

import pytest

from holdem import equity_table
from holdem.equity_table import preflop_equity
from holdem.ranges import NUM_HAND_CLASSES, class_from_name
from holdem.realization import (
    ALL_IN_EQUITY,
    RealizationModel,
    flop_share_matrix,
    realization_factors,
    removal_rows,
)

pytestmark = pytest.mark.skipif(
    not equity_table.is_available(),
    reason="翻前权益表尚未生成，先跑 scripts/build_preflop_equity.py",
)

SAMPLE = ["AA", "KK", "AKs", "AKo", "QQ", "77", "A5s", "76s", "T9o", "J8o", "72o", "32o"]


def share(matrix, hero: str, villain: str) -> float:
    return matrix[class_from_name(hero) * NUM_HAND_CLASSES + class_from_name(villain)]


# ------------------------------------------------------------------ 守恒


def test_shares_of_both_sides_add_to_one():
    """底池份额必须守恒：谁多拿一分，对面就少一分。终局零和是求解器的地基。"""
    hero_ip = flop_share_matrix(RealizationModel(), hero_in_position=True, spr=10.0)
    hero_oop = flop_share_matrix(RealizationModel(), hero_in_position=False, spr=10.0)
    for hero in SAMPLE:
        for villain in SAMPLE:
            total = share(hero_ip, hero, villain) + share(hero_oop, villain, hero)
            assert total == pytest.approx(1.0, abs=1e-6), f"{hero} vs {villain} 份额不守恒"


def test_shares_stay_inside_the_pot():
    matrix = flop_share_matrix(RealizationModel(), hero_in_position=True, spr=10.0)
    assert all(0.0 <= value <= 1.0 for value in matrix)


# ------------------------------------------------------------------ 退化


def test_degenerates_to_all_in_equity():
    """关掉全部修正项之后，份额就是全下权益——模型的下界行为。"""
    matrix = flop_share_matrix(ALL_IN_EQUITY, hero_in_position=True, spr=10.0)
    for hero in SAMPLE:
        for villain in SAMPLE:
            expected = preflop_equity(class_from_name(hero), class_from_name(villain))
            assert share(matrix, hero, villain) == pytest.approx(expected, abs=1e-6)


def test_no_modifiers_at_zero_spr():
    """底池后面没筹码了，隐含赔率无从谈起，牌型修正全部归零。"""
    model = RealizationModel()
    factors = realization_factors(model, in_position=True, spr=0.0)
    assert set(factors) == {model.in_position}


def test_modifiers_saturate_beyond_reference_spr():
    model = RealizationModel()
    at_reference = realization_factors(model, in_position=True, spr=model.reference_spr)
    much_deeper = realization_factors(model, in_position=True, spr=model.reference_spr * 10)
    assert at_reference == much_deeper


# ------------------------------------------------------------------ 单调


def test_position_beats_no_position():
    model = RealizationModel()
    ip = realization_factors(model, in_position=True, spr=10.0)
    oop = realization_factors(model, in_position=False, spr=10.0)
    assert all(a > b for a, b in zip(ip, oop))


def test_hand_type_modifiers_go_the_right_way():
    factors = realization_factors(RealizationModel(), in_position=True, spr=10.0)

    def value(name: str) -> float:
        return factors[class_from_name(name)]

    assert value("76s") > value("76o"), "同花有听牌潜力，兑现更好"
    assert value("88") > value("83o"), "对子能中暗三条"
    assert value("76o") > value("KQo"), "高张不同花容易被压制，反向隐含赔率"
    assert value("76o") > value("72o"), "连张能凑顺子"


def test_sharpening_pulls_strong_and_weak_apart():
    """γ 越大，强牌拿到的份额越高、弱牌越低；权益本身不变。"""
    flat = RealizationModel(sharpening=0.0)
    sharp = RealizationModel(sharpening=0.8)
    flat_matrix = flop_share_matrix(flat, hero_in_position=True, spr=20.0)
    sharp_matrix = flop_share_matrix(sharp, hero_in_position=True, spr=20.0)

    assert share(sharp_matrix, "AA", "72o") > share(flat_matrix, "AA", "72o")
    assert share(sharp_matrix, "72o", "AA") < share(flat_matrix, "72o", "AA")


def test_equal_hands_split_when_positions_match():
    """同一手牌对同一手牌、位置修正相同时，份额恰好一半。"""
    model = RealizationModel(in_position=1.0, out_of_position=1.0)
    matrix = flop_share_matrix(model, hero_in_position=True, spr=10.0)
    for name in SAMPLE:
        assert share(matrix, name, name) == pytest.approx(0.5, abs=1e-9)


# ------------------------------------------------------------------ 共牌权重


def test_removal_rows_sum_to_the_number_of_villain_combos():
    """拿走两张牌之后对手恰好有 C(50,2)=1225 种组合，与我方牌型无关。"""
    rows = removal_rows()
    for index in (0, 42, 168):
        assert sum(rows[index]) == pytest.approx(1225.0, abs=1e-6)


# ------------------------------------------------------------------ 入参


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"in_position": 0}, "位置基准"),
        ({"out_of_position": -1}, "位置基准"),
        ({"sharpening": -0.1}, "锐化"),
        ({"reference_spr": 0}, "参考 SPR"),
    ],
)
def test_model_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RealizationModel(**kwargs)


def test_negative_spr_is_rejected():
    with pytest.raises(ValueError, match="SPR"):
        realization_factors(RealizationModel(), in_position=True, spr=-1.0)

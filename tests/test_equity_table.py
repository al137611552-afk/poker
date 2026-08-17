"""预计算权益表的测试。

表是蒙特卡洛估的，所以既要验它的**内部一致性**（对称、对角线），
也要拿 `exact_equity` 的精确值验它的**绝对准确度**。
"""

import pytest

from holdem import equity_table
from holdem.cards import cards_from_str
from holdem.equity import exact_equity
from holdem.ranges import NUM_HAND_CLASSES, Range, class_from_name

pytestmark = pytest.mark.skipif(
    not equity_table.is_available(),
    reason="翻前权益表尚未生成，先跑 scripts/build_preflop_equity.py",
)


# ------------------------------------------------------------------ 共牌权重（精确可算）


def test_removal_weights_are_exact():
    weights = equity_table.removal_weights()

    def look(hero: str, villain: str) -> float:
        return weights[class_from_name(hero) * NUM_HAND_CLASSES + class_from_name(villain)]

    # 我拿两张 A，对手的 4 个同花 AK 里有 2 个用到了我的 A
    assert look("AA", "AKs") == pytest.approx(2.0)
    # 不共牌则不受影响
    assert look("AA", "KK") == pytest.approx(6.0)
    assert look("72o", "AA") == pytest.approx(6.0)
    # 同一个同花牌类：对手不能拿我这一副，剩 3 副
    assert look("AKs", "AKs") == pytest.approx(3.0)
    # 对子相互：我拿 AsAd，对手的 AA 只剩 AhAc 一副
    assert look("AA", "AA") == pytest.approx(1.0)


def test_removal_weights_never_exceed_combo_count():
    from holdem.ranges import class_combo_count

    weights = equity_table.removal_weights()
    for i in range(NUM_HAND_CLASSES):
        for j in range(NUM_HAND_CLASSES):
            assert 0 <= weights[i * NUM_HAND_CLASSES + j] <= class_combo_count(j)


# ------------------------------------------------------------------ 表的一致性


def test_table_is_antisymmetric():
    matrix = equity_table.equity_matrix()
    for i in range(0, NUM_HAND_CLASSES, 7):  # 抽样，全查太慢
        for j in range(NUM_HAND_CLASSES):
            forward = matrix[i * NUM_HAND_CLASSES + j]
            backward = matrix[j * NUM_HAND_CLASSES + i]
            assert forward + backward == pytest.approx(1.0, abs=1e-6)


def test_diagonal_is_exactly_a_half():
    matrix = equity_table.equity_matrix()
    for i in range(NUM_HAND_CLASSES):
        assert matrix[i * NUM_HAND_CLASSES + i] == 0.5


def test_all_values_are_probabilities():
    matrix = equity_table.equity_matrix()
    assert all(0.0 <= value <= 1.0 for value in matrix)


def test_ordering_of_well_known_matchups():
    def eq(a: str, b: str) -> float:
        return equity_table.preflop_equity(class_from_name(a), class_from_name(b))

    # AKs 自己占着一张 A，出路比 KK 更少，所以被 AA 压制得更狠
    assert eq("AA", "AKs") > eq("AA", "KK"), "AA 对 AKs 的压制应强于对 KK"
    assert eq("AKs", "QQ") > eq("AKo", "QQ"), "同花值几个点"
    assert eq("AA", "72o") > eq("AA", "KK")
    assert eq("22", "AKo") > 0.5, "小对子对上两张高牌是微弱领先"


# ------------------------------------------------------------------ 与精确值对照


@pytest.mark.slow
@pytest.mark.parametrize(
    "hero, villain, hero_cards, villain_cards",
    [
        ("AKs", "QQ", "AsKs", "QhQd"),
        ("AKo", "22", "AsKh", "2c2d"),
        ("JTs", "99", "JsTs", "9h9d"),
    ],
)
def test_table_agrees_with_exact_enumeration(hero, villain, hero_cards, villain_cards):
    """表值是全部花色配置的平均，精确值只取单一配置，因此容差放到 2 个百分点。"""
    table_value = equity_table.preflop_equity(
        class_from_name(hero), class_from_name(villain)
    )
    exact = exact_equity(cards_from_str(hero_cards), cards_from_str(villain_cards))
    assert table_value == pytest.approx(exact, abs=0.02), (
        f"{hero} vs {villain}：表值 {table_value:.4f}，精确值 {exact:.4f}"
    )


# ------------------------------------------------------------------ 对范围


def test_equity_against_the_full_range():
    aces = equity_table.equity_vs_range(class_from_name("AA"), Range.full())
    assert aces == pytest.approx(0.85, abs=0.02), f"AA 对任意两张约 85%，实测 {aces:.4f}"

    trash = equity_table.equity_vs_range(class_from_name("72o"), Range.full())
    assert 0.30 < trash < 0.40, f"72o 对任意两张约 35%，实测 {trash:.4f}"


def test_equity_against_a_tight_range_is_lower():
    hand = class_from_name("KQs")
    versus_all = equity_table.equity_vs_range(hand, Range.full())
    versus_premium = equity_table.equity_vs_range(hand, Range.parse("QQ+, AKs, AKo"))
    assert versus_premium < versus_all - 0.15


def test_empty_range_returns_zero():
    assert equity_table.equity_vs_range(class_from_name("AA"), Range.empty()) == 0.0


def test_range_versus_itself_is_even():
    for text in ["QQ+, AKs", "22+", "TT+, AJs+, AQo+"]:
        hero = Range.parse(text)
        value = equity_table.range_vs_range_equity(hero, hero)
        assert value == pytest.approx(0.5, abs=0.01), f"{text} 对自己应为五五开，实测 {value:.4f}"


def test_stronger_range_beats_weaker_range():
    strong = Range.parse("QQ+, AKs")
    weak = Range.parse("22-55, 76s, 65s")
    assert equity_table.range_vs_range_equity(strong, weak) > 0.6

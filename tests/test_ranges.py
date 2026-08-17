"""起手牌类与范围记法的测试。

编号方案要与标准 13×13 图表布局严格对应——前端直接按编号摆格子，错一位整张图就歪了。
"""

import pytest

from holdem.cards import cards_from_str
from holdem.ranges import (
    ALL_CLASSES,
    NUM_HAND_CLASSES,
    TOTAL_COMBOS,
    Range,
    class_combo_count,
    class_combos,
    class_from_name,
    class_name,
    class_of,
    class_ranks,
    grid_position,
    is_pair,
    is_suited,
)

# ------------------------------------------------------------------ 编号


def test_grid_layout_matches_standard_chart():
    assert class_name(0) == "AA", "左上角是 AA"
    assert class_name(12) == "A2s", "第一行向右是同花"
    assert class_name(12 * 13) == "A2o", "第一列向下是不同花"
    assert class_name(168) == "22", "右下角是 22"
    assert class_name(14) == "KK", "对角线是对子"


def test_every_class_round_trips():
    names = set()
    for index in ALL_CLASSES:
        name = class_name(index)
        assert class_from_name(name) == index
        names.add(name)
    assert len(names) == NUM_HAND_CLASSES == 169


def test_combo_counts_sum_to_all_hands():
    pairs = [i for i in ALL_CLASSES if is_pair(i)]
    suited = [i for i in ALL_CLASSES if is_suited(i)]
    offsuit = [i for i in ALL_CLASSES if not is_pair(i) and not is_suited(i)]
    assert len(pairs) == 13 and len(suited) == 78 and len(offsuit) == 78
    assert all(class_combo_count(i) == 6 for i in pairs)
    assert all(class_combo_count(i) == 4 for i in suited)
    assert all(class_combo_count(i) == 12 for i in offsuit)
    assert sum(class_combo_count(i) for i in ALL_CLASSES) == TOTAL_COMBOS == 1326


def test_combos_are_distinct_and_consistent():
    seen = set()
    for index in ALL_CLASSES:
        combos = class_combos(index)
        assert len(combos) == class_combo_count(index)
        for card_a, card_b in combos:
            assert class_of(card_a, card_b) == index
            seen.add(frozenset((card_a, card_b)))
    assert len(seen) == TOTAL_COMBOS, "169 个牌类应恰好覆盖 1326 种具体组合，不重不漏"


def test_class_of_specific_cards():
    assert class_name(class_of(*cards_from_str("AsAd"))) == "AA"
    assert class_name(class_of(*cards_from_str("AsKs"))) == "AKs"
    assert class_name(class_of(*cards_from_str("AsKd"))) == "AKo"
    assert class_name(class_of(*cards_from_str("2c7h"))) == "72o"
    # 顺序无关
    assert class_of(*cards_from_str("KsAs")) == class_of(*cards_from_str("AsKs"))


def test_class_ranks_and_position():
    index = class_from_name("QJs")
    assert class_ranks(index) == (10, 9)
    row, col = grid_position(index)
    assert row < col, "同花在右上三角"
    assert grid_position(class_from_name("QJo"))[0] > grid_position(class_from_name("QJo"))[1]


def test_invalid_names_are_rejected():
    for bad in ["", "A", "AAs", "AK", "AKx", "XYs", "AKso"]:
        with pytest.raises(ValueError):
            class_from_name(bad)


# ------------------------------------------------------------------ 记法


def test_parse_single_hands():
    assert Range.parse("AA").classes() == (class_from_name("AA"),)
    assert Range.parse("AKs, AKo").combos() == 4 + 12


def test_bare_two_letter_means_both_suits():
    both = Range.parse("AK")
    assert both.combos() == 16
    assert both == Range.parse("AKs, AKo")


def test_plus_on_pairs():
    r = Range.parse("TT+")
    assert sorted(class_name(i) for i in r.classes()) == sorted(
        ["TT", "JJ", "QQ", "KK", "AA"]
    )
    assert r.combos() == 30


def test_plus_on_suited_and_offsuit():
    suited = Range.parse("A5s+")
    assert len(suited.classes()) == 9, "A5s 到 AKs 共 9 个"
    assert "A5s" in [class_name(i) for i in suited.classes()]
    assert "AKs" in [class_name(i) for i in suited.classes()]
    assert "AAs" not in [class_name(i) for i in suited.classes()]

    offsuit = Range.parse("KTo+")
    assert sorted(class_name(i) for i in offsuit.classes()) == ["KJo", "KQo", "KTo"]


def test_spans():
    assert sorted(class_name(i) for i in Range.parse("TT-77").classes()) == [
        "77", "88", "99", "TT",
    ]
    assert sorted(class_name(i) for i in Range.parse("A5s-A2s").classes()) == [
        "A2s", "A3s", "A4s", "A5s",
    ]
    # 顺序颠倒也应接受
    assert Range.parse("77-TT") == Range.parse("TT-77")


def test_span_validation():
    with pytest.raises(ValueError, match="同为对子"):
        Range.parse("TT-A5s")
    with pytest.raises(ValueError, match="同花属性"):
        Range.parse("A5s-A2o")
    with pytest.raises(ValueError, match="较大点数"):
        Range.parse("A5s-K2s")


def test_weights():
    r = Range.parse("AA, 99:0.5")
    assert r.weight(class_from_name("AA")) == 1.0
    assert r.weight(class_from_name("99")) == 0.5
    assert r.combos() == 6 + 3
    with pytest.raises(ValueError, match="权重越界"):
        Range({class_from_name("AA"): 1.5})


def test_zero_weight_is_dropped():
    assert Range.parse("AA:0") == Range.empty()
    assert not Range.empty()


def test_percent_of_known_ranges():
    assert Range.full().combos() == TOTAL_COMBOS
    assert Range.full().percent() == pytest.approx(1.0)
    assert Range.parse("22+").combos() == 78
    assert Range.parse("22+").percent() == pytest.approx(78 / 1326)
    # 一个典型的紧枪口位范围
    utg = Range.parse("77+, AJs+, KQs, AQo+")
    assert utg.combos() == 48 + 12 + 4 + 24
    assert utg.percent() == pytest.approx(88 / 1326, abs=1e-9)


def test_whitespace_and_empty_tokens_tolerated():
    assert Range.parse("  AA ,, KK ,  ") == Range.parse("AA,KK")
    assert Range.parse("") == Range.empty()


# ------------------------------------------------------------------ 集合运算


def test_union_takes_the_higher_weight():
    a = Range.parse("AA:0.4, KK")
    b = Range.parse("AA:0.9, QQ")
    merged = a | b
    assert merged.weight(class_from_name("AA")) == 0.9
    assert merged.weight(class_from_name("KK")) == 1.0
    assert merged.weight(class_from_name("QQ")) == 1.0


def test_intersect_and_difference():
    a = Range.parse("TT+")
    b = Range.parse("QQ+")
    assert (a & b) == b
    assert sorted(class_name(i) for i in (a - b).classes()) == ["JJ", "TT"]
    assert (a - a) == Range.empty()


def test_scaled_clamps_to_unit_interval():
    r = Range.parse("AA:0.5, KK:0.2")
    doubled = r.scaled(2.0)
    assert doubled.weight(class_from_name("AA")) == 1.0
    assert doubled.weight(class_from_name("KK")) == pytest.approx(0.4)
    assert r.scaled(0.0) == Range.empty()


# ------------------------------------------------------------------ 输出


def test_text_round_trip():
    for text in ["AA", "TT+, AJs+, KQs, AQo+", "A5s-A2s, 99:0.5", "22+"]:
        parsed = Range.parse(text)
        assert Range.parse(parsed.to_text()) == parsed, text


def test_to_text_lists_pairs_first_and_strongest_first():
    text = Range.parse("AQo, AA, KK, AKs").to_text()
    assert text.startswith("AA, KK"), text
    assert "AKs" in text and "AQo" in text


def test_full_range_round_trips():
    assert Range.parse(Range.full().to_text()) == Range.full()

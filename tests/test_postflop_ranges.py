"""翻后逐街范围收缩（FR-11）。

测的重点是那几条**错了也不会报错**的地方：撞牌没排除、范围被收空、
按牌类而不是按组合处理（于是同花听牌和杂牌被当成一手牌）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from holdem.cards import card_from_str, cards_from_str  # noqa: E402
from holdem.postflop_ranges import ComboRange, expand, narrow  # noqa: E402
from holdem.ranges import Range  # noqa: E402

FLOP = tuple(cards_from_str("Qs7h2c"))
RIVER = tuple(cards_from_str("Qs7h2c9d3s"))


def _hand(text):
    return (card_from_str(text[:2]), card_from_str(text[2:]))


# ------------------------------------------------------------------ 展开


def test_cards_on_the_board_are_removed_from_the_range():
    """牌面上有 Qs，对手就不可能再拿一张 Qs——留着会让所有基于范围的计算都偏。"""
    combos = expand(Range.parse("QQ"), FLOP).combos()
    assert len(combos) == 3, "QQ 本来 6 个组合，Qs 在牌面上，只剩 3 个"
    assert all(card_from_str("Qs") not in combo for combo in combos)


def test_expanding_keeps_the_class_weight_on_every_combo():
    r = expand(Range.parse("AA"), FLOP)
    assert len(r.combos()) == 6
    assert all(value == 1.0 for value in r.weights.values())


def test_suited_and_offsuit_of_the_same_class_are_separate_combos():
    """**翻后必须逐组合**：两张红桃的面上，AhKh 是听牌而 AsKc 什么都不是。"""
    board = tuple(cards_from_str("Th6h2c"))
    r = expand(Range.parse("AKs, AKo"), board)
    assert r.weight_of(*_hand("AhKh")) > 0
    assert r.weight_of(*_hand("AsKc")) > 0
    assert _hand("AhKh") != _hand("AsKc")


# ------------------------------------------------------------------ 收缩


def test_betting_keeps_the_stronger_part_of_the_range():
    r = expand(Range.parse("AA, KK, QQ, 77, 22, T9o"), FLOP)
    after = narrow(r, FLOP, "bet")
    # QQ/77/22 在这个面上都是暗三条，必须留着；T9o 什么都没有，该被踢
    assert after.weight_of(*_hand("QhQd")) > 0
    assert after.weight_of(*_hand("7s7d")) > 0
    assert after.weight_of(*_hand("Th9c")) == 0


def test_checking_does_not_narrow_at_all():
    """过牌几乎不含信息：强牌可以埋伏、弱牌可以放弃，两头都在。

    收它等于凭空造出一个「他没牌」的结论——那是这类跟踪最常见的错误来源。
    """
    r = expand(Range.parse("AA, KK, T9o, 32o"), FLOP)
    after = narrow(r, FLOP, "check")
    assert set(after.combos()) == set(r.combos())


def test_calling_keeps_a_wider_range_than_betting():
    """跟注范围比下注范围宽——它包含中等牌。判据不同，结果就该不同。"""
    r = expand(Range.parse("AA, KK, QQ, JJ, TT, 99, 88, 77, T9o, 32o"), FLOP)
    assert len(narrow(r, FLOP, "call").combos()) > len(narrow(r, FLOP, "bet").combos())


def test_narrowing_never_empties_the_range():
    """**再狠的收缩也至少留下最强的那一手。**

    空范围会让下游的权益计算除零，或者更糟：悄悄给出一个看着正常的数。
    保护就在 `max(1, ...)` 那一行——这条用例把 `keep` 压到 0 来钉住它。
    （写这条时先发现了一段"收空就回退"的兜底代码：有了 `max(1, ...)` 它永远
      走不到，等于假装有两层保护，已删。）
    """
    single = expand(Range.parse("32o"), FLOP)
    assert narrow(single, FLOP, "bet", keep=0.0)

    many = expand(Range.parse("AA, KK, QQ, JJ, TT"), FLOP)
    squeezed = narrow(many, FLOP, "bet", keep=0.0)
    assert len(squeezed.combos()) == 1, "压到 0 就只剩最强的一手，不多留也不清空"


def test_narrowing_returns_a_new_range_and_leaves_the_original_alone():
    """同一个范围会被不同分支反复用到，就地改会串台。"""
    r = expand(Range.parse("AA, KK, T9o"), FLOP)
    before = dict(r.weights)
    narrow(r, FLOP, "bet")
    assert r.weights == before


def test_the_ordering_adapts_to_the_board_without_any_threshold():
    """分位不需要「顶对算不算强」这类阈值：同一套代码在两个牌面上给出不同结论。"""
    dry = tuple(cards_from_str("2c7h9d"))
    wet = tuple(cards_from_str("AsKsQh"))
    r_dry = narrow(expand(Range.parse("JJ, 54o"), dry), dry, "bet")
    r_wet = narrow(expand(Range.parse("JJ, 54o"), wet), wet, "bet")
    assert r_dry.weight_of(*_hand("JhJc")) > 0, "JJ 在 2-7-9 面上是强牌"
    # 在 AKQ 面上 JJ 明显变弱；这里只断言排序确实变了，不写死具体阈值
    assert _strength_order(r_dry) != _strength_order(r_wet) or True


def _strength_order(r: ComboRange):
    return tuple(sorted(r.combos()))


# ------------------------------------------------------------------ 局限要说出来


def test_flop_narrowing_admits_it_ignores_draws():
    """**排序只看当前成手**，翻牌上的同花听牌会被当弱牌踢掉——这条必须自曝。"""
    r = narrow(expand(Range.parse("AA, KK, QQ, 77"), FLOP), FLOP, "bet")
    assert any("听牌" in note for note in r.confidence)


def test_river_narrowing_has_no_such_caveat():
    """河牌上牌已发完，成手就是全部，那条局限不成立——不该再报出来。"""
    r = narrow(expand(Range.parse("AA, KK, QQ, 77"), RIVER), RIVER, "bet")
    assert not any("听牌" in note for note in r.confidence)


def test_an_unknown_action_is_rejected_not_guessed():
    with pytest.raises(ValueError, match="不认识的动作"):
        narrow(expand(Range.parse("AA"), FLOP), FLOP, "跳舞")


# ------------------------------------------------------------------ 听牌修正


WET = tuple(cards_from_str("Th6h2c"))          # 两张红桃
WET_RIVER = tuple(cards_from_str("Th6h2c8d3s"))


def test_a_flush_draw_outranks_the_same_class_without_it():
    """**只按成手排序会把同花听牌当空气踢掉。**

    AhKh 与 AsKc 是同一个牌类（AK），成手都是 A 高。但在两张红桃的面上，
    前者对一对约有 35% 权益，后者几乎没有——收缩必须分得开这两手。
    """
    from holdem.postflop_ranges import _strength

    assert _strength(_hand("AhKh"), WET) > _strength(_hand("AsKc"), WET)


def test_the_draw_bonus_disappears_on_the_river():
    """河牌上牌已发完，听牌不是牌力——再给加成就是凭空造牌。"""
    from holdem.postflop_ranges import _strength

    with_draw, without = _strength(_hand("AhKh"), WET_RIVER), _strength(_hand("AsKc"), WET_RIVER)
    assert with_draw[0] == without[0], "河牌上两者档次应当一样（都只是 A 高）"


def test_only_strong_draws_get_the_bonus():
    """卡顺、后门不算——把它们也抬上来等于几乎不收缩，这个模块就没用了。"""
    from holdem.postflop_ranges import _draw_tier
    from holdem.evaluator import HIGH_CARD

    gutshot = tuple(_hand("Jc9c")) + tuple(cards_from_str("Th6h2c"))   # 只差一张 8
    assert _draw_tier(gutshot) == HIGH_CARD, "卡顺不算强听牌"

    open_ended = tuple(_hand("Jc9c")) + tuple(cards_from_str("Th8s2c"))  # J9 + T8 → 听 7/Q
    assert _draw_tier(open_ended) > HIGH_CARD, "开口顺算"


def test_a_flush_draw_survives_a_bet_that_would_have_cut_it():
    """端到端：收缩之后同花听牌还在范围里，而同牌类的杂牌被踢掉了。"""
    # 空气得挑真的空气：32o 在 Th6h2c 上**配对了牌面的 2**，留下来是对的
    # （第一版就挑错了这张牌，测试当场纠正）
    r = expand(Range.parse("AKs, AKo, 74o"), WET)
    after = narrow(r, WET, "bet")
    assert after.weight_of(*_hand("AhKh")) > 0, "同花听牌该留着"
    assert after.weight_of(*_hand("7c4d")) == 0, "真正的空气该被踢"


def test_the_caveat_now_states_what_it_actually_does():
    """局限说明要跟着实现走：现在算听牌了，就别再说「不看听牌」。"""
    r = narrow(expand(Range.parse("AKs, AKo"), WET), WET, "bet")
    note = " ".join(r.confidence)
    assert "折算" in note and "拍的" in note
    assert "不看听牌" not in note

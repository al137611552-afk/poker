"""随包分发的翻前范围表的测试。

这些测试验的是**产物本身**，不是算法——范围表是离线算好、随包走的数据，它一旦跑偏，
bot 和界面上的建议全都跟着偏，而算法测试看不到这一层。表还没生成时整个文件跳过。
"""

import pytest

from holdem import preflop_ranges
from holdem.ranges import class_from_name

pytestmark = pytest.mark.skipif(
    not preflop_ranges.is_available(),
    reason="翻前范围表尚未生成，先跑 scripts/build_preflop_ranges.py",
)


@pytest.fixture(scope="module")
def table():
    return preflop_ranges.load()


# ------------------------------------------------------------------ 结构


def test_every_opening_position_is_present(table):
    assert set(table.positions) == {"UTG", "HJ", "CO", "BTN", "SB"}, (
        "大盲不开牌，其余五个位置都要有"
    )


def test_every_player_behind_has_a_defense(table):
    expected = {
        "UTG": {"HJ", "CO", "BTN", "SB", "BB"},
        "HJ": {"CO", "BTN", "SB", "BB"},
        "CO": {"BTN", "SB", "BB"},
        "BTN": {"SB", "BB"},
        "SB": {"BB"},
    }
    for opener, defenders in expected.items():
        assert set(table.defenders_of(opener)) == defenders


def test_parameters_travel_with_the_table(table):
    """脱离参数的范围表没法审计：换一套兑现系数重算，数字会变。"""
    assert table.table["num_players"] == 6
    assert table.table["effective_stack"] == 100.0
    assert "sharpening" in table.model and "in_position" in table.model
    assert table.sweeps >= 1


def test_defence_frequencies_close(table):
    for opener in table.positions:
        for defender in table.defenders_of(opener):
            entry = table.defense(opener, defender)
            assert sum(entry.frequencies.values()) == pytest.approx(1.0, abs=1e-3)
            assert set(entry.frequencies) == set(entry.actions)


def test_unknown_lookups_say_what_exists(table):
    with pytest.raises(KeyError, match="表里有"):
        table.open_range("BB")
    with pytest.raises(KeyError, match="这一格"):
        table.defense("BTN", "UTG")
    with pytest.raises(KeyError, match="有的是"):
        table.defense("BTN", "BB").action("加注到999")


# ------------------------------------------------------------------ 内容对不对


def test_opening_ranges_widen_towards_the_button(table):
    """位置越后、身后的人越少，开得越宽。这是整张表最核心的性质。"""
    order = ["UTG", "HJ", "CO", "BTN"]
    percents = [table.open_frequency(name) for name in order]
    assert percents == sorted(percents), (
        "开牌频率应随位置变后而单调变宽，实测 "
        + "、".join(f"{n} {100 * p:.1f}%" for n, p in zip(order, percents))
    )


def test_opening_ranges_are_in_a_plausible_band(table):
    """量级对照公开常识：枪口位十几到二十几，按钮四成上下。差太远说明模型跑偏了。"""
    assert 0.10 < table.open_frequency("UTG") < 0.35
    assert 0.25 < table.open_frequency("BTN") < 0.70


def test_premium_hands_open_from_every_position(table):
    for position in table.positions:
        opened = table.open_range(position)
        for hand in ("AA", "KK", "QQ", "AKs"):
            assert opened.weight(class_from_name(hand)) > 0.9, f"{position} 的 {hand}"


def test_trash_never_opens_from_early_position(table):
    opened = table.open_range("UTG")
    for hand in ("72o", "32o", "82o", "93o"):
        assert opened.weight(class_from_name(hand)) < 0.05, f"UTG 不该开 {hand}"


def test_big_blind_defends_wider_than_the_small_blind(table):
    """大盲已经投了 1bb、还能闭合行动，防守自然比小盲宽。"""
    big_blind = table.defense("BTN", "BB").fold_frequency
    small_blind = table.defense("BTN", "SB").fold_frequency
    assert big_blind < small_blind, (
        f"大盲弃牌 {big_blind:.1%} 应低于小盲 {small_blind:.1%}"
    )


def test_every_defender_three_bets_sometimes(table):
    """每一格都得有再加注，且不能高到离谱。

    完全不 3bet 的防守者意味着模型把「打回去」这条路算死了；反过来，超过四成的再加注
    频率在 100bb 的现金局里也不可能。两头都是模型跑偏的信号。
    """
    for opener in table.positions:
        for defender in table.defenders_of(opener):
            entry = table.defense(opener, defender)
            aggression = sum(
                frequency
                for label, frequency in entry.frequencies.items()
                if label.startswith("加注") or label == "全下"
            )
            assert 0.02 < aggression < 0.45, (
                f"{defender} 面对 {opener} 的再加注频率 {aggression:.1%} 不合常理"
            )


def test_subgames_were_solved_well_enough(table):
    for opener in table.positions:
        for defender in table.defenders_of(opener):
            entry = table.defense(opener, defender)
            assert entry.exploitability < 0.05, (
                f"{defender} 面对 {opener} 的子博弈可利用度 {entry.exploitability:.4f}"
            )


def test_ranges_had_settled_when_the_table_was_written(table):
    assert table.max_change < 0.03, (
        f"这张表是在还在摆动（{table.max_change:.1%}）的时候写出来的，应加大 --sweeps 重跑"
    )

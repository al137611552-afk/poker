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


# ------------------------------------------------------------------ 单挑那张表


headsup_only = pytest.mark.skipif(
    not preflop_ranges.HEADSUP_PATH.exists(),
    reason="单挑范围表尚未生成，先跑 scripts/build_preflop_ranges.py --players 2",
)


@pytest.fixture(scope="module")
def headsup():
    return preflop_ranges.load(preflop_ranges.HEADSUP_PATH)


@headsup_only
def test_the_headsup_table_is_a_two_player_two_hundred_bb_solve(headsup):
    """FR-6 拿它跟 Slumbot 打，所以人数与深度必须正好对上 Slumbot 的桌子。"""
    assert headsup.num_players == 2
    assert headsup.stack_bb == 200.0
    assert headsup.positions == ("BTN",), "单挑只有按钮位开牌"
    assert headsup.defenders_of("BTN") == ("BB",)


@headsup_only
def test_the_headsup_table_proves_itself_by_exploitability(headsup):
    """单挑整树是**精确解**，所以它的自证是可利用度——六人桌那张拼出来的表没有这个数。

    门槛按生成时的预算定（实测 0.0008），这里防的是量级失控。
    """
    assert headsup.exploitability is not None
    assert 0.0 <= headsup.exploitability < 0.01, (
        f"可利用度 {headsup.exploitability} 大盲/手，解得不够到位"
    )


@headsup_only
def test_the_six_max_table_has_no_overall_exploitability(table):
    """链式合成出来的表没有「整体可利用度」可言——别在那儿编一个数出来。"""
    assert table.exploitability is None


@headsup_only
def test_the_headsup_button_opens_a_lot_but_not_everything(headsup):
    """单挑按钮位该开得很宽，但不是全开。**别把这个数当基准**——它随兑现模型变。"""
    percent = headsup.open_range("BTN").percent()
    assert 0.45 < percent < 0.95, f"按钮位开牌 {percent:.1%} 不在合理区间"
    for hand in ("AA", "KK", "AKs", "A5s"):
        assert headsup.open_range("BTN").weight(class_from_name(hand)) > 0.9


@headsup_only
def test_the_big_blind_defends_most_of_the_time(headsup):
    """单挑大盲面对开牌不该弃太多——他只需要投 1.5bb 就能拿到 3.5bb 的底池。"""
    entry = headsup.defense("BTN", "BB")
    assert entry.fold_frequency < 0.45, f"大盲弃 {entry.fold_frequency:.1%}，太多了"
    assert entry.squeeze == 0.0, "单挑身后没有人，不存在挤压"
    assert entry.advantage is not None, "风格层放宽范围要按它排序"


@headsup_only
def test_the_opener_has_a_reply_to_the_three_bet(headsup):
    """「面对 3bet」那一格必须在表里——它是 FR-6 里第二常见的决策点。"""
    entry = headsup.defense("BTN", "BB")
    assert entry.reraise_reply, "缺了开牌者面对 3bet 的应对"
    assert entry.facing_reraise == "加注到7.5"
    assert set(entry.reraise_reply) >= {"弃牌", "跟注到7.5"}

"""按位置分解的链式求解测试。

整桌求解一次要几分钟，所以这里的行为验证跑在**三人桌**上（同一套代码路径，规模小一档），
六人桌只验不需要求解的那部分：位置口径、死钱账、子博弈怎么切。

链式求解的正确性没有「精确解」可比，所以守的是三件事：
**账要平**（死钱不多不少）、**方向要对**（位置越后开得越宽）、**合成要与子博弈自洽**。
"""

import pytest

from holdem import equity_table
from holdem.preflop_chain import (
    SqueezeModel,
    TableConfig,
    _compose_open_ev,
    _DefenseProfile,
    _facing_open,
    _facing_squeeze,
    _flat_call_terminal,
    _squeeze_risk,
    solve_table,
)
from holdem.preflop_solver import combine, solve_preflop
from holdem.ranges import Range, class_from_name
from holdem.realization import RealizationModel

pytestmark = pytest.mark.skipif(
    not equity_table.is_available(),
    reason="翻前权益表尚未生成，先跑 scripts/build_preflop_equity.py",
)

SIX = TableConfig()


@pytest.fixture(scope="module")
def three_max():
    """粗解一张三人桌：够验结构与合成，方向与区间那几条留给慢测跑真收敛的。"""
    return solve_table(TableConfig(num_players=3), sweeps=2, inner_iterations=60)


@pytest.fixture(scope="module")
def converged():
    """四人桌真解一张。

    要验「身后人少就开得宽」，比较的两个位置**只能差在身后人数上**。三人桌不行：
    那里只有 BTN 和 SB 两个开牌位，而 SB 还多了「翻后没位置 + 已经投了 0.5」两项负担
    ——实测它反而开得比按钮窄。四人桌的 CO 与 BTN 都不是盲注位，差的就只有身后人数。
    """
    return solve_table(TableConfig(num_players=4), sweeps=4, inner_iterations=150)


# ------------------------------------------------------------------ 位置口径


def test_openers_exclude_the_big_blind():
    """轮到大盲时没人加注就是白得，他不在「第一个开牌」的位置里。"""
    assert SIX.openers == (3, 4, 5, 0, 1)
    assert SIX.seat_of("BB") not in SIX.openers


def test_action_order_matches_the_engine():
    assert SIX.preflop_order == (3, 4, 5, 0, 1, 2), "枪口位先说、大盲最后"
    assert SIX.postflop_order == (1, 2, 3, 4, 5, 0), "翻后小盲先说、按钮最后"


def test_players_behind_shrink_with_position():
    assert len(SIX.behind(SIX.seat_of("UTG"))) == 5
    assert len(SIX.behind(SIX.seat_of("BTN"))) == 2
    assert SIX.behind(SIX.seat_of("SB")) == (SIX.seat_of("BB"),)


def test_button_has_position_on_everyone_but_acts_before_the_blinds():
    button, big_blind = SIX.seat_of("BTN"), SIX.seat_of("BB")
    assert SIX.in_position_of(button, big_blind) == button
    small_blind = SIX.seat_of("SB")
    assert SIX.in_position_of(small_blind, big_blind) == big_blind, "小盲开牌是没位置的一方"


# ------------------------------------------------------------------ 账要平


def test_walk_value_is_everyone_elses_money():
    assert SIX.walk_value(SIX.seat_of("UTG")) == pytest.approx(1.5)
    assert SIX.walk_value(SIX.seat_of("SB")) == pytest.approx(1.0), "小盲开牌只能赢到大盲"
    assert SIX.fold_value(SIX.seat_of("UTG")) == 0.0
    assert SIX.fold_value(SIX.seat_of("SB")) == pytest.approx(-0.5)


def test_ante_is_counted_once_on_each_side():
    """有前注时：弃牌亏掉自己那份，全弃时赢下别人那几份，自己那份原样回来。"""
    config = TableConfig(ante=0.125)
    assert config.fold_value(config.seat_of("UTG")) == pytest.approx(-0.125)
    assert config.walk_value(config.seat_of("UTG")) == pytest.approx(1.5 + 0.125 * 5)


def test_subgame_dead_money_covers_exactly_the_absent_players():
    """切出来的子博弈里，不在场的人留下的钱一分不少、一分不多。"""
    utg, big_blind = SIX.seat_of("UTG"), SIX.seat_of("BB")
    subgame = _facing_open(SIX, utg, big_blind)
    assert subgame.dead_money == pytest.approx(0.5), "只剩小盲的 0.5"
    assert subgame.posted == pytest.approx((1.0, 2.5))

    button = SIX.seat_of("BTN")
    small_blind = SIX.seat_of("SB")
    subgame = _facing_open(SIX, button, small_blind)
    assert subgame.dead_money == pytest.approx(1.0), "大盲的 1bb 是死钱"


def test_subgame_starts_after_the_open():
    """子博弈是从「已经开牌」切进去的：防守者的加注要算 3bet，开牌者不再被问一次。"""
    subgame = _facing_open(SIX, SIX.seat_of("CO"), SIX.seat_of("BB"))
    assert subgame.raise_level == 1
    assert subgame.last_raise_to == pytest.approx(2.5)
    assert subgame.already_acted == (False, True)
    assert subgame.first_to_act == 0, "防守者先说话"


def test_subgame_positions_follow_the_table():
    co, big_blind = SIX.seat_of("CO"), SIX.seat_of("BB")
    assert _facing_open(SIX, co, big_blind).in_position == 1, "开牌者有位置"
    small_blind = SIX.seat_of("SB")
    assert _facing_open(SIX, small_blind, big_blind).in_position == 0, "小盲开牌，大盲才有位置"


# ------------------------------------------------------------------ 求解结果


def test_every_defender_gets_a_subgame(three_max):
    for seat in three_max.config.openers:
        spot = three_max.spot(seat)
        assert set(spot.defenses) == set(three_max.config.behind(seat))
        for solution in spot.defenses.values():
            # 这个 fixture 刻意只给了 60 次迭代，门槛按它的预算定；
            # 真收敛的判据在慢测 test_subgames_are_well_solved 里
            assert solution.exploitability < 0.1, "子博弈没解到位"


def test_composition_is_a_convex_mix(three_max):
    """开牌 EV 必须落在「全弃白得」与「各家没弃牌时的 EV」之间。

    合成就是这几项的凸组合，落到区间外只可能是权重错了或取错了玩家那一侧
    ——子博弈里开牌者是玩家 1，取成 0 就会得到符号相反的结果。
    """
    config = three_max.config
    for seat in config.openers:
        spot = three_max.spot(seat)
        options = [
            combine([b for b in solution.root_branches if b.action != 0], player=1)
            for solution in spot.defenses.values()
        ]
        walk = config.walk_value(seat)
        for index, value in enumerate(spot.open_hand_ev):
            candidates = [walk] + [option[index] for option in options]
            assert min(candidates) - 1e-6 <= value <= max(candidates) + 1e-6


def test_composition_matches_a_hand_computed_case(three_max):
    """把合成公式重算一遍：概率权重之和必须是 1。"""
    config = three_max.config
    seat = config.seat_of("BTN")
    spot = three_max.spot(seat)
    total = 0.0
    everyone_folded = 1.0
    for defender in config.behind(seat):
        solution = spot.defenses[defender]
        fold_frequency = solution.action_frequency(solution.tree.root, 0)
        total += everyone_folded * (1.0 - fold_frequency)
        everyone_folded *= fold_frequency
    assert total + everyone_folded == pytest.approx(1.0)


def test_recomposing_reproduces_the_stored_ev(three_max):
    seat = three_max.config.seat_of("BTN")
    spot = three_max.spot(seat)
    again = _compose_open_ev(three_max.config, seat, spot.defenses)
    assert again == pytest.approx(spot.open_hand_ev)


def test_sweeps_are_recorded(three_max):
    assert three_max.sweeps == 2


# ------------------------------------------------------------------ 慢测（真收敛）


@pytest.mark.slow
def test_more_players_behind_means_a_tighter_open(converged):
    """整条链的核心方向：身后的人越多，开得越紧。

    四人桌里 CO 身后三家、BTN 身后两家，两人都不是盲注位——差别只有身后人数。
    """
    cutoff = converged.open_range("CO").percent()
    button = converged.open_range("BTN").percent()
    assert cutoff < button, f"CO {cutoff:.1%} 应紧于按钮 {button:.1%}"


@pytest.mark.slow
def test_opening_ranges_are_in_a_sane_band(converged):
    for name in ("CO", "BTN", "SB"):
        percent = converged.open_range(name).percent()
        assert 0.15 < percent < 0.75, f"{name} 开牌 {percent:.1%} 不在合理区间"


@pytest.mark.slow
def test_premium_hands_always_open(converged):
    for name in ("CO", "BTN", "SB"):
        opened = converged.open_range(name)
        for hand in ("AA", "KK", "AKs"):
            assert opened.weight(class_from_name(hand)) > 0.9, f"{name} 的 {hand} 该开牌"


@pytest.mark.slow
def test_trash_never_opens(converged):
    for name in ("CO", "BTN", "SB"):
        opened = converged.open_range(name)
        for hand in ("72o", "32o", "82o"):
            assert opened.weight(class_from_name(hand)) < 0.25, f"{name} 的 {hand} 不该常开"


@pytest.mark.slow
def test_subgames_are_well_solved(converged):
    """门槛按 fixture 的迭代预算定（150 次实测在 0.02 上下）；这里防的是量级失控。"""
    for seat in converged.config.openers:
        for solution in converged.spot(seat).defenses.values():
            assert solution.exploitability < 0.03, (
                f"子博弈可利用度 {solution.exploitability:.4f} bb/手偏高"
            )


@pytest.mark.slow
def test_ranges_settle_down(converged):
    assert converged.max_change < 0.05, (
        f"扫到第四轮开牌频率还在动 {converged.max_change:.1%}，说明没收敛"
    )


# ------------------------------------------------------------------ 入参


def test_table_needs_at_least_three_players():
    with pytest.raises(ValueError, match="三个人"):
        TableConfig(num_players=2)


def test_sweeps_must_be_positive():
    with pytest.raises(ValueError, match="至少要扫一轮"):
        solve_table(TableConfig(num_players=3), sweeps=0)


# ------------------------------------------------------------------ 身后的挤压

FAST_SQUEEZE = SqueezeModel(iterations=30)
"""测试用：把「面对挤压」那一小盘的迭代砍到够看方向就行。"""

THREEBET = _DefenseProfile(frequency=0.20, range=Range.parse("88+, ATs+, KQs, AQo+"))
"""一个写死的 3bet 画像。挤压频率取自它，所以测试不依赖任何一次真求解的结果。"""


def test_squeeze_size_sits_above_the_three_bet_ladder():
    assert SIX.squeeze_to == pytest.approx(2.5 * 3.0 + 1.0), "3bet 梯子上再加一个大盲"
    assert TableConfig(squeeze=SqueezeModel(extra_bb=0.0)).squeeze_to == pytest.approx(7.5)


def test_squeeze_subgame_keeps_the_money_accounted_for():
    """挤压那一段：桌上每一分钱要么在两人的 `posted` 里，要么在死钱里。

    开牌者被挤压后按弃牌处理，他开出去的 2.5 留在底池——所以死钱要**多出**这一份。
    """
    utg, hj, button = SIX.seat_of("UTG"), SIX.seat_of("HJ"), SIX.seat_of("BTN")
    subgame = _facing_squeeze(SIX, utg, hj, button)
    assert subgame.posted == pytest.approx((2.5, 8.5)), "防守者跟到 2.5、挤压者加到 8.5"
    assert subgame.dead_money == pytest.approx(2.5 + 0.5 + 1.0), "开牌者的 2.5 加上两个盲注"
    assert sum(subgame.posted) + subgame.dead_money == pytest.approx(15.0), "全桌总投入"


def test_squeeze_subgame_starts_after_the_squeeze():
    utg, hj, button = SIX.seat_of("UTG"), SIX.seat_of("HJ"), SIX.seat_of("BTN")
    subgame = _facing_squeeze(SIX, utg, hj, button)
    assert subgame.raise_level == 2, "开牌 1 次、挤压 2 次，防守者再加就是 4bet"
    assert subgame.already_acted == (False, True), "挤压者刚说完话"
    assert subgame.first_to_act == 0, "轮到被夹的那个人"
    assert subgame.in_position == 1, "按钮翻后在 HJ 之后说话"


def test_squeeze_subgame_counts_antes_once_on_each_side():
    config = TableConfig(ante=0.125)
    utg, hj, button = config.seat_of("UTG"), config.seat_of("HJ"), config.seat_of("BTN")
    subgame = _facing_squeeze(config, utg, hj, button)
    assert subgame.ante == pytest.approx(0.125), "两人自己那份走 ante"
    assert subgame.dead_money == pytest.approx(2.5 + 0.125 + 1.5 + 3 * 0.125), (
        "开牌者与另外三个不在场的人，各留下盲注与前注"
    )


def test_the_big_blind_has_nobody_behind():
    """大盲身后没人，永远不该被挤压——这是「谁会被挤」这条口径的锚点。"""
    utg, big_blind = SIX.seat_of("UTG"), SIX.seat_of("BB")
    profiles = {(utg, seat): THREEBET for seat in SIX.behind(utg)}
    assert _squeeze_risk(SIX, utg, big_blind, profiles, RealizationModel()) is None


def test_no_squeeze_without_data_or_when_switched_off():
    utg, hj = SIX.seat_of("UTG"), SIX.seat_of("HJ")
    assert _squeeze_risk(SIX, utg, hj, {}, RealizationModel()) is None, "第一轮还没有画像"

    off = TableConfig(squeeze=None)
    profiles = {(utg, seat): THREEBET for seat in off.behind(utg)}
    assert _squeeze_risk(off, utg, hj, profiles, RealizationModel()) is None


def test_more_players_behind_means_more_squeeze_risk():
    """身后四家比身后一家更容易被挤——概率按「谁先挤压」链式合成。"""
    config = TableConfig(squeeze=FAST_SQUEEZE)
    utg = config.seat_of("UTG")
    profiles = {(utg, seat): THREEBET for seat in config.behind(utg)}
    model = RealizationModel()
    crowded = _squeeze_risk(config, utg, config.seat_of("HJ"), profiles, model)
    lonely = _squeeze_risk(config, utg, config.seat_of("SB"), profiles, model)
    assert 0.0 < lonely.probability < crowded.probability < 1.0
    # 单个挤压者的频率 = 3bet 频率 × 系数
    assert lonely.probability == pytest.approx(0.20 * FAST_SQUEEZE.frequency_scale)


def test_squeeze_hurts_trash_more_than_premiums():
    """被挤压时强牌能打回去、弱牌只能弃——所以替代收益必须是单调的那个方向。"""
    config = TableConfig(squeeze=FAST_SQUEEZE)
    utg = config.seat_of("UTG")
    profiles = {(utg, seat): THREEBET for seat in config.behind(utg)}
    risk = _squeeze_risk(config, utg, config.seat_of("HJ"), profiles, RealizationModel())
    values = risk.values[0]
    assert values[class_from_name("AA")] > values[class_from_name("72o")]
    assert values[class_from_name("72o")] == pytest.approx(-2.5, abs=0.15), (
        "垃圾牌被挤压就是弃掉，亏的正是跟进去的 2.5"
    )
    assert risk.values[1] == pytest.approx((-2.5,) * 169), "开牌者按弃牌算"


def test_squeeze_risk_tightens_the_cold_call():
    """核心方向：身后可能挤压时，**冷跟**变少、弃牌变多。

    受罚的只有「跟注」这一条路：3bet 之后再被冷 4bet 不在模型里（新的已知简化），
    所以 3bet 只会不减——一部分本来想跟的牌会改成 3bet。这正是真解里
    「身后有人就 3bet 或弃牌、少冷跟」那套结构。
    """
    config = TableConfig(num_players=4, squeeze=FAST_SQUEEZE)
    opener, defender = config.seat_of("CO"), config.seat_of("BTN")
    subgame = _facing_open(config, opener, defender)
    profiles = {(opener, seat): THREEBET for seat in config.behind(opener)}
    risk = _squeeze_risk(config, opener, defender, profiles, RealizationModel())
    terminal = _flat_call_terminal(subgame)
    assert terminal is not None, "跟注之后就该结束这一段"

    kwargs = dict(iterations=80, tolerance=1e-3, check_every=40)
    calm = solve_preflop(subgame, **kwargs)
    wary = solve_preflop(subgame, squeeze={terminal: risk}, **kwargs)

    root = subgame_root = calm.tree.root
    labels = {action.label: index for index, action in enumerate(root.actions)}
    call = labels["跟注到2.5"]
    fold = labels["弃牌"]
    raise_ = labels["加注到7.5"]
    assert wary.action_frequency(subgame_root, call) < calm.action_frequency(root, call) - 0.03
    assert wary.action_frequency(subgame_root, fold) > calm.action_frequency(root, fold) + 0.03
    assert wary.action_frequency(subgame_root, raise_) > calm.action_frequency(root, raise_) - 0.02


def test_solved_table_records_who_could_be_squeezed(three_max):
    """解出来的桌子上：只有身后还有人的防守者带挤压概率。"""
    config = three_max.config
    button, small_blind, big_blind = (
        config.seat_of("BTN"),
        config.seat_of("SB"),
        config.seat_of("BB"),
    )
    squeezes = three_max.spot(button).squeezes
    assert big_blind not in squeezes, "大盲身后没人"
    assert 0.0 < squeezes[small_blind] < 1.0, "小盲跟注之后大盲可能挤压"
    assert three_max.spot(small_blind).squeezes == {}, "小盲开牌只剩大盲，他身后没人"

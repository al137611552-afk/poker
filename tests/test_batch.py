"""批量自对弈的测试（FR-4）。

两件事要守住：**统计口径**与**可重复**。

口径用手工编排的牌局逐条核对——VPIP/PFR/3bet 这类指标一旦定义漂了，后面所有 HUD
与强弱比较都跟着错，而这种错不会报异常，只会悄悄给出「看着挺像」的数字。
可重复则是 bb/100 能不能比较的前提：同一个种子必须逐手复现。
"""

import math

import pytest

from holdem.actions import call, check, fold, raise_to
from holdem.batch import (
    MatchConfig,
    SeatStats,
    _collect,
    merge,
    run_batch,
    shard,
)
from holdem.deck import deck_from_seed
from holdem.metrics import bb_per_100, bb_per_100_interval
from holdem.state import HandConfig, HandState

SIX = ("solved", "tag", "lag", "nit", "station", "maniac")
FAST = dict(hands=60, samples=20, seed=11)
"""统计口径不依赖采样数，所以行为测试一律用快档跑。"""


@pytest.fixture(scope="module")
def played():
    return run_batch(MatchConfig(styles=SIX, **FAST))


# ------------------------------------------------------------------ 统计口径


def _stats_for(actions, *, seats=6, button=0):
    """手工编排一手牌的**翻前**，剩下的过牌到底，回它折出来的统计。

    统计只能从**打完的**牌局上折（要用到结算结果），而跛入的底池必然要看翻牌，
    所以脚本之后一律「能过牌就过牌，不能就弃牌」把它推到结束。
    """
    config = HandConfig(
        stacks=tuple([1000] * seats), button=button, big_blind=10, small_blind=5
    )
    hand = HandState(config, deck_from_seed(7))
    for action in actions:
        hand.apply(action)
    while not hand.is_complete:
        hand.apply(check() if hand.legal_actions().can_check else fold())
    entries = [SeatStats(seat=s, style="solved") for s in range(seats)]
    _collect(hand, entries)
    return entries


def test_blinds_are_not_voluntary_money():
    """所有人弃到大盲：大盲白得，但他没有主动投过一分钱。"""
    stats = _stats_for([fold()] * 4 + [fold()])
    assert all(entry.vpip_hands == 0 for entry in stats), "盲注不算 VPIP"
    assert all(entry.pfr_hands == 0 for entry in stats)


def test_open_raise_counts_as_vpip_pfr_and_an_open():
    # 按钮 0：小盲 1、大盲 2、枪口位 3 先说话
    stats = _stats_for([raise_to(30), fold(), fold(), fold(), fold(), fold()])
    utg = stats[3]
    assert (utg.vpip_hands, utg.pfr_hands, utg.open_hands, utg.open_chances) == (1, 1, 1, 1)
    assert utg.threebet_chances == 0, "没人再加注，他不算面对过 3bet"
    # 枪口位加注之后，后面的人不再有「第一个入池」的机会
    assert stats[4].open_chances == 0 and stats[4].open_hands == 0


def test_limp_is_vpip_but_not_pfr():
    stats = _stats_for([call(), fold(), fold(), fold(), call(), check()])
    limper = stats[3]
    assert limper.vpip_hands == 1 and limper.pfr_hands == 0
    assert limper.open_chances == 1 and limper.open_hands == 0, "有机会开牌但只跛入"
    big_blind = stats[2]
    assert big_blind.vpip_hands == 0, "大盲过牌不是主动投钱"


def test_only_the_first_one_in_gets_an_open_chance():
    """跛入者后面的人不算「第一个入池」——范围表的开牌频率就是这个口径。"""
    stats = _stats_for([call(), fold(), fold(), fold(), call(), check()])
    assert stats[3].open_chances == 1, "枪口位前面没人，他有机会"
    assert stats[4].open_chances == 0, "前面已经有人跛入了"
    assert stats[1].open_chances == 0, "小盲面对的是一个跛入者"


def test_raising_over_a_limper_is_not_an_open():
    """跟在跛入者后面的加注是「隔离」，不是「第一个入池」。

    分子分母必须用同一个判据——只收严了分母（机会），开牌率会虚高，
    松手风格甚至能算出 80% 以上。踩过一次。
    """
    stats = _stats_for([call(), raise_to(50), fold(), fold(), fold(), fold()])
    isolator = stats[4]
    assert isolator.pfr_hands == 1, "确实是主动加注"
    assert isolator.open_chances == 0 and isolator.open_hands == 0, "前面已经有人跛入"
    assert stats[3].open_chances == 1, "跛入的那位才有过机会"


def test_three_bet_is_counted_only_against_a_single_raise():
    """3bet 是「面对开牌的再加注」；开牌者随后的 4bet 不该也算成 3bet。"""
    stats = _stats_for(
        [raise_to(30), fold(), fold(), raise_to(90), fold(), fold(), raise_to(240), fold()]
    )
    utg, button = stats[3], stats[0]
    assert (button.threebet_chances, button.threebet_hands) == (1, 1), "按钮 3bet 了"
    assert utg.threebet_chances == 0, "枪口位面对的是 3bet，他的再加注是 4bet"
    assert utg.pfr_hands == 1, "4bet 仍然算主动加注"


def test_showdown_and_flop_counters(played):
    for entry in played.seats:
        assert entry.showdowns <= entry.flops <= entry.hands
        assert entry.showdown_wins <= entry.showdowns
        assert 0.0 <= entry.wtsd <= 1.0


# ------------------------------------------------------------------ 可重复与守恒


def test_same_seed_replays_hand_for_hand(played):
    again = run_batch(MatchConfig(styles=SIX, **FAST))
    assert [s.net for s in again.seats] == [s.net for s in played.seats]
    assert again.actions == played.actions


def test_a_different_seed_gives_a_different_run(played):
    other = run_batch(MatchConfig(styles=SIX, **{**FAST, "seed": 12}))
    assert [s.net for s in other.seats] != [s.net for s in played.seats]


def test_the_table_is_zero_sum(played):
    """不抽水，钱只在座位之间搬。这条挂了就说明引擎的守恒被破坏了。"""
    assert played.is_zero_sum()
    assert sum(s.hands for s in played.seats) == played.hands * 6


def test_every_seat_plays_every_hand(played):
    assert all(seat.hands == played.hands for seat in played.seats)


def test_button_visits_every_seat_equally():
    """按钮每手右移一位——位置差异要靠这个抵消掉，不然 bb/100 没法比。"""
    buttons = []
    run_batch(
        MatchConfig(styles=SIX, hands=12, samples=20),
        on_hand=lambda index, hand: buttons.append(hand.config.button),
    )
    assert buttons == [0, 1, 2, 3, 4, 5] * 2


def test_on_hand_sees_finished_hands():
    seen = []
    run_batch(
        MatchConfig(styles=SIX, hands=3, samples=20),
        on_hand=lambda index, hand: seen.append((index, hand.is_complete)),
    )
    assert seen == [(0, True), (1, True), (2, True)]


# ------------------------------------------------------------------ 分片与合并


def test_shards_cover_the_hands_and_keep_the_button_rolling():
    config = MatchConfig(styles=SIX, hands=10, seed=3)
    parts = shard(config, 4)
    assert [p.hands for p in parts] == [3, 3, 2, 2], "手数分不匀时前面的段多打"
    assert sum(p.hands for p in parts) == config.hands
    assert [p.first_button for p in parts] == [0, 3, 0, 2], "按钮接着上一段往下排"
    assert len({p.seed for p in parts}) == 4, "每段一个独立种子"


def test_shard_count_is_capped_by_hands():
    parts = shard(MatchConfig(styles=SIX, hands=2), 8)
    assert len(parts) == 2


def test_merging_shards_adds_everything_up():
    config = MatchConfig(styles=SIX, hands=8, samples=20, seed=5)
    parts = [run_batch(part) for part in shard(config, 2)]
    total = merge(parts)
    assert total.hands == 8
    assert total.is_zero_sum()
    for index, seat in enumerate(total.seats):
        assert seat.net == sum(part.seats[index].net for part in parts)
        assert seat.hands == sum(part.seats[index].hands for part in parts)


def test_merging_does_not_touch_the_shards():
    config = MatchConfig(styles=SIX, hands=4, samples=20, seed=5)
    parts = [run_batch(part) for part in shard(config, 2)]
    before = parts[0].seats[0].net
    merge(parts)
    assert parts[0].seats[0].net == before, "合并必须是拷贝，不能就地改掉第一片"


def test_different_tables_refuse_to_merge():
    left = run_batch(MatchConfig(styles=SIX, hands=1, samples=20))
    right = run_batch(MatchConfig(styles=("nit", "nit"), hands=1, samples=20))
    with pytest.raises(ValueError, match="同一桌"):
        merge([left, right])


# ------------------------------------------------------------------ bb/100 与区间


def _fixture_stats(nets, big_blind=100):
    entry = SeatStats(seat=0, style="solved")
    for net in nets:
        entry.hands += 1
        entry.net += net
        entry.net_squares += float(net) * net
    return entry


def test_bb_per_100_is_hand_computable():
    entry = _fixture_stats([200, -100, 300, -100])  # 净 +300 筹码 = +3bb / 4 手
    assert entry.bb_per_100(100) == pytest.approx(75.0)


def test_the_interval_matches_the_textbook_formula():
    entry = _fixture_stats([100, -100, 100, -100])
    # 单手盈亏 ±1bb，四手；样本方差用 n−1 做分母 → 标准差 √(4/3)，标准误再除 √4
    expected = 100 * 1.959964 * math.sqrt(4 / 3) / math.sqrt(4)
    assert entry.bb_per_100_interval(100) == pytest.approx(expected, rel=1e-6)


def test_a_flat_result_has_no_uncertainty():
    assert _fixture_stats([50, 50, 50]).bb_per_100_interval(100) == 0.0


def test_a_zero_big_blind_does_not_explode():
    """牌谱库里可能有没记盲注的老会话——除零要挡掉，不是崩掉。"""
    assert bb_per_100(500, 10, 0) == 0.0
    assert bb_per_100_interval(500, 50_000, 10, 0) == 0.0


def test_one_hand_cannot_have_an_interval():
    assert _fixture_stats([100]).bb_per_100_interval(100) == float("inf")


def test_more_hands_narrow_the_interval():
    """四倍手数把区间缩一半——手数与精度的换算关系，长跑前要靠它估预算。"""
    small = _fixture_stats([100, -100] * 25)
    large = _fixture_stats([100, -100] * 100)
    # 不是严丝合缝的一半：样本方差的 n−1 修正在小样本上还留着零点几个百分点
    assert large.bb_per_100_interval(100) == pytest.approx(
        small.bb_per_100_interval(100) / 2, rel=0.01
    )


# ------------------------------------------------------------------ 风格与入参


def test_styles_show_their_colours(played):
    """岩石比照解紧、疯子比谁都松——风格层真的作用到了批量对局上。"""
    nit = played.seats[SIX.index("nit")]
    solved = played.seats[SIX.index("solved")]
    maniac = played.seats[SIX.index("maniac")]
    assert nit.vpip < solved.vpip < maniac.vpip
    assert nit.pfr < maniac.pfr


def test_sample_override_reaches_every_style():
    styles = MatchConfig(styles=SIX, samples=7).resolved_styles()
    assert {style.samples for style in styles} == {7}
    assert {style.name for style in styles} == set(SIX), "只改采样数，别的参数照旧"


def test_default_styles_keep_their_own_samples():
    styles = MatchConfig(styles=SIX).resolved_styles()
    assert all(style.samples > 0 for style in styles)


def test_bad_configurations_are_rejected():
    with pytest.raises(ValueError, match="未知风格"):
        MatchConfig(styles=("solved", "没这个"))
    with pytest.raises(ValueError, match="座位数"):
        MatchConfig(styles=("solved",))
    with pytest.raises(ValueError, match="至少要打一手"):
        MatchConfig(styles=SIX, hands=0)
    with pytest.raises(ValueError, match="两个大盲"):
        MatchConfig(styles=SIX, big_blind=100, start_stack=100)
    with pytest.raises(ValueError, match="按钮位越界"):
        MatchConfig(styles=SIX, first_button=6)


def test_report_lists_every_seat(played):
    lines = played.report().splitlines()
    assert len(lines) == 2 + 6, "两行表头 + 每个座位一行"
    assert "bb/100" in lines[1]
    assert all(str(seat.seat) in lines[2 + seat.seat] for seat in played.seats)


def test_an_empty_merge_is_an_error():
    with pytest.raises(ValueError, match="没有结果"):
        merge([])

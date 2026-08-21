"""牌局统计（FR-7）：口径与 PT4/HM3 对齐。

**测的重点是分母，不是分子。** 「3bet 12%」这个数，分母取「所有手」还是
「面对加注的手」，差出一个数量级，而两边都能算出一个看着合理的百分比。
所以每条用例都打一手**真牌局**，然后核对「这次算不算一次机会」。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from holdem.actions import bet, call, check, fold, raise_to  # noqa: E402
from holdem.deck import deck_from_seed  # noqa: E402
from holdem.state import HandConfig, HandState  # noqa: E402
from holdem.stats import Chance, StatLine, accumulate, hand_stats  # noqa: E402


def _play(actions, *, stacks=None, button=0, bb=10, sb=5, seed=7, finish=True):
    config = HandConfig(
        stacks=tuple(stacks or [1000] * 6), button=button, big_blind=bb, small_blind=sb
    )
    hand = HandState(config, deck_from_seed(seed))
    for action in actions:
        hand.apply(action)
    if finish:
        _finish(hand)
    return hand


def _finish(hand):
    """把剩下的街用「能过牌就过牌」推到底。

    统计只对**打完的**牌成立，所以每条用例都得把牌打完——但用例关心的动作
    只发生在前面几街，后面怎么打无所谓，只要别改变已经发生的事实。
    过牌不会改变任何一个被测指标（它既不是主动投钱，也不是下注或跟注）。
    """
    guard = 0
    while not hand.is_complete:
        legal = hand.legal_actions()
        hand.apply(check() if legal.can_check else fold())
        guard += 1
        if guard > 200:
            raise AssertionError("推不完这手牌")


# ------------------------------------------------------------------ 没机会 ≠ 没做


def test_no_chance_is_not_zero_percent():
    """「从没面对过 3bet」和「面对 3bet 从不弃牌」是两件事，压成 0% 就分不开了。"""
    assert Chance().rate is None
    c = Chance()
    c.observe(False)
    assert c.rate == 0.0, "有机会但没做，才是 0%"


def test_aggression_factor_without_calls_is_unknown_not_infinite():
    """一次没跟注过时 AF 没有意义；给 inf 只会让它在报告里排到第一名。"""
    line = StatLine()
    line.postflop_aggressive = 3
    assert line.aggression_factor is None
    line.postflop_calls = 2
    assert line.aggression_factor == 1.5


# ------------------------------------------------------------------ 翻前


def test_rfi_denominator_is_folded_to_me_not_all_hands():
    """RFI 的分母是「前面全弃」的手，不是所有手。

    六人桌按钮 0：小盲 1、大盲 2、UTG 3。UTG 开牌，后面的人**都不算有 RFI 机会**
    ——他们面对的是一个加注，不是一张白纸。
    """
    hand = _play([raise_to(30), fold(), fold(), fold(), fold(), fold()])
    stats = hand_stats(hand)
    assert stats[3].rfi.chances == 1 and stats[3].rfi.hits == 1
    for seat in (4, 5, 0, 1):
        assert stats[seat].rfi.chances == 0, f"座位 {seat} 面对的是加注，没有开牌机会"


def test_a_raise_behind_a_limper_is_isolation_not_rfi():
    """跟在跛入者后面的加注是**隔离**，不是开牌——分子分母必须同一个判据。"""
    hand = _play([call(), raise_to(40), fold(), fold(), fold(), fold(), fold()])
    stats = hand_stats(hand)
    assert stats[3].rfi.chances == 1 and stats[3].rfi.hits == 0, "跛入者有机会但没开"
    assert stats[4].rfi.chances == 0, "前面已有人入池，这就不是开牌了"
    assert stats[4].pfr.hits == 1, "但它确确实实是一次翻前加注"


def test_threebet_only_counts_facing_exactly_one_raise():
    hand = _play([raise_to(30), raise_to(90), fold(), fold(), fold(), fold(), fold()])
    stats = hand_stats(hand)
    assert stats[4].threebet.chances == 1 and stats[4].threebet.hits == 1
    assert stats[3].threebet.chances == 0, "开牌那个人不是在 3bet"


def test_fold_to_threebet_denominator_is_i_opened_and_got_reraised():
    """分母是「**我**开了牌又被 3bet」，不是「所有被 3bet 的场合」。"""
    hand = _play([raise_to(30), raise_to(90), fold(), fold(), fold(), fold(), fold()])
    stats = hand_stats(hand)
    assert stats[3].fold_to_threebet.chances == 1
    assert stats[3].fold_to_threebet.hits == 1, "UTG 开牌后被 3bet 且弃牌了"
    assert stats[4].fold_to_threebet.chances == 0, "3bet 的人自己不在这个分母里"


def test_blinds_are_not_voluntary_money():
    """大盲被动过牌进翻牌，VPIP 必须是 0——盲注不是主动投钱。"""
    hand = _play([call(), fold(), fold(), fold(), fold(), check()])
    stats = hand_stats(hand)
    assert stats[2].vpip.hits == 0, "大盲只是过牌，没主动投过钱"
    assert stats[2].vpip.chances == 1, "但这手牌照样计入他的分母"
    assert stats[3].vpip.hits == 1, "跛入是主动投钱"


# ------------------------------------------------------------------ 翻后


def _to_flop():
    """UTG 开牌、大盲跟，两人看翻牌。翻前进攻方是座位 3。"""
    return [raise_to(30), fold(), fold(), fold(), fold(), call()]


def test_cbet_chance_requires_nobody_bet_first():
    """被人抢先下注时你**没有**持续下注的机会——把它算进分母＝把「没机会」记成「放弃」。"""
    hand = _play(_to_flop() + [bet(40)])          # 大盲（座位 2）先下注
    stats = hand_stats(hand)
    assert stats[3].cbet_flop.chances == 0, "翻前进攻方被抢先下注了，没有 cbet 机会"


def test_cbet_counted_when_the_preflop_aggressor_bets_the_flop():
    hand = _play(_to_flop() + [check(), bet(40)])
    stats = hand_stats(hand)
    assert stats[3].cbet_flop.chances == 1 and stats[3].cbet_flop.hits == 1


def test_checking_back_the_flop_is_a_missed_cbet_not_a_missing_stat():
    hand = _play(_to_flop() + [check(), check()])
    stats = hand_stats(hand)
    assert stats[3].cbet_flop.chances == 1 and stats[3].cbet_flop.hits == 0


def test_fold_to_cbet_is_measured_on_the_player_facing_it():
    hand = _play(_to_flop() + [check(), bet(40), fold()])
    stats = hand_stats(hand)
    assert stats[2].fold_to_cbet_flop.chances == 1
    assert stats[2].fold_to_cbet_flop.hits == 1
    assert stats[3].fold_to_cbet_flop.chances == 0, "下注的人不在这个分母里"


def test_wtsd_denominator_is_saw_the_flop():
    """没看到翻牌的人不进 WTSD 的分母——否则「弃牌多」会被读成「不爱摊牌」。"""
    hand = _play(_to_flop() + [check(), check()])
    stats = hand_stats(hand)
    assert stats[3].wtsd.chances == 1 and stats[2].wtsd.chances == 1
    for seat in (4, 5, 0, 1):
        assert stats[seat].wtsd.chances == 0, f"座位 {seat} 翻前就弃了"


# ------------------------------------------------------------------ 按位置


def test_by_position_keeps_the_same_seat_apart():
    """开牌率混在一起会随对手松紧漂移，所以要能按位置拆。"""
    into = {}
    accumulate(_play([raise_to(30), fold(), fold(), fold(), fold(), fold()]),
               into, by_position=True)
    keys = sorted(into)
    assert all(isinstance(k, tuple) and len(k) == 2 for k in keys)
    assert (3, "UTG") in into and into[(3, "UTG")].rfi.hits == 1


def test_lines_merge_without_losing_denominators():
    a, b = StatLine(), StatLine()
    a.hands = 1; a.rfi.observe(True); a.postflop_calls = 2
    b.hands = 1; b.rfi.observe(False); b.postflop_aggressive = 3
    a.add(b)
    assert a.hands == 2 and a.rfi.chances == 2 and a.rfi.hits == 1
    assert a.aggression_factor == 1.5


def test_an_unfinished_hand_says_so_instead_of_crashing():
    """没打完的牌算不出摊牌类指标——硬算会把「还没到摊牌」记成「没走到摊牌」。"""
    import pytest

    hand = _play([raise_to(30)], finish=False)
    with pytest.raises(ValueError, match="还没打完"):
        hand_stats(hand)


# ------------------------------------------------------------------ 与 batch 的口径对账


def test_the_shared_metrics_match_the_ones_batch_already_computes():
    """**两份统计必须给出同一批数。**

    `batch.py` 里有一份边跑边累计的统计，口径是这份的子集。合并成一份是下一段的事，
    但在合并之前得先钉住它们**没有漂**——两个模块各算各的、谁也不知道对方变了，
    才是这类重复实现真正的代价。

    这条用例一红，要么是有人改了判据（那得两边一起改、并说明历史数据不可比），
    要么就是漂了。
    """
    from holdem.batch import SeatStats, _collect

    hands = [
        _play([raise_to(30), fold(), fold(), fold(), fold(), fold()]),
        _play([call(), raise_to(40), fold(), fold(), fold(), fold(), fold()]),
        _play([raise_to(30), raise_to(90), fold(), fold(), fold(), fold(), fold()]),
        _play([raise_to(30), fold(), fold(), fold(), fold(), call(), check(), bet(40), fold()]),
    ]

    batch_stats = [SeatStats(seat=i, style="x") for i in range(6)]
    mine: dict = {}
    for hand in hands:
        _collect(hand, batch_stats)
        accumulate(hand, mine)

    for seat in range(6):
        theirs, ours = batch_stats[seat], mine[seat]
        assert ours.hands == theirs.hands, seat
        assert ours.vpip.hits == theirs.vpip_hands, f"座位 {seat} 的 VPIP 漂了"
        assert ours.pfr.hits == theirs.pfr_hands, f"座位 {seat} 的 PFR 漂了"
        assert ours.rfi.chances == theirs.open_chances, f"座位 {seat} 的开牌**机会**漂了"
        assert ours.rfi.hits == theirs.open_hands, f"座位 {seat} 的开牌漂了"
        assert ours.threebet.chances == theirs.threebet_chances, f"座位 {seat} 的 3bet 机会漂了"
        assert ours.threebet.hits == theirs.threebet_hands, f"座位 {seat} 的 3bet 漂了"
        assert ours.wtsd.chances == theirs.flops, f"座位 {seat} 的 WTSD 分母漂了"
        assert ours.wsd.chances == theirs.showdowns, f"座位 {seat} 的摊牌数漂了"
        assert ours.wsd.hits == theirs.showdown_wins, f"座位 {seat} 的摊牌胜漂了"

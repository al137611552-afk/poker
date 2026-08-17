"""权益估算与占位 bot 的测试。

bot 只是占位实现，所以测的不是「打得好」，而是「不违规、不卡死、风格参数确实有效」。
"""

import random

import pytest

from holdem.actions import ActionKind
from holdem.bots import STYLES, RuleBot, play_out
from holdem.cards import cards_from_str
from holdem.deck import deck_from_seed
from holdem.equity import monte_carlo_equity
from holdem.history import action_records
from holdem.state import PREFLOP, HandConfig, HandState

# ------------------------------------------------------------------ 权益


def test_aces_beat_a_random_hand_about_85_percent():
    equity = monte_carlo_equity(
        cards_from_str("AsAd"), num_opponents=1, samples=4000, rng=random.Random(1)
    )
    assert 0.82 <= equity <= 0.88, f"AA 对一个随机手牌的胜率约 85%，实测 {equity:.3f}"


def test_worst_hand_is_much_weaker():
    equity = monte_carlo_equity(
        cards_from_str("7s2h"), num_opponents=1, samples=4000, rng=random.Random(2)
    )
    assert 0.30 <= equity <= 0.40, f"72o 约 35%，实测 {equity:.3f}"


def test_equity_falls_with_more_opponents():
    kwargs = dict(samples=2000, rng=random.Random(3))
    heads_up = monte_carlo_equity(cards_from_str("AsAd"), num_opponents=1, **kwargs)
    five_way = monte_carlo_equity(
        cards_from_str("AsAd"), num_opponents=5, samples=2000, rng=random.Random(3)
    )
    assert five_way < heads_up - 0.2


def test_made_nuts_on_the_river_is_certain():
    equity = monte_carlo_equity(
        cards_from_str("AsKs"),
        cards_from_str("QsJsTs2c3d"),
        num_opponents=3,
        samples=400,
        rng=random.Random(4),
    )
    assert equity == 1.0, "皇家同花顺在河牌不可能被超越"


def test_identical_seed_reproduces_estimate():
    a = monte_carlo_equity(cards_from_str("9h9d"), num_opponents=2, samples=300, rng=random.Random(9))
    b = monte_carlo_equity(cards_from_str("9h9d"), num_opponents=2, samples=300, rng=random.Random(9))
    assert a == b


def test_equity_input_validation():
    with pytest.raises(ValueError):
        monte_carlo_equity(cards_from_str("As"), samples=10)
    with pytest.raises(ValueError):
        monte_carlo_equity(cards_from_str("AsAs"), samples=10)
    with pytest.raises(ValueError):
        monte_carlo_equity(cards_from_str("AsKs"), num_opponents=0, samples=10)


# ------------------------------------------------------------------ bot


def _table(num_seats, style="tag", seed=1, stacks=1000):
    config = HandConfig(
        stacks=(stacks,) * num_seats, button=0, big_blind=10, small_blind=5
    )
    hand = HandState(config, deck_from_seed(seed))
    bots = {seat: RuleBot(style, seed=seed * 100 + seat) for seat in range(num_seats)}
    return hand, bots


def test_bot_only_produces_legal_actions():
    for seed in range(30):
        hand, bots = _table(6, seed=seed)
        while not hand.is_complete:
            legal = hand.legal_actions()
            action = bots[hand.to_act].act(hand)
            assert legal.contains(action), f"bot 给出了非法动作 {action}"
            hand.apply(action)


def test_bots_finish_hands_across_table_sizes():
    for num_seats in range(2, 10):
        for seed in range(5):
            hand, bots = _table(num_seats, seed=seed * 7 + num_seats)
            play_out(hand, bots)
            assert hand.is_complete
            assert sum(hand.stacks) == num_seats * 1000


def test_bots_handle_short_stacks():
    config = HandConfig(stacks=(12, 1000, 35, 7), button=0, big_blind=10, small_blind=5)
    for seed in range(20):
        hand = HandState(config, deck_from_seed(seed))
        bots = {seat: RuleBot("lag", seed=seed + seat) for seat in range(4)}
        play_out(hand, bots)
        assert sum(hand.stacks) == 12 + 1000 + 35 + 7


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError, match="未知风格"):
        RuleBot("超级高手")


def test_styles_differ_in_looseness():
    """疯子应该比岩石更常主动投钱——不比强弱，只验证参数确实起作用。"""

    def voluntary_rate(style):
        entered = 0
        total = 0
        for seed in range(60):
            hand, _ = _table(6, seed=seed)
            bots = {seat: RuleBot(style, seed=seed * 10 + seat) for seat in range(6)}
            # 只看枪口位第一个决策
            action = bots[hand.to_act].act(hand)
            total += 1
            if action.kind is not ActionKind.FOLD:
                entered += 1
        return entered / total

    nit = voluntary_rate("nit")
    maniac = voluntary_rate("maniac")
    assert maniac > nit + 0.2, f"疯子 {maniac:.2f} 应明显松于岩石 {nit:.2f}"


def _preflop_stats(style, hands=120):
    """跑一批六人桌自对弈，量出 VPIP 与 PFR。"""
    rng = random.Random(7)
    voluntary = raises = opportunities = 0
    for index in range(hands):
        config = HandConfig(
            stacks=(1000,) * 6, button=index % 6, big_blind=10, small_blind=5
        )
        hand = HandState(config, deck_from_seed(rng.randrange(1 << 30)))
        bots = {s: RuleBot(style, seed=rng.randrange(1 << 30)) for s in range(6)}
        play_out(hand, bots)
        seen = set()
        for record in action_records(hand):
            if record.street != PREFLOP or record.seat in seen:
                continue
            seen.add(record.seat)
            opportunities += 1
            if record.kind in ("call", "bet", "raise"):
                voluntary += 1
            if record.kind == "raise":
                raises += 1
    return voluntary / opportunities, raises / opportunities


def test_style_archetypes_are_recognisable():
    """VPIP/PFR 要能把风格区分开——这是 M1 风格层的回归基线。"""
    nit_vpip, nit_pfr = _preflop_stats("nit")
    tag_vpip, tag_pfr = _preflop_stats("tag")
    lag_vpip, _ = _preflop_stats("lag")
    station_vpip, station_pfr = _preflop_stats("station")
    maniac_vpip, maniac_pfr = _preflop_stats("maniac")

    assert nit_vpip < tag_vpip < lag_vpip < maniac_vpip, (
        f"入池率应随风格变松：岩石 {nit_vpip:.2f} < 紧凶 {tag_vpip:.2f} "
        f"< 松凶 {lag_vpip:.2f} < 疯子 {maniac_vpip:.2f}"
    )
    assert nit_vpip < 0.15, f"岩石入池率过高: {nit_vpip:.2f}"
    assert maniac_vpip > 0.55, f"疯子入池率过低: {maniac_vpip:.2f}"
    assert station_vpip > 0.4 and station_pfr < 0.12, (
        f"跟注站应松而不凶，实测 VPIP {station_vpip:.2f} / PFR {station_pfr:.2f}"
    )
    assert maniac_pfr > tag_pfr > nit_pfr


def test_bot_is_deterministic_given_a_seed():
    def run():
        hand, bots = _table(4, seed=5)
        play_out(hand, bots)
        return hand.stacks

    assert run() == run()


def test_play_out_rejects_missing_bot():
    hand, bots = _table(3)
    del bots[hand.to_act]
    with pytest.raises(KeyError):
        play_out(hand, bots)


def test_all_styles_are_playable():
    for style in STYLES:
        hand, bots = _table(3, style=style, seed=11)
        play_out(hand, bots)
        assert hand.is_complete

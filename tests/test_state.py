"""状态机测试：位置与行动顺序、加注规则、边池、筹码守恒。"""

import random

import pytest

from holdem.actions import bet, call, check, fold, raise_to
from holdem.cards import cards_from_str
from holdem.deck import deck_from_seed, stacked_deck
from holdem.state import COMPLETE, FLOP, PREFLOP, RIVER, TURN, HandConfig, HandState


def make_hand(stacks, button=0, bb=10, sb=5, ante=0, deck=None, seed=1):
    config = HandConfig(
        stacks=tuple(stacks), button=button, big_blind=bb, small_blind=sb, ante=ante
    )
    return HandState(config, deck if deck is not None else deck_from_seed(seed))


# ------------------------------------------------------------------ 位置与顺序


def test_heads_up_button_is_small_blind_and_acts_first_preflop():
    hand = make_hand([1000, 1000], button=0)
    assert hand.sb_seat == 0 and hand.bb_seat == 1
    assert hand.committed_street == [5, 10]
    assert hand.to_act == 0


def test_heads_up_big_blind_acts_first_postflop():
    hand = make_hand([1000, 1000], button=0)
    hand.apply(call())
    assert hand.to_act == 1, "大盲应有后位选择权"
    hand.apply(check())
    assert hand.street == FLOP
    assert hand.to_act == 1, "翻后由大盲先行动"


def test_six_max_blind_positions_and_utg():
    hand = make_hand([1000] * 6, button=0)
    assert hand.sb_seat == 1 and hand.bb_seat == 2
    assert hand.committed_street == [0, 5, 10, 0, 0, 0]
    assert hand.to_act == 3, "六人桌由大盲左手（枪口位）先行动"


def test_postflop_order_starts_left_of_button():
    hand = make_hand([1000] * 6, button=0)
    for _ in range(4):  # UTG, HJ, CO 跟注，庄家跟注
        hand.apply(call())
    hand.apply(call())  # 小盲补齐
    hand.apply(check())  # 大盲过牌
    assert hand.street == FLOP
    assert hand.to_act == 1, "翻后从庄家左手第一位开始"


# ------------------------------------------------------------------ 下注规则


def test_min_raise_is_previous_raise_size():
    hand = make_hand([1000] * 3, button=0)
    legal = hand.legal_actions()
    assert legal.min_raise_to == 20, "首次加注最小为两倍大盲"
    hand.apply(raise_to(30))
    legal = hand.legal_actions()
    assert legal.min_raise_to == 50, "再加注的增量不得小于上一次加注的增量"
    assert legal.call_to == 30


def test_raise_below_minimum_is_rejected():
    hand = make_hand([1000] * 3, button=0)
    hand.apply(raise_to(30))
    with pytest.raises(ValueError):
        hand.apply(raise_to(40))


def test_short_all_in_does_not_reopen_betting():
    # 大盲（座位 2）只有 45 筹码，全下到 45，增量 15 < 上一次加注的 20
    # 座位 0 保持存活，否则「无人可跟」这条规则会先一步禁掉加注，测不到重开权
    hand = make_hand([1000, 1000, 45, 1000], button=0)
    hand.apply(raise_to(30))  # 座位 3（枪口位）
    hand.apply(call())  # 座位 0（庄家）
    hand.apply(fold())  # 座位 1（小盲）
    assert hand.to_act == 2
    assert hand.legal_actions().max_raise_to == 45
    hand.apply(raise_to(45))

    assert hand.to_act == 3
    legal = hand.legal_actions()
    assert legal.call_to == 45
    assert not legal.can_raise, "面对未达最小加注额的全下，已行动过的玩家不能再加注"
    hand.apply(call())

    legal = hand.legal_actions()
    assert legal.seat == 0
    assert legal.call_to == 45
    assert not legal.can_raise, "同样已行动过的座位 0 也只能跟或弃"


def test_full_all_in_raise_does_reopen_betting():
    hand = make_hand([1000, 1000, 60, 1000], button=0)
    hand.apply(raise_to(30))  # 座位 3
    hand.apply(call())  # 座位 0
    hand.apply(fold())  # 座位 1
    hand.apply(raise_to(60))  # 座位 2 全下，增量 30 >= 20，属完整加注
    legal = hand.legal_actions()
    assert legal.seat == 3
    assert legal.can_raise, "面对完整加注应重新获得加注权"
    assert legal.min_raise_to == 90


def test_cannot_raise_when_no_opponent_can_act():
    hand = make_hand([1000, 40], button=0)
    hand.apply(raise_to(40))  # 座位 0 把对手打成必须全下的额度
    legal = hand.legal_actions()
    assert legal.seat == 1
    assert legal.call_to == 40
    assert not legal.can_raise, "对手已无筹码可再战时不应允许加注"


def test_check_not_allowed_when_facing_bet():
    hand = make_hand([1000, 1000], button=0)
    legal = hand.legal_actions()
    assert not legal.can_check
    assert legal.can_fold and legal.can_call
    with pytest.raises(ValueError):
        hand.apply(check())


def test_bet_and_raise_kinds_are_not_interchangeable():
    hand = make_hand([1000, 1000], button=0)
    with pytest.raises(ValueError):
        hand.apply(bet(30))  # 翻前已有盲注，应为 raise
    hand.apply(call())
    hand.apply(check())
    assert hand.street == FLOP
    with pytest.raises(ValueError):
        hand.apply(raise_to(30))  # 翻后无人下注，应为 bet
    hand.apply(bet(30))
    assert hand.current_bet == 30


# ------------------------------------------------------------------ 结算


def test_fold_around_returns_uncalled_blind():
    hand = make_hand([1000] * 6, button=0)
    for _ in range(4):  # UTG..BTN 全部弃牌
        hand.apply(fold())
    hand.apply(fold())  # 小盲弃牌
    assert hand.is_complete
    result = hand.result
    assert result.net[1] == -5
    assert result.net[2] == 5, "大盲赢下小盲，未被跟注的部分应退回"
    assert sum(result.net) == 0
    assert not result.went_to_showdown


def test_all_in_preflop_runs_out_the_board():
    deck = stacked_deck(
        hole={0: "AsAd", 1: "KsKd"}, board="2c3d4h5s7c", num_seats=2, button=0
    )
    hand = make_hand([500, 500], button=0, deck=deck)
    hand.apply(raise_to(500))
    hand.apply(call())
    assert hand.is_complete
    assert len(hand.board) == 5
    assert hand.board == cards_from_str("2c3d4h5s7c")
    assert hand.result.went_to_showdown
    assert hand.stacks == [1000, 0]


def test_split_pot_returns_even_stacks():
    deck = stacked_deck(
        hole={0: "AcKc", 1: "AdKd"}, board="AhKh2s7d9c", num_seats=2, button=0
    )
    hand = make_hand([500, 500], button=0, deck=deck)
    hand.apply(raise_to(500))
    hand.apply(call())
    assert hand.stacks == [500, 500]
    assert sum(hand.result.net) == 0


def test_side_pot_short_stack_cannot_win_main_pot_excess():
    # 座位 2 短筹码全下并拿到最好的牌，只能赢主池
    deck = stacked_deck(
        hole={0: "7c2d", 1: "8c3d", 2: "AsAd"},
        board="AhKd9s4c2h",
        num_seats=3,
        button=0,
    )
    hand = make_hand([1000, 1000, 100], button=0, deck=deck, bb=10, sb=5)
    hand.apply(raise_to(100))  # 座位 0（三人桌的庄家先行动）
    hand.apply(call())  # 座位 1 小盲跟注
    hand.apply(call())  # 座位 2 大盲全下 100

    # 座位 2 全下后牌局并未结束：0 与 1 仍有筹码，翻后还要继续下注
    assert not hand.is_complete
    assert hand.street == FLOP
    for _ in range(6):  # 翻牌、转牌、河牌各过牌一轮
        hand.apply(check())
    assert hand.is_complete

    # 三人投入相同，不产生边池；三条 A 通吃
    assert sum(p.amount for p in hand.result.pots) == 300
    assert len(hand.result.pots) == 1
    assert hand.stacks[2] == 300


def test_side_pot_with_three_different_stacks():
    deck = stacked_deck(
        hole={0: "AsAd", 1: "KsKd", 2: "QsQd"},
        board="2c3d4h5s7c",
        num_seats=3,
        button=0,
    )
    hand = make_hand([1000, 300, 100], button=0, deck=deck)
    # 加注额封顶在对手能跟到的 300，而不是自己的 1000
    assert hand.legal_actions().max_raise_to == 300
    hand.apply(raise_to(300))
    hand.apply(call())  # 座位 1 全下 300
    hand.apply(call())  # 座位 2 全下 100
    result = hand.result
    assert sum(result.net) == 0
    assert len(result.pots) == 2, "应有主池与一层边池"
    # AA 通吃两个池
    assert hand.stacks == [1400, 0, 0]


# ------------------------------------------------------------------ 随机对局的不变量


def _random_action(hand, rng):
    legal = hand.legal_actions()
    choices = []
    if legal.can_fold:
        choices.append(fold())
    if legal.can_check:
        choices.append(check())
    if legal.can_call:
        choices.append(call())
    if legal.can_raise:
        sizes = {legal.min_raise_to, legal.max_raise_to}
        mid = (legal.min_raise_to + legal.max_raise_to) // 2
        if legal.min_raise_to <= mid <= legal.max_raise_to:
            sizes.add(mid)
        for size in sizes:
            choices.append(
                bet(size) if legal.is_opening_bet else raise_to(size)
            )
    assert choices, f"座位 {legal.seat} 无任何合法动作: {legal}"
    return rng.choice(choices)


def _play_random_hand(rng, num_seats, ante=0):
    stacks = [rng.choice([2, 7, 25, 100, 400, 1000, 5000]) for _ in range(num_seats)]
    hand = make_hand(
        stacks,
        button=rng.randrange(num_seats),
        ante=ante,
        deck=deck_from_seed(rng.randrange(1 << 30)),
    )
    steps = 0
    while not hand.is_complete:
        hand.apply(_random_action(hand, rng))
        steps += 1
        assert steps < 400, "牌局未能收敛，可能存在行动顺序死循环"
    return stacks, hand


def test_random_hands_conserve_chips():
    rng = random.Random(20260817)
    for _ in range(3000):
        num_seats = rng.randint(2, 9)
        stacks, hand = _play_random_hand(rng, num_seats)
        assert sum(hand.stacks) == sum(stacks), "筹码总量必须守恒"
        assert all(s >= 0 for s in hand.stacks), "筹码不得为负"
        assert sum(hand.result.net) == 0
        for seat in range(num_seats):
            assert hand.stacks[seat] == stacks[seat] + hand.result.net[seat]


def test_random_hands_with_antes_conserve_chips():
    rng = random.Random(99)
    for _ in range(500):
        num_seats = rng.randint(2, 9)
        stacks, hand = _play_random_hand(rng, num_seats, ante=1)
        assert sum(hand.stacks) == sum(stacks)
        assert all(s >= 0 for s in hand.stacks)


def test_random_hands_have_consistent_structure():
    rng = random.Random(5)
    for _ in range(500):
        num_seats = rng.randint(2, 9)
        _, hand = _play_random_hand(rng, num_seats)
        assert hand.street == COMPLETE
        assert hand.to_act == -1
        board_len = len(hand.board)
        assert board_len in (0, 3, 4, 5)
        if hand.result.went_to_showdown:
            assert board_len == 5, "摊牌时必须发满五张公共牌"
            assert len(hand.result.showdown_scores) >= 2
        all_cards = list(hand.board) + [c for h in hand.hole for c in h]
        assert len(set(all_cards)) == len(all_cards), "发出的牌不得重复"

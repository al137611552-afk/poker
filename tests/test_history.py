"""决策点上下文还原测试。

统计指标全部建在 ActionRecord 上，所以这里逐字段核对，而不是只看「没报错」。
"""

import random

from holdem.actions import bet, call, check, fold, raise_to
from holdem.deck import deck_from_seed
from holdem.history import action_records
from holdem.state import FLOP, PREFLOP, HandConfig, HandState


def _play(stacks, button, actions, bb=10, sb=5, ante=0, seed=7):
    config = HandConfig(
        stacks=tuple(stacks), button=button, big_blind=bb, small_blind=sb, ante=ante
    )
    hand = HandState(config, deck_from_seed(seed))
    for action in actions:
        hand.apply(action)
    return hand


def test_preflop_context_fields():
    # 六人桌，按钮在座位 0：小盲 1、大盲 2、枪口位 3
    hand = _play(
        [1000] * 6,
        button=0,
        actions=[raise_to(30), fold(), fold(), call(), fold(), call()],
    )
    records = action_records(hand)
    preflop = [r for r in records if r.street == PREFLOP]
    assert len(preflop) == 6

    first = preflop[0]
    assert (first.seat, first.position, first.kind) == (3, "UTG", "raise")
    assert first.pot_before == 15, "行动前底池只有两个盲注"
    assert first.bet_before == 10 and first.to_call == 10
    assert first.to == 30 and first.amount == 30
    assert first.actors_before == 6
    assert first.is_voluntary

    button_call = preflop[3]
    assert (button_call.seat, button_call.position, button_call.kind) == (0, "BTN", "call")
    assert button_call.pot_before == 45
    assert button_call.to_call == 30 and button_call.amount == 30
    assert button_call.actors_before == 4

    bb_call = preflop[5]
    assert (bb_call.seat, bb_call.position, bb_call.kind) == (2, "BB", "call")
    assert bb_call.pot_before == 75
    assert bb_call.to_call == 20, "大盲已投入 10，只需补 20"
    assert bb_call.amount == 20 and bb_call.to == 30
    assert bb_call.actors_before == 3


def test_blinds_are_not_voluntary():
    hand = _play([1000] * 3, button=0, actions=[fold(), fold()])
    records = action_records(hand)
    assert all(not r.is_voluntary for r in records), "只有弃牌，不该有主动投钱"
    assert [r.kind for r in records] == ["fold", "fold"]


def test_street_reset_and_flop_context():
    hand = _play(
        [1000] * 3,
        button=0,
        actions=[call(), call(), check(), check(), check(), bet(40), fold(), fold()],
    )
    flop = [r for r in action_records(hand) if r.street == FLOP]
    assert [r.kind for r in flop] == ["check", "check", "bet", "fold", "fold"]
    assert flop[0].bet_before == 0, "新的一街下注额应清零"
    assert flop[0].to_call == 0
    assert flop[0].pot_before == 30
    the_bet = flop[2]
    assert the_bet.to == 40 and the_bet.amount == 40
    assert the_bet.pot_before == 30
    assert flop[3].to_call == 40 and flop[3].pot_before == 70


def test_ante_counted_in_pot_but_not_in_street_bet():
    hand = _play([1000] * 6, button=0, actions=[fold()] * 5, ante=1)
    first = action_records(hand)[0]
    assert first.pot_before == 6 + 15, "前注进底池，但不影响本街需要跟注的额度"
    assert first.bet_before == 10
    assert first.to_call == 10


def test_stack_before_tracks_remaining_chips():
    hand = _play([1000, 1000, 200], button=0, actions=[raise_to(60), call(), call()])
    records = action_records(hand)
    bb_action = next(r for r in records if r.seat == 2)
    assert bb_action.stack_before == 190, "大盲已投入 10，还剩 190"


def test_amounts_reconcile_with_final_pot():
    rng = random.Random(31337)
    for _ in range(300):
        n = rng.randint(2, 9)
        config = HandConfig(
            stacks=tuple(rng.choice([25, 100, 400, 1000]) for _ in range(n)),
            button=rng.randrange(n),
            big_blind=10,
            small_blind=5,
            ante=rng.choice([0, 0, 1]),
        )
        hand = HandState(config, deck_from_seed(rng.randrange(1 << 30)))
        while not hand.is_complete:
            legal = hand.legal_actions()
            choices = []
            if legal.can_check:
                choices.append(check())
            if legal.can_call:
                choices.append(call())
            if legal.can_fold:
                choices.append(fold())
            if legal.can_raise:
                size = rng.choice([legal.min_raise_to, legal.max_raise_to])
                choices.append(bet(size) if legal.is_opening_bet else raise_to(size))
            hand.apply(rng.choice(choices))

        records = action_records(hand)
        blinds_and_antes = sum(
            e.amount for e in hand.events if e.kind in ("blind", "ante")
        )
        voluntary = sum(r.amount for r in records)
        assert blinds_and_antes + voluntary == sum(hand.result.contributions)
        assert all(r.pot_before >= 0 and r.stack_before >= 0 for r in records)
        assert all(r.actors_before >= 1 for r in records)

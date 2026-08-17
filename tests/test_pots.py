import pytest

from holdem.pots import Pot, award, build_pots, refund_uncalled


def test_no_refund_when_matched():
    adjusted, refunds = refund_uncalled([100, 100, 100])
    assert adjusted == [100, 100, 100]
    assert refunds == [0, 0, 0]


def test_refunds_uncalled_excess():
    adjusted, refunds = refund_uncalled([300, 100, 0])
    assert adjusted == [100, 100, 0]
    assert refunds == [200, 0, 0]


def test_folded_players_count_toward_call_level():
    # 座位 1 弃牌但已投入 100，座位 0 的 100 因此算被跟到
    adjusted, refunds = refund_uncalled([100, 100])
    assert refunds == [0, 0]
    assert adjusted == [100, 100]


def test_single_pot_when_all_equal():
    pots = build_pots([50, 50, 50], [False, False, False])
    assert pots == [Pot(150, (0, 1, 2))]


def test_side_pot_from_short_all_in():
    # 座位 0 全下 100，另两人打到 300
    pots = build_pots([100, 300, 300], [False, False, False])
    assert pots == [Pot(300, (0, 1, 2)), Pot(400, (1, 2))]


def test_folded_money_stays_in_pot():
    # 座位 2 投入 100 后弃牌，钱留在主池但没有资格
    pots = build_pots([300, 300, 100], [False, False, True])
    # 资格相同的相邻档位会合并成一个池
    assert pots == [Pot(700, (0, 1))]


def test_two_side_pots():
    pots = build_pots([50, 150, 400, 400], [False] * 4)
    assert pots == [
        Pot(200, (0, 1, 2, 3)),
        Pot(300, (1, 2, 3)),
        Pot(500, (2, 3)),
    ]
    assert sum(p.amount for p in pots) == 1000


def test_conservation_and_eligibility_are_consistent():
    contributions = [20, 75, 75, 200, 0]
    folded = [True, False, False, False, False]
    pots = build_pots(contributions, folded)
    assert sum(p.amount for p in pots) == sum(contributions)
    for pot in pots:
        assert pot.eligible, "每个池都必须有可分配的对象"
        assert all(not folded[s] for s in pot.eligible)


def test_award_single_winner():
    pots = [Pot(300, (0, 1, 2))]
    payouts = award(pots, {0: 10, 1: 50, 2: 20}, first_seat_left_of_button=0, num_seats=3)
    assert payouts == [0, 300, 0]


def test_award_split_with_odd_chip_to_first_left_of_button():
    pots = [Pot(101, (0, 1))]
    payouts = award(pots, {0: 7, 1: 7}, first_seat_left_of_button=1, num_seats=2)
    assert payouts == [50, 51]
    payouts = award(pots, {0: 7, 1: 7}, first_seat_left_of_button=0, num_seats=2)
    assert payouts == [51, 50]


def test_award_side_pot_goes_to_eligible_only():
    pots = [Pot(300, (0, 1, 2)), Pot(400, (1, 2))]
    # 短筹码 0 牌最大，只能拿主池
    payouts = award(pots, {0: 99, 1: 50, 2: 20}, first_seat_left_of_button=0, num_seats=3)
    assert payouts == [300, 400, 0]


def test_award_rejects_pot_without_contender():
    with pytest.raises(ValueError):
        award([Pot(100, (0,))], {1: 5}, first_seat_left_of_button=0, num_seats=2)

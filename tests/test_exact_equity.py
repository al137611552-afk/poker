"""精确权益枚举的测试，以及用它去校验蒙特卡洛估计。

`exact_equity` 是全项目权益数值的基准，所以它自己必须用「独立重算」来验，
而不是跟蒙特卡洛互相印证——两者都错的时候互相印证会一起通过。
"""

from itertools import combinations

import pytest

from holdem.cards import FULL_DECK, cards_from_str
from holdem.equity import exact_equity, monte_carlo_equity
from holdem.evaluator import evaluate

import random


def test_completed_board_gives_a_definite_result():
    board = cards_from_str("2c3d4h5s7c")
    assert exact_equity(cards_from_str("AsAd"), cards_from_str("KsKd"), board) == 1.0
    assert exact_equity(cards_from_str("KsKd"), cards_from_str("AsAd"), board) == 0.0


def test_completed_board_split_pot_is_a_half():
    # 公共牌成顺，双方都只能用公共牌
    board = cards_from_str("AhKhQsJd Tc")
    assert exact_equity(cards_from_str("2c2d"), cards_from_str("3c3d"), board) == 0.5


def test_single_missing_card_matches_an_independent_enumeration():
    hole_a = cards_from_str("AsAd")
    hole_b = cards_from_str("7h8h")
    board = cards_from_str("2c9dTh")[:0] + cards_from_str("2c9dThJs")

    # 独立重算：直接遍历 44 张河牌
    known = set(hole_a + hole_b + board)
    rivers = [c for c in FULL_DECK if c not in known]
    assert len(rivers) == 44
    score = 0.0
    for river in rivers:
        full = board + [river]
        a, b = evaluate(hole_a + full), evaluate(hole_b + full)
        score += 1.0 if a > b else 0.5 if a == b else 0.0
    expected = score / len(rivers)

    assert exact_equity(hole_a, hole_b, board) == pytest.approx(expected)


def test_equities_are_complementary():
    hole_a = cards_from_str("QsQd")
    hole_b = cards_from_str("AhKh")
    board = cards_from_str("2c9dTh")
    forward = exact_equity(hole_a, hole_b, board)
    backward = exact_equity(hole_b, hole_a, board)
    assert forward + backward == pytest.approx(1.0)


def test_flop_equity_is_cheap_and_sane():
    # AA 在无危险面对 78s：领先但不是必胜
    equity = exact_equity(
        cards_from_str("AsAd"), cards_from_str("7h8h"), cards_from_str("2c9dKs")
    )
    assert 0.75 < equity < 0.95


def test_input_validation():
    with pytest.raises(ValueError, match="两张"):
        exact_equity(cards_from_str("As"), cards_from_str("KsKd"))
    with pytest.raises(ValueError, match="重复"):
        exact_equity(cards_from_str("AsAd"), cards_from_str("AsKd"))
    with pytest.raises(ValueError, match="五张"):
        exact_equity(
            cards_from_str("AsAd"), cards_from_str("KsKc"), cards_from_str("2c3d4h5s7c8d")
        )


def _exact_equity_vs_random_opponent(hole, board):
    """穷举全部对手底牌与剩余公共牌，得到「对上随机手牌」的精确权益。

    只在缺一张公共牌时才划算（约 4 万次求值），正好够用来校验蒙特卡洛。
    """
    known = set(hole) | set(board)
    unseen = [c for c in FULL_DECK if c not in known]
    total = 0.0
    count = 0
    for opponent in combinations(unseen, 2):
        rest = [c for c in unseen if c not in opponent]
        for river in rest:
            full = list(board) + [river]
            mine = evaluate(list(hole) + full)
            theirs = evaluate(list(opponent) + full)
            total += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
            count += 1
    return total / count


def test_monte_carlo_agrees_with_exact_ground_truth():
    """用精确值校验蒙特卡洛估计器——这是 bots.py 决策数值可信的前提。"""
    hole = cards_from_str("AsKs")
    board = cards_from_str("Qh7d2c9s")
    truth = _exact_equity_vs_random_opponent(hole, board)
    estimate = monte_carlo_equity(
        hole, board, num_opponents=1, samples=20000, rng=random.Random(11)
    )
    assert estimate == pytest.approx(truth, abs=0.01), (
        f"蒙特卡洛 {estimate:.4f} 与精确值 {truth:.4f} 偏差过大"
    )


@pytest.mark.slow
def test_preflop_matchups_match_published_values():
    """翻前全枚举（每个对局约 171 万个牌面），对照公开的权益数值。

    公开值是同一牌类下全部花色组合的平均，这里只算单一组合，
    因此允许约 1 个百分点的偏差。
    """
    aces_vs_kings = exact_equity(cards_from_str("AsAd"), cards_from_str("KsKd"))
    assert 0.80 < aces_vs_kings < 0.84, f"AA vs KK 公开值约 81.9%，实测 {aces_vs_kings:.4f}"

    ak_vs_queens = exact_equity(cards_from_str("AsKs"), cards_from_str("QhQd"))
    assert 0.44 < ak_vs_queens < 0.48, f"AKs vs QQ 公开值约 46%，实测 {ak_vs_queens:.4f}"

    ak_vs_deuces = exact_equity(cards_from_str("AsKh"), cards_from_str("2c2d"))
    assert 0.44 < ak_vs_deuces < 0.50, f"AKo vs 22 公开值约 47%，实测 {ak_vs_deuces:.4f}"

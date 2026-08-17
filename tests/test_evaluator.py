"""求值器测试。

核心手段是「双实现交叉验证」：这里另写一份慢速参考实现（枚举 7 选 5、用
Counter 分类），与 evaluator 的位运算快路径对比。两份实现的思路不同，
同时错成一样的概率很低。
"""

import random
from collections import Counter
from itertools import combinations

from holdem.cards import FULL_DECK, cards_from_str
from holdem.evaluator import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    evaluate,
    score_category,
)

# ------------------------------------------------------------------ 参考实现


def _reference_five(cards):
    """独立实现的 5 张牌分类，返回 (类别, 决胜点数元组)。"""
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    suits = {c & 3 for c in cards}
    counter = Counter(ranks)
    # 先按出现次数排，再按点数排
    grouped = sorted(counter.items(), key=lambda kv: (-kv[1], -kv[0]))
    shape = [count for _, count in grouped]
    ordered_ranks = [rank for rank, _ in grouped]

    is_flush = len(suits) == 1
    unique = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = -1
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            is_straight, straight_high = True, unique[0]
        elif unique == [12, 3, 2, 1, 0]:  # 轮子顺
            is_straight, straight_high = True, 3

    if is_flush and is_straight:
        return STRAIGHT_FLUSH, (straight_high,)
    if shape == [4, 1]:
        return QUADS, tuple(ordered_ranks)
    if shape == [3, 2]:
        return FULL_HOUSE, tuple(ordered_ranks)
    if is_flush:
        return FLUSH, tuple(ranks)
    if is_straight:
        return STRAIGHT, (straight_high,)
    if shape == [3, 1, 1]:
        return TRIPS, tuple(ordered_ranks)
    if shape == [2, 2, 1]:
        return TWO_PAIR, tuple(ordered_ranks)
    if shape == [2, 1, 1, 1]:
        return PAIR, tuple(ordered_ranks)
    return HIGH_CARD, tuple(ranks)


def _pack(category, kickers):
    score = 0
    for rank in kickers:
        score = (score << 4) | rank
    score <<= 4 * (5 - len(kickers))
    return (category << 20) | score


def reference_evaluate(cards):
    """慢速参考：枚举所有 5 张组合取最好的。"""
    return max(_pack(*_reference_five(five)) for five in combinations(cards, 5))


# ------------------------------------------------------------------ 已知牌型


def _score(text):
    return evaluate(cards_from_str(text))


def test_category_ordering():
    hands = [
        ("2c4d6h8sTc", HIGH_CARD),
        ("2c2d6h8sTc", PAIR),
        ("2c2d6h6sTc", TWO_PAIR),
        ("2c2d2h6sTc", TRIPS),
        ("2c3d4h5s6c", STRAIGHT),
        ("2c5c7c9cJc", FLUSH),
        ("2c2d2h6s6c", FULL_HOUSE),
        ("2c2d2h2s6c", QUADS),
        ("2c3c4c5c6c", STRAIGHT_FLUSH),
    ]
    scores = [_score(text) for text, _ in hands]
    for (text, expected), score in zip(hands, scores):
        assert score_category(score) == expected, text
    assert scores == sorted(scores), "牌型强弱顺序错误"


def test_wheel_straight():
    wheel = _score("As2c3d4h5s")
    six_high = _score("2c3d4h5s6c")
    assert score_category(wheel) == STRAIGHT
    assert wheel < six_high, "A-5 顺应为最小顺子"


def test_wheel_straight_flush_vs_higher():
    assert _score("Ac2c3c4c5c") < _score("2c3c4c5c6c")


def test_royal_flush_is_top():
    royal = _score("AsKsQsJsTs")
    assert score_category(royal) == STRAIGHT_FLUSH
    assert royal == max(
        evaluate(hand) for hand in combinations(cards_from_str("AsKsQsJsTs9s8s"), 5)
    )


def test_seven_card_picks_best_five():
    # 手牌完全没用，公共牌自成同花
    board_flush = evaluate(cards_from_str("2h3d" + "AsKsQsJs9s"))
    assert score_category(board_flush) == FLUSH


def test_two_pair_kicker_from_third_pair():
    # 三组对子时，第三组的点数只能当踢脚
    score = evaluate(cards_from_str("AcAdKcKdQcQd2s"))
    assert score_category(score) == TWO_PAIR
    assert score == _pack(TWO_PAIR, (12, 11, 10))


def test_two_trips_makes_full_house():
    score = evaluate(cards_from_str("AcAdAhKcKdKh2s"))
    assert score_category(score) == FULL_HOUSE
    assert score == _pack(FULL_HOUSE, (12, 11))


def test_quads_kicker():
    score = evaluate(cards_from_str("7c7d7h7s2c3d4h"))
    assert score == _pack(QUADS, (5, 2))


def test_straight_flush_beats_quads():
    sf = evaluate(cards_from_str("5c6c7c8c9c2d3d"))
    quads = evaluate(cards_from_str("8c8d8h8s2c3d4h"))
    assert score_category(sf) == STRAIGHT_FLUSH
    assert score_category(quads) == QUADS
    assert sf > quads


def test_trips_plus_straight_prefers_straight():
    # 5-9 顺子与三条同时成立时必须取顺子
    score = evaluate(cards_from_str("5c6d7h8s9d9h9s"))
    assert score_category(score) == STRAIGHT


def test_flush_uses_five_highest_of_suit():
    score = evaluate(cards_from_str("AcKcQcJc9c2c3c"))
    assert score == _pack(FLUSH, (12, 11, 10, 9, 7))


def test_split_pot_equality():
    board = cards_from_str("AcKdQhJs2c")
    a = evaluate([*cards_from_str("Th9h"), *board])
    b = evaluate([*cards_from_str("Td9d"), *board])
    assert a == b


def test_five_and_six_card_input():
    five = cards_from_str("AcKcQcJcTc")
    assert score_category(evaluate(five)) == STRAIGHT_FLUSH
    six = cards_from_str("AcKcQcJcTc2d")
    assert evaluate(six) == evaluate(five)


# ------------------------------------------------------------------ 交叉验证


def test_cross_check_random_seven_card_hands():
    rng = random.Random(20260817)
    deck = list(FULL_DECK)
    for _ in range(4000):
        hand = rng.sample(deck, 7)
        assert evaluate(hand) == reference_evaluate(hand), [
            f"{c:02d}" for c in sorted(hand)
        ]


def test_cross_check_biased_hands():
    """刻意制造重复点数与同花色，压向边界分支。"""
    rng = random.Random(7)
    for _ in range(2000):
        ranks = [rng.randrange(13) for _ in range(7)]
        used = set()
        hand = []
        for rank in ranks:
            for suit in rng.sample(range(4), 4):
                card = rank * 4 + suit
                if card not in used:
                    used.add(card)
                    hand.append(card)
                    break
        if len(hand) != 7:
            continue
        assert evaluate(hand) == reference_evaluate(hand)

"""5–7 张牌的牌力求值。

`evaluate(cards)` 返回一个整数分数，分数越大牌越强；同分即为平分底池。
分数布局：`category << 20 | 五个决胜点数各占 4 bit`，因此可直接用 `<` `>` 比较，
也可以安全地存进数据库当排序键。

这里刻意不引外部求值库：M0 阶段要的是可单测的纯逻辑，热路径的替换（查表或 Rust）
留到自对弈吞吐成为瓶颈时再做，届时用本文件的实现作为交叉验证基准。
"""

from __future__ import annotations

from .cards import NUM_RANKS

# 牌型类别，数值越大越强
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8

CATEGORY_NAMES = {
    HIGH_CARD: "高牌",
    PAIR: "一对",
    TWO_PAIR: "两对",
    TRIPS: "三条",
    STRAIGHT: "顺子",
    FLUSH: "同花",
    FULL_HOUSE: "葫芦",
    QUADS: "四条",
    STRAIGHT_FLUSH: "同花顺",
}

# 轮子顺（A-5-4-3-2）的点数掩码：A 与 5,4,3,2
_WHEEL_MASK = (1 << 12) | (1 << 3) | (1 << 2) | (1 << 1) | 1


def _pack(category: int, kickers: object) -> int:
    """把类别与决胜点数打包成单个可比较整数。"""
    score = 0
    for rank in kickers:  # type: ignore[union-attr]
        score = (score << 4) | rank
    # 补齐到 5 个槽位，保证不同长度的决胜序列之间比较仍然正确
    slots = 5 - len(kickers)  # type: ignore[arg-type]
    score <<= 4 * slots
    return (category << 20) | score


def _straight_high(rank_mask: int) -> int:
    """给定点数掩码，返回顺子的最大点数；无顺子返回 -1。轮子顺返回 3（五高）。"""
    for high in range(NUM_RANKS - 1, 3, -1):
        window = 0b11111 << (high - 4)
        if rank_mask & window == window:
            return high
    if rank_mask & _WHEEL_MASK == _WHEEL_MASK:
        return 3
    return -1


def evaluate(cards: object) -> int:
    """求 5–7 张牌中最好的五张的分数。传入的牌必须互不相同。"""
    cards = list(cards)  # type: ignore[arg-type]
    if not 5 <= len(cards) <= 7:
        raise ValueError(f"求值需要 5–7 张牌，收到 {len(cards)} 张")

    counts = [0] * NUM_RANKS
    suit_ranks = [0, 0, 0, 0]  # 每种花色的点数掩码
    suit_counts = [0, 0, 0, 0]
    rank_mask = 0

    for card in cards:
        rank = card >> 2
        suit = card & 3
        counts[rank] += 1
        rank_mask |= 1 << rank
        suit_ranks[suit] |= 1 << rank
        suit_counts[suit] += 1

    # ---- 同花 / 同花顺 ----
    flush_suit = -1
    for suit in range(4):
        if suit_counts[suit] >= 5:
            flush_suit = suit
            break

    if flush_suit >= 0:
        flush_mask = suit_ranks[flush_suit]
        sf_high = _straight_high(flush_mask)
        if sf_high >= 0:
            return _pack(STRAIGHT_FLUSH, [sf_high])

    # ---- 按重复数分组 ----
    quads: list[int] = []
    trips: list[int] = []
    pairs: list[int] = []
    singles: list[int] = []
    for rank in range(NUM_RANKS - 1, -1, -1):
        count = counts[rank]
        if count == 4:
            quads.append(rank)
        elif count == 3:
            trips.append(rank)
        elif count == 2:
            pairs.append(rank)
        elif count == 1:
            singles.append(rank)

    if quads:
        quad = quads[0]
        kicker = max(r for r in range(NUM_RANKS) if counts[r] > 0 and r != quad)
        return _pack(QUADS, [quad, kicker])

    if trips and (len(trips) >= 2 or pairs):
        # 两组三条时，较小的一组当作对子；三条 + 对子取最大的对子
        top = trips[0]
        second = trips[1] if len(trips) >= 2 else -1
        if pairs and pairs[0] > second:
            second = pairs[0]
        return _pack(FULL_HOUSE, [top, second])

    if flush_suit >= 0:
        flush_mask = suit_ranks[flush_suit]
        flush_ranks = [r for r in range(NUM_RANKS - 1, -1, -1) if flush_mask & (1 << r)]
        return _pack(FLUSH, flush_ranks[:5])

    straight = _straight_high(rank_mask)
    if straight >= 0:
        return _pack(STRAIGHT, [straight])

    if trips:
        trip = trips[0]
        kickers = [r for r in range(NUM_RANKS - 1, -1, -1) if counts[r] > 0 and r != trip][:2]
        return _pack(TRIPS, [trip, *kickers])

    if len(pairs) >= 2:
        high, low = pairs[0], pairs[1]
        kicker = max(r for r in range(NUM_RANKS) if counts[r] > 0 and r != high and r != low)
        return _pack(TWO_PAIR, [high, low, kicker])

    if pairs:
        pair = pairs[0]
        return _pack(PAIR, [pair, *singles[:3]])

    return _pack(HIGH_CARD, singles[:5])


def score_category(score: int) -> int:
    """从分数中取回牌型类别。"""
    return score >> 20


def describe(score: int) -> str:
    """给分数一个中文短描述，用于牌谱与复盘展示。"""
    return CATEGORY_NAMES[score_category(score)]

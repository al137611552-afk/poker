"""牌堆与随机性。

整个引擎里唯一制造随机数的地方。种子由调用方给定，因此任何一手牌都能凭
（种子, 手数）精确复现——这是牌谱回放、bug 复现和「同一副牌换个打法看结果」
这类复盘功能的前提。
"""

from __future__ import annotations

import random

from .cards import FULL_DECK


def shuffled_deck(rng: random.Random) -> list[int]:
    deck = list(FULL_DECK)
    rng.shuffle(deck)
    return deck


def deck_from_seed(seed: int) -> list[int]:
    """由整数种子直接得到一副洗好的牌。"""
    return shuffled_deck(random.Random(seed))


def stacked_deck(
    hole: dict[int, str],
    board: str = "",
    num_seats: int | None = None,
    button: int = 0,
    rest_seed: int = 0,
) -> list[int]:
    """构造一副「指定发牌结果」的牌，用于测试与教学场景重现。

    `hole` 形如 {0: "AsKs", 1: "7d7c"}，`board` 形如 "Ks7h2c" 或完整五张。
    未指定的部分用剩余牌按种子随机填充。

    `button` 必须与 HandState 使用的按钮位一致——发牌顺序从庄家左手起算，
    按钮位不同，同一副牌发到的手就不同。
    """
    from .cards import cards_from_str

    if not hole:
        raise ValueError("至少要指定一个座位的底牌")
    if num_seats is None:
        num_seats = max(hole) + 1
    if max(hole) >= num_seats:
        raise ValueError("hole 中的座位号超出 num_seats")
    assigned: list[int] = []
    hole_cards: dict[int, list[int]] = {}
    for seat, text in hole.items():
        cards = cards_from_str(text)
        if len(cards) != 2:
            raise ValueError(f"座位 {seat} 的底牌必须是两张: {text!r}")
        hole_cards[seat] = cards
        assigned.extend(cards)

    board_cards = cards_from_str(board) if board else []
    assigned.extend(board_cards)
    if len(set(assigned)) != len(assigned):
        raise ValueError("指定的牌中存在重复")

    remaining = [c for c in FULL_DECK if c not in set(assigned)]
    random.Random(rest_seed).shuffle(remaining)
    filler = iter(remaining)

    # 按 HandState 的发牌顺序回填：从庄家左手起两轮底牌，然后是公共牌
    deal_order = [(button + 1 + i) % num_seats for i in range(num_seats)]
    deck: list[int] = []
    for round_index in range(2):
        for seat in deal_order:
            cards = hole_cards.get(seat)
            deck.append(cards[round_index] if cards else next(filler))
    board_iter = iter(board_cards)
    for _ in range(5):
        card = next(board_iter, -1)
        deck.append(card if card >= 0 else next(filler))
    deck.extend(filler)
    return deck

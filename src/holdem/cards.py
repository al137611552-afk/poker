"""牌的表示与解析。

编码：card = rank * 4 + suit，取值 0..51。
  rank 0..12 对应 2,3,...,K,A
  suit 0..3  对应 c,d,h,s（与 PHH 标准一致的小写字母顺序）

选用整数编码而非对象，是为了让求值器和自对弈循环里的热路径只做位运算。
"""

from __future__ import annotations

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

NUM_RANKS = 13
NUM_SUITS = 4
NUM_CARDS = 52

FULL_DECK = tuple(range(NUM_CARDS))


def make_card(rank: int, suit: int) -> int:
    """由 rank(0..12) 与 suit(0..3) 组成一张牌。"""
    if not 0 <= rank < NUM_RANKS:
        raise ValueError(f"rank 越界: {rank}")
    if not 0 <= suit < NUM_SUITS:
        raise ValueError(f"suit 越界: {suit}")
    return rank * NUM_SUITS + suit


def card_rank(card: int) -> int:
    return card >> 2


def card_suit(card: int) -> int:
    return card & 3


def card_to_str(card: int) -> str:
    """0..51 -> 形如 'As' / 'Td' / '2c' 的两字符表示。"""
    if not 0 <= card < NUM_CARDS:
        raise ValueError(f"card 越界: {card}")
    return RANK_CHARS[card >> 2] + SUIT_CHARS[card & 3]


def card_from_str(text: str) -> int:
    """'As' -> 51。大小写敏感：点数大写、花色小写，与 PHH 一致。"""
    if len(text) != 2:
        raise ValueError(f"牌面格式错误: {text!r}")
    rank = RANK_CHARS.find(text[0])
    suit = SUIT_CHARS.find(text[1])
    if rank < 0 or suit < 0:
        raise ValueError(f"无法识别的牌面: {text!r}")
    return rank * NUM_SUITS + suit


def cards_to_str(cards: object) -> str:
    """一组牌 -> 'AsKd7c' 形式的连写字符串。"""
    return "".join(card_to_str(c) for c in cards)  # type: ignore[union-attr]


def cards_from_str(text: str) -> list[int]:
    """'AsKd7c' -> [51, 46, 20]。允许用空格分隔。"""
    compact = text.replace(" ", "")
    if len(compact) % 2 != 0:
        raise ValueError(f"牌面串长度必须为偶数: {text!r}")
    return [card_from_str(compact[i : i + 2]) for i in range(0, len(compact), 2)]

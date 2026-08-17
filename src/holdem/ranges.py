"""起手牌类与范围。

翻前的一切——范围表、风格层、HUD、复盘里的 13×13 网格——都以「169 个起手牌类」为单位，
所以这里定死两件事：**编号方案**与**记法**。

## 编号

169 个牌类编号 0..168，直接对应标准的 13×13 图表布局：

```
行 = 较大的点数（第 0 行是 A，第 12 行是 2），列 = 较小的点数
对角线   = 对子      AA KK QQ …
右上三角 = 同花      AKs AQs …
左下三角 = 不同花    AKo AQo …
index = 行 * 13 + 列
```

这样前端拿到编号就能直接摆格子，不用再转换一次。

## 记法

`"TT+, AJs+, KQo, A5s-A2s, 99:0.5"`——与主流工具一致：`+` 向上延伸，`-` 表示区间，
`:` 后跟 0..1 的权重（混合策略）。纯逻辑，无依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .cards import NUM_RANKS, RANK_CHARS, card_rank, card_suit, make_card

NUM_HAND_CLASSES = 169
TOTAL_COMBOS = 1326


# ------------------------------------------------------------------ 牌类编号


def class_index(hi_rank: int, lo_rank: int, suited: bool) -> int:
    """由点数与是否同花得到牌类编号。`hi_rank` 必须不小于 `lo_rank`。"""
    if not 0 <= lo_rank <= hi_rank < NUM_RANKS:
        raise ValueError(f"点数越界或顺序颠倒: {hi_rank}, {lo_rank}")
    if hi_rank == lo_rank:
        if suited:
            raise ValueError("对子不可能同花")
        row = col = NUM_RANKS - 1 - hi_rank
        return row * NUM_RANKS + col
    row, col = NUM_RANKS - 1 - hi_rank, NUM_RANKS - 1 - lo_rank
    return row * NUM_RANKS + col if suited else col * NUM_RANKS + row


def class_of(card_a: int, card_b: int) -> int:
    """由两张具体的牌得到牌类编号。"""
    if card_a == card_b:
        raise ValueError("两张牌不能相同")
    rank_a, rank_b = card_rank(card_a), card_rank(card_b)
    hi, lo = max(rank_a, rank_b), min(rank_a, rank_b)
    suited = hi != lo and card_suit(card_a) == card_suit(card_b)
    return class_index(hi, lo, suited)


def grid_position(index: int) -> tuple[int, int]:
    """牌类编号 -> (行, 列)，供前端摆 13×13 网格。"""
    _check_index(index)
    return divmod(index, NUM_RANKS)


def is_pair(index: int) -> bool:
    row, col = grid_position(index)
    return row == col


def is_suited(index: int) -> bool:
    row, col = grid_position(index)
    return row < col


def class_ranks(index: int) -> tuple[int, int]:
    """牌类编号 -> (较大点数, 较小点数)。"""
    row, col = grid_position(index)
    hi_row, lo_row = min(row, col), max(row, col)
    return NUM_RANKS - 1 - hi_row, NUM_RANKS - 1 - lo_row


def class_name(index: int) -> str:
    hi, lo = class_ranks(index)
    if hi == lo:
        return RANK_CHARS[hi] * 2
    return RANK_CHARS[hi] + RANK_CHARS[lo] + ("s" if is_suited(index) else "o")


def class_from_name(name: str) -> int:
    """`"AKs"` -> 编号。对子写作 `"AA"`，同花/不同花后缀不可省略。"""
    text = name.strip()
    if len(text) not in (2, 3):
        raise ValueError(f"无法识别的牌类: {name!r}")
    hi = RANK_CHARS.find(text[0].upper())
    lo = RANK_CHARS.find(text[1].upper())
    if hi < 0 or lo < 0:
        raise ValueError(f"无法识别的点数: {name!r}")
    if hi < lo:
        hi, lo = lo, hi
    if hi == lo:
        if len(text) == 3:
            raise ValueError(f"对子不能带同花后缀: {name!r}")
        return class_index(hi, lo, False)
    if len(text) != 3:
        raise ValueError(f"非对子必须写明 s 或 o: {name!r}")
    suffix = text[2].lower()
    if suffix not in ("s", "o"):
        raise ValueError(f"后缀只能是 s 或 o: {name!r}")
    return class_index(hi, lo, suffix == "s")


def class_combos(index: int) -> list[tuple[int, int]]:
    """牌类编号 -> 全部具体组合。对子 6 个、同花 4 个、不同花 12 个。"""
    hi, lo = class_ranks(index)
    if hi == lo:
        cards = [make_card(hi, suit) for suit in range(4)]
        return list(combinations(cards, 2))
    if is_suited(index):
        return [(make_card(hi, suit), make_card(lo, suit)) for suit in range(4)]
    return [
        (make_card(hi, suit_a), make_card(lo, suit_b))
        for suit_a in range(4)
        for suit_b in range(4)
        if suit_a != suit_b
    ]


def class_combo_count(index: int) -> int:
    if is_pair(index):
        return 6
    return 4 if is_suited(index) else 12


ALL_CLASSES = tuple(range(NUM_HAND_CLASSES))


def _check_index(index: int) -> None:
    if not 0 <= index < NUM_HAND_CLASSES:
        raise ValueError(f"牌类编号越界: {index}")


# ------------------------------------------------------------------ 范围


@dataclass(frozen=True)
class Range:
    """一组带权重的起手牌类。权重 0..1，用于表达混合策略。"""

    weights: dict[int, float]

    def __post_init__(self) -> None:
        for index, weight in self.weights.items():
            _check_index(index)
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"{class_name(index)} 的权重越界: {weight}")
        # 丢掉零权重，保证相等性判断不受写法影响
        cleaned = {i: w for i, w in sorted(self.weights.items()) if w > 0}
        object.__setattr__(self, "weights", cleaned)

    # -------------------------------------------------------------- 构造

    @classmethod
    def parse(cls, text: str) -> "Range":
        """解析 `"TT+, AJs+, A5s-A2s, 99:0.5"` 这样的记法。"""
        weights: dict[int, float] = {}
        for chunk in text.split(","):
            token = chunk.strip()
            if not token:
                continue
            body, _, weight_text = token.partition(":")
            weight = float(weight_text) if weight_text else 1.0
            for index in _expand(body.strip()):
                weights[index] = max(weights.get(index, 0.0), weight)
        return cls(weights)

    @classmethod
    def empty(cls) -> "Range":
        return cls({})

    @classmethod
    def full(cls) -> "Range":
        return cls({index: 1.0 for index in ALL_CLASSES})

    # -------------------------------------------------------------- 查询

    def weight(self, index: int) -> float:
        _check_index(index)
        return self.weights.get(index, 0.0)

    def weight_of_hand(self, card_a: int, card_b: int) -> float:
        return self.weight(class_of(card_a, card_b))

    def __contains__(self, index: int) -> bool:
        return self.weights.get(index, 0.0) > 0

    def __bool__(self) -> bool:
        return bool(self.weights)

    def combos(self) -> float:
        """加权组合数。全范围为 1326。"""
        return sum(w * class_combo_count(i) for i, w in self.weights.items())

    def percent(self) -> float:
        """占全部起手牌的比例，0..1。这是 VPIP 之类指标对得上的口径。"""
        return self.combos() / TOTAL_COMBOS

    def classes(self) -> tuple[int, ...]:
        return tuple(self.weights)

    # -------------------------------------------------------------- 组合

    def union(self, other: "Range") -> "Range":
        merged = dict(self.weights)
        for index, weight in other.weights.items():
            merged[index] = max(merged.get(index, 0.0), weight)
        return Range(merged)

    def intersect(self, other: "Range") -> "Range":
        return Range(
            {
                index: min(weight, other.weight(index))
                for index, weight in self.weights.items()
                if index in other
            }
        )

    def difference(self, other: "Range") -> "Range":
        return Range(
            {
                index: max(0.0, weight - other.weight(index))
                for index, weight in self.weights.items()
            }
        )

    def scaled(self, factor: float) -> "Range":
        """整体缩放权重并截断到 0..1，风格层用它做粗粒度的松紧调整。"""
        return Range({i: min(1.0, max(0.0, w * factor)) for i, w in self.weights.items()})

    __or__ = union
    __and__ = intersect
    __sub__ = difference

    # -------------------------------------------------------------- 输出

    def to_text(self) -> str:
        """回写成记法。不做区间合并，保证可逆、便于 diff。"""
        parts = []
        for index in sorted(self.weights, key=_sort_key):
            weight = self.weights[index]
            name = class_name(index)
            parts.append(name if weight >= 1.0 else f"{name}:{weight:g}")
        return ", ".join(parts)

    def __str__(self) -> str:
        return self.to_text()


def _sort_key(index: int) -> tuple:
    """显示顺序：对子在前，点数大的在前，同花在不同花之前。不影响语义。"""
    hi, lo = class_ranks(index)
    return (not is_pair(index), -hi, -lo, not is_suited(index))


# ------------------------------------------------------------------ 记法展开


def _expand(token: str) -> list[int]:
    if not token:
        return []
    if "-" in token:
        return _expand_span(token)
    if token.endswith("+"):
        return _expand_plus(token[:-1].strip())
    # "AK" 不带后缀时表示同花与不同花都要
    stripped = token.strip()
    if len(stripped) == 2 and stripped[0].upper() != stripped[1].upper():
        hi = RANK_CHARS.find(stripped[0].upper())
        lo = RANK_CHARS.find(stripped[1].upper())
        if hi < 0 or lo < 0:
            raise ValueError(f"无法识别的牌类: {token!r}")
        if hi < lo:
            hi, lo = lo, hi
        return [class_index(hi, lo, True), class_index(hi, lo, False)]
    return [class_from_name(stripped)]


def _expand_plus(base: str) -> list[int]:
    """`TT+` -> TT 及以上的对子；`A5s+` -> A5s 到 AKs；`KTo+` -> KTo 到 KQo。"""
    index = class_from_name(base)
    hi, lo = class_ranks(index)
    if hi == lo:
        return [class_index(r, r, False) for r in range(hi, NUM_RANKS)]
    suited = is_suited(index)
    # 固定较大的点数，让较小的点数向上走到「差一位」为止
    return [class_index(hi, r, suited) for r in range(lo, hi)]


def _expand_span(token: str) -> list[int]:
    left_text, _, right_text = token.partition("-")
    left = class_from_name(left_text.strip())
    right = class_from_name(right_text.strip())
    left_hi, left_lo = class_ranks(left)
    right_hi, right_lo = class_ranks(right)

    if is_pair(left) != is_pair(right):
        raise ValueError(f"区间两端必须同为对子或同为非对子: {token!r}")
    if is_pair(left):
        low, high = min(left_hi, right_hi), max(left_hi, right_hi)
        return [class_index(r, r, False) for r in range(low, high + 1)]

    if is_suited(left) != is_suited(right):
        raise ValueError(f"区间两端的同花属性必须一致: {token!r}")
    if left_hi != right_hi:
        raise ValueError(f"区间两端的较大点数必须相同: {token!r}")
    low, high = min(left_lo, right_lo), max(left_lo, right_lo)
    return [class_index(left_hi, r, is_suited(left)) for r in range(low, high + 1)]

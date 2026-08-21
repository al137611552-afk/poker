"""翻后逐街收缩对手范围（FR-11）。纯逻辑，不碰 IO、不需要求解器。

翻牌时双方的范围由 `range_tracking.flop_ranges` 给出（来自离线解好的翻前表）。
这个模块接着往后走：**每观察到一个翻后动作，就把那一方的范围收一次**。

## 为什么必须逐组合，不能按牌类

`Range` 是 169 个牌类的粒度，翻前够用（翻前 AhKh 与 AsKc 确实等价），
**翻后完全不够**：两张红桃的牌面上，`AhKh` 是同花听牌而 `AsKc` 什么都不是。
按牌类收缩会把这两个当成一手牌，于是「他跟注了，所以有听牌」这种推断整个失效。
所以这里用 `ComboRange`（逐组合权重）。

## 收缩判据：分位，不是绝对牌力

判据是**这手牌在他自己当前范围里排第几**，不是「有没有成对」。
理由是可证伪性：绝对牌力要定一堆阈值（「顶对算强吗」取决于牌面），
而分位只需要一个排序，且天然随牌面自适应——A 高牌在 `AKQ` 面上是弱牌，
在 `722` 面上是强牌，同一套代码不用改。

## 已知局限（写在明面上）

**排序只看当前成手，不看听牌。** 于是翻牌上的同花听牌会被当作弱牌，
「他跟注了」会把听牌错误地排除掉。河牌上没有这个问题（牌已发完，成手就是全部）。
补听牌修正是下一段的事——**在补上之前，别拿翻牌上的收缩结果当准的**，
`ComboRange.confidence` 会如实报出这一点。

## 一条不肯让步的默认值

**收缩不许把范围收空。** 收空说明判据与实际打法矛盾（比如对手打了一手我们认为
他不可能有的牌），这时**退回不收缩并记下原因**，而不是交出一个空范围——
空范围会让下游的权益计算变成除零，或者更糟：悄悄给出一个看着正常的数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .evaluator import evaluate
from .ranges import Range, class_combos

__all__ = ["ComboRange", "narrow", "expand"]

# 各动作保留范围里最强的百分之多少。**这些数是拍的**，没有实测支撑，
# 调它们要拿批量对局的 bb/100 调，别拿直觉调（ADR 里记着这一条）。
KEEP_ON_BET = 0.55
"""主动下注/加注：保留较强的一半多一点。留这么宽是因为**下注里有诈唬**。"""
KEEP_ON_CALL = 0.75
"""跟注：踢掉最弱的四分之一。跟注范围比下注范围宽——它包含中等牌。"""
KEEP_ON_CHECK = 1.0
"""过牌：**一点都不收。** 过牌几乎不含信息：强牌可以埋伏、弱牌可以放弃，
两头都在。收它等于凭空造出一个「他没牌」的结论。"""


@dataclass
class ComboRange:
    """逐组合的范围：`(小牌, 大牌) -> 权重`。牌用整数表示，组内已排序。"""

    weights: "dict[tuple[int, int], float]"
    confidence: "list[str]" = field(default_factory=list)
    """收缩过程中攒下的保留意见，给上层照实展示（PRD 的「诚实」那条）。"""

    def __bool__(self) -> bool:
        return any(value > 0 for value in self.weights.values())

    @property
    def total(self) -> float:
        return sum(self.weights.values())

    def combos(self) -> "list[tuple[int, int]]":
        """还留在范围里的组合（权重为正的）。"""
        return [combo for combo, value in self.weights.items() if value > 0]

    def weight_of(self, card_a: int, card_b: int) -> float:
        return self.weights.get(_key(card_a, card_b), 0.0)


def _key(card_a: int, card_b: int) -> "tuple[int, int]":
    return (card_a, card_b) if card_a < card_b else (card_b, card_a)


def expand(hand_range: Range, board: "tuple[int, ...]") -> ComboRange:
    """把 169 牌类的范围摊成逐组合，**并去掉与公共牌撞牌的组合**。

    撞牌这一步不能省：牌面上有 `Qs` 时，对手不可能再拿一张 `Qs`，
    把它留在范围里会让所有基于范围的计算都偏。
    """
    blocked = set(board)
    weights: dict = {}
    for index in range(169):
        weight = hand_range.weight(index)
        if weight <= 0:
            continue
        for card_a, card_b in class_combos(index):
            if card_a in blocked or card_b in blocked:
                continue
            weights[_key(card_a, card_b)] = weight
    return ComboRange(weights=weights)


def narrow(
    current: ComboRange, board: "tuple[int, ...]", kind: str, *, keep: "float | None" = None
) -> ComboRange:
    """按一个翻后动作收缩范围。

    - `kind`：`"bet"` / `"raise"` / `"call"` / `"check"`（`fold` 不用收，人已经走了）
    - `keep`：保留最强的多少比例；不给就按动作取默认值

    返回**新的** `ComboRange`，不改原来的——同一个范围会被不同分支反复用到，
    就地改会串台。
    """
    share = keep if keep is not None else _default_keep(kind)
    notes = list(current.confidence)

    if share >= 1.0:
        return ComboRange(weights=dict(current.weights), confidence=notes)

    live = current.combos()
    if not live:
        notes.append("范围已经是空的，没什么可收的")
        return ComboRange(weights=dict(current.weights), confidence=notes)

    if len(board) < 5:
        notes.append(
            "翻/转牌上的排序只看当前成手、不看听牌，同花听牌会被当成弱牌——"
            "这一步的收缩结果偏紧"
        )

    ranked = sorted(live, key=lambda combo: _strength(combo, board), reverse=True)
    # **`max(1, ...)` 就是「不许收成空」那条保护本身**，不是防御性的凑数：
    # 哪怕 share 给到 0，也至少留下最强的那一手。空范围会让下游的权益计算除零，
    # 或者更糟——悄悄给出一个看着正常的数。
    # （这里原本还跟着一段"收空就回退"的兜底，但有了这一行它永远走不到；
    #   留着等于假装有两层保护，实际只有一层。）
    cut = max(1, round(len(ranked) * share))
    kept = set(ranked[:cut])

    weights = {
        combo: (value if combo in kept else 0.0)
        for combo, value in current.weights.items()
    }
    return ComboRange(weights=weights, confidence=notes)


def _default_keep(kind: str) -> float:
    if kind in ("bet", "raise"):
        return KEEP_ON_BET
    if kind == "call":
        return KEEP_ON_CALL
    if kind == "check":
        return KEEP_ON_CHECK
    raise ValueError(f"不认识的动作：{kind}")


def _strength(combo: "tuple[int, int]", board: "tuple[int, ...]") -> int:
    """这手牌在这个牌面上的成手强度。数越大越强（`evaluate` 的口径）。"""
    return evaluate(tuple(combo) + tuple(board))

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

from .evaluator import HIGH_CARD, PAIR, evaluate, score_category
from .ranges import Range, class_combos

__all__ = ["ComboRange", "KeepProfile", "DEFAULT_PROFILE", "narrow", "expand"]

# 各动作保留范围里最强的百分之多少。**这些数是拍的**，没有实测支撑，
# 调它们要拿批量对局的 bb/100 调，别拿直觉调（ADR 里记着这一条）。
KEEP_ON_BET = 0.55
"""主动下注/加注：保留较强的一半多一点。留这么宽是因为**下注里有诈唬**。"""
KEEP_ON_CALL = 0.75
"""跟注：踢掉最弱的四分之一。跟注范围比下注范围宽——它包含中等牌。"""
KEEP_ON_CHECK = 1.0
"""过牌：**一点都不收。** 过牌几乎不含信息：强牌可以埋伏、弱牌可以放弃，
两头都在。收它等于凭空造出一个「他没牌」的结论。"""


@dataclass(frozen=True)
class KeepProfile:
    """收缩参数的一套取值。

    做成可注入的对象、而不是直接读模块常量：**校准要并排跑好几套参数**，
    改全局常量的做法在多进程里会串台，而且改错了不会报错、只会让某一组的结果
    悄悄用上另一组的参数。
    """

    bet: float = KEEP_ON_BET
    call: float = KEEP_ON_CALL
    check: float = KEEP_ON_CHECK
    draw_tier: int = PAIR
    """强听牌折算到第几档成手。见 `_draw_tier`。"""

    def keep_for(self, kind: str) -> float:
        if kind in ("bet", "raise"):
            return self.bet
        if kind == "call":
            return self.call
        if kind == "check":
            return self.check
        raise ValueError(f"不认识的动作：{kind}")


DEFAULT_PROFILE = KeepProfile()


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
    current: ComboRange, board: "tuple[int, ...]", kind: str, *,
    keep: "float | None" = None, profile: KeepProfile = DEFAULT_PROFILE,
) -> ComboRange:
    """按一个翻后动作收缩范围。

    - `kind`：`"bet"` / `"raise"` / `"call"` / `"check"`（`fold` 不用收，人已经走了）
    - `keep`：保留最强的多少比例；不给就按动作取默认值

    返回**新的** `ComboRange`，不改原来的——同一个范围会被不同分支反复用到，
    就地改会串台。
    """
    share = keep if keep is not None else profile.keep_for(kind)
    notes = list(current.confidence)

    if share >= 1.0:
        return ComboRange(weights=dict(current.weights), confidence=notes)

    live = current.combos()
    if not live:
        notes.append("范围已经是空的，没什么可收的")
        return ComboRange(weights=dict(current.weights), confidence=notes)

    if len(board) < 5:
        notes.append(
            "翻/转牌上强听牌（同花听/开口顺）按「相当于一对」折算，卡顺与后门不算——"
            "折算档次是拍的，没有实测支撑"
        )

    ranked = sorted(
        live, key=lambda combo: _strength(combo, board, profile.draw_tier), reverse=True
    )
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


def _strength(combo: "tuple[int, int]", board: "tuple[int, ...]",
              draw_tier: int = PAIR) -> "tuple[int, int]":
    """这手牌在这个牌面上有多强。越大越强，用作排序键。

    返回 `(有效档次, 原始分)`：
    - **有效档次**＝成手档次与听牌折算档次取大的。牌发完之后（河牌）听牌不再是牌力，
      折算部分自动消失。
    - **原始分**＝`evaluate` 的完整打包分，用来在同档次内比大小（踢脚也算数）。

    为什么要有听牌那一档：只按成手排序，翻牌上的同花听牌会排在一对之后被踢掉，
    而它的实际权益（对一对约 35%）明显高于一手空气。**忽略它会让收缩系统性偏紧**。
    """
    cards = tuple(combo) + tuple(board)
    raw = evaluate(cards)
    made = score_category(raw)
    if len(board) >= 5:
        return (made, raw)          # 河牌：牌已发完，听牌不是牌力
    return (max(made, _draw_tier(cards, draw_tier)), raw)


def _draw_tier(cards: "tuple[int, ...]", tier_value: int = PAIR) -> int:
    """把听牌折算成一个「相当于第几档成手」的数。

    **只认强听牌**（同花听牌、开口顺听）：它们的权益够格跟一对掰腕子。
    卡顺、后门这类不算——把它们也抬上来等于几乎不收缩，那这个模块就没用了。

    折算到 `PAIR` 是**拍的**（ADR-0008 记着）：强听牌对一对约 35% 权益，
    比一对弱、比空气强得多，落在这一档最接近。要调就用批量对局的 bb/100 调。

    识别用位运算，不遍历补牌：`evaluate` 一次 4.5µs，一个决策里几百个组合各数一遍
    补牌是几万次调用（实测约 60ms/决策），bot 实时用不起。
    """
    tier = HIGH_CARD

    suits = [0, 0, 0, 0]
    ranks = 0
    for card in cards:
        suits[card % 4] += 1
        ranks |= 1 << (card // 4)
    if max(suits) == 4:             # 恰好 4 张同花色＝听同花（5 张就是成手了）
        tier = tier_value

    # 顺子听牌：把 A 同时当 1（轮子）；任意 5 连窗口里凑齐 4 张就算
    wheel = ranks | ((ranks >> 12) & 1)
    for low in range(0, 10):
        window = (wheel >> low) & 0b11111
        if bin(window).count("1") == 4:
            tier = max(tier, tier_value)
            break
    return tier

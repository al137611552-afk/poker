"""一个决策点的分数**有多可信**（FR-9 的 A/B/C 分级）。

PRD 的原则是硬的：**非求解得出的建议必须标注置信度，不得冒充精确解**。
逐点 EV 损失已经算得出来了（`review.py`），但那个数背后的可信度差别很大——
同样是「你亏了 2.3bb」，可能是干净解上的确凿结论，也可能整个落在求解器自己的残差里。
把这两种混在一张表里给人看，比不给更糟：它会让人照着噪声改打法。

## 三档与判据

| 档 | 含义 | 判据 |
|---|---|---|
| A | 可据此改打法 | 下面两档一条都不沾 |
| B | 有保留 | 这手牌的牌类被聚合伤得重；或范围经过滚动聚合（转牌及以后） |
| C | 只能参考 | 英雄的牌不在假设范围里；或损失落在噪声底以内 |

**取最差的那条**，不做加权：置信度不是打分，是「最弱的一环有多弱」。

## 两条不肯让步的默认值

1. **不知道 ≠ 很好**。噪声底量不到（没记收敛度、老报告）就**不许给 A**——
   与 `leaks.py` 那条同源。判据缺失时给出乐观结论，报告会在最该示警的时候显得最干净。
2. **落在噪声里就是 C，哪怕别的都干净**。解自己离最优还差 0.5bb 时，
   一个 0.3bb 的「漏洞」量的是求解器的残差，不是打法。
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Confidence", "grade", "why"]


class Confidence(Enum):
    """三档。值就是给人看的字母。"""

    A = "A"
    B = "B"
    C = "C"

    @property
    def usable(self) -> bool:
        """能不能据此改打法。只有 A 能。"""
        return self is Confidence.A


# 判据命中时给出的说法。报告里要**把原因一起显示**——
# 光给个字母，读的人无从判断这一条该不该信。
_C_OFF_RANGE = "英雄这手牌不在假设的范围里，解没给它频率"
_C_IN_NOISE = "损失落在解自己的残差以内，分不出是漏洞还是噪声"
_C_NO_FLOOR = "量不到噪声底（没记收敛度），无法判断这个数站不站得住"
_B_FLAGGED = "这手牌的牌类被聚合伤得重（类内落差大），同花信息已经没了"
_B_ROLLED = "范围是滚过来的，逐街聚合过，保真度低一档"


def grade(
    *,
    in_range: bool,
    loss: "float | None",
    noise_floor: "float | None",
    hand_class_flagged: bool = False,
    rolled_streets: int = 0,
) -> "tuple[Confidence, tuple[str, ...]]":
    """给一个决策点定档，连同**所有**命中的理由。

    - `loss`：这个决策的 EV 损失（大盲）。`None` = 没打上分。
    - `noise_floor`：这个点上「小到多少就不值得信」（大盲）。`None` = 量不到。
    - `hand_class_flagged`：英雄这手牌的牌类是否在 `RolledRange.flagged()` 里。
    - `rolled_streets`：范围滚了几街才到这个点（翻牌 0，转牌 1，河牌 2）。

    理由是**全部**命中的，不只是定档的那一条：读报告的人需要知道有几处不确定，
    而不是只看到最重的那一处。
    """
    reasons: list[str] = []

    if not in_range:
        reasons.append(_C_OFF_RANGE)
    if noise_floor is None:
        reasons.append(_C_NO_FLOOR)
    elif loss is not None and loss <= noise_floor:
        reasons.append(_C_IN_NOISE)

    if hand_class_flagged:
        reasons.append(_B_FLAGGED)
    if rolled_streets > 0:
        reasons.append(_B_ROLLED)

    if _C_OFF_RANGE in reasons or _C_IN_NOISE in reasons or _C_NO_FLOOR in reasons:
        return Confidence.C, tuple(reasons)
    if reasons:
        return Confidence.B, tuple(reasons)
    return Confidence.A, ()


def why(grade_and_reasons: "tuple[Confidence, tuple[str, ...]]") -> str:
    """一行人话。A 档没有理由可说，就说它凭什么是 A。"""
    level, reasons = grade_and_reasons
    if level is Confidence.A:
        return "A：解收敛、牌在范围里、没经过聚合——可以据此改打法"
    head = "B：可以参考，但" if level is Confidence.B else "C：只能看看，因为"
    return head + "；".join(reasons)

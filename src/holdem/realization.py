"""翻后收益的压缩模型：权益兑现。

翻前树的终局分三类（见 `preflop_tree.py`）：只剩一人、全下摊牌、**进翻牌**。前两类的收益
是精确的——弃牌终局纯粹是筹码搬运，全下终局直接查翻前权益表。第三类不精确，也不可能
精确：那要把整个翻后解出来。这里把翻后压缩成**一个数**——权益兑现系数。

## 模型

一手牌很少能兑现它的全部权益：位置差、被压制、难打的牌打不出应有的胜率，而位置好、
有听牌潜力的牌能打出超过胜率的份额。记兑现系数 `R`，进翻牌时双方对底池的期望份额为

```
share_hero = (R_hero · eq)^γ / ((R_hero · eq)^γ + (R_villain · (1 − eq))^γ)
```

写成**归一化的比值**，而不是 `R · eq · P`，是为了守住一条硬约束：**筹码必须守恒**。
两边份额恒好加成 1，终局是严格零和的，不会凭空造钱——这在求解器里是可以被测试验证的性质。
代价是 `R` 只有**相对**意义：两人系数同时乘 2，结果不变。

## γ：为什么份额比权益更极端

只按权益分池会得出一个明显错误的结论：**弱牌能百分之百兑现自己的权益**。第一版就是这样，
解出来 SB 面对 3bet 只弃 26.7%、连 72o 都跟四成——因为「有 30% 权益就能拿 30% 底池」。
真实牌局不是这样：弱牌的权益集中在少数牌面上，多街下注里它多数时候要弃牌，而强牌能
把底池打大。公开求解器里这体现为**强牌兑现超过 100%、弱牌不足 70%**。

指数 γ ≥ 1 就干这件事：把胜负比 `eq/(1−eq)` 整体拉开，γ=1 时退化成按权益分池。它同样
按 SPR 打折——街数越多、筹码越深，拉开得越厉害；全下摊牌（没有后续街）根本不经过这一层。

## 系数怎么定

```
R = 位置基准 + 牌型修正 × 隐含赔率折扣
```

- **位置基准**：翻后最后说话的人拿 `in_position`，其余拿 `out_of_position`。位置是翻后
  最稳定的优势来源，不随筹码深度消失。
- **牌型修正**：对子（能中暗三条）、同花、连张按顺序拿正修正；高张不同花拿负修正
  ——它们的权益里有相当一部分来自「对手弃牌」，真打起来容易被压制（反向隐含赔率）。
- **隐含赔率折扣**：牌型修正讲的全是「后面能赢下大底池」，所以要按 **SPR**（底池后面
  还剩多少筹码）打折。4bet 底池里 SPR 只有两三倍，暗三条也换不来一整个身家；
  单次加注底池 SPR 二十倍，同花听牌才有隐含赔率可言。位置优势不打折。

## 这些参数是假设，不是测量结果

**必须说清楚**：默认参数取自公开求解器的常识量级（位置差约 10 个百分点、修正项几个
百分点），**没有经过本项目的实测校准**。它们决定的是范围的松紧与结构，所以：

- 测试只验**性质**（守恒、单调、退化），不把某个具体数字钉成基准；
- 参数全部集中在 `RealizationModel` 里，可整体替换；
- 校准计划见 ADR-0003：用 Slumbot 免费 API 打单挑实测，反推兑现系数。

把所有修正项设成 0、两人位置相同时，模型精确退化成「按全下权益分池」，
这是它的下界行为，也由测试守着。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .cards import NUM_RANKS
from .equity_table import equity_matrix, removal_weights
from .ranges import NUM_HAND_CLASSES, class_ranks, is_pair, is_suited

__all__ = [
    "RealizationModel",
    "realization_factors",
    "flop_share_matrix",
    "showdown_share_matrix",
]

_TEN = NUM_RANKS - 5  # 点数编号：0 是 2，8 是 T


@dataclass(frozen=True)
class RealizationModel:
    """权益兑现系数的参数。默认值是常识量级的假设，待实测校准（见模块 docstring）。"""

    in_position: float = 1.05
    out_of_position: float = 0.93
    pair_bonus: float = 0.05
    suited_bonus: float = 0.04
    connector_bonus: float = 0.03
    """无间隔连张拿满，间隔 1–2 张拿一半。"""
    broadway_offsuit_penalty: float = 0.03
    """两张都是 T 以上的不同花牌：权益里靠弃牌的部分多，真打容易被压制。"""
    sharpening: float = 0.35
    """γ = 1 + sharpening × SPR 折扣。0 表示按权益分池，越大则强弱牌的份额差越被拉开。"""
    reference_spr: float = 8.0
    """牌型修正与 γ 都按 `min(1, SPR / reference_spr)` 打折；SPR 更深不再加成。"""

    def __post_init__(self) -> None:
        if self.in_position <= 0 or self.out_of_position <= 0:
            raise ValueError("位置基准必须为正")
        if self.sharpening < 0:
            raise ValueError("锐化系数不能为负")
        if self.reference_spr <= 0:
            raise ValueError("参考 SPR 必须为正")

    def key(self) -> tuple:
        """可哈希的参数指纹，供缓存使用。"""
        return (
            self.in_position,
            self.out_of_position,
            self.pair_bonus,
            self.suited_bonus,
            self.connector_bonus,
            self.broadway_offsuit_penalty,
            self.sharpening,
            self.reference_spr,
        )


ALL_IN_EQUITY = RealizationModel(
    in_position=1.0,
    out_of_position=1.0,
    pair_bonus=0.0,
    suited_bonus=0.0,
    connector_bonus=0.0,
    broadway_offsuit_penalty=0.0,
    sharpening=0.0,
)
"""退化模型：份额就是全下权益。用来把「翻后模型」这一层从实验里摘掉。"""


def _hand_modifier(model: RealizationModel, index: int) -> float:
    """牌型修正（未打折）。"""
    hi, lo = class_ranks(index)
    if is_pair(index):
        return model.pair_bonus
    bonus = model.suited_bonus if is_suited(index) else 0.0
    gap = hi - lo - 1
    if gap == 0:
        bonus += model.connector_bonus
    elif gap <= 2:
        bonus += model.connector_bonus * 0.5
    if not is_suited(index) and lo >= _TEN:
        bonus -= model.broadway_offsuit_penalty
    return bonus


def realization_factors(
    model: RealizationModel, *, in_position: bool, spr: float
) -> tuple[float, ...]:
    """169 个牌类在给定位置与 SPR 下的兑现系数。"""
    if spr < 0:
        raise ValueError("SPR 不能为负")
    base = model.in_position if in_position else model.out_of_position
    discount = min(1.0, spr / model.reference_spr)
    return tuple(
        base + _hand_modifier(model, index) * discount
        for index in range(NUM_HAND_CLASSES)
    )


# ------------------------------------------------------------------ 份额矩阵


def showdown_share_matrix() -> tuple[float, ...]:
    """全下摊牌时英雄拿到的底池份额——就是翻前权益本身，不经过兑现模型。"""
    return _showdown_matrix()


@lru_cache(maxsize=1)
def _showdown_matrix() -> tuple[float, ...]:
    return tuple(equity_matrix())


def flop_share_matrix(
    model: RealizationModel, *, hero_in_position: bool, spr: float
) -> tuple[float, ...]:
    """`M[i * 169 + j]` = 英雄持牌类 i、对手持牌类 j 时，英雄进翻牌后的底池期望份额。

    单挑口径；多人底池的份额留给六人桌那一段处理。
    """
    return _flop_matrix(model.key(), hero_in_position, round(float(spr), 3))


@lru_cache(maxsize=32)
def _flop_matrix(key: tuple, hero_in_position: bool, spr: float) -> tuple[float, ...]:
    model = RealizationModel(*key)
    equity = equity_matrix()
    hero_r = realization_factors(model, in_position=hero_in_position, spr=spr)
    villain_r = realization_factors(model, in_position=not hero_in_position, spr=spr)
    gamma = 1.0 + model.sharpening * min(1.0, spr / model.reference_spr)
    shares = [0.0] * (NUM_HAND_CLASSES * NUM_HAND_CLASSES)
    for i in range(NUM_HAND_CLASSES):
        base = i * NUM_HAND_CLASSES
        mine = hero_r[i]
        for j in range(NUM_HAND_CLASSES):
            eq = equity[base + j]
            hero_weight = (mine * eq) ** gamma
            total = hero_weight + (villain_r[j] * (1.0 - eq)) ** gamma
            shares[base + j] = hero_weight / total if total > 0 else 0.5
    return tuple(shares)


# ------------------------------------------------------------------ 共牌权重


@lru_cache(maxsize=1)
def removal_rows() -> tuple[tuple[float, ...], ...]:
    """把共牌权重按行切好，让求解器每一步只做点积。

    `removal_rows()[i][j]` = 我方持牌类 i 时，对手仍可能持有的牌类 j 的组合数。
    每一行的和恒为 C(50,2)=1225——我方拿走两张牌之后，对手的组合数与我方牌型无关。
    """
    flat = removal_weights()
    return tuple(
        tuple(flat[i * NUM_HAND_CLASSES + j] for j in range(NUM_HAND_CLASSES))
        for i in range(NUM_HAND_CLASSES)
    )


VILLAIN_COMBOS = 1225.0
"""任一手具体底牌拿走两张之后，对手可能的组合数。`removal_rows()` 每行的和。"""

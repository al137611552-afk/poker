"""短筹码推/弃纳什均衡求解。

这是整个项目里**唯一能精确求解**的一块策略：把动作限制成「小盲全下或弃牌、大盲跟注或
弃牌」之后，博弈只剩每人 169 个二选一的决策点，可以解到均衡。深筹码的翻前范围没有这个
待遇——那需要翻后价值，只能靠求解器或范围表近似。

## 收益（以大盲为单位，站在小盲一侧计净得失）

设有效筹码 `S`，双方各下前注 `a`，小盲 0.5、大盲 1：

| 结果 | 小盲净得失 |
|---|---|
| 小盲弃牌 | `-(a + 0.5)` |
| 小盲全下、大盲弃牌 | `+(a + 1)` |
| 小盲全下、大盲跟注 | `2S·eq − S` |

## 共牌效应

我方持有某牌类会削减对手持有相关牌类的组合数（拿着 AA，对手能有的 AK 就少一半）。
求解时按 `equity_table.removal_weights()` 加权——忽略这一项会让门槛系统性偏移，
与公开的纳什表对不上。

## 算法

用 **CFR+**（遗憾匹配，遗憾截零 + 线性加权平均）。先前试过虚拟对局，收敛是 O(1/√t)，
400 次迭代后可利用度仍有 0.005 bb/手（相当于 0.5 bb/100，对一个号称「纳什」的模块
来说太大）。CFR+ 在同样的每步开销下收敛快一个量级以上。

## 正确性怎么保证

不依赖任何外部图表：解完直接算**可利用度**（双方各自最佳应对能多赚多少），
可利用度趋近于零就说明确实是均衡。公开的纳什表只作旁证。
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import mul

from .equity_table import equity_matrix, removal_weights
from .ranges import NUM_HAND_CLASSES, TOTAL_COMBOS, Range, class_combo_count

_CLASS_PROBABILITY = tuple(
    class_combo_count(i) / TOTAL_COMBOS for i in range(NUM_HAND_CLASSES)
)


@dataclass(frozen=True)
class PushFoldSolution:
    """一个有效筹码下的推弃均衡。"""

    effective_stack: float
    ante: float
    push: Range
    """小盲的全下范围，权重即混合频率。"""
    call: Range
    """大盲面对全下的跟注范围。"""
    exploitability: float
    """双方最佳应对的总收益，单位大盲/手。越接近 0 越是均衡。"""
    small_blind_ev: float
    """均衡下小盲每手的期望得失（大盲/手）。"""
    iterations: int

    @property
    def push_percent(self) -> float:
        return self.push.percent()

    @property
    def call_percent(self) -> float:
        return self.call.percent()


class _Game:
    """把权益表与共牌权重预处理成按行组织的向量，让每一步只做点积。"""

    __slots__ = ("stack", "ante", "removal_rows", "weighted_rows", "totals")

    def __init__(self, stack: float, ante: float) -> None:
        equity = equity_matrix()
        removal = removal_weights()
        self.stack = stack
        self.ante = ante
        self.removal_rows: list[tuple[float, ...]] = []
        self.weighted_rows: list[tuple[float, ...]] = []
        self.totals: list[float] = []
        for hero in range(NUM_HAND_CLASSES):
            base = hero * NUM_HAND_CLASSES
            row = tuple(removal[base + j] for j in range(NUM_HAND_CLASSES))
            self.removal_rows.append(row)
            self.weighted_rows.append(
                tuple(row[j] * equity[base + j] for j in range(NUM_HAND_CLASSES))
            )
            self.totals.append(sum(row))

    def push_ev(self, hero: int, call_probs: list[float]) -> float:
        """小盲持 hero 全下的期望。"""
        called = sum(map(mul, self.removal_rows[hero], call_probs))
        total = self.totals[hero]
        if called <= 0:
            return self.ante + 1.0
        equity_sum = sum(map(mul, self.weighted_rows[hero], call_probs))
        folded = total - called
        return (
            folded * (self.ante + 1.0)
            + 2.0 * self.stack * equity_sum
            - self.stack * called
        ) / total

    def call_ev(self, hero: int, push_probs: list[float]) -> float:
        """大盲持 hero 面对全下时跟注的期望（站在大盲一侧计）。"""
        shoved = sum(map(mul, self.removal_rows[hero], push_probs))
        if shoved <= 0:
            return 0.0
        equity_sum = sum(map(mul, self.weighted_rows[hero], push_probs))
        return 2.0 * self.stack * (equity_sum / shoved) - self.stack

    def push_chance(self, hero: int, push_probs: list[float]) -> float:
        return sum(map(mul, self.removal_rows[hero], push_probs)) / self.totals[hero]


def _regret_matching(positive: float, negative: float) -> float:
    total = positive + negative
    if total <= 0:
        return 0.5
    return positive / total


def solve_push_fold(
    effective_stack: float,
    *,
    ante: float = 0.0,
    iterations: int = 600,
    tolerance: float = 1e-4,
) -> PushFoldSolution:
    """求解给定有效筹码下的推弃均衡。

    可利用度低于 `tolerance`（大盲/手）即提前停止。
    """
    if effective_stack <= 0:
        raise ValueError("有效筹码必须为正")
    if ante < 0:
        raise ValueError("前注不能为负")

    game = _Game(float(effective_stack), ante)
    fold_value = -(ante + 0.5)
    bb_fold_value = -(ante + 1.0)
    classes = range(NUM_HAND_CLASSES)

    push_regret_yes = [0.0] * NUM_HAND_CLASSES
    push_regret_no = [0.0] * NUM_HAND_CLASSES
    call_regret_yes = [0.0] * NUM_HAND_CLASSES
    call_regret_no = [0.0] * NUM_HAND_CLASSES

    push_sum = [0.0] * NUM_HAND_CLASSES
    call_sum = [0.0] * NUM_HAND_CLASSES
    weight_sum = 0.0

    push_probs = [0.5] * NUM_HAND_CLASSES
    call_probs = [0.5] * NUM_HAND_CLASSES
    average_push = list(push_probs)
    average_call = list(call_probs)
    used = 0

    for step in range(1, iterations + 1):
        used = step

        push_values = [game.push_ev(i, call_probs) for i in classes]
        call_values = [game.call_ev(j, push_probs) for j in classes]

        for i in classes:
            current = push_probs[i] * push_values[i] + (1.0 - push_probs[i]) * fold_value
            push_regret_yes[i] = max(0.0, push_regret_yes[i] + push_values[i] - current)
            push_regret_no[i] = max(0.0, push_regret_no[i] + fold_value - current)
        for j in classes:
            current = call_probs[j] * call_values[j] + (1.0 - call_probs[j]) * bb_fold_value
            call_regret_yes[j] = max(0.0, call_regret_yes[j] + call_values[j] - current)
            call_regret_no[j] = max(0.0, call_regret_no[j] + bb_fold_value - current)

        push_probs = [_regret_matching(push_regret_yes[i], push_regret_no[i]) for i in classes]
        call_probs = [_regret_matching(call_regret_yes[j], call_regret_no[j]) for j in classes]

        # CFR+ 的线性加权平均：越晚的迭代权重越大
        weight = float(step)
        weight_sum += weight
        for i in classes:
            push_sum[i] += weight * push_probs[i]
            call_sum[i] += weight * call_probs[i]

        if step % 25 == 0 or step == iterations:
            average_push = [value / weight_sum for value in push_sum]
            average_call = [value / weight_sum for value in call_sum]
            gap, _ = _exploitability(game, average_push, average_call)
            if gap < tolerance:
                break
    else:
        average_push = [value / weight_sum for value in push_sum]
        average_call = [value / weight_sum for value in call_sum]

    gap, sb_ev = _exploitability(game, average_push, average_call)
    return PushFoldSolution(
        effective_stack=game.stack,
        ante=ante,
        push=_to_range(average_push),
        call=_to_range(average_call),
        exploitability=gap,
        small_blind_ev=sb_ev,
        iterations=used,
    )


def _exploitability(
    game: _Game, push_probs: list[float], call_probs: list[float]
) -> tuple[float, float]:
    """返回 (可利用度, 小盲的期望)，都以大盲/手为单位。"""
    fold_value = -(game.ante + 0.5)
    bb_fold_value = -(game.ante + 1.0)

    actual = 0.0
    best_sb = 0.0
    for i in range(NUM_HAND_CLASSES):
        share = _CLASS_PROBABILITY[i]
        value = game.push_ev(i, call_probs)
        actual += share * (push_probs[i] * value + (1.0 - push_probs[i]) * fold_value)
        best_sb += share * max(value, fold_value)

    bb_actual = 0.0
    bb_best = 0.0
    for j in range(NUM_HAND_CLASSES):
        chance = game.push_chance(j, push_probs)
        if chance <= 0:
            continue
        share = _CLASS_PROBABILITY[j] * chance
        value = game.call_ev(j, push_probs)
        bb_actual += share * (
            call_probs[j] * value + (1.0 - call_probs[j]) * bb_fold_value
        )
        bb_best += share * max(value, bb_fold_value)

    return (best_sb - actual) + (bb_best - bb_actual), actual


def _to_range(probs: list[float]) -> Range:
    return Range({i: round(p, 4) for i, p in enumerate(probs) if p > 1e-4})

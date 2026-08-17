"""翻前树的 CFR+ 求解器。

树来自 `preflop_tree.py`，进翻牌的终局收益来自 `realization.py`，这里只管把两者接起来
求解。算法与 `pushfold.py` 同源（CFR+：遗憾截零 + 线性加权平均），区别是从「一个决策点」
变成了「一棵树」：牌类不再对应一个二选一，而是每个决策节点各带 169 个信息集。

## 向量形式

树是公共的，牌是私有的，所以一次遍历同时算 169 个牌类：每个节点上传下去的是**到达概率
向量**，回传的是**反事实价值向量**。终局的价值是

```
v_hero[i] = Σ_j 共牌权重(i, j) · 对手到达概率[j] · 收益(i, j)
```

也就是一次矩阵乘向量。把「共牌权重 × 收益」预乘好之后，每个牌类只剩一次点积——这是
纯 Python 下唯一能把整棵树跑起来的写法（每次遍历约 9000 次长度 169 的点积）。

## 正确性怎么保证

和推弃求解一样，**不依赖任何外部数据**：

1. **可利用度自证**：解完对每个玩家算一次最佳应对，看还能多赚多少（单位大盲/手）。
2. **退化交叉验证**：把树退化成「全下或弃牌」，解应当与 `pushfold.solve_push_fold`
   逐手一致——这一条同时验证了树、终局收益、CFR+ 实现与共牌处理。

## 只支持单挑

多人桌的终局要在三个以上对手的联合分布上积分，共牌处理与多人权益都是另一码事，
留给六人桌那一段（见 ADR-0003）。这里遇到多人配置直接报错，不假装能算。
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import mul

from .preflop_tree import (
    DecisionNode,
    PreflopConfig,
    PreflopTree,
    SubgameConfig,
    TerminalNode,
    build_tree,
)
from .ranges import NUM_HAND_CLASSES, Range, class_combo_count
from .realization import (
    RealizationModel,
    flop_share_matrix,
    removal_rows,
    showdown_share_matrix,
)

__all__ = ["BranchValue", "PreflopSolution", "solve_preflop"]

_CLASSES = range(NUM_HAND_CLASSES)
_COMBO_COUNT = tuple(class_combo_count(i) for i in range(NUM_HAND_CLASSES))


# ------------------------------------------------------------------ 结果


@dataclass(frozen=True)
class BranchValue:
    """根节点某个动作分支下的价值，**未归一化**地存着。

    链式求解要问的是「对手没弃牌时我这手牌值多少」，也就是若干分支的合并。合并必须在
    归一化之前做：每个分支的除数（对手在该分支里还剩多少组合）按牌类不同，先除再加权
    会引入误差。所以这里把分子分母分开存，由 `combine` 合并。
    """

    action: int
    values: tuple[tuple[float, ...], tuple[float, ...]]
    """两人的反事实价值（分子）。"""
    weights: tuple[tuple[float, ...], tuple[float, ...]]
    """两人的归一化权重（分母）＝对手在这个分支里还剩多少组合。"""

    def hand_ev(self, player: int) -> tuple[float, ...]:
        return _divide(self.values[player], self.weights[player])


def combine(branches: "list[BranchValue] | tuple[BranchValue, ...]", player: int) -> tuple[float, ...]:
    """把若干分支合成一条逐牌类 EV（条件在「这些分支之一发生了」）。"""
    if not branches:
        raise ValueError("没有分支可合并")
    values = [0.0] * NUM_HAND_CLASSES
    weights = [0.0] * NUM_HAND_CLASSES
    for branch in branches:
        for i in _CLASSES:
            values[i] += branch.values[player][i]
            weights[i] += branch.weights[player][i]
    return _divide(values, weights)


def _divide(values, weights) -> tuple[float, ...]:
    return tuple(
        values[i] / weights[i] if weights[i] > 1e-12 else 0.0 for i in _CLASSES
    )


# ------------------------------------------------------------------ 解


@dataclass(frozen=True)
class PreflopSolution:
    """一棵翻前树的解。策略按「节点 → 169 个牌类 → 各动作概率」组织。"""

    tree: PreflopTree
    model: RealizationModel
    strategies: dict[int, tuple[tuple[float, ...], ...]]
    reaches: dict[int, tuple[float, ...]]
    """每个决策节点上，**行动者**走到这里的概率（按牌类）。"""
    exploitability: float
    """两名玩家最佳应对的总收益，大盲/手。越接近 0 越是均衡。

    给了先验范围时，这是**范围内**每手的可利用度。
    """
    player_ev: tuple[float, ...]
    """均衡下每人每手的期望得失（大盲/手）。

    两者之和等于死钱（`SubgameConfig.dead_money`）——弃掉的人留下的钱要有人拿走。
    整桌树里死钱为 0，于是就是零和。
    """
    hand_ev: tuple[tuple[float, ...], tuple[float, ...]]
    """逐牌类的期望得失（大盲/手）。链式求解靠它把子博弈的价值传回上一层。"""
    root_branches: tuple[BranchValue, ...]
    """根节点每个动作分支下的价值，供链式求解合并（比如「防守者没弃牌」）。"""
    iterations: int

    @staticmethod
    def _id(node: DecisionNode | int) -> int:
        return node if isinstance(node, int) else node.node_id

    def strategy_at(self, node: DecisionNode | int) -> tuple[tuple[float, ...], ...]:
        """原始策略：每个牌类**若走到这里**各动作的频率。

        走不到这里的牌类（比如根本没开牌的 72o 面对 3bet）在这里的数字没有意义，
        统计与画图请用 `arriving_range` / `action_range` / `action_frequency`。
        """
        return self.strategies[self._id(node)]

    def arriving_range(self, node: DecisionNode | int) -> Range:
        """行动者带到这个节点的范围。根节点是全范围。"""
        reach = self.reaches[self._id(node)]
        return Range({i: round(reach[i], 4) for i in _CLASSES if reach[i] > 1e-4})

    def action_range(self, node: DecisionNode | int, action: int) -> Range:
        """某个动作对应的范围＝到达概率 × 该动作频率。

        乘上到达概率是要害：深处节点上「没走到这里的牌」不该出现在范围里。
        """
        node_id = self._id(node)
        strategy = self.strategies[node_id]
        reach = self.reaches[node_id]
        weights = {}
        for i in _CLASSES:
            weight = reach[i] * strategy[i][action]
            if weight > 1e-4:
                weights[i] = round(weight, 4)
        return Range(weights)

    def action_frequency(self, node: DecisionNode | int, action: int) -> float:
        """**到达这个节点的范围里**有多大比例选这个动作（按组合数加权）。

        直接用未取整的到达概率算——`action_range` 会为了好看四舍五入，
        拿它来做除法会让一组频率加不回 1。
        """
        node_id = self._id(node)
        strategy = self.strategies[node_id]
        reach = self.reaches[node_id]
        arriving = 0.0
        taken = 0.0
        for i in _CLASSES:
            combos = reach[i] * _COMBO_COUNT[i]
            arriving += combos
            taken += combos * strategy[i][action]
        return taken / arriving if arriving > 0 else 0.0


# ------------------------------------------------------------------ 终局


class _TerminalEval:
    """把一个终局的收益预乘成「每个牌类一行」，让求解时只剩点积。

    `rows[p][i][j]` = 共牌权重(i, j) × 玩家 p 持牌类 i、对手持牌类 j 时的净得失。
    """

    __slots__ = ("rows",)

    def __init__(self, rows: tuple[tuple[tuple[float, ...], ...], ...]) -> None:
        self.rows = rows

    def values(self, reach: list[list[float]]) -> tuple[list[float], list[float]]:
        rows0, rows1 = self.rows
        reach0, reach1 = reach
        return (
            [sum(map(mul, rows0[i], reach1)) for i in _CLASSES],
            [sum(map(mul, rows1[j], reach0)) for j in _CLASSES],
        )

    def values_for(self, player: int, reach_other: list[float]) -> list[float]:
        rows = self.rows[player]
        return [sum(map(mul, rows[i], reach_other)) for i in _CLASSES]


def _build_terminal(node: TerminalNode, config: PreflopConfig, model: RealizationModel):
    removal = removal_rows()

    if node.kind == "fold":
        payoffs = node.fold_payoffs()
        rows = tuple(
            tuple(tuple(weight * payoffs[player] for weight in removal[i]) for i in _CLASSES)
            for player in range(2)
        )
        return _TerminalEval(rows)

    hero, villain = node.alive
    pot = node.pot
    if node.kind == "showdown":
        hero_share = showdown_share_matrix()
    else:
        # 进翻牌：兑现系数按位置与 SPR 取，SPR = 底池后面还剩多少筹码
        remaining = config.effective_stack - node.contributions[hero]
        spr = remaining / pot if pot > 0 else 0.0
        hero_share = flop_share_matrix(
            model, hero_in_position=node.in_position == hero, spr=spr
        )

    rows_hero = tuple(
        tuple(
            removal[i][j] * (pot * hero_share[i * NUM_HAND_CLASSES + j] - node.contributions[hero])
            for j in _CLASSES
        )
        for i in _CLASSES
    )
    rows_villain = tuple(
        tuple(
            removal[j][i]
            * (pot * (1.0 - hero_share[i * NUM_HAND_CLASSES + j]) - node.contributions[villain])
            for i in _CLASSES
        )
        for j in _CLASSES
    )
    ordered = (rows_hero, rows_villain) if hero == 0 else (rows_villain, rows_hero)
    return _TerminalEval(ordered)


# ------------------------------------------------------------------ 求解


def solve_preflop(
    config: PreflopConfig | SubgameConfig | None = None,
    *,
    model: RealizationModel | None = None,
    priors: tuple[Range | None, Range | None] = (None, None),
    iterations: int = 400,
    tolerance: float = 1e-3,
    check_every: int = 25,
) -> PreflopSolution:
    """求解一棵翻前树。整桌给 `PreflopConfig`，两人子博弈给 `SubgameConfig`。

    `priors` 给每个玩家一个**先验范围**：走进这棵树时他手上可能是些什么牌。链式求解里
    「面对开牌」的子博弈就是这么用的——开牌者的范围由上一层定死，不在子博弈里重解。
    默认两边都是全范围。

    可利用度低于 `tolerance`（大盲/手）即提前停止。
    """
    cfg = config or PreflopConfig()
    if isinstance(cfg, PreflopConfig) and cfg.num_players != 2:
        raise NotImplementedError(
            f"整桌树只支持单挑，收到 {cfg.num_players} 人；"
            f"六人桌按位置拆成 SubgameConfig 求解，见 ADR-0003"
        )
    if iterations < 1:
        raise ValueError("迭代次数至少为 1")

    tree = build_tree(cfg)
    realization = model or RealizationModel()
    solver = _Solver(tree, realization, priors)
    return solver.run(iterations=iterations, tolerance=tolerance, check_every=check_every)


class _Solver:
    def __init__(
        self,
        tree: PreflopTree,
        model: RealizationModel,
        priors: tuple[Range | None, Range | None] = (None, None),
    ) -> None:
        self.tree = tree
        self.model = model
        self.priors = tuple(_prior_vector(prior) for prior in priors)
        # 每个牌类对上「对手先验范围」时的可用组合数，换算每手 EV 时要除掉它
        removal = removal_rows()
        self.normalizers = tuple(
            tuple(
                max(sum(map(mul, removal[i], self.priors[1 - player])), 1e-12)
                for i in _CLASSES
            )
            for player in (0, 1)
        )
        self.terminals = {
            node.node_id: _build_terminal(node, tree.config, model) for node in tree.terminals
        }
        self.regrets = {
            node.node_id: [[0.0] * len(node.actions) for _ in _CLASSES]
            for node in tree.decisions
        }
        self.strategy_sums = {
            node.node_id: [[0.0] * len(node.actions) for _ in _CLASSES]
            for node in tree.decisions
        }
        self.average: dict[int, tuple[tuple[float, ...], ...]] = {}

    # -------------------------------------------------------------- 主循环

    def run(self, *, iterations: int, tolerance: float, check_every: int) -> PreflopSolution:
        used = 0
        for step in range(1, iterations + 1):
            used = step
            reach = [list(self.priors[0]), list(self.priors[1])]
            self._traverse(self.tree.root, reach, float(step))
            if step % check_every == 0 or step == iterations:
                self._freeze_average()
                gap, _, _ = self._exploitability()
                if gap < tolerance:
                    break

        self._freeze_average()
        gap, evs, hand_ev = self._exploitability()
        return PreflopSolution(
            tree=self.tree,
            model=self.model,
            strategies=dict(self.average),
            reaches=self._node_reaches(),
            exploitability=gap,
            player_ev=evs,
            hand_ev=hand_ev,
            root_branches=self._root_branches(),
            iterations=used,
        )

    def _root_branches(self) -> tuple[BranchValue, ...]:
        """根节点每个动作各走一遍，留下未归一化的价值。≤4 次遍历，很便宜。"""
        root = self.tree.root
        if root.is_terminal:
            return ()
        actor = root.player
        other = 1 - actor
        strategy = self.average[root.node_id]
        removal = removal_rows()
        branches = []
        for action in range(len(root.actions)):
            scaled = [self.priors[actor][i] * strategy[i][action] for i in _CLASSES]
            reach = [None, None]
            reach[actor] = scaled
            reach[other] = list(self.priors[other])
            values = self._average_values(root.children[action], reach)
            # 行动者自己的除数不变（对手的到达概率没被限制），另一边要用被限制后的
            weights = [None, None]
            weights[actor] = self.normalizers[actor]
            weights[other] = tuple(sum(map(mul, removal[i], scaled)) for i in _CLASSES)
            branches.append(
                BranchValue(
                    action=action,
                    values=(tuple(values[0]), tuple(values[1])),
                    weights=(weights[0], weights[1]),
                )
            )
        return tuple(branches)

    def _node_reaches(self) -> dict[int, tuple[float, ...]]:
        """每个决策节点上行动者的到达概率，只看他自己的动作，不含对手的。"""
        reaches: dict[int, tuple[float, ...]] = {}

        def walk(node, own: list[list[float]]) -> None:
            if node.is_terminal:
                return
            actor = node.player
            reaches[node.node_id] = tuple(own[actor])
            strategy = self.average[node.node_id]
            for action, child in enumerate(node.children):
                branch = list(own)
                branch[actor] = [own[actor][i] * strategy[i][action] for i in _CLASSES]
                walk(child, branch)

        walk(self.tree.root, [list(self.priors[0]), list(self.priors[1])])
        return reaches

    def _traverse(
        self, node, reach: list[list[float]], weight: float
    ) -> tuple[list[float], list[float]]:
        if node.is_terminal:
            return self.terminals[node.node_id].values(reach)

        actor = node.player
        other = 1 - actor
        count = len(node.actions)
        strategy = self._current_strategy(node.node_id, count)
        own_reach = reach[actor]

        child_values: list[tuple[list[float], list[float]]] = []
        for action in range(count):
            scaled = [own_reach[i] * strategy[i][action] for i in _CLASSES]
            branch = [None, None]
            branch[actor] = scaled
            branch[other] = reach[other]
            child_values.append(self._traverse(node.children[action], branch, weight))

        actor_values = [0.0] * NUM_HAND_CLASSES
        other_values = [0.0] * NUM_HAND_CLASSES
        for action in range(count):
            child_actor = child_values[action][actor]
            child_other = child_values[action][other]
            for i in _CLASSES:
                actor_values[i] += strategy[i][action] * child_actor[i]
                other_values[i] += child_other[i]

        regrets = self.regrets[node.node_id]
        sums = self.strategy_sums[node.node_id]
        for i in _CLASSES:
            baseline = actor_values[i]
            row = regrets[i]
            for action in range(count):
                # CFR+：遗憾截零，负遗憾不累积
                row[action] = max(0.0, row[action] + child_values[action][actor][i] - baseline)
            share = own_reach[i] * weight
            if share > 0:
                target = sums[i]
                current = strategy[i]
                for action in range(count):
                    target[action] += share * current[action]

        return (actor_values, other_values) if actor == 0 else (other_values, actor_values)

    def _current_strategy(self, node_id: int, count: int) -> list[list[float]]:
        regrets = self.regrets[node_id]
        strategy = []
        uniform = 1.0 / count
        for row in regrets:
            total = sum(row)
            if total > 0:
                strategy.append([value / total for value in row])
            else:
                strategy.append([uniform] * count)
        return strategy

    def _freeze_average(self) -> None:
        for node in self.tree.decisions:
            count = len(node.actions)
            uniform = 1.0 / count
            frozen = []
            for row in self.strategy_sums[node.node_id]:
                total = sum(row)
                if total > 0:
                    frozen.append(tuple(value / total for value in row))
                else:
                    frozen.append((uniform,) * count)
            self.average[node.node_id] = tuple(frozen)

    # -------------------------------------------------------------- 可利用度

    def _exploitability(self):
        """返回 (可利用度, 两人的总体 EV, 两人的逐牌类 EV)。"""
        reach = [list(self.priors[0]), list(self.priors[1])]
        values = self._average_values(self.tree.root, reach)
        hand_ev = tuple(self._per_hand(player, values[player]) for player in (0, 1))
        evs = tuple(self._aggregate(player, hand_ev[player]) for player in (0, 1))

        gap = 0.0
        for player in (0, 1):
            best = self._best_response(self.tree.root, list(self.priors[1 - player]), player)
            gap += self._aggregate(player, self._per_hand(player, best)) - evs[player]
        return gap, evs, hand_ev

    def _per_hand(self, player: int, values: list[float]) -> tuple[float, ...]:
        """反事实价值 → 每手期望。

        `values[i]` 是按对手的可用组合数加权过的，所以除以「对手在先验范围内还剩多少
        组合」才换算回一手牌的期望。对手范围窄时这个除数按牌类不同——共牌效应使然。
        """
        norm = self.normalizers[player]
        return tuple(values[i] / norm[i] for i in _CLASSES)

    def _aggregate(self, player: int, hand_ev: tuple[float, ...]) -> float:
        """先验范围内每手的平均期望。先验是全范围时就是全体起手牌的平均。"""
        prior = self.priors[player]
        total = 0.0
        weight = 0.0
        for i in _CLASSES:
            share = _COMBO_COUNT[i] * prior[i]
            weight += share
            total += share * hand_ev[i]
        return total / weight if weight > 0 else 0.0

    def _average_values(self, node, reach: list[list[float]]):
        """按平均策略走一遍，回传双方的反事实价值。"""
        if node.is_terminal:
            return self.terminals[node.node_id].values(reach)

        actor = node.player
        other = 1 - actor
        strategy = self.average[node.node_id]
        own_reach = reach[actor]
        actor_values = [0.0] * NUM_HAND_CLASSES
        other_values = [0.0] * NUM_HAND_CLASSES

        for action in range(len(node.actions)):
            scaled = [own_reach[i] * strategy[i][action] for i in _CLASSES]
            branch = [None, None]
            branch[actor] = scaled
            branch[other] = reach[other]
            child = self._average_values(node.children[action], branch)
            child_actor, child_other = child[actor], child[other]
            for i in _CLASSES:
                actor_values[i] += strategy[i][action] * child_actor[i]
                other_values[i] += child_other[i]

        return (actor_values, other_values) if actor == 0 else (other_values, actor_values)

    def _best_response(self, node, reach_other: list[float], hero: int) -> list[float]:
        """英雄按最佳应对、对手按平均策略走，回传英雄的反事实价值。"""
        if node.is_terminal:
            return self.terminals[node.node_id].values_for(hero, reach_other)

        count = len(node.actions)
        if node.player == hero:
            branches = [
                self._best_response(node.children[action], reach_other, hero)
                for action in range(count)
            ]
            return [max(branch[i] for branch in branches) for i in _CLASSES]

        strategy = self.average[node.node_id]
        values = [0.0] * NUM_HAND_CLASSES
        for action in range(count):
            scaled = [reach_other[i] * strategy[i][action] for i in _CLASSES]
            branch = self._best_response(node.children[action], scaled, hero)
            for i in _CLASSES:
                values[i] += branch[i]
        return values


def _prior_vector(prior: Range | None) -> tuple[float, ...]:
    """先验范围 → 169 长的权重向量；`None` 表示全范围。"""
    if prior is None:
        return (1.0,) * NUM_HAND_CLASSES
    if not prior:
        raise ValueError("先验范围不能是空的")
    return tuple(prior.weight(i) for i in _CLASSES)

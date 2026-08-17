"""六人桌翻前范围表：按位置分解成一串两人子博弈，再链式合成。

完整的六人翻前树有 62 万个节点，纯 Python 的向量 CFR 跑不动（ADR-0003 的已知限制）。
但六人桌的翻前局面绝大多数本来就是**一个人开牌、一个人应对**：所以这里把整桌拆开——

```
对每个位置 p（前面的人都弃了）：
    对 p 身后的每个防守者 q：
        解一局两人子博弈「p 已开牌到 2.5，q 应对」（p 的范围是输入，不在子博弈里重解）
    把各个 q 的结果合成 p 的开牌 EV，再更新 p 的开牌范围
反复扫，直到范围不再变
```

## 合成怎么算

设 p 身后依次是 q₁…q_k，各自的（聚合）弃牌频率为 f₁…f_k：

```
EV_开牌(i) = Πf_m · W                                  ← 全都弃了，赢下死钱
           + Σ_m (Π_{l<m} f_l)(1 − f_m) · EV_继续_m(i)  ← q_m 是第一个不弃的人
```

`EV_继续_m(i)` 从子博弈里取「防守者没弃牌」的那几个分支合并而来。**合并必须在归一化
之前做**（见 `preflop_solver.BranchValue`）：每个分支的除数按牌类不同，先除再加权会错。

## 外层怎么收敛

外层对「开牌 / 弃牌」这个二选一跑 CFR+（与 `pushfold.py` 同一套遗憾匹配），每一轮用
**平均**开牌范围去重解子博弈。严格说这是「迭代最佳应对 + CFR+ 平滑」，不是对合成博弈
的收敛性证明；判据是**范围本身还变不变**（`TableSolution.max_change`）。

## 刻意的简化（都是抽象，不是疏漏）

1. **只有第一个不弃牌的人继续**：冷跟之后再有人加注（挤压）、多人底池，都不建模。
   多人底池的终局要在三个以上对手的联合分布上积分，那是另一码事。
2. **弃牌的人不带走牌**：不做「他弃牌所以他没有大牌」的共牌/条件化修正，各家弃牌频率
   按聚合值处理，与英雄的牌无关。
3. **身后的人对防守者不构成威胁**：q 在子博弈里不用担心 q 之后还有人。这会让靠后位置的
   防守略偏松。

这三条都会让解偏离真正的六人均衡；它们的方向已知（① ③ 使范围偏松），先记在这里，
校准之后再评估要不要补。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .positions import position_names
from .preflop_solver import PreflopSolution, combine, solve_preflop
from .preflop_tree import SubgameConfig
from .ranges import NUM_HAND_CLASSES, Range
from .realization import RealizationModel

__all__ = ["TableConfig", "OpenSpot", "TableSolution", "solve_table", "defender_advantage"]

_CLASSES = range(NUM_HAND_CLASSES)


# ------------------------------------------------------------------ 配置


@dataclass(frozen=True)
class TableConfig:
    """整桌参数。座位编号沿用 `positions.py`：0 是按钮、1 小盲、2 大盲。"""

    num_players: int = 6
    effective_stack: float = 100.0
    small_blind: float = 0.5
    big_blind: float = 1.0
    ante: float = 0.0
    open_to: float = 2.5
    reraise_multiples: tuple[float, ...] = (3.0, 2.2)
    jam_from_level: int = 2

    def __post_init__(self) -> None:
        if self.num_players < 3:
            raise ValueError("整桌链式求解至少要三个人；单挑直接解整棵树")
        if self.effective_stack <= self.open_to:
            raise ValueError("有效筹码必须大于开牌尺度")
        if self.big_blind <= 0 or self.small_blind < 0 or self.ante < 0:
            raise ValueError("盲注或前注不合法")

    # ---------------------------------------------------------- 位置

    def position_name(self, seat: int) -> str:
        return position_names(self.num_players)[seat]

    def seat_of(self, name: str) -> int:
        return position_names(self.num_players).index(name)

    def posted(self, seat: int) -> float:
        """开牌前已经放进底池的盲注。"""
        if seat == 1:
            return self.small_blind
        if seat == 2:
            return self.big_blind
        return 0.0

    @property
    def preflop_order(self) -> tuple[int, ...]:
        return tuple(range(3, self.num_players)) + (0, 1, 2)

    @property
    def postflop_order(self) -> tuple[int, ...]:
        return tuple(range(1, self.num_players)) + (0,)

    @property
    def openers(self) -> tuple[int, ...]:
        """能「第一个开牌」的位置。大盲不在其列——轮到他时没人加注就是白得。"""
        return self.preflop_order[:-1]

    def behind(self, opener: int) -> tuple[int, ...]:
        order = self.preflop_order
        return order[order.index(opener) + 1 :]

    def in_position_of(self, first: int, second: int) -> int:
        """两人里翻后最后说话的那个座位。"""
        order = self.postflop_order
        return first if order.index(first) > order.index(second) else second

    # ---------------------------------------------------------- 收益

    def walk_value(self, opener: int) -> float:
        """开牌之后所有人都弃了，开牌者的净得（他自己的盲注与前注会回到手里）。"""
        return sum(
            self.posted(seat) + self.ante
            for seat in range(self.num_players)
            if seat != opener
        )

    def fold_value(self, opener: int) -> float:
        """不开牌直接弃掉的净得失。前位是 0，盲注位要亏掉已投入的钱。"""
        return -(self.posted(opener) + self.ante)


# ------------------------------------------------------------------ 结果


@dataclass(frozen=True)
class OpenSpot:
    """一个「第一个开牌」的位置：它的开牌范围，以及身后每个人的应对。"""

    opener: int
    name: str
    open_range: Range
    defenses: dict[int, PreflopSolution]
    """防守者座位 → 「他面对开牌」的子博弈解。"""
    open_hand_ev: tuple[float, ...]
    """逐牌类的开牌 EV（大盲/手）。"""
    fold_value: float

    @property
    def open_frequency(self) -> float:
        return self.open_range.percent()

    def defense(self, defender: int) -> PreflopSolution:
        return self.defenses[defender]


@dataclass(frozen=True)
class TableSolution:
    config: TableConfig
    spots: dict[int, OpenSpot]
    sweeps: int
    max_change: float
    """最后一轮里开牌频率的最大变动，收敛判据。"""

    def spot(self, position: int | str) -> OpenSpot:
        seat = position if isinstance(position, int) else self.config.seat_of(position)
        return self.spots[seat]

    def open_range(self, position: int | str) -> Range:
        return self.spot(position).open_range

    def summary(self) -> str:
        """一行一个位置的开牌频率，给构建脚本打印用。"""
        lines = []
        for seat in self.config.openers:
            spot = self.spots[seat]
            defenses = " ".join(
                f"{self.config.position_name(d)}弃{100 * s.action_frequency(s.tree.root, 0):.0f}%"
                for d, s in spot.defenses.items()
            )
            lines.append(f"{spot.name:>4s} 开牌 {100 * spot.open_frequency:5.1f}%   {defenses}")
        return "\n".join(lines)


# ------------------------------------------------------------------ 求解


def solve_table(
    config: TableConfig | None = None,
    *,
    model: RealizationModel | None = None,
    sweeps: int = 8,
    inner_iterations: int = 200,
    inner_tolerance: float = 1e-3,
    progress=None,
) -> TableSolution:
    """解出整桌的翻前范围表。

    `progress` 可传一个 `callable(text)`，长时间跑的时候用来看进度。
    """
    cfg = config or TableConfig()
    if sweeps < 1:
        raise ValueError("至少要扫一轮")
    realization = model or RealizationModel()

    openers = cfg.openers
    outer = {seat: _OuterStrategy(cfg.fold_value(seat)) for seat in openers}
    spots: dict[int, OpenSpot] = {}
    max_change = float("inf")

    for sweep in range(1, sweeps + 1):
        max_change = 0.0
        for seat in openers:
            previous = spots[seat].open_frequency if seat in spots else None
            defenses = {}
            for defender in cfg.behind(seat):
                defenses[defender] = solve_preflop(
                    _facing_open(cfg, seat, defender),
                    model=realization,
                    priors=(None, outer[seat].average_range()),
                    iterations=inner_iterations,
                    tolerance=inner_tolerance,
                    check_every=max(inner_iterations // 4, 1),
                )
            open_ev = _compose_open_ev(cfg, seat, defenses)
            outer[seat].update(open_ev, sweep)
            spots[seat] = OpenSpot(
                opener=seat,
                name=cfg.position_name(seat),
                open_range=outer[seat].average_range(),
                defenses=defenses,
                open_hand_ev=open_ev,
                fold_value=cfg.fold_value(seat),
            )
            if previous is not None:
                max_change = max(max_change, abs(spots[seat].open_frequency - previous))
            if progress is not None:
                progress(
                    f"第 {sweep} 轮 · {spots[seat].name} 开牌 "
                    f"{100 * spots[seat].open_frequency:.1f}%"
                )

    return TableSolution(config=cfg, spots=spots, sweeps=sweeps, max_change=max_change)


def defender_advantage(solution: PreflopSolution) -> tuple[float, ...]:
    """防守者「继续」比「弃牌」每手好多少（大盲/手），逐牌类。

    风格层放宽范围时要有个顺序：先纳进来的应该是**最接近该打**的牌。这个顺序不该靠
    「权益高低」之类的外部猜测——求解器本来就算出了每手牌两条路的价值，差值就是答案。
    """
    root = solution.tree.root
    fold_index = next(
        (i for i, action in enumerate(root.actions) if action.kind == "fold"), None
    )
    if fold_index is None:
        return (0.0,) * NUM_HAND_CLASSES
    fold_ev = solution.root_branches[fold_index].hand_ev(0)
    others = [b for b in solution.root_branches if b.action != fold_index]
    if not others:
        return (0.0,) * NUM_HAND_CLASSES
    continue_ev = combine(others, player=0)
    return tuple(continue_ev[i] - fold_ev[i] for i in _CLASSES)


def _facing_open(config: TableConfig, opener: int, defender: int) -> SubgameConfig:
    """切出「开牌者已经开到 open_to，防守者应对」这一段。"""
    dead = sum(
        config.posted(seat) + config.ante
        for seat in range(config.num_players)
        if seat not in (opener, defender)
    )
    return SubgameConfig(
        posted=(config.posted(defender), config.open_to),
        dead_money=dead,
        ante=config.ante,
        first_to_act=0,
        in_position=1 if config.in_position_of(opener, defender) == opener else 0,
        names=(config.position_name(defender), config.position_name(opener)),
        raise_level=1,
        last_raise_to=config.open_to,
        already_acted=(False, True),
        effective_stack=config.effective_stack,
        big_blind=config.big_blind,
        open_to=config.open_to,
        reraise_multiples=config.reraise_multiples,
        jam_from_level=config.jam_from_level,
    )


def _compose_open_ev(
    config: TableConfig, opener: int, defenses: dict[int, PreflopSolution]
) -> tuple[float, ...]:
    """把身后各家的子博弈结果合成开牌者的逐牌类 EV。"""
    total = [0.0] * NUM_HAND_CLASSES
    everyone_folded = 1.0
    for defender in config.behind(opener):
        solution = defenses[defender]
        root = solution.tree.root
        fold_frequency = solution.action_frequency(root, 0)
        continues = [branch for branch in solution.root_branches if branch.action != 0]
        # 子博弈里开牌者是玩家 1
        continue_ev = combine(continues, player=1)
        weight = everyone_folded * (1.0 - fold_frequency)
        for i in _CLASSES:
            total[i] += weight * continue_ev[i]
        everyone_folded *= fold_frequency

    walk = config.walk_value(opener)
    for i in _CLASSES:
        total[i] += everyone_folded * walk
    return tuple(total)


@dataclass
class _OuterStrategy:
    """外层「开牌 / 弃牌」的 CFR+ 状态。"""

    fold_value: float
    regrets: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0] for _ in _CLASSES]
    )
    sums: list[list[float]] = field(
        default_factory=lambda: [[0.5, 0.5] for _ in _CLASSES]
    )
    weight_sum: float = 1.0
    seeded: bool = True
    """第一轮还没有 EV 可依据，先用 50/50 当先验；它一进平均就会给每手垃圾牌垫上
    一个永不消失的开牌频率，所以第一次更新时把种子丢掉。"""

    def current(self) -> list[float]:
        """当前（未平均的）开牌概率。"""
        probabilities = []
        for row in self.regrets:
            total = row[0] + row[1]
            probabilities.append(row[0] / total if total > 0 else 0.5)
        return probabilities

    def update(self, open_ev: tuple[float, ...], step: int) -> None:
        if self.seeded:
            self.sums = [[0.0, 0.0] for _ in _CLASSES]
            self.weight_sum = 0.0
            self.seeded = False
        current = self.current()
        for i in _CLASSES:
            value = current[i] * open_ev[i] + (1.0 - current[i]) * self.fold_value
            row = self.regrets[i]
            row[0] = max(0.0, row[0] + open_ev[i] - value)
            row[1] = max(0.0, row[1] + self.fold_value - value)
        # 累加的是**更新后**的策略：这一轮的 EV 已经算过了，用它修正过的策略更贴近
        # 当前的对手，也避免把毫无依据的初始均分算进平均
        updated = self.current()
        weight = float(step)
        self.weight_sum += weight
        for i in _CLASSES:
            self.sums[i][0] += weight * updated[i]
            self.sums[i][1] += weight * (1.0 - updated[i])

    def average_range(self) -> Range:
        weights = {}
        for i in _CLASSES:
            value = self.sums[i][0] / self.weight_sum
            if value > 1e-4:
                weights[i] = round(value, 4)
        if not weights:
            raise RuntimeError("开牌范围收敛到了空集，检查收益模型")
        return Range(weights)

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
3. ~~**身后的人对防守者不构成威胁**~~：**已补上**，见下。

① ② 都会让解偏离真正的六人均衡，方向已知（① 使范围偏松），先记在这里，校准之后再
评估要不要补。

## 身后的挤压（补上的那条简化）

只在两人子博弈里想事情，防守者跟注就永远没有后顾之忧——于是冷跟得远比真解宽，
整桌的入池率也跟着虚高。真实的六人桌里，跟注之后身后每个人都可能**挤压**（squeeze），
把跟注者夹在中间。所以这里给「防守者跟注、行动结束」那个终局挂上 `SqueezeRisk`：

```
以概率 S 被挤压 → 收益换成「防守者面对挤压」那一小盘的解
以概率 1 − S 照常进翻牌
```

- **挤压者是谁**：q 身后的每一个人 r 各算一次，按「谁先挤压」的概率加权合成。
  某个 r 的挤压频率取他**面对同一个开牌的 3bet 频率** × `SqueezeModel.frequency_scale`
  ——面对「开牌 + 一个跟注」的挤压比单纯 3bet 要少，这个折扣是假设，不是测量值。
- **被挤压之后怎么打**：再切一段子博弈（`_facing_squeeze`），q 可以弃/跟/4bet/全下，
  照常用 CFR+ 解，所以强牌不会白白被挤走，弱牌该弃就弃。
- **开牌者怎么办**：假设他弃牌，他投进去的 `open_to` 全部变成死钱由 q 与挤压者去争。
  这条同时决定了开牌者在这条支路上的收益（−`open_to`），所以开牌范围也跟着收紧一点。
  （他其实可以跟/4bet，那只会比弃牌更好，所以这里给的是他的下界。）
- **用的是上一轮的 3bet 频率**：q 身后那几家的解在同一轮里还没算到，所以取上一轮的
  ——与外层用平均范围重解是同一个套路。**第一轮没有可用数据，等于不建模挤压**。

挤压之后**这条支路不再是两人零和**：底池有一部分被第三方拿走，`player_ev` 之和会小于
死钱，差额就是挤压者的所得。这是模型的应有之义，不是账没平。

## 还没建模的（新的已知简化）

- q **3bet 之后**身后再有人冷 4bet，不建模（频率低一个量级）。
- 挤压者只被建模成「加注」，他冷跟造成的多人底池仍然按 ① 处理。
- 挤压尺度固定在 3bet 梯子上再加 `extra_bb`，不按人数/位置变化。

## 幅度过头了（评估结果时记住这条）

补上挤压之后，开牌范围比公开解**紧 2–4 个百分点**（此前是松 2–3 个）。方向对、幅度过头，
最大嫌疑就是上面那条「被挤压时开牌者一律弃牌」——那是他的下界。
**别拿公开图表去回调 `frequency_scale`**：那等于把别人的表当真值，与本项目
「正确性靠可利用度自证、公开图表只作量级对照」的口径相悖。真校准走 ADR-0003 的 B 档。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .positions import position_names
from .preflop_solver import (
    PreflopSolution,
    SqueezeRisk,
    action_advantage,
    combine,
    solve_preflop,
)
from .preflop_tree import SubgameConfig, build_tree
from .ranges import NUM_HAND_CLASSES, Range
from .realization import RealizationModel

__all__ = [
    "TableConfig",
    "SqueezeModel",
    "OpenSpot",
    "TableSolution",
    "solve_table",
    "defender_advantage",
]

_CLASSES = range(NUM_HAND_CLASSES)


# ------------------------------------------------------------------ 配置


@dataclass(frozen=True)
class SqueezeModel:
    """身后挤压的建模参数。**和 `realization.py` 一样，这些是假设，不是测量值。**

    所以测试只验性质（有人在身后才有挤压、越多人越紧、概率为 0 时退化成原来的解），
    不把具体数字钉成基准；将来用实解校准（ADR-0003 的 B 档）时整体替换。
    """

    frequency_scale: float = 0.6
    """挤压频率 = 该位置面对开牌的 3bet 频率 × 这个系数。

    面对「开牌 + 一个跟注」时人们挤压得比单纯 3bet 少（底池更大但要过两关），
    公开统计里大约是 3bet 频率的一半到六成。取 0.6 偏保守一侧（略多算挤压）。
    """
    extra_bb: float = 1.0
    """挤压尺度在 3bet 梯子之上再加多少——底池里多了一个跟注者，主流打法要加价。

    与树里「每有一个跛入者开牌 +1bb」是同一条惯例。
    """
    iterations: int = 120
    """「面对挤压」那一小盘的 CFR+ 迭代数。树只有几个节点，比外面那盘便宜得多。"""

    def __post_init__(self) -> None:
        if not 0.0 <= self.frequency_scale <= 2.0:
            raise ValueError(f"挤压频率系数越界: {self.frequency_scale}")
        if self.extra_bb < 0:
            raise ValueError("挤压加价不能为负")
        if self.iterations < 1:
            raise ValueError("迭代次数至少为 1")


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
    squeeze: SqueezeModel | None = SqueezeModel()
    """身后挤压的建模；`None` 表示关掉（回到 ADR-0004 原来的简化 ③）。"""

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

    @property
    def squeeze_to(self) -> float:
        """挤压（面对「开牌 + 一个跟注」再加注）的目标额：3bet 梯子上再加价。"""
        settings = self.squeeze or SqueezeModel()
        ladder = (
            self.open_to * self.reraise_multiples[0]
            if self.reraise_multiples
            else self.effective_stack
        )
        return min(ladder + settings.extra_bb, self.effective_stack)


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
    squeezes: dict[int, float] = field(default_factory=dict)
    """防守者座位 → 他跟注之后身后挤压的概率。没建模挤压时是空的。"""

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
                + (f"(挤{100 * spot.squeezes[d]:.0f}%)" if spot.squeezes.get(d) else "")
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
    # 「谁面对谁的开牌 3bet 多少」——建模挤压要用身后那几家的这个数。他们在同一轮里
    # 还没轮到，所以取上一轮的；第一轮没有，等于第一轮不建模挤压。
    profiles: dict[tuple[int, int], _DefenseProfile] = {}
    max_change = float("inf")

    for sweep in range(1, sweeps + 1):
        max_change = 0.0
        for seat in openers:
            previous = spots[seat].open_frequency if seat in spots else None
            defenses = {}
            squeezes: dict[int, float] = {}
            for defender in cfg.behind(seat):
                subgame = _facing_open(cfg, seat, defender)
                risk = _squeeze_risk(cfg, seat, defender, profiles, realization)
                terminal = _flat_call_terminal(subgame) if risk is not None else None
                if terminal is not None:
                    squeezes[defender] = risk.probability
                solution = solve_preflop(
                    subgame,
                    model=realization,
                    priors=(None, outer[seat].average_range()),
                    squeeze={terminal: risk} if terminal is not None else None,
                    iterations=inner_iterations,
                    tolerance=inner_tolerance,
                    check_every=max(inner_iterations // 4, 1),
                )
                defenses[defender] = solution
                profiles[(seat, defender)] = _reraise_profile(solution)
            open_ev = _compose_open_ev(cfg, seat, defenses)
            outer[seat].update(open_ev, sweep)
            spots[seat] = OpenSpot(
                opener=seat,
                name=cfg.position_name(seat),
                open_range=outer[seat].average_range(),
                defenses=defenses,
                open_hand_ev=open_ev,
                fold_value=cfg.fold_value(seat),
                squeezes=squeezes,
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

    子博弈的根节点就是防守者的决策点，所以直接取根节点那一份
    （通用版在 `preflop_solver.action_advantage`，单挑整树解要指名到防守者那个节点）。
    """
    return action_advantage(solution)


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


@dataclass(frozen=True)
class _DefenseProfile:
    """一个防守者面对开牌时的再加注（3bet）画像，用来估他会不会挤压。"""

    frequency: float
    range: Range


def _reraise_profile(solution: PreflopSolution) -> _DefenseProfile:
    """把「面对开牌」的解压成 3bet 频率 + 3bet 范围。

    各个加注尺度是互斥的分支，权重直接相加（`union` 取最大值会漏掉一半）。
    """
    root = solution.tree.root
    weights: dict[int, float] = {}
    frequency = 0.0
    for index, action in enumerate(root.actions):
        if not action.is_raise:
            continue
        frequency += solution.action_frequency(root, index)
        for hand, weight in solution.action_range(root, index).weights.items():
            weights[hand] = min(1.0, weights.get(hand, 0.0) + weight)
    return _DefenseProfile(frequency=frequency, range=Range(weights))


def _flat_call_terminal(subgame: SubgameConfig) -> int | None:
    """「防守者跟注、行动结束」那个终局的编号；挤压就挂在它上面。

    这里重建一次树只为拿编号（十来个节点，几乎不要钱）。编号是按固定顺序发的，
    所以与 `solve_preflop` 内部那棵树对得上——两处用的是同一个 `build_tree`。
    """
    root = build_tree(subgame).root
    if root.is_terminal:
        return None
    for index, action in enumerate(root.actions):
        if action.kind == "call":
            child = root.children[index]
            return child.node_id if child.is_terminal else None
    return None


def _facing_squeeze(
    config: TableConfig, opener: int, defender: int, squeezer: int
) -> SubgameConfig:
    """切出「防守者跟了开牌，身后某一家挤压」的一段。玩家 0 是防守者、1 是挤压者。

    开牌者按「弃牌」处理：他投进去的 `open_to`（含他的盲注与前注）原样留在底池里变成
    死钱，由这两个人去争。账因此是平的——桌上每一分钱要么在某个人的 `posted` 里，
    要么在 `dead_money` 里。
    """
    dead = config.open_to + config.ante
    dead += sum(
        config.posted(seat) + config.ante
        for seat in range(config.num_players)
        if seat not in (opener, defender, squeezer)
    )
    return SubgameConfig(
        posted=(config.open_to, config.squeeze_to),
        dead_money=dead,
        ante=config.ante,
        first_to_act=0,
        in_position=1 if config.in_position_of(defender, squeezer) == squeezer else 0,
        names=(config.position_name(defender), config.position_name(squeezer)),
        raise_level=2,
        last_raise_to=config.squeeze_to,
        already_acted=(False, True),
        effective_stack=config.effective_stack,
        big_blind=config.big_blind,
        open_to=config.open_to,
        reraise_multiples=config.reraise_multiples,
        jam_from_level=config.jam_from_level,
    )


def _squeeze_risk(
    config: TableConfig,
    opener: int,
    defender: int,
    profiles: dict[tuple[int, int], _DefenseProfile],
    model: RealizationModel,
) -> SqueezeRisk | None:
    """防守者跟注之后被身后挤压的风险；没人在身后（或没建模）时是 `None`。

    身后每一家各解一小盘「面对他的挤压」，再按「谁是第一个挤压的人」的概率合成。
    """
    settings = config.squeeze
    if settings is None:
        return None
    squeezers = config.behind(defender)
    if not squeezers:
        return None
    if config.squeeze_to >= config.effective_stack:
        # 筹码浅到挤压就是全下：那是推弃的地盘，不在这个模型里
        return None

    values = [0.0] * NUM_HAND_CLASSES
    remaining = 1.0
    total = 0.0
    for seat in squeezers:
        profile = profiles.get((opener, seat))
        if profile is None or not profile.range:
            continue
        chance = min(1.0, settings.frequency_scale * profile.frequency)
        if chance < 1e-4:
            continue
        weight = remaining * chance
        remaining -= weight
        total += weight
        solution = solve_preflop(
            _facing_squeeze(config, opener, defender, seat),
            model=model,
            priors=(None, profile.range),
            iterations=settings.iterations,
            tolerance=1e-3,
            check_every=max(settings.iterations // 4, 1),
        )
        defender_ev = solution.hand_ev[0]
        for i in _CLASSES:
            values[i] += weight * defender_ev[i]

    if total < 1e-4:
        return None
    # 条件在「确实被挤压了」上，所以要除掉总概率
    defender_values = tuple(values[i] / total for i in _CLASSES)
    # 开牌者弃牌，亏掉他开出去的钱（这是他的下界：跟或 4bet 只会更好）
    opener_value = -(config.open_to + config.ante)
    return SqueezeRisk(
        probability=total,
        values=(defender_values, (opener_value,) * NUM_HAND_CLASSES),
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

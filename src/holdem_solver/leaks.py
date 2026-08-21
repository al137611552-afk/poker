"""按场景聚合 EV 损失并排序——漏洞报告（FR-10）。**纯逻辑，不碰 IO。**

复盘给的是「这一个决策亏了多少」。可是一手一手看不出毛病：真正有用的是
**「你的钱主要从哪类局面漏出去」**。这一层把逐点的 EV 损失按场景归类、加总、排序。

```python
report = build_report(results, hands=200)
for leak in report.leaks[:5]:
    print(leak.scenario.title, leak.spots, leak.per_100_hands)
```

## 场景怎么分

四个维度，组合起来就是场景：**街** × **翻前角色**（进攻方／防守方）× **位置** ×
**面对什么**（无人下注／面对下注／面对加注）。常见的那几格给了通俗别名（持续下注、
面对持续下注、转牌第二枪…），但归类用的始终是四个维度本身——别名只是给人看的。

**角色必须按翻前最后加注的人分**：同一个「翻牌下注」，进攻方打出来是持续下注、
防守方打出来是领打，两者的正确频率差着数量级，混在一格里等于把信号平均掉。

## 排序按「总漏损」，不按「平均每次亏多少」

平均值最大的那格往往是「河牌面对全下」这种一年遇不上几次的局面，改它不值钱。
**钱是「每次亏多少 × 多常遇到」漏掉的**，所以排序键是总漏损，报告里同时给出
平均值与次数，让人看得出是「亏得狠」还是「亏得勤」。

## 报告必须自曝覆盖率

打不了分的点（翻前线路没覆盖、尺度不在树里、dump 不够）**逐条计数并写进报告**。
一份「按 12 个决策算出来的漏洞报告」跟一份「按 1200 个决策算出来的」长得一模一样，
不写覆盖率就没人分得清。

## 报告还必须自曝**解自己收敛了没有**

覆盖率管的是「算了多少个点」，收敛度管的是「算出来的数当不当得了真」。EV 损失是拿实战
动作跟**解**比出来的差；解要是没收敛，那个差里有一部分根本不是漏洞，是求解器自己的残差。
所以 `Convergence` 把每个解的可利用度收进报告，并折成**大盲口径的噪声底**
（`可利用度% × 底池`）跟平均每点漏损直接比——比值小于 3 就别照着排行去改打法。

噪声底有两个来源，**取大的那个**：

- 求解器自报的**整局面**可利用度（`Convergence`，日志里那一行，量不到就没有）；
- 解在**我们真正打分的那几个点上**离最优差多少（`DecisionScore.gap`，我们自己在树上
  算出来的）。后者更贴题——报告只在那几个点上做减法，解在别处收没收敛不重要。

两个都是「同一把尺子上的残差」：漏损跟它一个量级，那个数就不是漏洞，是没解干净。

## 为什么这里的「/100 手」不走 `metrics.py`

`metrics.py` 是**胜率**口径（带 95% 置信区间，因为单手盈亏方差极大）。这里的数不是
赢了多少，而是**给定解之后确定算出来的 EV 损失合计**——同一批牌重算一遍分毫不差，
没有「区间」可言。两者别混着读：漏损 2bb/100 不等于「打好了就能多赢 2bb/100」，
它是**上界**（假设对手照解走、且我们把这些决策全改对）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from holdem.state import FLOP, RIVER, TURN

from .review import DecisionPoint, ReviewResult, ScoredDecision

__all__ = [
    "Convergence",
    "Scenario",
    "ScenarioLeak",
    "LeakReport",
    "scenario_of",
    "build_report",
]

_STREETS = {FLOP: "翻牌", TURN: "转牌", RIVER: "河牌"}

NO_BET, FACING_BET, FACING_RAISE = "无人下注", "面对下注", "面对加注"
AGGRESSOR, DEFENDER = "进攻方", "防守方"
IN_POSITION, OUT_OF_POSITION = "有位置", "没位置"

_ALIASES = {
    ("翻牌", AGGRESSOR, NO_BET): "持续下注",
    ("翻牌", DEFENDER, FACING_BET): "面对持续下注",
    ("转牌", AGGRESSOR, NO_BET): "转牌第二枪",
    ("转牌", DEFENDER, FACING_BET): "面对转牌第二枪",
    ("河牌", AGGRESSOR, NO_BET): "河牌下注",
    ("河牌", DEFENDER, FACING_BET): "河牌面对下注",
    ("翻牌", DEFENDER, NO_BET): "领打机会",
}


@dataclass(frozen=True)
class Scenario:
    """一类翻后局面。四个维度定死一格。"""

    street: str
    role: str
    position: str
    facing: str

    @property
    def alias(self) -> str | None:
        """常用叫法，没有就是 `None`。"""
        return _ALIASES.get((self.street, self.role, self.facing))

    @property
    def title(self) -> str:
        """写给人看的名字：有别名就带上。"""
        core = f"{self.street}·{self.role}·{self.position}·{self.facing}"
        return f"{self.alias}（{core}）" if self.alias else core


@dataclass(frozen=True)
class ScenarioLeak:
    """一格场景漏了多少。"""

    scenario: Scenario
    spots: int
    total_loss: float
    off_range_spots: int
    """其中英雄的牌不在假设范围里的点——风格层打的表外牌，别跟正常点一起解读。"""

    @property
    def mean_loss(self) -> float:
        return self.total_loss / self.spots if self.spots else 0.0


@dataclass(frozen=True)
class Convergence:
    """这批解**自己**收敛到什么程度。可利用度是求解器报的，单位是「%底池」。"""

    solves: int
    """这批报告一共动用了多少次求解。"""
    exploitability: tuple[float, ...]
    """量得到可利用度的那些解，逐个记着（%底池）。"""
    unmeasured: int
    """求解器没报可利用度的次数（旧缓存、日志里没那一行）——**当成未知，不当成收敛**。"""
    noise_bb: tuple[float, ...]
    """把每个解的可利用度折成大盲：`可利用度% × 底池 / 100`。跟漏损同一把尺子才比得了。"""
    accuracy: float | None
    """当初要求的收敛门槛（%底池）。`None`＝没记，判不了收没收敛。"""

    @property
    def worst(self) -> float | None:
        return max(self.exploitability) if self.exploitability else None

    @property
    def median(self) -> float | None:
        if not self.exploitability:
            return None
        values = sorted(self.exploitability)
        return values[len(values) // 2]

    @property
    def unconverged(self) -> int:
        """超过门槛的解有几个。门槛没记就返回 0（**别把「不知道」报成「都过了」**，
        判断可信度请一并看 `unmeasured` 与 `noise_floor`）。"""
        if self.accuracy is None:
            return 0
        return sum(1 for value in self.exploitability if value > self.accuracy)

    @property
    def noise_floor(self) -> float | None:
        """平均每个解剩多少残差（大盲）。**这就是「一个决策点的漏损小到多少就不值得信」**。"""
        return sum(self.noise_bb) / len(self.noise_bb) if self.noise_bb else None


@dataclass(frozen=True)
class LeakReport:
    """一批牌的漏洞报告。金额一律是大盲。"""

    hands: int
    """一共看了多少手（含没打上分的）。"""
    leaks: tuple[ScenarioLeak, ...]
    """按**总漏损**倒序。"""
    scored_spots: int
    skipped: tuple[tuple[str, int], ...]
    """打不了分的原因 → 次数，次数多的在前。"""
    uncovered_hands: tuple[tuple[str, int], ...]
    """连计划都做不出来的手牌（翻前线路没覆盖等）→ 次数。"""
    solver_gaps: tuple[float, ...] = ()
    """逐点记着「解自己离最优差多少」（大盲，`DecisionScore.gap`）。
    只收英雄的牌在范围里、且解确实给了频率的点——表外牌没有频率，那里的 gap 没意义。"""
    convergence: Convergence | None = None
    """这批解收敛到什么程度。`None`＝没记（老报告）。"""

    @property
    def total_loss(self) -> float:
        return sum(leak.total_loss for leak in self.leaks)

    @property
    def skipped_spots(self) -> int:
        return sum(count for _, count in self.skipped)

    @property
    def reviewed_hands(self) -> int:
        """真正做出计划、进了求解的手数。"""
        return self.hands - sum(count for _, count in self.uncovered_hands)

    @property
    def coverage(self) -> float:
        """打上分的决策点占比——**报告可信到什么程度，全看这个数**。"""
        total = self.scored_spots + self.skipped_spots
        return self.scored_spots / total if total else 0.0

    def per_100_hands(self, loss: float) -> float:
        """把一笔漏损折成「每 100 手漏多少大盲」。看的是**看过的手数**，不是打上分的点数。"""
        return 100.0 * loss / self.hands if self.hands else 0.0

    def share(self, leak: ScenarioLeak) -> float:
        total = self.total_loss
        return leak.total_loss / total if total > 0 else 0.0

    @property
    def mean_loss(self) -> float:
        """平均每个打上分的决策点漏多少大盲。"""
        return self.total_loss / self.scored_spots if self.scored_spots else 0.0

    @property
    def mean_solver_gap(self) -> float | None:
        """打分的那些点上，解自己平均离最优差多少（大盲）。"""
        if not self.solver_gaps:
            return None
        return sum(self.solver_gaps) / len(self.solver_gaps)

    @property
    def noise_floor(self) -> float | None:
        """一个点的漏损小到多少就不值得信（大盲）。两路残差**取大的**：
        求解器自报的整局面可利用度、以及打分那几个点上解自己的 gap。两路都没有就是 `None`。
        """
        floors = [
            value
            for value in (
                self.convergence.noise_floor if self.convergence else None,
                self.mean_solver_gap,
            )
            if value is not None
        ]
        return max(floors) if floors else None

    @property
    def signal_to_noise(self) -> float | None:
        """平均每点漏损 ÷ 噪声底。两路残差都量不到就是 `None`（**不是「很好」**）。"""
        floor = self.noise_floor
        if floor is None or floor <= 0.0:
            return None
        return self.mean_loss / floor

    def ranking_trustworthy(self, *, margin: float = 3.0) -> bool | None:
        """这份排行照着改打法靠不靠谱。`None`＝判不了（没记收敛度）。

        判据只有一条：**平均每点漏损要比解自己的残差（`noise_floor`）大出 `margin` 倍**。
        差不到这个数，排在前面的多半是「这格出现得多」而不是「这格漏得狠」——因为负差按 0
        算（见 `_loss_of`），噪声在合计里只加不减，格子越常见累计噪声越高。
        """
        ratio = self.signal_to_noise
        return None if ratio is None else ratio >= margin


def scenario_of(point: DecisionPoint, *, aggressor_seat: int) -> Scenario:
    """一个决策点属于哪一格。"""
    street_steps = [
        step
        for step in point.prefix
        if step.street == point.street and not step.is_deal
    ]
    aggressive = [step for step in street_steps if step.kind in ("bet", "raise")]
    if not aggressive:
        facing = NO_BET
    elif len(aggressive) == 1:
        facing = FACING_BET
    else:
        facing = FACING_RAISE
    return Scenario(
        street=_STREETS.get(point.street, str(point.street)),
        role=AGGRESSOR if point.seat == aggressor_seat else DEFENDER,
        position=IN_POSITION if point.hero == 1 else OUT_OF_POSITION,
        facing=facing,
    )


def build_report(
    results, *, hands: int, uncovered=(), exploitability=(), accuracy: float | None = None
) -> LeakReport:
    """把一批复盘结果聚成漏洞报告。

    `hands` 是**一共看了多少手**（含 `uncovered` 里那些连计划都做不出来的），
    这样报告里的「每 100 手漏多少」才是对着真实牌局密度说的。

    `exploitability` 是**与 `results` 一一对应**的可利用度（%底池，求解器自己报的，
    量不到就给 `None`），`accuracy` 是当初要求的收敛门槛。两个都给了，报告才判得了
    自己可不可信——**不给不报错，但 `ranking_trustworthy()` 会返回「判不了」。**
    """
    results = list(results)
    totals: dict[Scenario, list] = {}
    skipped: Counter = Counter()
    scored_spots = 0
    solver_gaps: list[float] = []

    for result in results:
        aggressor = result.plan.setup.aggressor_seat
        for decision in result.decisions:
            if decision.score is None:
                skipped[decision.skipped or "没说原因"] += 1
                continue
            scenario = scenario_of(decision.point, aggressor_seat=aggressor)
            bucket = totals.setdefault(scenario, [0, 0.0, 0])
            bucket[0] += 1
            bucket[1] += _loss_of(decision)
            bucket[2] += 0 if decision.in_range else 1
            scored_spots += 1
            gap = _solver_gap(decision)
            if gap is not None:
                solver_gaps.append(gap)

    leaks = tuple(
        sorted(
            (
                ScenarioLeak(
                    scenario=scenario,
                    spots=spots,
                    total_loss=loss,
                    off_range_spots=off_range,
                )
                for scenario, (spots, loss, off_range) in totals.items()
            ),
            key=lambda leak: (-leak.total_loss, leak.scenario.title),
        )
    )
    return LeakReport(
        hands=hands,
        leaks=leaks,
        scored_spots=scored_spots,
        skipped=tuple(skipped.most_common()),
        uncovered_hands=tuple(Counter(uncovered).most_common()),
        solver_gaps=tuple(solver_gaps),
        convergence=_convergence(results, exploitability, accuracy),
    )


def _convergence(results, exploitability, accuracy) -> Convergence | None:
    """把逐个解的可利用度收成一份收敛度。一个都没给就返回 `None`（老报告没这一栏）。"""
    values = list(exploitability)
    if not values:
        return None
    measured, noise = [], []
    for index, value in enumerate(values):
        if value is None:
            continue
        measured.append(value)
        if index < len(results):
            # 可利用度是「%底池」，乘上这个局面的底池才跟漏损同一把尺子（大盲）
            noise.append(value / 100.0 * results[index].plan.request.pot)
    return Convergence(
        solves=len(values),
        exploitability=tuple(measured),
        unmeasured=len(values) - len(measured),
        noise_bb=tuple(noise),
        accuracy=accuracy,
    )


def _solver_gap(decision: ScoredDecision) -> float | None:
    """解在这个点上离最优差多少（大盲）。**没有频率就没有 gap**——英雄打的是表外牌时
    `strategy` 是空的，`solved_ev` 会算成 0，那个「gap」等于最优 EV 本身，纯属胡说。"""
    score = decision.score
    if score is None or not decision.in_range or not score.strategy:
        return None
    return max(0.0, score.gap)


def _loss_of(decision: ScoredDecision) -> float:
    """EV 损失。**负数按 0 算**：解自己没收敛干净时会出现极小的负差，
    那不是「打得比解还好」，把它当成 0 才不会在合计里凭空造钱。"""
    loss = decision.loss or 0.0
    return max(0.0, loss)

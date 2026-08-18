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

__all__ = ["Scenario", "ScenarioLeak", "LeakReport", "scenario_of", "build_report"]

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
    results, *, hands: int, uncovered=()
) -> LeakReport:
    """把一批复盘结果聚成漏洞报告。

    `hands` 是**一共看了多少手**（含 `uncovered` 里那些连计划都做不出来的），
    这样报告里的「每 100 手漏多少」才是对着真实牌局密度说的。
    """
    totals: dict[Scenario, list] = {}
    skipped: Counter = Counter()
    scored_spots = 0

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
    )


def _loss_of(decision: ScoredDecision) -> float:
    """EV 损失。**负数按 0 算**：解自己没收敛干净时会出现极小的负差，
    那不是「打得比解还好」，把它当成 0 才不会在合计里凭空造钱。"""
    loss = decision.loss or 0.0
    return max(0.0, loss)

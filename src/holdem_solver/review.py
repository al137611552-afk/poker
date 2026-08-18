"""把**打完的一手牌**接到求解器上：逐个翻后决策点算 EV 损失（FR-9）。

前面两块各就各位之后，这里只做「接线」——而接线恰恰是最容易接歪的地方：

| 已有 | 干什么 |
|---|---|
| `holdem.range_tracking` | 翻前线路 → 翻牌时双方范围、底池、有效筹码 |
| `request` / `backend` | 局面 → 命令文件 → 解出来的树 |
| `evaluate` | 树 + 英雄的两张牌 → 各条路的 EV |

```python
plan = plan_review(hand, hero_seat=3)      # 纯逻辑：要解哪个局面、要打分哪几个点
report = solver.solve(plan.request)        # 唯一碰进程与磁盘的一步
result = score_plan(plan, report.root)     # 纯逻辑：逐点 EV 损失
result.total_loss                          # 这手牌翻后一共漏了多少大盲
```

## 三条纪律

1. **说不了就说不了**。翻前线路表里没有（`NotCovered`）、实战尺度树里没有
   （`LineNotInTree`）、dump 层数不够——一律**明说原因**，不拿半棵树糊弄出一个数。
   一个说不清来路的「你亏了 2.3bb」比没有数字更糟。
2. **实战打出的尺度必须先并进树里**（`BetSizes.with_size`），否则那个动作在解里根本
   不存在，也就没法给它打分。加注尺度并不进去（求解器的 raise 百分比口径没实测过），
   所以**实战的加注尺度对不上时会如实报出来**，不悄悄换成最近的那个。
3. **英雄的牌可能不在我们假设的范围里**（风格层会打表外的牌）。这不是错误：EV 照样
   算得出来（英雄的牌是固定的），但解在那个点上没有给这手牌的频率——
   `ScoredDecision.in_range` 标出来，报告里别把这种点跟正常点混着平均。

## dump 必须导满三层

**只导一层连翻牌上的决策都算不了 EV**：任何一条走到转牌的路在树里都没有下文，而 EV 是
把后面的期望积出来的。所以 `plan_review` 默认 `dump_rounds=3`（翻牌局面＝翻/转/河三条街）,
代价是产物大、解得慢——这一层本来就只服务复盘，不进实战。

层数不够时**不是整手牌作废**：够不着的那个点带上原因跳过，其余的点照常出分。
"""

from __future__ import annotations

from dataclasses import dataclass

from holdem.cards import card_to_str
from holdem.range_tracking import FlopRanges, NotCovered, flop_ranges
from holdem.state import FLOP, PREFLOP, HandState

from .evaluate import DecisionScore, Spot, score_decision
from .request import BetSizes, SolveRequest
from .result import SolvedNode

__all__ = [
    "DecisionPoint",
    "LineNotInTree",
    "ReviewPlan",
    "ReviewResult",
    "ScoredDecision",
    "Step",
    "plan_review",
    "score_plan",
]

_STREET_NAMES = {FLOP: "flop", 2: "turn", 3: "river"}


class LineNotInTree(RuntimeError):
    """实战这条线在解出来的树里走不通（多半是尺度对不上，或者 dump 层数不够）。"""


@dataclass(frozen=True)
class Step:
    """牌局里的一步，已经翻成求解器的口径。金额是**大盲**、是「本街投到多少」。"""

    kind: str
    """`check` / `call` / `fold` / `bet` / `raise` / `deal`。"""
    seat: int | None = None
    amount: float | None = None
    card: int | None = None
    street: int = FLOP

    @property
    def is_deal(self) -> bool:
        return self.kind == "deal"


@dataclass(frozen=True)
class DecisionPoint:
    """一个待打分的翻后决策点：走到这儿的路 + 英雄实际打了什么。"""

    street: int
    seat: int
    hero: int
    """求解器口径：0 = OOP、1 = IP。"""
    hero_cards: tuple[int, int]
    prefix: tuple[Step, ...]
    """从翻牌根节点起、双方依次走过的每一步（含发牌）。"""
    taken: Step

    @property
    def street_name(self) -> str:
        return _STREET_NAMES.get(self.street, str(self.street))


@dataclass(frozen=True)
class ReviewPlan:
    """这手牌要解哪个局面、要给哪几个点打分。**纯数据**，解之前就能全部算出来。"""

    request: SolveRequest
    setup: FlopRanges
    hero_seat: int
    points: tuple[DecisionPoint, ...]

    @property
    def spot(self) -> Spot:
        return Spot.from_request(self.request)


@dataclass(frozen=True)
class ScoredDecision:
    """一个决策点的打分结果——**要么有分，要么有说不出分的原因**。"""

    point: DecisionPoint
    score: DecisionScore | None = None
    label: str | None = None
    """实际动作在树里的标签。"""
    skipped: str | None = None
    """打不了分的原因（尺度不在树里、dump 截断、走不通…）。"""
    in_range: bool = True
    """英雄这手牌在我们假设的范围里吗——不在的话解没给它频率。"""

    @property
    def loss(self) -> float | None:
        if self.score is None or self.label is None:
            return None
        return self.score.loss(self.label)


@dataclass(frozen=True)
class ReviewResult:
    """一手牌的翻后复盘。"""

    plan: ReviewPlan
    decisions: tuple[ScoredDecision, ...]

    @property
    def scored(self) -> tuple[ScoredDecision, ...]:
        return tuple(d for d in self.decisions if d.score is not None)

    @property
    def total_loss(self) -> float:
        """翻后一共漏了多少大盲（只算打得了分的点）。"""
        return sum(d.loss or 0.0 for d in self.scored)

    @property
    def worst(self) -> ScoredDecision | None:
        """漏得最多的那个决策，FR-10 的漏洞报告从这里长出来。"""
        candidates = self.scored
        return max(candidates, key=lambda d: d.loss or 0.0) if candidates else None

    def skipped_reasons(self) -> tuple[str, ...]:
        return tuple(d.skipped for d in self.decisions if d.skipped)


# ------------------------------------------------------------------ 规划（纯逻辑）


def plan_review(
    hand: HandState,
    hero_seat: int,
    *,
    bet_sizes: BetSizes = BetSizes(),
    tables=None,
    **request_options,
) -> ReviewPlan:
    """从打完的一手牌算出复盘计划。翻前线路表里没有就抛 `NotCovered`。"""
    if not 0 <= hero_seat < hand.config.num_seats:
        raise ValueError(f"没有 {hero_seat} 号座位")
    request_options.setdefault("dump_rounds", 3)
    setup = flop_ranges(hand, tables)
    if hero_seat not in (setup.oop_seat, setup.ip_seat):
        raise NotCovered(f"{hero_seat} 号座位没看到翻牌，没什么可复盘的")

    hole = hand.hole[hero_seat]
    if len(hole) != 2:
        raise NotCovered(f"{hero_seat} 号座位没有底牌")
    hero_cards = (hole[0], hole[1])
    hero = setup.player_index(hero_seat)

    steps, sizes = _postflop_steps(hand, bet_sizes)
    points = []
    prefix: list[Step] = []
    for step in steps:
        if not step.is_deal and step.seat == hero_seat:
            points.append(
                DecisionPoint(
                    street=step.street,
                    seat=hero_seat,
                    hero=hero,
                    hero_cards=hero_cards,
                    prefix=tuple(prefix),
                    taken=step,
                )
            )
        prefix.append(step)

    board = tuple(hand.board[:3])
    request = SolveRequest(
        board=board,
        pot=setup.pot,
        effective_stack=setup.effective_stack,
        oop_range=setup.oop,
        ip_range=setup.ip,
        bet_sizes=sizes,
        **request_options,
    )
    return ReviewPlan(
        request=request, setup=setup, hero_seat=hero_seat, points=tuple(points)
    )


def _postflop_steps(hand: HandState, sizes: BetSizes) -> "tuple[list[Step], BetSizes]":
    """把翻后的事件流翻成求解器口径的一串步子，顺带把实战尺度并进树里。

    尺度按「下注额占**行动前底池**的百分比」算——求解器的 `set_bet_sizes` 就是这个口径。
    """
    big_blind = hand.config.big_blind
    steps: list[Step] = []
    pot = 0
    street = PREFLOP

    for event in hand.events:
        kind = event.kind
        if kind in ("ante", "blind"):
            pot += event.amount
            continue
        if kind == "refund":
            pot -= event.amount
            continue
        if kind == "deal_board":
            street = event.street
            if street > FLOP:
                for card in event.cards:
                    steps.append(Step(kind="deal", card=card, street=street))
            continue
        if kind not in ("fold", "check", "call", "bet", "raise"):
            continue

        if street >= FLOP:
            if kind in ("bet", "raise"):
                percent = 100.0 * event.amount / pot if pot else 0.0
                if kind == "bet":
                    sizes = sizes.with_size(_STREET_NAMES[street], percent)
                steps.append(
                    Step(
                        kind=kind,
                        seat=event.seat,
                        amount=event.to / big_blind,
                        street=street,
                    )
                )
            else:
                steps.append(Step(kind=kind, seat=event.seat, street=street))

        if kind in ("call", "bet", "raise"):
            pot += event.amount

    return steps, sizes


# ------------------------------------------------------------------ 打分（纯逻辑）


def score_plan(plan: ReviewPlan, root: SolvedNode) -> ReviewResult:
    """在解出来的树上给每个决策点打分。**单个点打不了分不会让整手牌作废**。"""
    spot = plan.spot
    tolerance = plan.request.rounding
    scored = []
    for point in plan.points:
        in_range = plan.setup.range_of(point.seat).weight_of_hand(*point.hero_cards) > 0.0
        try:
            line = resolve_line(root, point.prefix, tolerance=tolerance)
            label = _label_of(root, line, point.taken, tolerance=tolerance)
        except LineNotInTree as exc:
            scored.append(ScoredDecision(point=point, skipped=str(exc), in_range=in_range))
            continue
        try:
            score = score_decision(
                spot,
                root,
                line=line,
                hero=point.hero,
                hero_cards=point.hero_cards,
                taken=label,
            )
        except (KeyError, ValueError) as exc:
            scored.append(
                ScoredDecision(point=point, skipped=f"算不了 EV：{exc}", in_range=in_range)
            )
            continue
        scored.append(
            ScoredDecision(point=point, score=score, label=label, in_range=in_range)
        )
    return ReviewResult(plan=plan, decisions=tuple(scored))


def resolve_line(
    root: SolvedNode, prefix: "tuple[Step, ...]", *, tolerance: float = 0.05
) -> tuple[str, ...]:
    """把实战走过的每一步翻成树里的标签。走不通就抛 `LineNotInTree`，并说清卡在哪。

    `tolerance` 是金额对得上算「同一个尺度」的宽容度（大盲）。默认给的是求解器把下注额
    取整之后的最大偏差（`SolveRequest.rounding`）——**不是用来把对不上的尺度圆过去的**。
    """
    labels: list[str] = []
    node = root
    for step in prefix:
        label = _match(node, step, tolerance)
        child = node.children.get(label)
        if child is None:
            raise LineNotInTree(f"树里「{label}」之后没有子节点（dump 层数不够？）")
        labels.append(label)
        node = child
    return tuple(labels)


def _label_of(
    root: SolvedNode, line: "tuple[str, ...]", step: Step, tolerance: float = 0.05
) -> str:
    """走到 `line` 尽头那个节点，把 `step` 翻成那里的标签。"""
    node = root
    for label in line:
        node = node.children.get(label)
        if node is None:
            raise LineNotInTree(f"树里「{label}」之后没有子节点（dump 层数不够？）")
    return _match(node, step, tolerance)


def _match(node: SolvedNode, step: Step, tolerance: float = 0.05) -> str:
    """一步 → 这个节点上的标签。"""
    if node.kind == "chance":
        if not step.is_deal:
            raise LineNotInTree("轮到发牌，实战这一步却是个动作")
        label = card_to_str(step.card)
        if label not in node.children:
            raise LineNotInTree(f"发牌节点上没有 {label}")
        return label
    if step.is_deal:
        raise LineNotInTree("实战这一步是发牌，树里却轮到有人说话")

    index = node.action_index(step.kind, step.amount)
    if index is None:
        available = "、".join(a.label for a in node.actions) or "（没有动作）"
        raise LineNotInTree(f"树里没有这一步（{step.kind} {step.amount}）；有的是：{available}")
    action = node.actions[index]
    if action.amount is not None and step.amount is not None:
        if abs(action.amount - step.amount) > tolerance + 1e-9:
            raise LineNotInTree(
                f"实战打的是 {step.amount:.2f}bb，树里最近的只有 {action.amount:.2f}bb"
                "——这个尺度不在树里，没法给它打分"
            )
    return action.label

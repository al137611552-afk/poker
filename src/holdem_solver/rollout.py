"""把上一街的解滚成下一街的范围——逐街分别解（方案 B / ADR-0005）的接缝。

整棵翻牌树导满三层是 GB 量级（pilot 实测 822.8MB），所以复盘不再解一棵大树，而是
**每街一棵 `dump_rounds=1` 的小树**：翻牌解完，沿实战走过的那条线把双方范围往前滚，
滚出来的范围当作转牌那棵树的输入，再解一次。这个模块只负责「滚」这一步，纯逻辑。

```python
rolled = roll_forward(flop_root, prefix, request=flop_request)
turn_request = rolled.to_request(flop_request)   # 底池、有效筹码、公共牌都推进好了
```

## 为什么会丢东西——先把话说在前面

求解器解出来的策略是**逐组合**的（`AhKh` 与 `AsKc` 在两张红桃的翻牌上是两回事），
但它的范围输入**只认 169 个牌类**：喂 `set_range_oop AhKs` 会抛
`range str AhKs len not valid` 然后 SIGABRT（实测，见 ADR-0005）。于是滚回去的时候
只能按牌类聚合权重——**同一牌类里的同花听牌被抹平**（`AKs` 四个组合只有一个是听牌）。

这是方案 B 的保真度上限，不是实现细节，所以这里**不把它藏起来**：聚合的同时算出每个
牌类的「类内落差」（`spread`），落差大的牌类由 `flagged()` 点名——那正是聚合伤得最重的
地方，打分报告拿它去标低置信度（PRD 的 A/B/C 分级）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from holdem.cards import card_to_str
from holdem.ranges import Range, class_combos, class_of

from .request import SolveRequest
from .result import IP, OOP, SolvedNode
from .review import LineNotInTree, Step, _match as _match_step

# 类内落差超过这个值就点名：一个牌类里最强的组合与最弱的组合，滚完之后差了三分之一以上，
# 说明求解器在**花色**上做了实质区分，而这正是聚合传不过去的东西。
SPREAD_THRESHOLD = 0.35

_EPSILON = 1e-4


def _key(card_a: int, card_b: int) -> str:
    return card_to_str(card_a) + card_to_str(card_b)


@dataclass(frozen=True)
class RolledRange:
    """滚到下一街的范围：既有逐组合的真相，也有喂得回求解器的那份近似。"""

    combos: Mapping[str, float]
    """具体组合 → 滚出来的权重（`输入权重 × 沿途各步的动作频率`），**没有归一化**。"""

    hand_range: Range
    """按牌类聚合并归一化后的范围——**喂给求解器的就是这一份**，同花信息已经没了。"""

    spread: Mapping[int, float]
    """牌类 → 类内落差 `(最大−最小)/最大`。0 = 类里各组合一致，聚合没损失。"""

    def flagged(self, threshold: float = SPREAD_THRESHOLD) -> tuple[int, ...]:
        """聚合伤得最重的牌类（还留在范围里、且类内落差超过门槛的）。"""
        return tuple(
            index
            for index in sorted(self.spread)
            if self.spread[index] >= threshold and self.hand_range.weight(index) > 0.0
        )

    @property
    def total_weight(self) -> float:
        return sum(self.combos.values())


@dataclass(frozen=True)
class Rollout:
    """一条线滚完之后，下一街开局的样子。"""

    oop: RolledRange
    ip: RolledRange
    board: tuple[int, ...]
    pot: float
    effective_stack: float

    def player(self, seat: int) -> RolledRange:
        return self.oop if seat == OOP else self.ip

    def to_request(self, previous: SolveRequest, **overrides) -> SolveRequest:
        """按滚出来的范围与推进后的底池，造下一街那棵树的请求。

        下注尺度、精度、`dump_rounds` 这些跑法参数默认沿用上一街——要改就用 `overrides`。
        """
        fields = dict(
            board=self.board,
            oop_range=self.oop.hand_range,
            ip_range=self.ip.hand_range,
            pot=self.pot,
            effective_stack=self.effective_stack,
        )
        fields.update(overrides)
        return replace(previous, **fields)


def roll_forward(
    root: SolvedNode,
    prefix: Sequence[Step],
    *,
    request: SolveRequest,
    tolerance: float | None = None,
) -> Rollout:
    """沿实战走过的 `prefix` 把双方范围滚到下一街。

    `prefix` 与 FR-9 打分用的是同一种步骤序列（含发牌那一步），匹配动作也走
    `review` 里那套口径——**滚范围与打分必须对同一条线有同一个理解**，否则报告里
    「这一步亏了多少」和「下一街从什么范围开始」说的就不是同一手牌了。

    这条线上有人弃牌就没有下一街，抛 `LineNotInTree`。
    """
    if tolerance is None:
        tolerance = request.rounding

    board = list(request.board)
    dead = set(board)
    weights = {
        OOP: _expand(request.oop_range, dead),
        IP: _expand(request.ip_range, dead),
    }
    pot = request.pot
    stack = request.effective_stack
    invested = {OOP: 0.0, IP: 0.0}

    node = root
    for step in prefix:
        label = _match_step(node, step, tolerance)
        if node.kind == "chance":
            # 街的边界：本街投进去的钱进底池，双方筹码同步扣减（能走到这儿说明跟平了）
            pot += invested[OOP] + invested[IP]
            stack -= max(invested[OOP], invested[IP])
            invested = {OOP: 0.0, IP: 0.0}
            card = step.card
            board.append(card)
            dead.add(card)
            weights = {seat: _drop(combos, card) for seat, combos in weights.items()}
        else:
            player = node.player
            if player is None:
                raise LineNotInTree("动作节点上没写是谁在说话，解读不了")
            index = _action_index(node, label)
            weights[player] = _apply(weights[player], node, index)
            _record_bet(invested, player, node.actions[index])

        child = node.children.get(label)
        if child is None:
            raise LineNotInTree(f"树里「{label}」之后没有子节点（dump 层数不够？）")
        node = child

    return Rollout(
        oop=_aggregate(weights[OOP]),
        ip=_aggregate(weights[IP]),
        board=tuple(board),
        pot=pot,
        effective_stack=stack,
    )


# ------------------------------------------------------------------ 内部


def _expand(hand_range: Range, dead: "set[int]") -> dict[str, float]:
    """牌类权重 → 逐组合权重。牌面上已经有的牌挡掉的组合直接不要。"""
    combos: dict[str, float] = {}
    for index, weight in hand_range.weights.items():
        for card_a, card_b in class_combos(index):
            if card_a in dead or card_b in dead:
                continue
            combos[_key(card_a, card_b)] = weight
    if not combos:
        raise ValueError("范围里的组合全被公共牌挡掉了")
    return combos


def _drop(combos: Mapping[str, float], card: int) -> dict[str, float]:
    """发出来的这张牌，谁手里有就删掉谁——牌只有一张。"""
    text = card_to_str(card)
    return {key: value for key, value in combos.items() if text not in (key[:2], key[2:4])}


def _action_index(node: SolvedNode, label: str) -> int:
    for index, action in enumerate(node.actions):
        if action.label == label:
            return index
    raise LineNotInTree(f"节点上没有「{label}」这个动作")


def _apply(combos: Mapping[str, float], node: SolvedNode, index: int) -> dict[str, float]:
    """乘上这一步的动作频率：走这条线的概率，逐组合各不相同。

    解里查不到的组合直接删掉（求解器压根没让它走到这儿）；**滚成 0 的组合要留着**，
    留着才看得出「这个牌类里有一支被单独摘出去了」——那正是聚合损失最大的地方。
    """
    rolled: dict[str, float] = {}
    for key, weight in combos.items():
        frequency = node.strategy.get(key) or node.strategy.get(key[2:4] + key[:2])
        if frequency is None:
            continue
        rolled[key] = weight * frequency[index]
    return rolled


def _record_bet(invested: dict, player: int, action) -> None:
    """记本街谁投到了多少（求解器给的金额就是「本街投到多少」）。"""
    if action.kind == "fold":
        raise LineNotInTree("这条线上有人弃牌，牌局到此为止，没有下一街可滚")
    if action.kind in ("bet", "raise"):
        if action.amount is None:
            raise LineNotInTree(f"「{action.label}」没有金额，推不出底池")
        invested[player] = action.amount
    elif action.kind == "call":
        invested[player] = max(invested.values())


def _aggregate(combos: Mapping[str, float]) -> RolledRange:
    """逐组合 → 169 牌类（求解器只认这个），同时把聚合伤到的地方记下来。"""
    buckets: dict[int, list[float]] = {}
    for key, weight in combos.items():
        index = class_of(*_cards(key))
        buckets.setdefault(index, []).append(weight)

    # 均值要把滚成 0 的组合算进来：一个牌类四个组合里死了一个，这个类就该轻四分之一
    means = {index: sum(values) / len(values) for index, values in buckets.items()}
    top = max(means.values(), default=0.0)
    if top <= 0.0:
        raise ValueError("滚出来的范围是空的——这条线在解里走不通")

    # 归一化成「最重的牌类＝1」：求解器看的是相对权重，这样写出来的范围最好读
    weights = {
        index: round(value / top, 4)
        for index, value in means.items()
        if value / top > _EPSILON
    }
    spread = {
        index: (max(values) - min(values)) / max(values) if max(values) > 0.0 else 0.0
        for index, values in buckets.items()
        if len(values) > 1
    }
    return RolledRange(combos=dict(combos), hand_range=Range(weights), spread=spread)


def _cards(key: str) -> "tuple[int, int]":
    from holdem.cards import card_from_str

    return card_from_str(key[:2]), card_from_str(key[2:4])

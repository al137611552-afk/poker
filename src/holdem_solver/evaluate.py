"""在解出来的树上算 EV。**纯逻辑，不碰进程也不碰磁盘。**

求解器的产物里**没有 EV**（只有策略，见 `result` 模块说明），可是 FR-9 要回答的正是
「你这个决策亏了多少大盲」。所以这一层自己算：**固定英雄的两张牌**，让对手按解出来的
策略走，在树上把期望积出来。

```python
score = score_decision(spot, root, line=("CHECK", "BET 3.000000"), hero=1, hero_cards=(qh, qd))
score.evs          # {"CALL": 1.83, "FOLD": -0.0, "RAISE 12.0": -0.44}
score.loss("FOLD") # 弃牌比最好的那条路差多少（大盲）
```

## 口径

EV = **最终底池 × 我方份额 − 我方从这个局面起投入的钱**。所以：

- 根节点已经在底池里的钱算「赢得到的」，不算我方投入——这与主流求解器一致，
  也是「这个局面对我值多少」的自然读法；
- **弃牌的 EV 恒等于「−已经投进去的钱」**，这是个可以逐笔对账的硬约束（有测试守着）；
- 单位跟着 `SolveRequest` 走，也就是**大盲**。

## 为什么固定英雄的牌

复盘时英雄的两张牌是已知的，没必要对他的整个范围积分——固定之后每个摊牌终局只要把
英雄这一手与对手范围里的每一手比一次（O(n)），而不是范围对范围（O(n²)）。
再加上**按最终牌面缓存比牌结果**（同一条发牌分支下所有终局共用一份），转牌局面几万次
求值就够，河牌局面几千次。翻牌局面要枚举 45×44 种跑马，是几十倍的量——能算，但别在
交互里等它。

## 三条会算错的地方

1. **共牌**：对手不可能拿着英雄手里或牌面上的牌。每进一张新牌都要把冲突的组合清零，
   否则期望里会混进不存在的手牌。
2. **发牌节点的 52 张里有牌面上已经有的**（求解器就是这么导的），必须按牌面过滤，
   而且概率要按**实际可发的张数**归一化，不是 52。
3. **金额是「本街投到多少」的总额**，不是这次加了多少（与我们引擎的 raise-to 口径一致）。
   算投入时要减掉自己本街已经投过的部分。

## 「发牌节点下面是空的」有两种，别混

- **双方已经全下**：后面没有任何决策，求解器本来就不会往下导。这时**我们自己把剩下的
  公共牌枚举完再摊牌**——从转牌起 45 张、从翻牌起 45×44/2 种，是正常且必须算的。
- **dump 层数不够**（`set_dump_rounds` 给小了）：后面明明还有决策，只是没导出来。
  这时**必须报错**，不能拿半棵树糊弄出一个数。

**怎么区分**：光看「双方投入 == 有效筹码」不够——求解器的 `allin_threshold` 会把
「加注到 27、身后只剩 3」也当成没有后续决策，那时投入并没到顶。`deal_number` 也不行
（两种情况都是 0，实测）。所以再加一条**从树本身取证**的判据：**同一层要是有别的发牌
节点展开了**，说明 dump 覆盖到了这一层，那这个空的就是全下跑马；**整层都没展开**才是截断。
"""

from __future__ import annotations

from dataclasses import dataclass

from holdem.cards import card_from_str, card_to_str
from holdem.evaluator import evaluate
from holdem.ranges import Range, class_combos

from .request import SolveRequest
from .result import SolvedNode

__all__ = ["Spot", "DecisionScore", "evaluate_actions", "score_decision", "hand_ev"]


@dataclass(frozen=True)
class Spot:
    """求解的那个局面的静态信息。从 `SolveRequest` 来，也可以手工构造。"""

    board: tuple[int, ...]
    pot: float
    effective_stack: float
    oop_range: Range
    ip_range: Range

    @classmethod
    def from_request(cls, request: SolveRequest) -> "Spot":
        return cls(
            board=tuple(request.board),
            pot=request.pot,
            effective_stack=request.effective_stack,
            oop_range=request.oop_range,
            ip_range=request.ip_range,
        )

    def range_of(self, player: int) -> Range:
        return self.oop_range if player == 0 else self.ip_range


@dataclass(frozen=True)
class DecisionScore:
    """一个决策点上的打分。"""

    evs: dict[str, float]
    """动作标签 → EV（大盲）。"""
    strategy: dict[str, float]
    """解在这个决策点上、对这手牌给出的动作频率。"""
    taken: str | None
    """实际打出的动作；只是问「各条路值多少」时是 `None`。"""

    @property
    def best(self) -> str:
        return max(self.evs, key=self.evs.get)

    def loss(self, label: str | None = None) -> float:
        """某个动作比最好的那条路差多少（大盲）。0 表示这就是最优解之一。"""
        choice = label or self.taken
        if choice is None:
            raise ValueError("没给动作，也不知道实际打的是哪个")
        if choice not in self.evs:
            raise KeyError(f"这个决策点上没有「{choice}」；有的是：{'、'.join(self.evs)}")
        return self.evs[self.best] - self.evs[choice]

    @property
    def solved_ev(self) -> float:
        """照解的混合策略打，这手牌值多少。"""
        return sum(self.evs[label] * weight for label, weight in self.strategy.items())

    @property
    def gap(self) -> float:
        """解自己离最优差多少。收敛好的解上这个数应该很小——它是我们这套 EV 算法
        与求解器**互相印证**的地方：算错了，这个差会明显鼓起来。"""
        return self.evs[self.best] - self.solved_ev


# ------------------------------------------------------------------ 主入口


def score_decision(
    spot: Spot,
    root: SolvedNode,
    *,
    line: "tuple[str, ...]",
    hero: int,
    hero_cards: "tuple[int, int]",
    taken: str | None = None,
) -> DecisionScore:
    """沿着实际打出的动作序列走到某个决策点，算出那里各条路的 EV。

    `line` 是从根节点起、**双方**依次打出的动作标签（发牌节点用牌，比如 `"7d"`）。
    走到的那个节点必须轮到 `hero` 说话。
    """
    state = _State.at_root(spot, hero, hero_cards, root)
    node = root
    for step in line:
        node, state = state.advance(node, step)
    if node.kind != "action":
        raise ValueError("走到的不是决策点")
    if node.player != hero:
        raise ValueError(f"这个决策点轮到玩家 {node.player} 说话，不是 {hero}")

    evs = evaluate_actions(node, state)
    weights = node.for_combo(*hero_cards)
    strategy = (
        {action.label: weights[index] for index, action in enumerate(node.actions)}
        if weights
        else {}
    )
    return DecisionScore(evs=evs, strategy=strategy, taken=taken)


def evaluate_actions(node: SolvedNode, state: "_State") -> dict[str, float]:
    """英雄在这个决策点上，每个动作各值多少（大盲）。"""
    if node.player != state.hero:
        raise ValueError("这个决策点不是英雄的")
    evs = {}
    for index, action in enumerate(node.actions):
        after = state.take(node, index)
        child = node.children.get(action.label)
        evs[action.label] = after.value(child, action, node.player)
    return evs


def hand_ev(spot: Spot, root: SolvedNode, hero: int, hero_cards: "tuple[int, int]") -> float:
    """整个局面对英雄这手牌值多少（双方都照解走）。"""
    state = _State.at_root(spot, hero, hero_cards, root)
    return state.value(root, None)


# ------------------------------------------------------------------ 遍历


class _State:
    """遍历时随身携带的账：底池、双方本街投入、英雄总投入、对手范围的到达概率。"""

    __slots__ = (
        "spot", "hero", "hero_cards", "board", "pot", "street", "committed",
        "reach", "combos", "_showdown", "chance_depth", "expanded",
    )

    def __init__(
        self, spot, hero, hero_cards, board, pot, street, committed, reach, combos,
        showdown, chance_depth=0, expanded=frozenset(),
    ):
        self.spot = spot
        self.hero = hero
        self.hero_cards = hero_cards
        self.board = board
        self.pot = pot
        self.street = street
        self.committed = committed
        self.reach = reach
        self.combos = combos
        self._showdown = showdown
        self.chance_depth = chance_depth
        """走到这里经过了几个发牌节点。"""
        self.expanded = expanded
        """哪几层发牌节点在 dump 里真的展开了——用来区分「全下跑马」与「dump 截断」。"""

    # ---------------------------------------------------------- 构造

    @classmethod
    def at_root(
        cls, spot: Spot, hero: int, hero_cards: "tuple[int, int]", root: "SolvedNode | None" = None
    ) -> "_State":
        if len(set(hero_cards) | set(spot.board)) != len(hero_cards) + len(spot.board):
            raise ValueError("英雄的牌与公共牌撞了")
        villain = spot.range_of(1 - hero)
        combos, reach = _expand(villain, blocked=set(spot.board) | set(hero_cards))
        if not combos:
            raise ValueError("对手范围在这个牌面上一手都不剩")
        return cls(
            spot=spot,
            hero=hero,
            hero_cards=tuple(hero_cards),
            board=tuple(spot.board),
            pot=spot.pot,
            street=(0.0, 0.0),
            committed=(0.0, 0.0),
            reach=reach,
            combos=combos,
            showdown={},
            expanded=_expanded_depths(root) if root is not None else frozenset(),
        )

    def _clone(self, **changes) -> "_State":
        values = dict(
            spot=self.spot,
            hero=self.hero,
            hero_cards=self.hero_cards,
            board=self.board,
            pot=self.pot,
            street=self.street,
            committed=self.committed,
            reach=self.reach,
            combos=self.combos,
            showdown=self._showdown,
            chance_depth=self.chance_depth,
            expanded=self.expanded,
        )
        values.update(changes)
        return _State(**values)

    # ---------------------------------------------------------- 走一步

    def advance(self, node: SolvedNode, step: str) -> "tuple[SolvedNode, _State]":
        """按实际发生的一步（动作标签或一张牌）往下走。"""
        if node.kind == "chance":
            child = node.children.get(step)
            if child is None:
                raise KeyError(f"这个发牌节点上没有 {step}")
            return child, self.deal(card_from_str(step))
        index = next(
            (i for i, action in enumerate(node.actions) if action.label == step), None
        )
        if index is None:
            raise KeyError(f"这个决策点上没有「{step}」；有的是：{'、'.join(a.label for a in node.actions)}")
        state = self.take(node, index)
        child = node.children.get(step)
        if child is None:
            raise ValueError(f"「{step}」之后牌局就结束了，走不下去")
        return child, state

    def take(self, node: SolvedNode, index: int) -> "_State":
        """某人打出一个动作之后的账。对手打的动作还要把他的范围按策略收窄。"""
        action = node.actions[index]
        actor = node.player
        street = list(self.street)
        committed = list(self.committed)
        pot = self.pot

        if action.kind in ("bet", "raise"):
            target = action.amount or 0.0
            added = target - street[actor]
            street[actor] = target
        elif action.kind == "call":
            added = max(0.0, street[1 - actor] - street[actor])
            street[actor] = street[1 - actor]
        else:  # check / fold
            added = 0.0
        pot += added
        committed[actor] += added

        reach = self.reach
        if actor != self.hero:
            reach = self._filtered(node, index)

        return self._clone(
            pot=pot, street=tuple(street), committed=tuple(committed), reach=reach
        )

    @property
    def invested(self) -> float:
        """英雄从这个局面起投进去的钱。"""
        return self.committed[self.hero]

    @property
    def both_all_in(self) -> bool:
        """双方都推光了——后面只剩发牌，求解器不会再往下导，得我们自己跑马。"""
        stack = self.spot.effective_stack
        return all(value >= stack - 1e-9 for value in self.committed)

    def _filtered(self, node: SolvedNode, index: int) -> list[float]:
        """对手做了这个动作之后，他的范围还剩多少——按解出来的频率乘上去。"""
        scaled = []
        for position, combo in enumerate(self.combos):
            weight = self.reach[position]
            if weight <= 0.0:
                scaled.append(0.0)
                continue
            strategy = node.for_combo(*combo)
            scaled.append(weight * strategy[index] if strategy else 0.0)
        return scaled

    def deal(self, card: int) -> "_State":
        """发一张公共牌：底池结算到一起，本街投入清零，共牌再清一遍。"""
        pot = self.pot
        reach = [
            0.0 if (card in combo) else weight
            for weight, combo in zip(self.reach, self.combos)
        ]
        return self._clone(
            board=self.board + (card,),
            pot=pot,
            street=(0.0, 0.0),
            reach=reach,
        )

    def _runout_value(self) -> float:
        """双方全下之后自己把剩下的公共牌发完，再摊牌。

        跑马按均匀分布取平均，每条跑马内部再按「对手范围里还剩哪些手」归一化
        ——这与求解器的口径一致。
        """
        from itertools import combinations

        need = 5 - len(self.board)
        if need <= 0:
            return self._showdown_value()
        seen = set(self.board) | set(self.hero_cards)
        deck = [card for card in range(52) if card not in seen]
        total = 0.0
        count = 0
        for extra in combinations(deck, need):
            state = self
            for card in extra:
                state = state.deal(card)
            total += state._showdown_value()
            count += 1
        return total / count

    # ---------------------------------------------------------- 求值

    def value(self, node: "SolvedNode | None", action=None, actor: int | None = None) -> float:
        """从这里往下，英雄这手牌的期望（大盲）。

        `node` 是 `None` 表示这一步之后就没有节点了——那就是终局：弃牌，或者
        跟注/过牌把这条街关上之后直接摊牌（河牌上就是这样）。`actor` 是刚说完话的人。
        """
        if node is None:
            if action is not None and action.kind == "fold":
                # 谁弃的？刚打出这个动作的人。英雄弃牌就亏掉已投入，对手弃牌英雄赢下底池
                return -self.invested if actor == self.hero else self.pot - self.invested
            # 跟注/过牌把这条街关上了：牌面发完就摊牌，没发完（全下跑马）就自己发完
            return self._showdown_value() if len(self.board) == 5 else self._runout_value()
        if node.kind == "chance":
            return self._chance_value(node)
        if node.is_placeholder:
            raise ValueError("走到了空占位符节点（多半是牌面上已有的牌），这条路不该走")
        return self._action_value(node)

    def _action_value(self, node: SolvedNode) -> float:
        if node.player == self.hero:
            weights = node.for_combo(*self.hero_cards)
            if weights is None:
                raise ValueError("英雄这手牌走不到这个节点，解里没有它的策略")
            total = 0.0
            for index, action in enumerate(node.actions):
                if weights[index] <= 0.0:
                    continue
                after = self.take(node, index)
                child = node.children.get(action.label)
                total += weights[index] * after.value(child, action, node.player)
            return total

        # 对手的节点：按他打出各动作的概率（＝范围质量的比例）加权
        total = 0.0
        mass = _mass(self.reach)
        for index, action in enumerate(node.actions):
            after = self.take(node, index)
            share = _mass(after.reach)
            if share <= 0.0:
                continue
            child = node.children.get(action.label)
            total += after.value(child, action, node.player) * share / mass
        return total

    def _chance_value(self, node: SolvedNode) -> float:
        """发一张牌：在**真正能发出来的**牌上取平均。"""
        seen = set(self.board) | set(self.hero_cards)
        usable = [
            (card_from_str(card), child)
            for card, child in node.children.items()
            if card_from_str(card) not in seen and not child.is_placeholder
        ]
        if not usable:
            if self.both_all_in or self.chance_depth in self.expanded:
                # 双方全下，后面没有决策可导——这不是截断，是该我们自己跑马
                return self._runout_value()
            # 树被 dump 截断了（`set_dump_rounds` 太小），后面的策略根本没导出来
            raise ValueError(
                "发牌节点下面是空的：dump 的层数不够，这个局面算不了跨街 EV"
            )
        total = 0.0
        for card, child in usable:
            after = self.deal(card)._clone(chance_depth=self.chance_depth + 1)
            total += after.value(child)
        return total / len(usable)

    def _showdown_value(self) -> float:
        """摊牌：英雄这手牌与对手范围逐手比大小，按到达概率加权。"""
        if len(self.board) < 5:
            raise ValueError(
                f"牌面只有 {len(self.board)} 张就摊牌了——该走跑马那条路，不该直接比大小"
            )
        shares = self._showdown_shares()
        mass = 0.0
        won = 0.0
        for weight, share in zip(self.reach, shares):
            if weight <= 0.0:
                continue
            mass += weight
            won += weight * share
        if mass <= 0.0:
            return self.pot - self.invested
        return self.pot * (won / mass) - self.invested

    def _showdown_shares(self) -> list[float]:
        """英雄对每个对手组合能拿到的底池份额（1 / 0.5 / 0）。

        **按最终牌面缓存**：同一条跑马下所有终局共用同一份比牌结果，
        不然每个终局都要把整个范围重新求值一遍。
        """
        key = self.board
        cached = self._showdown.get(key)
        if cached is not None:
            return cached
        hero_score = evaluate(list(self.hero_cards) + list(self.board))
        shares = []
        for combo in self.combos:
            if combo[0] in self.board or combo[1] in self.board:
                shares.append(0.0)
                continue
            villain_score = evaluate(list(combo) + list(self.board))
            shares.append(1.0 if hero_score > villain_score else (0.5 if hero_score == villain_score else 0.0))
        self._showdown[key] = shares
        return shares


def _expanded_depths(root: SolvedNode) -> frozenset:
    """哪几层发牌节点在 dump 里真的展开了（0 = 第一个发牌节点那一层）。

    有一个展开了就算这一层覆盖到了：同一层的其他空发牌节点必然是「全下之后没有决策」，
    而不是被 dump 截断的。
    """
    found: set[int] = set()

    def walk(node: SolvedNode, depth: int) -> None:
        if node.kind == "chance":
            children = [c for c in node.children.values() if not c.is_placeholder]
            if children:
                found.add(depth)
            for child in children:
                walk(child, depth + 1)
            return
        for child in node.children.values():
            walk(child, depth)

    walk(root, 0)
    return frozenset(found)


def _mass(reach: "list[float]") -> float:
    return sum(reach)


def _expand(hand_range: Range, blocked: "set[int]") -> "tuple[list[tuple[int, int]], list[float]]":
    """范围 → 具体组合与权重，撞牌的直接扔掉。"""
    combos: list[tuple[int, int]] = []
    weights: list[float] = []
    for index, weight in sorted(hand_range.weights.items()):
        if weight <= 0.0:
            continue
        for combo in class_combos(index):
            if combo[0] in blocked or combo[1] in blocked:
                continue
            combos.append(combo)
            weights.append(weight)
    return combos, weights


def combo_label(cards: "tuple[int, int]") -> str:
    return card_to_str(cards[0]) + card_to_str(cards[1])

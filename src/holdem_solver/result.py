"""解析 TexasSolver 的策略树。**纯逻辑，不碰进程也不碰磁盘。**

## dump 里有什么、没有什么

有：每个决策点的**动作列表**与**逐具体组合的策略**（`AsKh → [0.72, 0.28]`）、发牌节点的
牌数。**没有 EV**——一个 `ev` 字段都没有（官方样例里也没有）。

这条决定了 FR-9 的形状：**「你这个动作亏了多少」没法从 dump 里读出来，得我们自己在解出来
的树上算**（终局收益 + 到达概率，与 `preflop_solver` 同一套办法，只是终局换成真实摊牌）。
拿「你打了均衡里频率很低的动作」冒充 EV 损失是不诚实的，PRD 的「诚实」那条不允许。

## 两个会把人坑到的口径

1. **求解器管 OOP 叫 player 1、IP 叫 player 0**（实测：给 IP 只放 AA、OOP 放小对子，
   根节点 `player=1` 里出现的是小对子）。这里一律翻译成**我们的口径：0 = OOP、1 = IP**，
   `SolvedNode.player` 拿到的已经是我们的编号。取错一侧的解看着仍然「像那么回事」，
   所以这里有测试守着。
2. **组合键的两张牌没有固定顺序**（`AsKh` 还是 `KhAs` 都可能）。查自己的牌时两种都试，
   别只试一种——差别是「查不到就退回兜底」，同样不报错。

## 发牌节点的两个坑

- **它的子节点挂在 `dealcards` 里，不是 `childrens`**（而且没有下划线）。读错了会看到一个
  「没有子节点的发牌节点」，进而误以为「跨街的策略导不出来」——实测导得出来：
  一个转牌局面 `dump_rounds=2`，发牌之后有 1920 个带策略的动作节点。
- **`dealcards` 里有 52 张牌，包括已经在牌面上的**。那些牌的子树是空占位符
  （动作节点但没有动作也没有策略）。**必须按牌面过滤**，否则会把不可能的转牌算进期望。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from holdem.cards import card_to_str
from holdem.ranges import NUM_HAND_CLASSES, class_of

__all__ = ["SolvedAction", "SolvedNode", "parse_result", "parse_action"]

OOP, IP = 0, 1
"""我们的口径。求解器自己的编号正好相反，读进来时就翻译掉。"""

_AMOUNT = re.compile(r"^([A-Z]+)(?:\s+([0-9.]+))?$")


@dataclass(frozen=True)
class SolvedAction:
    """解里的一个动作。`amount` 是「下注到多少」，单位与请求一致（大盲）。"""

    label: str
    kind: str
    """`check` / `call` / `fold` / `bet` / `raise`。"""
    amount: float | None

    @property
    def is_aggressive(self) -> bool:
        return self.kind in ("bet", "raise")


def parse_action(label: str) -> SolvedAction:
    """`"BET 30.000000"` → `SolvedAction("BET 30.000000", "bet", 30.0)`。"""
    match = _AMOUNT.match(label.strip())
    if not match:
        raise ValueError(f"看不懂的动作标签: {label!r}")
    word, amount = match.group(1), match.group(2)
    kind = word.lower()
    if kind not in ("check", "call", "fold", "bet", "raise"):
        raise ValueError(f"没见过的动作类型: {label!r}")
    return SolvedAction(label=label, kind=kind, amount=float(amount) if amount else None)


@dataclass(frozen=True)
class SolvedNode:
    """策略树上的一个节点。"""

    kind: str
    """`action`（有人要说话）/ `chance`（发牌）。"""
    player: int | None
    """**我们的口径**：0 = OOP、1 = IP。发牌节点是 `None`。"""
    actions: tuple[SolvedAction, ...]
    strategy: dict[str, tuple[float, ...]]
    """具体组合（`AsKh`）→ 各动作的概率，顺序与 `actions` 一致。"""
    children: dict[str, "SolvedNode"]
    """动作节点：动作标签 → 子节点。发牌节点：**牌** → 子节点（来自 `dealcards`）。"""
    deal_number: int | None = None

    @property
    def is_placeholder(self) -> bool:
        """空占位符：`dealcards` 里那些牌面上已经有的牌就长这样，遍历时要跳过。"""
        return self.kind == "action" and not self.actions and not self.children

    # ---------------------------------------------------------- 查询

    def action_index(self, kind: str, amount: float | None = None) -> int | None:
        """按类型（可选按尺度）找动作下标；没有就回 `None`。"""
        best = None
        for index, action in enumerate(self.actions):
            if action.kind != kind:
                continue
            if amount is None or action.amount is None:
                return index
            gap = abs(action.amount - amount)
            if best is None or gap < best[0]:
                best = (gap, index)
        return None if best is None else best[1]

    def for_combo(self, card_a: int, card_b: int) -> tuple[float, ...] | None:
        """这两张具体的牌在这个节点上的策略；不在范围里（走不到这儿）就回 `None`。

        组合键的两张牌顺序不固定，两种都试。
        """
        first, second = card_to_str(card_a), card_to_str(card_b)
        return self.strategy.get(first + second) or self.strategy.get(second + first)

    def for_class(self, index: int) -> tuple[float, ...] | None:
        """一个牌类（169 类之一）在这个节点上的平均策略，按组合数等权平均。

        画 13×13 图要用它；给某一手牌打分请用 `for_combo`——**别拿类平均去替代具体牌**，
        同一类里的不同花色在特定牌面上可以差出天壤（同花听牌与否）。
        """
        if not 0 <= index < NUM_HAND_CLASSES:
            raise ValueError(f"牌类编号越界: {index}")
        total = [0.0] * len(self.actions)
        count = 0
        for combo, weights in self.strategy.items():
            if _combo_class(combo) != index:
                continue
            count += 1
            for position, value in enumerate(weights):
                total[position] += value
        if not count:
            return None
        return tuple(value / count for value in total)

    def child(self, label: str) -> "SolvedNode | None":
        return self.children.get(label)

    def walk(self):
        """深度优先遍历自己与全部子孙，方便统计与断言。"""
        yield self
        for child in self.children.values():
            yield from child.walk()


def _combo_class(combo: str) -> int:
    from holdem.cards import card_from_str

    return class_of(card_from_str(combo[:2]), card_from_str(combo[2:4]))


def parse_result(document: dict, *, scale: float = 1.0) -> SolvedNode:
    """把 `dump_result` 出来的 JSON 转成 `SolvedNode` 树。

    `scale` 是命令文件里用的放大倍数（见 `SolveRequest.scale`）：金额除回去，
    于是树里的 `amount` 跟请求一样是**大盲**。标签**原样保留**——它是子节点的键。
    """
    kind = document.get("node_type", "")
    if kind == "action_node":
        strategy_block = document.get("strategy") or {}
        labels = tuple(strategy_block.get("actions") or ())
        actions = tuple(_scaled(parse_action(label), scale) for label in labels)
        raw = strategy_block.get("strategy") or {}
        strategy = {combo: tuple(float(v) for v in weights) for combo, weights in raw.items()}
        for combo, weights in strategy.items():
            if len(weights) != len(actions):
                raise ValueError(
                    f"{combo} 的策略有 {len(weights)} 个数，动作却有 {len(actions)} 个"
                )
        return SolvedNode(
            kind="action",
            player=_our_player(document.get("player")),
            actions=actions,
            strategy=strategy,
            children={
                label: parse_result(child, scale=scale)
                for label, child in (document.get("childrens") or {}).items()
            },
        )
    if kind == "chance_node":
        # 发牌节点的子节点在 dealcards 里（**没有下划线**），键是牌；
        # childrens 在这种节点上永远是空的，读错了会以为跨街策略没导出来
        dealt = document.get("dealcards") or document.get("childrens") or {}
        return SolvedNode(
            kind="chance",
            player=None,
            actions=(),
            strategy={},
            children={
                card: parse_result(child, scale=scale) for card, child in dealt.items()
            },
            deal_number=document.get("deal_number"),
        )
    raise ValueError(f"没见过的节点类型: {kind!r}")


def _scaled(action: SolvedAction, scale: float) -> SolvedAction:
    """金额除回大盲；`scale` 是 1 就原样返回。"""
    if scale == 1.0 or action.amount is None:
        return action
    return replace(action, amount=action.amount / scale)


def _our_player(raw) -> int | None:
    """求解器的 player 编号 → 我们的编号（0 = OOP、1 = IP）。它俩正好相反。"""
    if raw is None:
        return None
    number = int(raw)
    if number not in (0, 1):
        raise ValueError(f"看不懂的玩家编号: {raw!r}")
    return OOP if number == 1 else IP

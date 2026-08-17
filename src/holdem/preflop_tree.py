"""翻前抽象博弈树。

深筹码的翻前范围没法像推/弃那样精确求解：「跟注去看翻牌」值多少钱取决于翻后，而翻后
不可能在翻前算出来。这里的做法是把问题切成两半——**下注序列完整展开成一棵树**（本模块），
**翻后压缩成一个终局收益模型**（`realization.py`），再由 CFR+ 在树上求解
（`preflop_solver.py`）。

树是**公共**的：里面只有下注序列，没有任何牌的信息。私有牌在求解器里以 169 个牌类的
向量出现，一个决策节点对应 169 个信息集。

## 抽象在哪（这些是刻意的简化，不是疏漏）

1. **下注尺度离散化**：开牌加注到 `open_to`（有跛入者每人 +1bb），此后每次再加注按
   `reraise_multiples` 相对上一个加注额放大，梯子用尽只剩全下。
2. **有效筹码相等**：所有人同样深度，于是跟注永远跟得满，树里不必处理边池。
3. **不主动弃掉免费牌**：能过牌时不给「弃牌」这个动作。它永远不优于过牌，留着只会
   让树变大。
4. **前注是死钱**：计入底池与净得失，但不参与「跟到多少」的匹配。

## 为什么不复用 `state.py`

`state.py` 是带牌、带边池、带完整合法性判定的真实状态机；这里要的是一棵只有下注序列的
公共树，尺度还是离散的。硬合成一层会把两边都拖复杂。代价是下注轮的推进逻辑在这里重写了
一遍——所以它有独立的筹码守恒测试兜底（见 `tests/test_preflop_tree.py`）。

## 座位编号

沿用 `positions.py` 的口径：**编号 = 相对按钮的偏移**，0 是按钮，1 是小盲，2 是大盲。
单挑是特例，按钮即小盲、翻前先说话——与 `state.py` 一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .positions import position_names

__all__ = [
    "PreflopConfig",
    "SubgameConfig",
    "TreeAction",
    "DecisionNode",
    "TerminalNode",
    "PreflopTree",
    "build_tree",
]


# ------------------------------------------------------------------ 配置


@dataclass(frozen=True)
class PreflopConfig:
    """一棵翻前树的全部参数。单位一律是大盲。"""

    num_players: int = 2
    effective_stack: float = 100.0
    small_blind: float = 0.5
    big_blind: float = 1.0
    ante: float = 0.0
    """每人前注（死钱）。"""
    open_to: float = 2.5
    """首次加注的目标额；每有一个跛入者再加 1bb。"""
    reraise_multiples: tuple[float, ...] = (3.0, 2.2)
    """3bet、4bet…… 相对上一个加注额的倍数；用尽之后只剩全下。"""
    allow_limp: bool = True
    """关掉之后，无人加注时只能开牌或弃牌，不许只跟一个大盲（大盲本人过牌不受影响）。

    用来裁剪树，也用来构造推/弃这类退化博弈：`open_to` 设成有效筹码即得到「全下或弃牌」。
    """
    jam_from_level: int = 2
    """从第几次加注起，把「全下」也放进动作集（开牌算第 1 次）。"""

    def __post_init__(self) -> None:
        if self.num_players < 2:
            raise ValueError("至少要两个玩家")
        if self.effective_stack <= 0:
            raise ValueError("有效筹码必须为正")
        if self.big_blind <= 0 or self.small_blind < 0:
            raise ValueError("盲注不合法")
        if self.ante < 0:
            raise ValueError("前注不能为负")
        if self.effective_stack < self.big_blind:
            raise ValueError("有效筹码不足一个大盲")

    def position_name(self, player: int) -> str:
        return position_names(self.num_players)[player]

    def setup(self) -> "_Setup":
        n = self.num_players
        posted = [0.0] * n
        sb_seat, bb_seat = (0, 1) if n == 2 else (1, 2)
        posted[sb_seat] = min(self.small_blind, self.effective_stack)
        posted[bb_seat] = min(self.big_blind, self.effective_stack)
        return _Setup(
            players=n,
            posted=tuple(posted),
            ante=self.ante,
            dead_money=0.0,
            # 翻前：单挑按钮先说，三人以上从大盲左手（枪口位）起
            preflop_order=(0, 1) if n == 2 else tuple(range(3, n)) + (0, 1, 2),
            # 翻后：小盲先说、按钮最后（单挑退化成大盲先说）
            postflop_order=tuple(range(1, n)) + (0,),
            names=position_names(n),
            raise_level=0,
            last_raise_to=None,
            already_acted=(False,) * n,
            effective_stack=self.effective_stack,
            big_blind=self.big_blind,
            open_to=self.open_to,
            reraise_multiples=self.reraise_multiples,
            allow_limp=self.allow_limp,
            jam_from_level=self.jam_from_level,
        )


@dataclass(frozen=True)
class SubgameConfig:
    """从整桌里切出来的**两人子博弈**：比如「CO 开牌 vs 大盲防守」。

    六人桌的完整翻前树有六十万个节点，纯 Python 解不动；而绝大多数翻前局面本来就是
    「一个人开牌、一个人应对」。所以六人桌的范围表按位置拆成一串两人子博弈来解，
    中间弃掉的人只留下**死钱**（`dead_money`）。这是 ADR-0003 里那条「按位置分解」的地基。

    位置不再由座位编号推导，而是显式给出：谁翻前先说话、谁翻后最后说话。前者决定行动
    顺序，后者决定终局收益模型里的位置系数——大盲防守时两者是同一个人（先说话、也先
    在翻后说话），开牌者才是有位置的一方。
    """

    posted: tuple[float, float] = (1.0, 0.0)
    """两人已投入的、**参与匹配**的钱（大盲的 1bb 写在这里）。"""
    dead_money: float = 0.5
    """弃牌者留下的无主死钱（小盲、前注）。默认是「都弃到大盲」时小盲留下的 0.5bb。"""
    first_to_act: int = 1
    """翻前先说话的人。默认 1＝开牌者先说，0＝大盲后说。"""
    in_position: int = 1
    """翻后最后说话的人。大盲防守时是开牌者。"""
    names: tuple[str, str] = ("BB", "CO")
    ante: float = 0.0
    """两人各自的前注（死钱，不参与匹配）。其他人的前注算进 `dead_money`。"""
    raise_level: int = 0
    """进入子博弈时已经发生过几次加注。切「面对开牌」的一段时是 1，防守者的加注才算 3bet。"""
    last_raise_to: float | None = None
    """上一个加注的目标额；`None` 表示取已投入里的最大值。"""
    already_acted: tuple[bool, bool] = (False, False)
    """谁已经行动过。切「面对开牌」的一段时开牌者是 True，否则他会被多问一次。

    盲注不算行动——大盲永远从 False 开始，这样跛入之后他才有选择权。
    """
    effective_stack: float = 100.0
    big_blind: float = 1.0
    open_to: float = 2.5
    reraise_multiples: tuple[float, ...] = (3.0, 2.2)
    allow_limp: bool = False
    """子博弈默认不许跛入：开牌者要么开要么弃。"""
    jam_from_level: int = 2

    def __post_init__(self) -> None:
        if self.effective_stack <= 0:
            raise ValueError("有效筹码必须为正")
        if self.big_blind <= 0:
            raise ValueError("盲注不合法")
        if any(value < 0 for value in self.posted):
            raise ValueError("已投入的钱不能为负")
        if self.dead_money < 0:
            raise ValueError("死钱不能为负")
        if self.ante < 0:
            raise ValueError("前注不能为负")
        if self.first_to_act not in (0, 1) or self.in_position not in (0, 1):
            raise ValueError("位置只能是 0 或 1")
        if self.effective_stack <= max(self.posted):
            raise ValueError("有效筹码必须大于已投入的钱")
        if self.raise_level < 0:
            raise ValueError("加注次数不能为负")

    def position_name(self, player: int) -> str:
        return self.names[player]

    def setup(self) -> "_Setup":
        return _Setup(
            players=2,
            posted=(
                min(self.posted[0], self.effective_stack),
                min(self.posted[1], self.effective_stack),
            ),
            ante=self.ante,
            dead_money=self.dead_money,
            preflop_order=(self.first_to_act, 1 - self.first_to_act),
            postflop_order=(1 - self.in_position, self.in_position),
            names=self.names,
            raise_level=self.raise_level,
            last_raise_to=self.last_raise_to,
            already_acted=self.already_acted,
            effective_stack=self.effective_stack,
            big_blind=self.big_blind,
            open_to=self.open_to,
            reraise_multiples=self.reraise_multiples,
            allow_limp=self.allow_limp,
            jam_from_level=self.jam_from_level,
        )


@dataclass(frozen=True)
class _Setup:
    """把两种配置归一成建树真正需要的东西，下注轮逻辑只写一遍。"""

    players: int
    posted: tuple[float, ...]
    ante: float
    dead_money: float
    preflop_order: tuple[int, ...]
    postflop_order: tuple[int, ...]
    names: tuple[str, ...]
    raise_level: int
    last_raise_to: float | None
    already_acted: tuple[bool, ...]
    effective_stack: float
    big_blind: float
    open_to: float
    reraise_multiples: tuple[float, ...]
    allow_limp: bool
    jam_from_level: int


# ------------------------------------------------------------------ 节点


@dataclass(frozen=True)
class TreeAction:
    """一个离散动作。`to_amount` 是「加注到」的目标总额，跟注时即需要跟到的额度。"""

    kind: str
    """`"fold"` / `"call"`（含过牌）/ `"raise"`（含全下）。"""
    to_amount: float
    label: str

    @property
    def is_raise(self) -> bool:
        return self.kind == "raise"


@dataclass
class DecisionNode:
    node_id: int
    player: int
    actions: tuple[TreeAction, ...]
    children: list["DecisionNode | TerminalNode"] = field(default_factory=list)
    raise_level: int = 0
    """进入本节点时已经发生过几次加注（盲注不算）。"""

    @property
    def is_terminal(self) -> bool:
        return False


@dataclass(frozen=True)
class TerminalNode:
    node_id: int
    kind: str
    """`"fold"`（只剩一人）/ `"showdown"`（全下被跟，直接摊牌）/ `"flop"`（进翻牌）。"""
    contributions: tuple[float, ...]
    """每人投入的总额，含盲注与前注。"""
    alive: tuple[int, ...]
    """未弃牌的玩家，按座位编号升序。"""
    in_position: int
    """`alive` 里翻后最后说话的那个人；只剩一人时即那个人。"""
    dead_money: float = 0.0
    """无主的死钱（子博弈里先前弃掉的人留下的），计入底池但不属于任何人。"""

    @property
    def is_terminal(self) -> bool:
        return True

    @property
    def pot(self) -> float:
        return sum(self.contributions) + self.dead_money

    def fold_payoffs(self) -> tuple[float, ...]:
        """只剩一人时每人的净得失（大盲）。"""
        if self.kind != "fold":
            raise ValueError("只有 fold 终局能直接算收益")
        winner = self.alive[0]
        pot = self.pot
        return tuple(
            (pot - contribution) if player == winner else -contribution
            for player, contribution in enumerate(self.contributions)
        )


Node = DecisionNode | TerminalNode


@dataclass
class PreflopTree:
    config: "PreflopConfig | SubgameConfig"
    root: Node
    decisions: tuple[DecisionNode, ...]
    terminals: tuple[TerminalNode, ...]

    @property
    def size(self) -> int:
        return len(self.decisions) + len(self.terminals)


# ------------------------------------------------------------------ 构造


@dataclass(frozen=True)
class _State:
    """下注轮推进所需的全部信息。全部不可变，方便递归时分叉。"""

    bets: tuple[float, ...]
    """参与匹配的投入（盲注算在内，前注不算）。"""
    folded: tuple[bool, ...]
    allin: tuple[bool, ...]
    acted: tuple[bool, ...]
    """自上一次加注以来是否已经行动过。大盲的盲注不算行动，所以它有权选择。"""
    raise_level: int
    last_raise_to: float

    @property
    def current_bet(self) -> float:
        return max(self.bets)


def build_tree(config: "PreflopConfig | SubgameConfig | None" = None) -> PreflopTree:
    """展开一棵翻前树。整桌给 `PreflopConfig`，两人子博弈给 `SubgameConfig`。"""
    cfg = config or PreflopConfig()
    builder = _Builder(cfg.setup())
    root = builder.build(builder.initial_state())
    return PreflopTree(
        config=cfg,
        root=root,
        decisions=tuple(builder.decisions),
        terminals=tuple(builder.terminals),
    )


class _Builder:
    def __init__(self, setup: _Setup) -> None:
        self.config = setup
        self.decisions: list[DecisionNode] = []
        self.terminals: list[TerminalNode] = []
        self._next_id = 0
        self.preflop_order = list(setup.preflop_order)
        self.postflop_order = list(setup.postflop_order)

    # -------------------------------------------------------------- 初始状态

    def initial_state(self) -> _State:
        setup = self.config
        bets = list(setup.posted)
        allin = [bet >= setup.effective_stack for bet in bets]
        last_raise_to = setup.last_raise_to
        if last_raise_to is None:
            last_raise_to = max(setup.big_blind, max(bets))
        return _State(
            bets=tuple(bets),
            folded=(False,) * setup.players,
            allin=tuple(allin),
            acted=tuple(setup.already_acted),
            raise_level=setup.raise_level,
            last_raise_to=last_raise_to,
        )

    # -------------------------------------------------------------- 递归

    def build(self, state: _State) -> Node:
        alive = [p for p in range(self.config.players) if not state.folded[p]]
        if len(alive) == 1:
            return self._terminal(state, "fold", alive)

        actor = self._next_actor(state)
        if actor is None:
            kind = "showdown" if self._all_committed(state, alive) else "flop"
            return self._terminal(state, kind, alive)

        actions = self._actions_for(state, actor)
        node = DecisionNode(
            node_id=self._take_id(),
            player=actor,
            actions=actions,
            raise_level=state.raise_level,
        )
        self.decisions.append(node)
        for action in actions:
            node.children.append(self.build(self._apply(state, actor, action)))
        return node

    def _take_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    def _terminal(self, state: _State, kind: str, alive: list[int]) -> TerminalNode:
        contributions = tuple(bet + self.config.ante for bet in state.bets)
        last = max(alive, key=self.postflop_order.index)
        node = TerminalNode(
            node_id=self._take_id(),
            kind=kind,
            contributions=contributions,
            alive=tuple(alive),
            in_position=last,
            dead_money=self.config.dead_money,
        )
        self.terminals.append(node)
        return node

    # -------------------------------------------------------------- 下注轮

    def _next_actor(self, state: _State) -> int | None:
        """轮到谁说话；下注轮结束返回 None。"""
        current = state.current_bet
        for player in self.preflop_order:
            if state.folded[player] or state.allin[player]:
                continue
            if not state.acted[player] or state.bets[player] < current:
                return player
        return None

    def _all_committed(self, state: _State, alive: list[int]) -> bool:
        """活着的人是不是都已经全下——是的话直接摊牌，不必过翻后模型。"""
        return all(state.allin[player] for player in alive)

    def _actions_for(self, state: _State, actor: int) -> tuple[TreeAction, ...]:
        cfg = self.config
        current = state.current_bet
        owed = current - state.bets[actor]
        actions: list[TreeAction] = []

        if owed > 0:
            actions.append(TreeAction("fold", 0.0, "弃牌"))

        # 关掉跛入时，无人加注的局面只准开牌或弃牌；大盲过牌不算跛入
        limping = state.raise_level == 0 and owed > 0
        if cfg.allow_limp or not limping:
            call_to = min(current, cfg.effective_stack)
            actions.append(
                TreeAction("call", call_to, "过牌" if owed <= 0 else f"跟注到{call_to:g}")
            )

        for amount in self._raise_amounts(state, actor):
            label = "全下" if amount >= cfg.effective_stack else f"加注到{amount:g}"
            actions.append(TreeAction("raise", amount, label))
        return tuple(actions)

    def _raise_amounts(self, state: _State, actor: int) -> list[float]:
        """本节点允许的「加注到」额度，从小到大，已去重。"""
        cfg = self.config
        stack = cfg.effective_stack
        if state.bets[actor] >= stack:
            return []
        # 没人跟得起的加注不放进树：对手全下之后再加注毫无意义
        others = [
            p
            for p in range(cfg.players)
            if p != actor and not state.folded[p] and not state.allin[p]
        ]
        if not others:
            return []

        level = state.raise_level
        amounts: list[float] = []
        if level == 0:
            # 每有一个跛入者，开牌就抬高 1bb（主流开牌尺度惯例）
            limpers = sum(
                1
                for p in range(cfg.players)
                if p != actor
                and not state.folded[p]
                and state.acted[p]
                and state.bets[p] == cfg.big_blind
            )
            amounts.append(cfg.open_to + limpers * cfg.big_blind)
        elif level - 1 < len(cfg.reraise_multiples):
            amounts.append(state.last_raise_to * cfg.reraise_multiples[level - 1])

        if level + 1 >= cfg.jam_from_level or not amounts:
            amounts.append(stack)

        cleaned: list[float] = []
        for amount in sorted(amounts):
            capped = min(amount, stack)
            # 加注必须真的抬高价格，且不能高于对手跟得起的额度（与 state.py 同口径）
            if capped <= state.current_bet:
                continue
            if cleaned and abs(capped - cleaned[-1]) < 1e-9:
                continue
            cleaned.append(capped)
        return cleaned

    def _apply(self, state: _State, actor: int, action: TreeAction) -> _State:
        cfg = self.config
        bets = list(state.bets)
        folded = list(state.folded)
        allin = list(state.allin)
        acted = list(state.acted)
        raise_level = state.raise_level
        last_raise_to = state.last_raise_to

        if action.kind == "fold":
            folded[actor] = True
        else:
            bets[actor] = action.to_amount
            allin[actor] = action.to_amount >= cfg.effective_stack - 1e-9
        acted[actor] = True

        if action.is_raise:
            raise_level += 1
            last_raise_to = action.to_amount
            # 加注重开一圈：其他人都要重新面对这个价格
            acted = [player == actor for player in range(cfg.players)]

        return _State(
            bets=tuple(bets),
            folded=tuple(folded),
            allin=tuple(allin),
            acted=tuple(acted),
            raise_level=raise_level,
            last_raise_to=last_raise_to,
        )

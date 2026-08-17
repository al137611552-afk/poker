"""无限注德州扑克的手牌状态机。

纯逻辑：不摸随机数、不碰磁盘、不联网。牌由外部以「已洗好的一副牌」传入，
随机性因此完全属于调用方，回放与单测都能精确复现。

发牌顺序遵循真实牌桌：从庄家左手第一位起逐张发两轮底牌，再依次发翻牌 3 张、
转牌 1 张、河牌 1 张。不设烧牌（burn card）——它不影响任何策略，只会让牌谱噪声变大。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import Action, ActionKind, LegalActions
from .evaluator import evaluate
from .pots import Pot, award, build_pots, refund_uncalled

PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
COMPLETE = 4

STREET_NAMES = {PREFLOP: "preflop", FLOP: "flop", TURN: "turn", RIVER: "river"}
_BOARD_CARDS = {FLOP: 3, TURN: 1, RIVER: 1}


@dataclass(frozen=True)
class HandConfig:
    """一手牌的初始条件。筹码一律以最小单位（分/筹码点）计的整数表示。"""

    stacks: tuple[int, ...]
    button: int
    big_blind: int
    small_blind: int = 0
    ante: int = 0

    def __post_init__(self) -> None:
        if len(self.stacks) < 2:
            raise ValueError("至少需要两个座位")
        if not 0 <= self.button < len(self.stacks):
            raise ValueError(f"button 越界: {self.button}")
        if self.big_blind <= 0:
            raise ValueError("大盲必须为正")
        if any(s < 0 for s in self.stacks):
            raise ValueError("筹码不能为负")

    @property
    def num_seats(self) -> int:
        return len(self.stacks)

    @property
    def sb(self) -> int:
        return self.small_blind or self.big_blind // 2


@dataclass(frozen=True)
class Event:
    """牌局流水，供牌谱导出与复盘定位使用。"""

    street: int
    kind: str
    seat: int = -1
    amount: int = 0
    """实际付出的筹码（增量）。"""
    cards: tuple[int, ...] = ()
    to: int = 0
    """该动作之后此人本街的投入总额。牌谱与求解器都用这个口径，别和 amount 混淆。"""


@dataclass
class HandResult:
    payouts: list[int]
    """每个座位从底池中赢得的筹码（不含未跟注退款）。"""
    refunds: list[int]
    contributions: list[int]
    pots: list[Pot]
    showdown_scores: dict[int, int]
    went_to_showdown: bool

    @property
    def net(self) -> list[int]:
        """每个座位本手的净盈亏。"""
        return [
            self.payouts[i] + self.refunds[i] - self.contributions[i]
            for i in range(len(self.payouts))
        ]


class HandState:
    """一手牌的完整状态。可变对象；用 `clone()` 取快照做搜索或假设推演。"""

    def __init__(self, config: HandConfig, deck: list[int]) -> None:
        n = config.num_seats
        needed = 2 * n + 5
        if len(deck) < needed:
            raise ValueError(f"牌不够：需要 {needed} 张，收到 {len(deck)} 张")
        if len(set(deck[:needed])) != needed:
            raise ValueError("牌堆中存在重复的牌")

        self.config = config
        self._deck = list(deck)
        self._deck_pos = 0

        self.stacks: list[int] = list(config.stacks)
        self.committed_total: list[int] = [0] * n
        self.committed_street: list[int] = [0] * n
        self.folded: list[bool] = [False] * n
        self.all_in: list[bool] = [False] * n
        self.hole: list[tuple[int, ...]] = [() for _ in range(n)]
        self.board: list[int] = []
        self.events: list[Event] = []

        self.street = PREFLOP
        self.current_bet = 0
        self.last_raise_size = config.big_blind
        self.last_full_raise_level = 0
        self.acted_at_level: list[int] = [-1] * n
        self.to_act = -1
        self.sb_seat = -1
        self.bb_seat = -1
        self.result: HandResult | None = None

        self._deal_hole_cards()
        self._post_antes_and_blinds()
        self._open_preflop_action()

    # ------------------------------------------------------------------ 构建

    def _draw(self, count: int) -> tuple[int, ...]:
        cards = tuple(self._deck[self._deck_pos : self._deck_pos + count])
        self._deck_pos += count
        return cards

    def _seats_from(self, start: int) -> list[int]:
        n = self.config.num_seats
        return [(start + i) % n for i in range(n)]

    def _deal_hole_cards(self) -> None:
        n = self.config.num_seats
        order = self._seats_from((self.config.button + 1) % n)
        dealt: list[list[int]] = [[] for _ in range(n)]
        for _ in range(2):
            for seat in order:
                dealt[seat].extend(self._draw(1))
        for seat in range(n):
            self.hole[seat] = tuple(dealt[seat])
            self.events.append(
                Event(PREFLOP, "deal_hole", seat=seat, cards=self.hole[seat])
            )

    def _commit(self, seat: int, amount: int) -> int:
        """从座位筹码中投入 amount（自动按剩余筹码封顶），返回实际投入。"""
        amount = min(amount, self.stacks[seat])
        self.stacks[seat] -= amount
        self.committed_street[seat] += amount
        self.committed_total[seat] += amount
        if self.stacks[seat] == 0:
            self.all_in[seat] = True
        return amount

    def _post_antes_and_blinds(self) -> None:
        cfg = self.config
        n = cfg.num_seats

        if cfg.ante > 0:
            for seat in self._seats_from((cfg.button + 1) % n):
                posted = self._commit(seat, cfg.ante)
                if posted:
                    self.events.append(Event(PREFLOP, "ante", seat=seat, amount=posted))
            # 前注不参与本街的下注对齐，归入底池后重置本街投入
            self.committed_street = [0] * n

        if n == 2:
            sb_seat, bb_seat = cfg.button, (cfg.button + 1) % n
        else:
            sb_seat, bb_seat = (cfg.button + 1) % n, (cfg.button + 2) % n

        self.sb_seat, self.bb_seat = sb_seat, bb_seat
        posted_sb = self._commit(sb_seat, cfg.sb)
        self.events.append(Event(PREFLOP, "blind", seat=sb_seat, amount=posted_sb))
        posted_bb = self._commit(bb_seat, cfg.big_blind)
        self.events.append(Event(PREFLOP, "blind", seat=bb_seat, amount=posted_bb))

        self.current_bet = max(self.committed_street)
        self.last_raise_size = cfg.big_blind
        # 大盲构成第一个「完整加注」的基准，因此大盲本人保有后位选择权
        self.last_full_raise_level = cfg.big_blind

    def _open_preflop_action(self) -> None:
        n = self.config.num_seats
        first = self.config.button if n == 2 else (self.bb_seat + 1) % n
        self.to_act = self._next_actor(first, inclusive=True)
        self._maybe_advance(just_opened=True)

    # ------------------------------------------------------------------ 查询

    @property
    def is_complete(self) -> bool:
        return self.street == COMPLETE

    @property
    def pot_size(self) -> int:
        return sum(self.committed_total)

    def contenders(self) -> list[int]:
        """尚未弃牌的座位。"""
        return [i for i in range(self.config.num_seats) if not self.folded[i]]

    def _can_act(self, seat: int) -> bool:
        return not self.folded[seat] and not self.all_in[seat]

    def _actors(self) -> list[int]:
        return [i for i in range(self.config.num_seats) if self._can_act(i)]

    def _next_actor(self, start: int, inclusive: bool = False) -> int:
        n = self.config.num_seats
        offset = 0 if inclusive else 1
        for i in range(n):
            seat = (start + offset + i) % n
            if self._can_act(seat):
                return seat
        return -1

    def legal_actions(self) -> LegalActions:
        if self.is_complete:
            raise RuntimeError("手牌已结束，没有可行动作")
        seat = self.to_act
        committed = self.committed_street[seat]
        stack = self.stacks[seat]
        to_call = self.current_bet - committed

        call_to = min(self.current_bet, committed + stack)
        can_call = to_call > 0 and stack > 0
        can_check = to_call <= 0

        # 加注额封顶到「对手最多能跟到的额度」：超出的部分没人能跟，只会原样退回，
        # 记进牌谱还会被外部工具判为非法动作。这与主流平台的做法一致。
        opponents_reach = max(
            (
                self.committed_street[s] + self.stacks[s]
                for s in range(self.config.num_seats)
                if s != seat and not self.folded[s]
            ),
            default=0,
        )
        max_raise_to = min(committed + stack, opponents_reach)
        reopened = self.acted_at_level[seat] < self.last_full_raise_level
        can_raise = stack > 0 and reopened and max_raise_to > self.current_bet
        min_raise_to = min(self.current_bet + self.last_raise_size, max_raise_to)

        return LegalActions(
            seat=seat,
            can_fold=to_call > 0,
            can_check=can_check,
            can_call=can_call,
            call_to=call_to,
            call_cost=call_to - committed,
            can_raise=can_raise,
            is_opening_bet=self.current_bet == 0,
            min_raise_to=min_raise_to if can_raise else 0,
            max_raise_to=max_raise_to if can_raise else 0,
        )

    # ------------------------------------------------------------------ 推进

    def apply(self, action: Action) -> None:
        """执行一个动作并推进牌局。动作非法则抛 ValueError。"""
        if self.is_complete:
            raise RuntimeError("手牌已结束")
        legal = self.legal_actions()
        if not legal.contains(action):
            raise ValueError(f"非法动作 {action}，座位 {legal.seat} 的合法集为 {legal}")

        seat = self.to_act
        self.acted_at_level[seat] = self.current_bet

        if action.kind is ActionKind.FOLD:
            self.folded[seat] = True
            self.events.append(Event(self.street, "fold", seat=seat))

        elif action.kind is ActionKind.CHECK:
            self.events.append(Event(self.street, "check", seat=seat))

        elif action.kind is ActionKind.CALL:
            paid = self._commit(seat, legal.call_to - self.committed_street[seat])
            self.events.append(
                Event(
                    self.street,
                    "call",
                    seat=seat,
                    amount=paid,
                    to=self.committed_street[seat],
                )
            )

        else:  # BET / RAISE
            increment = action.to - self.current_bet
            paid = self._commit(seat, action.to - self.committed_street[seat])
            if increment >= self.last_raise_size:
                # 完整加注：重新打开加注权，并刷新最小加注增量
                self.last_raise_size = increment
                self.last_full_raise_level = action.to
            self.current_bet = max(self.current_bet, self.committed_street[seat])
            # 加注者自身在新的下注层级上已行动
            self.acted_at_level[seat] = self.current_bet
            kind = "bet" if action.kind is ActionKind.BET else "raise"
            self.events.append(
                Event(
                    self.street,
                    kind,
                    seat=seat,
                    amount=paid,
                    to=self.committed_street[seat],
                )
            )

        self._maybe_advance()

    def _betting_closed(self) -> bool:
        if len(self.contenders()) <= 1:
            return True
        for seat in self._actors():
            if self.acted_at_level[seat] < 0:
                return False
            if self.committed_street[seat] != self.current_bet:
                return False
        return True

    def _maybe_advance(self, just_opened: bool = False) -> None:
        if len(self.contenders()) <= 1:
            self._finish(went_to_showdown=False)
            return

        if not self._betting_closed():
            if not just_opened:
                self.to_act = self._next_actor(self.to_act)
            if self.to_act < 0:
                self._run_out_and_finish()
            return

        self._run_out_and_finish()

    def _run_out_and_finish(self) -> None:
        """当前下注轮结束：发完后续公共牌，能下注就停下来，否则一路发到河牌摊牌。"""
        while self.street < RIVER:
            self._start_street(self.street + 1)
            if len(self._actors()) >= 2:
                self.to_act = self._next_actor((self.config.button + 1) % self.config.num_seats, inclusive=True)
                return
        self._finish(went_to_showdown=True)

    def _start_street(self, street: int) -> None:
        self.street = street
        cards = self._draw(_BOARD_CARDS[street])
        self.board.extend(cards)
        self.events.append(Event(street, "deal_board", cards=cards))
        self.committed_street = [0] * self.config.num_seats
        self.acted_at_level = [-1] * self.config.num_seats
        self.current_bet = 0
        self.last_raise_size = self.config.big_blind
        self.last_full_raise_level = 0

    def _finish(self, went_to_showdown: bool) -> None:
        n = self.config.num_seats
        contributions, refunds = refund_uncalled(self.committed_total)
        for seat, amount in enumerate(refunds):
            if amount:
                self.stacks[seat] += amount
                self.events.append(Event(self.street, "refund", seat=seat, amount=amount))

        pots = build_pots(contributions, self.folded)
        live = self.contenders()

        if went_to_showdown and len(live) > 1:
            scores = {seat: evaluate([*self.hole[seat], *self.board]) for seat in live}
            for seat in live:
                self.events.append(
                    Event(self.street, "showdown", seat=seat, cards=self.hole[seat])
                )
        else:
            # 未摊牌：唯一存活者拿走全部，用哨兵分数表达
            scores = {seat: 1 for seat in live}

        payouts = award(pots, scores, (self.config.button + 1) % n, n)
        for seat, amount in enumerate(payouts):
            if amount:
                self.stacks[seat] += amount
                self.events.append(Event(self.street, "award", seat=seat, amount=amount))

        self.result = HandResult(
            payouts=payouts,
            refunds=refunds,
            contributions=list(self.committed_total),
            pots=pots,
            showdown_scores=scores if went_to_showdown and len(live) > 1 else {},
            went_to_showdown=went_to_showdown and len(live) > 1,
        )
        self.street = COMPLETE
        self.to_act = -1

    # ------------------------------------------------------------------ 工具

    def clone(self) -> "HandState":
        import copy

        return copy.deepcopy(self)

    def describe(self) -> str:
        """一行摘要，调试与日志用。"""
        if self.is_complete:
            return f"[结束] 底池 {self.pot_size} 净额 {self.result.net if self.result else []}"
        street = STREET_NAMES[self.street]
        return (
            f"[{street}] 底池 {self.pot_size} 当前下注 {self.current_bet} "
            f"轮到座位 {self.to_act}"
        )

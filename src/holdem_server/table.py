"""牌桌编排：一张桌子的生命周期。

刻意与 FastAPI 解耦——本文件不 import 任何 Web 框架，因此可以脱离 HTTP 单测。
服务端只是它的一层适配器。

对外只暴露两种推进方式：`step_bot()` 走一个 bot 动作，`apply_human()` 走一个人类动作。
前端据此逐步播放，而不是一次性把整条街结算完。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from holdem.actions import Action, ActionKind, bet, call, check, fold, raise_to
from holdem.bots import DEFAULT_STYLE, STYLES, RuleBot
from holdem.cards import card_to_str
from holdem.deck import shuffled_deck
from holdem.positions import position_of
from holdem.state import COMPLETE, STREET_NAMES, HandConfig, HandState
from holdem.store import HandStore

MAX_SEATS = 9
MIN_SEATS = 2


@dataclass
class SeatConfig:
    name: str
    is_human: bool = False
    style: str = DEFAULT_STYLE

    def __post_init__(self) -> None:
        if not self.is_human and self.style not in STYLES:
            raise ValueError(f"未知风格: {self.style}")


@dataclass
class TableConfig:
    seats: list[SeatConfig]
    starting_stack: int = 1000
    big_blind: int = 10
    small_blind: int = 5
    ante: int = 0
    seed: int | None = None
    auto_rebuy: bool = True
    """筹码打光后自动补回初始额度——练习场景下不希望被淘汰打断。"""

    def __post_init__(self) -> None:
        if not MIN_SEATS <= len(self.seats) <= MAX_SEATS:
            raise ValueError(f"座位数必须在 {MIN_SEATS}–{MAX_SEATS} 之间")
        if sum(1 for s in self.seats if s.is_human) > 1:
            raise ValueError("目前只支持一个真人座位")
        if self.big_blind <= 0 or self.starting_stack <= 0:
            raise ValueError("盲注与初始筹码必须为正")


@dataclass
class LogLine:
    street: str
    text: str


@dataclass
class TableSession:
    config: TableConfig
    store: HandStore | None = None
    store_session_id: int | None = None

    stacks: list[int] = field(default_factory=list)
    button: int = 0
    hand: HandState | None = None
    hand_no: int = 0
    log: list[LogLine] = field(default_factory=list)
    last_saved_hand_id: int | None = None

    def __post_init__(self) -> None:
        n = len(self.config.seats)
        self.stacks = [self.config.starting_stack] * n
        self.button = n - 1  # 第一手开始前先移动按钮，于是首手按钮落在座位 0
        self._rng = random.Random(self.config.seed)
        self._bots = {
            seat: RuleBot(cfg.style, seed=self._rng.randrange(1 << 30))
            for seat, cfg in enumerate(self.config.seats)
            if not cfg.is_human
        }
        self._logged_events = 0

    # ------------------------------------------------------------ 基本信息

    @property
    def num_seats(self) -> int:
        return len(self.config.seats)

    @property
    def hero_seat(self) -> int | None:
        for seat, cfg in enumerate(self.config.seats):
            if cfg.is_human:
                return seat
        return None

    @property
    def in_progress(self) -> bool:
        return self.hand is not None and not self.hand.is_complete

    @property
    def waiting_for_human(self) -> bool:
        return self.in_progress and self.hand.to_act == self.hero_seat  # type: ignore[union-attr]

    # ------------------------------------------------------------ 手牌生命周期

    def start_hand(self) -> None:
        if self.in_progress:
            raise RuntimeError("上一手还没结束")

        if self.config.auto_rebuy:
            for seat in range(self.num_seats):
                if self.stacks[seat] < self.config.big_blind:
                    self.stacks[seat] = self.config.starting_stack

        playable = [s for s in range(self.num_seats) if self.stacks[s] > 0]
        if len(playable) < MIN_SEATS:
            raise RuntimeError("能参与的座位不足两个")

        self.button = (self.button + 1) % self.num_seats
        self.hand_no += 1
        self.log = []
        self._logged_events = 0

        config = HandConfig(
            stacks=tuple(self.stacks),
            button=self.button,
            big_blind=self.config.big_blind,
            small_blind=self.config.small_blind,
            ante=self.config.ante,
        )
        self.hand = HandState(config, shuffled_deck(self._rng))
        self._collect_log()

    def step_bot(self) -> bool:
        """若当前该 bot 行动则走一步，返回是否真的走了。"""
        if not self.in_progress:
            return False
        seat = self.hand.to_act  # type: ignore[union-attr]
        if seat == self.hero_seat:
            return False
        bot = self._bots.get(seat)
        if bot is None:
            raise RuntimeError(f"座位 {seat} 既不是真人也没有 bot")
        self.hand.apply(bot.act(self.hand))  # type: ignore[union-attr]
        self._after_action()
        return True

    def apply_human(self, kind: str, amount: int | None = None) -> None:
        if not self.in_progress:
            raise RuntimeError("当前没有进行中的牌局")
        hero = self.hero_seat
        if hero is None:
            raise RuntimeError("这张桌子没有真人座位")
        if self.hand.to_act != hero:  # type: ignore[union-attr]
            raise RuntimeError("还没轮到你行动")
        self.hand.apply(self._build_action(kind, amount))  # type: ignore[union-attr]
        self._after_action()

    def _build_action(self, kind: str, amount: int | None) -> Action:
        legal = self.hand.legal_actions()  # type: ignore[union-attr]
        if kind == ActionKind.FOLD.value:
            return fold()
        if kind == ActionKind.CHECK.value:
            return check()
        if kind == ActionKind.CALL.value:
            return call()
        if kind in (ActionKind.BET.value, ActionKind.RAISE.value):
            if amount is None:
                raise ValueError("下注或加注必须给出金额")
            return bet(amount) if legal.is_opening_bet else raise_to(amount)
        raise ValueError(f"未知动作: {kind!r}")

    def _after_action(self) -> None:
        self._collect_log()
        if self.hand is not None and self.hand.is_complete:
            self._settle()

    def _settle(self) -> None:
        hand = self.hand
        assert hand is not None and hand.result is not None
        self.stacks = list(hand.stacks)
        if self.store is not None and self.store_session_id is not None:
            self.last_saved_hand_id = self.store.save_hand(
                hand,
                session_id=self.store_session_id,
                players=[s.name for s in self.config.seats],
                hand_no=self.hand_no,
            )

    # ------------------------------------------------------------ 日志

    def _collect_log(self) -> None:
        hand = self.hand
        assert hand is not None
        names = [s.name for s in self.config.seats]
        for event in hand.events[self._logged_events :]:
            street = STREET_NAMES.get(event.street, "showdown")
            who = names[event.seat] if event.seat >= 0 else ""
            text = None
            if event.kind == "blind":
                text = f"{who} 下盲注 {event.amount}"
            elif event.kind == "ante":
                text = f"{who} 下前注 {event.amount}"
            elif event.kind == "deal_board":
                cards = " ".join(card_to_str(c) for c in event.cards)
                text = f"发牌：{cards}"
            elif event.kind == "fold":
                text = f"{who} 弃牌"
            elif event.kind == "check":
                text = f"{who} 过牌"
            elif event.kind == "call":
                text = f"{who} 跟注 {event.amount}"
            elif event.kind == "bet":
                text = f"{who} 下注 {event.to}"
            elif event.kind == "raise":
                text = f"{who} 加注至 {event.to}"
            elif event.kind == "showdown":
                cards = " ".join(card_to_str(c) for c in event.cards)
                text = f"{who} 亮牌 {cards}"
            elif event.kind == "refund":
                text = f"{who} 收回未被跟注的 {event.amount}"
            elif event.kind == "award":
                text = f"{who} 赢得 {event.amount}"
            if text:
                self.log.append(LogLine(street, text))
        self._logged_events = len(hand.events)

    # ------------------------------------------------------------ 视图

    def view(self) -> dict:
        """给前端的公开状态。**除英雄本人与摊牌者外，底牌一律隐藏。**"""
        hand = self.hand
        hero = self.hero_seat
        payload: dict = {
            "handNo": self.hand_no,
            "button": self.button,
            "heroSeat": hero,
            "bigBlind": self.config.big_blind,
            "smallBlind": self.config.small_blind,
            "inProgress": self.in_progress,
            "waitingForHuman": self.waiting_for_human,
            "log": [{"street": line.street, "text": line.text} for line in self.log],
        }

        if hand is None:
            payload["seats"] = [
                {
                    "seat": seat,
                    "name": cfg.name,
                    "isHuman": cfg.is_human,
                    "style": None if cfg.is_human else STYLES[cfg.style].label,
                    "stack": self.stacks[seat],
                    "committed": 0,
                    "folded": False,
                    "allIn": False,
                    "cards": None,
                    "position": position_of(seat, self.button, self.num_seats),
                    "isActing": False,
                }
                for seat, cfg in enumerate(self.config.seats)
            ]
            payload["board"] = []
            payload["pot"] = 0
            payload["street"] = None
            payload["legal"] = None
            payload["result"] = None
            return payload

        revealed = set()
        if hand.is_complete and hand.result is not None:
            revealed = set(hand.result.showdown_scores)
        if hero is not None:
            revealed.add(hero)

        payload["seats"] = [
            {
                "seat": seat,
                "name": cfg.name,
                "isHuman": cfg.is_human,
                "style": None if cfg.is_human else STYLES[cfg.style].label,
                "stack": hand.stacks[seat],
                "committed": hand.committed_street[seat],
                "folded": hand.folded[seat],
                "allIn": hand.all_in[seat],
                "cards": (
                    [card_to_str(c) for c in hand.hole[seat]] if seat in revealed else None
                ),
                "position": position_of(seat, hand.config.button, self.num_seats),
                "isActing": self.in_progress and hand.to_act == seat,
            }
            for seat, cfg in enumerate(self.config.seats)
        ]
        payload["board"] = [card_to_str(c) for c in hand.board]
        payload["pot"] = hand.pot_size
        payload["street"] = STREET_NAMES.get(hand.street, "complete")

        if self.waiting_for_human:
            legal = hand.legal_actions()
            payload["legal"] = {
                "canFold": legal.can_fold,
                "canCheck": legal.can_check,
                "canCall": legal.can_call,
                "callCost": legal.call_cost,
                "canRaise": legal.can_raise,
                "isOpeningBet": legal.is_opening_bet,
                "minRaiseTo": legal.min_raise_to,
                "maxRaiseTo": legal.max_raise_to,
                "potSizedTo": self._pot_sized_raise(legal, hand),
            }
        else:
            payload["legal"] = None

        if hand.street == COMPLETE and hand.result is not None:
            payload["result"] = {
                "net": hand.result.net,
                "wentToShowdown": hand.result.went_to_showdown,
                "pots": [
                    {"amount": pot.amount, "eligible": list(pot.eligible)}
                    for pot in hand.result.pots
                ],
                "handId": self.last_saved_hand_id,
            }
        else:
            payload["result"] = None
        return payload

    @staticmethod
    def _pot_sized_raise(legal, hand: HandState) -> int:
        """底池大小的加注额，作为前端的一个快捷按钮。"""
        target = legal.call_to + hand.pot_size + legal.call_cost
        return max(legal.min_raise_to, min(target, legal.max_raise_to))

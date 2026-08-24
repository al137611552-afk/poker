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
from holdem.bots import DEFAULT_STYLE, STYLES, Bot
from holdem.cards import card_to_str
from holdem.deck import shuffled_deck
from holdem.positions import position_of
from holdem.preflop_review import review_preflop
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
            seat: Bot(cfg.style, seed=self._rng.randrange(1 << 30))
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

    def review(self) -> dict:
        """复盘**刚打完**那手牌（FR-9）。

        分两层，因为它们的精度不一样，**不能并排成一列数**：

        - **翻前**：用离线解好的范围表判频率分。不依赖求解器，装没装都有。
        - **翻后**：EV 损失要真解局面（`holdem_solver.review`）。没装求解器就
          如实说「没装」，**不降级成一个看着差不多的数**——那正是 PRD 的
          「不得冒充精确解」要防的事。
        """
        hand = self.hand
        if hand is None or hand.result is None:
            # 抛普通异常、由路由层转成 HTTP 状态码：**这一层不该认识 fastapi**
            # （牌桌逻辑要能脱离 web 单测，那是 CLAUDE.md 的分层约定）
            raise LookupError("还没有打完的牌可复盘")

        hero = self.hero_seat
        preflop: list = []
        note = None
        try:
            steps = review_preflop(hand, hero)
        except Exception as exc:                 # 表缺失、重放对不上……
            steps = []
            note = f"翻前复盘做不了：{exc}"
        for step in steps:
            verdict = step.verdict
            preflop.append({
                "index": step.index,
                "position": step.position,
                "potBefore": step.pot_before,
                "toCall": step.to_call,
                "action": step.action,
                "graded": step.graded,
                "verdict": verdict.verdict if verdict else "表里没有这一格，判不了",
                "frequency": verdict.frequency if verdict else None,
                "best": verdict.best if verdict else None,
                "taken": verdict.taken if verdict else None,
                "blunder": step.blunder,
                "weights": dict(verdict.weights) if verdict else {},
            })

        return {
            "handNo": self.hand_no,
            "heroSeat": hero,
            "heroCards": [card_to_str(c) for c in hand.hole[hero]],
            "board": [card_to_str(c) for c in hand.board],
            "net": hand.result.net[hero],
            "preflop": preflop,
            "preflopNote": note,
            # 翻后那半的门在这儿。**判据是「有没有求解器」，不是「要不要试试」**——
            # 试了再失败会让用户等上几十秒才看到一句「没装」。
            "postflop": {
                "available": _solver_ready(),
                "why": None if _solver_ready() else
                       "没装求解器（TEXAS_SOLVER_HOME），翻后 EV 损失算不了；"
                       "翻前那半照常给",
            },
        }

    def hud(self, *, scope: str = "session") -> dict:
        """牌桌浮层要的统计（FR-8）。口径来自 `stats.py`，这里一行都不重算。

        **每个指标都连样本量一起给。** HUD 最大的坑就是拿 5 手牌的 VPIP 当真——
        主流软件在样本少时会把数字标灰，我们把判断所需的原始计数直接交给前端，
        而不是替它决定「够不够」：够不够取决于看哪个指标（VPIP 几十手就稳，
        3bet 要几百手），塞死一个阈值反而会骗人。

        `scope="session"` 只看这一局（默认）：座位名跨会话会对应到不同风格的 bot，
        把它们混在一起统计等于把两个人的数据算给同一个人。
        `scope="all"` 看全库，适合长期跟同一批对手打。
        """
        if self.store is None:
            return {"scope": scope, "seats": [], "unavailable": "没有连数据库，统计不了"}

        names = [cfg.name for cfg in self.config.seats]
        session_id = None if scope == "all" else self.store_session_id
        lines = self.store.player_stats(session_id=session_id, players=tuple(names))

        seats = []
        for seat, name in enumerate(names):
            line = lines.get(name)
            seats.append({
                "seat": seat,
                "name": name,
                "hands": line.hands if line else 0,
                "stats": _hud_metrics(line),
            })
        return {"scope": scope, "seats": seats}

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


_HUD_METRICS = (
    ("vpip", "VPIP", "主动投钱"),
    ("pfr", "PFR", "翻前加注"),
    ("rfi", "开牌", "前面全弃时开牌"),
    ("threebet", "3bet", "面对加注再加注"),
    ("fold_to_threebet", "弃于3bet", "开牌后被 3bet 弃牌"),
    ("cbet_flop", "持续下注", "翻前进攻方在翻牌下注"),
    ("fold_to_cbet_flop", "弃于CB", "面对持续下注弃牌"),
    ("wtsd", "WTSD", "看到翻牌后走到摊牌"),
    ("wsd", "W$SD", "摊牌赢下"),
)


def _hud_metrics(line) -> list:
    """把一行统计摊成前端好画的形状。

    `rate` 为 `None` 表示**一次机会都没有过**——前端要把它显示成「—」而不是 0%。
    「从没面对过 3bet」和「面对 3bet 从不弃牌」是两件事（`stats.Chance` 那条）。
    """
    out = []
    for field, label, hint in _HUD_METRICS:
        chance = getattr(line, field) if line else None
        out.append({
            "key": field,
            "label": label,
            "hint": hint,
            "rate": chance.rate if chance else None,
            "chances": chance.chances if chance else 0,
            "hits": chance.hits if chance else 0,
        })
    # AF **不是**「机会/发生」那种结构，它是两个计数的比值。硬塞进 hits/chances
    # 会造出 `hits > chances` 这种自相矛盾的数据（第一版就是这样，被测试逮到），
    # 而前端只要照着通用逻辑画就会得出一个 >100% 的百分比。给它自己的字段名。
    out.append({
        "key": "aggression_factor",
        "label": "AF",
        "hint": "翻后（下注+加注）/跟注",
        "value": line.aggression_factor if line else None,
        "aggressive": line.postflop_aggressive if line else 0,
        "calls": line.postflop_calls if line else 0,
    })
    return out


def _solver_ready() -> bool:
    """本机有没有可用的求解器。**导入放在函数里**：`holdem_solver` 不是这个包的
    硬依赖，顶层导入会让没装求解器的机器连牌桌都起不来。"""
    try:
        from holdem_solver.backend import TexasSolver
    except Exception:
        return False
    try:
        return TexasSolver.available()
    except Exception:
        return False

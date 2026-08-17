"""占位规则 bot。

**这不是训练用的对手。** 它的作用是让牌桌能跑起来、让服务端与前端有东西可对接。
真正的对手（策略网络 + 风格层）在 M1 交付，届时本文件的接口保持不变、实现替换。

决策方式：蒙特卡洛权益 + 底池赔率 + 一组风格参数。够用、可解释、可单测，
但它对上真人会明显偏弱，也不该被当作水平参照。

纯逻辑：不碰 IO，随机性由调用方注入的 `rng` 提供。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .actions import Action, bet, call, check, fold, raise_to
from .equity import monte_carlo_equity
from .state import PREFLOP, HandState


@dataclass(frozen=True)
class BotStyle:
    """风格参数。M1 的风格层会扩成完整的向量，这里先放最小的一组。

    翻前与翻后用两套阈值，因为两处的权益尺度不同（见下方 `act` 的说明）。
    """

    name: str
    label: str
    preflop_open: float
    """翻前主动加注所需的**单挑**权益。0.62 大致相当于前 20% 的起手牌。"""
    preflop_call: float
    """翻前跟注所需的单挑权益，再按底池赔率上浮。"""
    open_equity: float
    """翻后主动下注所需的**多路**权益。"""
    call_margin: float
    """翻后跟注所需的权益，在底池赔率之上再加的余量。"""
    aggression: float
    """有优势时选择加注而非跟注的概率。"""
    bluff: float
    """无人下注且自己没牌时的诈唬概率。"""
    bet_fraction: float
    """下注尺度，占底池的比例。"""
    samples: int = 160


STYLES: dict[str, BotStyle] = {
    "tag": BotStyle("tag", "紧凶", 0.62, 0.55, 0.62, 0.04, 0.55, 0.12, 0.66),
    "lag": BotStyle("lag", "松凶", 0.55, 0.50, 0.52, 0.01, 0.70, 0.28, 0.75),
    "nit": BotStyle("nit", "岩石", 0.68, 0.60, 0.72, 0.10, 0.35, 0.03, 0.55),
    "station": BotStyle("station", "跟注站", 0.66, 0.44, 0.70, -0.08, 0.10, 0.02, 0.50),
    "maniac": BotStyle("maniac", "疯子", 0.48, 0.42, 0.44, -0.02, 0.85, 0.45, 0.95),
}

DEFAULT_STYLE = "tag"


class RuleBot:
    """按风格参数行动的占位 bot。"""

    def __init__(self, style: BotStyle | str = DEFAULT_STYLE, seed: int | None = None) -> None:
        if isinstance(style, str):
            if style not in STYLES:
                raise ValueError(f"未知风格: {style}，可选 {sorted(STYLES)}")
            style = STYLES[style]
        self.style = style
        self.rng = random.Random(seed)

    def act(self, hand: HandState) -> Action:
        """给出当前行动座位的动作。调用方保证 hand 未结束。

        翻前与翻后用两套尺度，这是刻意的：翻前若拿「对上全桌的多路权益」去比即时底池
        赔率，会算出「跟注需要 40% 权益」这种结论，结果是任何牌都弃——因为它忽略了
        对手会弃牌、忽略了后续街的隐含赔率。所以翻前改用**单挑权益**当牌力尺度，
        只在翻后才用多路权益配底池赔率。
        """
        legal = hand.legal_actions()
        seat = legal.seat

        if hand.street == PREFLOP:
            strength = monte_carlo_equity(
                hand.hole[seat],
                (),
                1,
                samples=self.style.samples,
                rng=self.rng,
            )
            return self._act_preflop(hand, legal, strength)

        opponents = max(1, len(hand.contenders()) - 1)
        equity = monte_carlo_equity(
            hand.hole[seat],
            hand.board,
            opponents,
            samples=self.style.samples,
            rng=self.rng,
        )
        if legal.can_check:
            return self._act_unopened(hand, legal, equity)
        return self._act_facing_bet(hand, legal, equity)

    # ------------------------------------------------------------ 翻前

    def _act_preflop(self, hand: HandState, legal, strength: float) -> Action:
        style = self.style
        pot_after_call = hand.pot_size + legal.call_cost
        pot_odds = legal.call_cost / pot_after_call if pot_after_call else 0.0
        # 面对越大的下注（3-bet、4-bet）要求越高，用底池赔率超出常规开局的部分度量
        required_call = style.preflop_call + max(0.0, pot_odds - 0.33)

        if strength >= style.preflop_open and legal.can_raise:
            if self.rng.random() < max(style.aggression, 0.5):
                return raise_to(self._preflop_raise_size(hand, legal))
        if legal.can_raise and self.rng.random() < style.bluff * 0.3:
            return raise_to(self._preflop_raise_size(hand, legal))
        if legal.can_check:
            return check()
        if strength >= required_call and legal.can_call:
            return call()
        return fold()

    def _preflop_raise_size(self, hand: HandState, legal) -> int:
        """开局约 3 倍大盲，面对加注则约 2.5 倍对手额度。"""
        bb = hand.config.big_blind
        target = 3 * bb if legal.call_to <= bb else int(round(legal.call_to * 2.5))
        return max(legal.min_raise_to, min(target, legal.max_raise_to))

    # ------------------------------------------------------------ 翻后无人下注

    def _act_unopened(self, hand: HandState, legal, equity: float) -> Action:
        style = self.style
        wants_value = equity >= style.open_equity
        wants_bluff = equity < style.open_equity and self.rng.random() < style.bluff

        if legal.can_raise and (wants_value or wants_bluff):
            size = self._bet_size(hand, legal, big=wants_value)
            return bet(size) if legal.is_opening_bet else raise_to(size)
        return check()

    # ------------------------------------------------------------ 翻后面对下注

    def _act_facing_bet(self, hand: HandState, legal, equity: float) -> Action:
        style = self.style
        pot_after_call = hand.pot_size + legal.call_cost
        pot_odds = legal.call_cost / pot_after_call if pot_after_call else 0.0
        required = pot_odds + style.call_margin

        strong = equity >= style.open_equity
        if strong and legal.can_raise and self.rng.random() < style.aggression:
            return raise_to(self._bet_size(hand, legal, big=True))

        if equity >= required and legal.can_call:
            return call()
        if legal.can_check:
            return check()
        return fold()

    # ------------------------------------------------------------ 尺度

    def _bet_size(self, hand: HandState, legal, *, big: bool) -> int:
        fraction = self.style.bet_fraction if big else self.style.bet_fraction * 0.6
        target = legal.call_to + int(round(hand.pot_size * fraction))
        return max(legal.min_raise_to, min(target, legal.max_raise_to))


def play_out(hand: HandState, bots: dict[int, RuleBot], max_steps: int = 400) -> None:
    """让 bot 把牌局打完（用于自对弈与测试）。座位没有对应 bot 时抛错。"""
    steps = 0
    while not hand.is_complete:
        seat = hand.to_act
        if seat not in bots:
            raise KeyError(f"座位 {seat} 没有对应的 bot")
        hand.apply(bots[seat].act(hand))
        steps += 1
        if steps > max_steps:
            raise RuntimeError("牌局未能收敛")

"""对手 bot：翻前照解出来的范围表打，翻后走启发式。

## 三层，各管一段

| 层 | 覆盖 | 依据 |
|---|---|---|
| **翻前查表** | 首个开牌、面对开牌、面对 3bet | `preflop_ranges` 的解（ADR-0003/0004） |
| **翻前兜底** | 跛入局、多人底池、4bet 之后、表缺的人数 | 单挑权益 + 底池赔率的规则 |
| **翻后** | 全部 | 蒙特卡洛权益 + 底池赔率 + 风格参数 |

**别把它当 GTO 对手**：翻前是模型解，而模型的兑现系数尚未校准（ADR-0003）；翻后完全是
规则。界面标注置信度时翻前算 B 级、翻后算 C 级——PRD 的「诚实」那条要求这么标。

## 风格层怎么作用

两个旋钮都作用在**解**上，而不是另写一套规则：

- `looseness` 沿「解认为这手牌多值得入池」的排序移动入池频率。**不是乘概率**——解里
  100% 弃掉的牌乘多少还是 0，那样「松」这个风格永远打不出解以外的牌。
- `aggression` 在入池之后把权重从跟注挪向加注。

排序用的是求解器算出的**逐牌类 EV**，所以放宽时先纳进来的是「最接近该打」的牌，
而不是「权益最高」的牌——后者会让松手风格打 K9o 却不打 65s，与真人完全不像。

纯逻辑：不碰 IO，随机性由调用方注入的 `rng` 提供。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .actions import Action, bet, call, check, fold, raise_to
from .equity import monte_carlo_equity
from .preflop_policy import PolicyDecision, PreflopTablePolicy, parse_label
from .state import PREFLOP, HandState

__all__ = ["BotStyle", "STYLES", "DEFAULT_STYLE", "Bot", "play_out", "shared_policy"]


@dataclass(frozen=True)
class BotStyle:
    """一种对手风格。前两项作用在解上，其余是表覆盖不到时的规则参数。"""

    name: str
    label: str
    looseness: float
    """入池频率相对解的倍数。1.0 = 照解走。"""
    aggression: float
    """入池之后加注相对跟注的倍数。1.0 = 照解走。"""
    preflop_open: float
    """兜底用：翻前主动加注所需的**单挑**权益。"""
    preflop_call: float
    """兜底用：翻前跟注所需的单挑权益，再按底池赔率上浮。"""
    open_equity: float
    """翻后主动下注所需的**多路**权益。"""
    call_margin: float
    """翻后跟注所需的权益，在底池赔率之上再加的余量。"""
    postflop_aggression: float
    """翻后有优势时选择加注而非跟注的概率。"""
    bluff: float
    """翻后无人下注且自己没牌时的诈唬概率。"""
    bet_fraction: float
    """下注尺度，占底池的比例。"""
    samples: int = 160


STYLES: dict[str, BotStyle] = {
    # 叫「照解」而不是「GTO」：这是我们模型的解，兑现系数还没校准，不是均衡（ADR-0003）
    "solved": BotStyle("solved", "照解", 1.00, 1.00, 0.62, 0.55, 0.62, 0.04, 0.55, 0.12, 0.66),
    "tag": BotStyle("tag", "紧凶", 0.70, 1.35, 0.64, 0.57, 0.62, 0.04, 0.60, 0.12, 0.70),
    "lag": BotStyle("lag", "松凶", 1.15, 1.80, 0.57, 0.52, 0.52, 0.01, 0.75, 0.28, 0.78),
    "nit": BotStyle("nit", "岩石", 0.42, 0.75, 0.70, 0.62, 0.72, 0.10, 0.35, 0.03, 0.55),
    "station": BotStyle("station", "跟注站", 1.45, 0.12, 0.66, 0.44, 0.70, -0.08, 0.10, 0.02, 0.50),
    "maniac": BotStyle("maniac", "疯子", 2.00, 3.00, 0.50, 0.44, 0.44, -0.02, 0.85, 0.45, 0.95),
}

DEFAULT_STYLE = "tag"

_RELAX_PER_STEP = 0.05
"""兜底阈值随 `looseness` 每偏离 1.0 放宽多少权益。实测标定，见 DEVLOG。"""
_BASE_RAISE_SHARE = 0.55
"""兜底里「够强就加注」的基准比例，再乘风格的 `aggression`。"""

_SHARED: PreflopTablePolicy | None = None
_TRIED = False


def shared_policy() -> PreflopTablePolicy | None:
    """全进程共用一份范围表——它有几十 KB，还带着按局面预算好的排序，不该每个 bot 一份。

    表没生成时返回 `None`，所有 bot 自动退回规则策略（引擎不依赖产物也能跑）。
    """
    global _SHARED, _TRIED
    if not _TRIED:
        _TRIED = True
        if PreflopTablePolicy.available():
            _SHARED = PreflopTablePolicy()
    return _SHARED


class Bot:
    """按风格行动的对手。翻前尽量照解走，解覆盖不到的局面退回规则。"""

    def __init__(
        self,
        style: BotStyle | str = DEFAULT_STYLE,
        seed: int | None = None,
        policy: PreflopTablePolicy | None = None,
    ) -> None:
        if isinstance(style, str):
            if style not in STYLES:
                raise ValueError(f"未知风格: {style}，可选 {sorted(STYLES)}")
            style = STYLES[style]
        self.style = style
        self.rng = random.Random(seed)
        self.policy = policy if policy is not None else shared_policy()
        self.table_hits = 0
        """照解走了多少次决策——自对弈时用来看解的覆盖率。"""
        self.fallback_hits = 0

    def act(self, hand: HandState) -> Action:
        """给出当前行动座位的动作。调用方保证 hand 未结束。"""
        legal = hand.legal_actions()
        seat = legal.seat

        if hand.street == PREFLOP:
            decision = None
            if self.policy is not None:
                decision = self.policy.decide(
                    hand,
                    looseness=self.style.looseness,
                    aggression=self.style.aggression,
                )
            if decision is not None:
                self.table_hits += 1
                return self._from_table(hand, legal, decision)
            self.fallback_hits += 1
            strength = monte_carlo_equity(
                hand.hole[seat], (), 1, samples=self.style.samples, rng=self.rng
            )
            return self._act_preflop_fallback(hand, legal, strength)

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

    # ------------------------------------------------------------ 翻前查表

    def _from_table(self, hand: HandState, legal, decision: PolicyDecision) -> Action:
        """按解给出的概率抽一个动作，再翻译成引擎能接受的合法动作。"""
        label = self._sample(decision.weights)
        kind, to_big_blinds = parse_label(label)

        if kind == "fold":
            # 能免费过牌时绝不弃牌——解里的「弃牌」是面对下注时的选择
            return check() if legal.can_check else fold()
        if kind == "call":
            return call() if legal.can_call else check()

        if legal.can_raise:
            if kind == "allin" or to_big_blinds is None:
                return raise_to(legal.max_raise_to)
            target = int(round(to_big_blinds * hand.config.big_blind))
            return raise_to(max(legal.min_raise_to, min(target, legal.max_raise_to)))
        # 加注不合法（筹码不够开新价位）时退而求其次
        if legal.can_call:
            return call()
        return check() if legal.can_check else fold()

    def _sample(self, weights: dict[str, float]) -> str:
        total = sum(weights.values())
        if total <= 0:
            return next(iter(weights))
        threshold = self.rng.random() * total
        cumulative = 0.0
        for label, weight in weights.items():
            cumulative += weight
            if threshold <= cumulative:
                return label
        return label

    # ------------------------------------------------------------ 翻前兜底

    def _act_preflop_fallback(self, hand: HandState, legal, strength: float) -> Action:
        """表覆盖不到的翻前局面（跛入、多人底池、4bet 之后）走这条老路。

        翻前用**单挑权益**当牌力尺度，不用多路权益比即时底池赔率——后者会算出「跟注要
        40% 权益」而把所有牌都弃掉，因为它忽略了对手会弃牌、也忽略了隐含赔率。
        """
        style = self.style
        pot_after_call = hand.pot_size + legal.call_cost
        pot_odds = legal.call_cost / pot_after_call if pot_after_call else 0.0

        # 两个旋钮在兜底里也必须生效，否则松风格会「一半按解、一半按另一套脾气」打。
        # 松紧线性挪阈值（别用开方——它对大 looseness 太钝、对小的又太猛），
        # 凶度决定入池之后加注的比例。
        relax = _RELAX_PER_STEP * (style.looseness - 1.0)
        open_threshold = style.preflop_open - relax
        call_threshold = style.preflop_call - relax + max(0.0, pot_odds - 0.33)
        raise_chance = min(1.0, _BASE_RAISE_SHARE * style.aggression)

        if strength >= open_threshold and legal.can_raise:
            if self.rng.random() < raise_chance:
                return raise_to(self._preflop_raise_size(hand, legal))
        if legal.can_raise and self.rng.random() < style.bluff * 0.3 * style.aggression:
            return raise_to(self._preflop_raise_size(hand, legal))
        if legal.can_check:
            return check()
        if strength >= call_threshold and legal.can_call:
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
        if strong and legal.can_raise and self.rng.random() < style.postflop_aggression:
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


def play_out(hand: HandState, bots: dict[int, Bot], max_steps: int = 400) -> None:
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

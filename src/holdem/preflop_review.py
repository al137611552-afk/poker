"""复盘打完的一手牌里**翻前**那几个决策（FR-9 的翻前那半）。

翻后的 EV 损失要真解局面（`holdem_solver.review`），但**翻前不用**——
离线解好的范围表就在那儿。所以这一层不依赖求解器：没装求解器的机器上，
用户照样拿得到翻前复盘。

```python
for step in review_preflop(hand, hero_seat=3):
    step.verdict          # 明显错误 / 次优 / 照解走
    step.frequency        # 解给这个动作的频率
```

## 打的是频率分，不是 EV 损失

跟 `training` 那边同一条纪律：翻前范围表里只有概率、没有 EV。
**别在界面上把它跟翻后的 EV 损失并排成一列数**——那会让人以为它们是同一把尺子。

## 为什么要重放

打完的 `HandState` 是终局，没法倒带到「英雄第三次说话之前」。所以这里用同一副牌
（`stacked_deck` 照抄底牌与公共牌）**重新打一遍**，每到英雄该说话就先判一次卷，
再照实战的动作走下去。

**重放必须逐动作对齐**：中途只要有一步对不上（座位不符、动作非法），
立刻抛错而不是跳过——跳过的结果是后面的判卷全都在错误的局面上做的，
而界面上还写着「这是你第三次决策」。
"""

from __future__ import annotations

from dataclasses import dataclass

from .actions import Action, bet, call, check, fold, raise_to
from .cards import card_to_str, cards_to_str
from .deck import stacked_deck
from .history import ActionRecord, action_records
from .preflop_policy import PreflopTablePolicy
from .state import PREFLOP, HandConfig, HandState
from .training import Verdict, grade

__all__ = ["PreflopDecision", "review_preflop"]


@dataclass(frozen=True)
class PreflopDecision:
    """英雄的一个翻前决策及其判卷。"""

    index: int
    """这是英雄在这手牌里第几次说话（从 1 起）。"""
    position: str
    pot_before: int
    to_call: int
    action: str
    """实战打出的动作，写给人看。"""
    verdict: "Verdict | None"
    """判卷结果。`None` = 表里没有这一格，**判不了**。"""

    @property
    def graded(self) -> bool:
        return self.verdict is not None

    @property
    def blunder(self) -> bool:
        return self.verdict is not None and self.verdict.blunder


def review_preflop(
    hand: HandState, hero_seat: int, *, policy: "PreflopTablePolicy | None" = None
) -> "list[PreflopDecision]":
    """复盘英雄的翻前决策。牌局没打完就抛错——复盘的对象是打完的牌。"""
    if hand.result is None:
        raise ValueError("这手牌还没打完，复盘不了")

    policy = policy or PreflopTablePolicy()
    replayed = _rebuild(hand)
    out: list[PreflopDecision] = []
    index = 0

    for record in action_records(hand):
        if record.street != PREFLOP:
            break
        action = _action_of(record)
        if replayed.to_act != record.seat:
            raise RuntimeError(
                f"重放对不上：牌谱说座位 {record.seat} 行动，重放到的是 {replayed.to_act}"
            )
        if record.seat == hero_seat:
            index += 1
            out.append(PreflopDecision(
                index=index,
                position=record.position,
                pot_before=record.pot_before,
                to_call=record.to_call,
                action=str(action),
                verdict=grade(replayed, action, policy=policy),
            ))
        replayed.apply(action)
    return out


def _rebuild(hand: HandState) -> HandState:
    """用同一副牌重开一局。底牌与公共牌照抄，其余按默认种子填（用不到）。"""
    config = hand.config
    deck = stacked_deck(
        hole={seat: cards_to_str(cards) for seat, cards in enumerate(hand.hole)},
        board="".join(card_to_str(c) for c in hand.board),
        num_seats=config.num_seats,
        button=config.button,
    )
    return HandState(
        HandConfig(
            stacks=config.stacks,
            button=config.button,
            big_blind=config.big_blind,
            small_blind=config.small_blind,
            ante=config.ante,
        ),
        deck,
    )


def _action_of(record: ActionRecord) -> Action:
    """`ActionRecord` → 可以再打一遍的 `Action`。"""
    if record.kind == "fold":
        return fold()
    if record.kind == "check":
        return check()
    if record.kind == "call":
        return call()
    if record.kind == "bet":
        return bet(record.to)
    if record.kind == "raise":
        return raise_to(record.to)
    raise ValueError(f"不认识的动作：{record.kind}")

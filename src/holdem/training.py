"""场景训练（FR-12）：指定局面反复练，每手即时打分。

想练「BB 面对 BTN 开牌」，就该能连着抽二十手这样的局面，而不是打两百手真牌
碰运气遇上几次。这个模块负责**造题**与**判卷**，纯逻辑、不碰 IO。

```python
spot = deal_open(hero="UTG", rng=rng)      # 造一道「UTG 第一个开牌」
verdict = grade(spot.hand, raise_to(30))   # 判这一手
verdict.frequency                          # 解给这个动作的频率
verdict.best                               # 解最推荐什么
```

## 打的是频率分，不是 EV 损失

翻前范围表里**只有各动作的概率，没有 EV**。所以这里给的是
「解在这个局面下给你选的这个动作多少频率」，**不假装是 EV 损失**——
那是翻后求解器才给得出的东西（`holdem_solver.review`）。

这个区别不是文字游戏：频率 0.3 的动作可能只比最优差一丁点，也可能差很多，
光看频率分不出来。所以判词只分三档（明显错、次优、照解走），不给一个精确的亏损数。
**能给的精度到哪儿，就说到哪儿。**

## 造题：脚本化，不用 bot

前置动作全部写死（该弃的弃、该开的按标准尺度开），不让 bot 参与。
bot 会按自己的风格偏离，于是「同一个场景」每次的前置局面都不同——
那样练的就不是一个场景了。造完用 `identify` 反查一遍，**确认造出来的局面
正是想要的那个**，对不上就抛错而不是将就。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .actions import Action, fold, raise_to
from .deck import deck_from_seed
from .positions import position_of
from .preflop_policy import (
    DEFEND, OPEN, VS_RERAISE, PreflopSpot, PreflopTablePolicy, identify, parse_label,
)
from .state import HandConfig, HandState

__all__ = [
    "TrainingSpot", "Verdict", "deal_open", "deal_defend", "deal_threebet", "grade",
]

SEATS = 6
BIG_BLIND = 10
SMALL_BLIND = 5
STACK = 100 * BIG_BLIND
OPEN_TO = 25          # 2.5bb，六人桌常见开牌尺度
THREEBET_TO = 80      # 8bb


@dataclass(frozen=True)
class TrainingSpot:
    """一道练习题：推进到英雄该说话的牌局 + 它是什么场景。"""

    hand: HandState
    spot: PreflopSpot
    hero_seat: int

    @property
    def label(self) -> str:
        return self.spot.label


@dataclass(frozen=True)
class Verdict:
    """一次判卷。

    **没有 EV 损失这一项**，因为翻前表里没有 EV（见模块说明）。
    """

    frequency: float
    """解给你选的这个动作多少频率（0~1）。"""
    best: str
    """解最推荐的动作标签。"""
    weights: "dict[str, float]"
    """完整分布，给界面画条形图用。"""
    taken: str
    """你选的动作翻成表里的标签。"""

    @property
    def blunder(self) -> bool:
        """解**几乎不这么打**。这是三档里唯一能确定的一档。"""
        return self.frequency < 0.02

    @property
    def on_solution(self) -> bool:
        """就是解最常选的那个。"""
        return self.taken == self.best

    @property
    def verdict(self) -> str:
        if self.blunder:
            return "明显错误：解几乎不这么打"
        if self.on_solution:
            return "照解走"
        return f"次优：解给它 {self.frequency:.0%}，更常选 {self.best}"


def deal_open(hero: str, *, rng: "random.Random | None" = None) -> TrainingSpot:
    """造一道「hero 第一个开牌」：他前面的人全弃。"""
    return _deal(hero, kind=OPEN, rng=rng)


def deal_defend(hero: str, opener: str, *, rng: "random.Random | None" = None) -> TrainingSpot:
    """造一道「hero 面对 opener 开牌」。"""
    return _deal(hero, kind=DEFEND, opener=opener, rng=rng)


def deal_threebet(
    hero: str, reraiser: str, *, rng: "random.Random | None" = None
) -> TrainingSpot:
    """造一道「hero 开牌后面对 reraiser 再加注」。"""
    return _deal(hero, kind=VS_RERAISE, reraiser=reraiser, rng=rng)


def grade(hand: HandState, action: Action, *,
          policy: "PreflopTablePolicy | None" = None) -> "Verdict | None":
    """给一个动作判卷。表里没有这一格就返回 `None`——**说不了就说不了**。"""
    # **先确认这一手打得出来。** 不查的话，用户会"练"一个根本非法的动作
    # （比如面对 2.5bb 开牌时"加注到 2.5bb"），还拿到一句像模像样的判词。
    # 判卷的前提是这确实是个可选项，否则评的不是决策、是笔误。
    if not hand.legal_actions().contains(action):
        raise ValueError(f"这个动作在当前局面不合法：{action}")

    policy = policy or PreflopTablePolicy()
    decision = policy.decide(hand)
    if decision is None:
        return None

    weights = decision.weights
    taken = _match_label(hand, action, weights)
    return Verdict(
        frequency=weights.get(taken, 0.0),
        best=max(weights, key=weights.get),
        weights=dict(weights),
        taken=taken,
    )


# ------------------------------------------------------------------ 造题


def _deal(hero: str, *, kind: str, opener: "str | None" = None,
          reraiser: "str | None" = None, rng: "random.Random | None" = None) -> TrainingSpot:
    rng = rng or random.Random()
    seed = rng.randrange(1 << 30)
    hand = HandState(
        HandConfig(stacks=(STACK,) * SEATS, button=0,
                   big_blind=BIG_BLIND, small_blind=SMALL_BLIND),
        deck_from_seed(seed),
    )
    seat_of = {position_of(seat, 0, SEATS): seat for seat in range(SEATS)}
    hero_seat = _require(seat_of, hero, "英雄")

    if kind == OPEN:
        _advance_until(hand, hero_seat, lambda _: fold())
    elif kind == DEFEND:
        opener_seat = _require(seat_of, opener, "开牌者")
        _advance_until(
            hand, hero_seat,
            lambda seat: raise_to(OPEN_TO) if seat == opener_seat else fold(),
        )
    elif kind == VS_RERAISE:
        reraiser_seat = _require(seat_of, reraiser, "再加注者")
        # 先让英雄开牌，再让指定的人 3bet，中间的人弃牌，转回英雄
        _advance_until(hand, hero_seat, lambda _: fold())
        hand.apply(raise_to(OPEN_TO))
        _advance_until(
            hand, hero_seat,
            lambda seat: raise_to(THREEBET_TO) if seat == reraiser_seat else fold(),
        )
    else:
        raise ValueError(f"不认识的场景：{kind}")

    spot = identify(hand, SEATS)
    if spot is None or spot.kind != kind:
        # **造出来的不是想要的那个就抛错，别将就**：将就的结果是练了半天
        # 练的是另一个场景，而界面上还写着原来那个名字
        raise RuntimeError(f"造题失败：想要 {kind}，`identify` 认出来的是 {spot}")
    return TrainingSpot(hand=hand, spot=spot, hero_seat=hero_seat)


def _advance_until(hand: HandState, hero_seat: int, choose) -> None:
    """推进到轮到英雄。`choose(seat)` 给出每个非英雄座位该做什么。"""
    guard = 0
    while hand.to_act != hero_seat:
        hand.apply(choose(hand.to_act))
        guard += 1
        if guard > 3 * SEATS:
            raise RuntimeError("推不到英雄该说话的位置")


def _require(seat_of: "dict[str, int]", name: "str | None", what: str) -> int:
    if name is None:
        raise ValueError(f"缺少{what}的位置")
    if name not in seat_of:
        raise ValueError(f"没有这个位置：{name}，可选 {sorted(seat_of)}")
    return seat_of[name]


def _match_label(hand: HandState, action: Action, labels) -> str:
    """把引擎的动作对到**表里那一格实际有的**标签上。

    不自己拼标签字符串：表里的标签是中文且带尺度（「加注到 2.5bb」），
    格式是那张表的事，猜一份出来迟早对不上（第一版就是这么错的）。
    改成**反着解析表给的标签**（`parse_label`），跟 `bots.py` 走同一条路。

    加注的金额按大盲比对，容差半个大盲——表里的尺度是离散的几档，
    实战打出的数额只要落在某一档附近就算那一档。
    """
    kind = action.kind.value
    if kind == "check":
        kind = "call"          # 表里没有「过牌」这一支，它与跟注同格

    legal = hand.legal_actions()
    is_allin = kind in ("bet", "raise") and action.to >= legal.max_raise_to
    to_bb = action.to / hand.config.big_blind if action.to else None

    best = None
    for label in labels:
        try:
            label_kind, label_to = parse_label(label)
        except ValueError:
            continue
        if label_kind == "allin" and is_allin:
            return label
        if label_kind != ("raise" if kind in ("bet", "raise") else kind):
            continue
        if label_kind != "raise":
            return label
        if to_bb is not None and label_to is not None and abs(label_to - to_bb) <= 0.5:
            if best is None or abs(label_to - to_bb) < best[0]:
                best = (abs(label_to - to_bb), label)
    if best is not None:
        return best[1]
    # 对不到任何一格＝**解在这个局面下根本没有这个动作**，如实报出来，
    # 频率自然是 0（`Verdict.blunder`）。别硬塞进最近的那一档。
    return f"（表里没有这个动作：{action}）"

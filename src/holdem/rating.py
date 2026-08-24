"""水平评级（FR-14）：测验轨 + 对局轨，对局轨做方差缩减。

## 为什么必须做方差缩减

单手扑克的方差大到离谱：一万手的 bb/100 置信区间常常有 ±10，而高手与鱼的差距
也就 10 bb/100 上下。**不缩减方差，评级量的是运气不是水平**——
这不是精度问题，是「这个数根本没有意义」的问题。

用的是最标准也最有效的一招：**全下摊牌按权益结算，不按实际结果**。
两个人在河牌前推光筹码，谁赢完全看后面几张牌；把那一手记成「按权益应得多少」，
一次性去掉扑克里最大的一块运气。

## 两条轨的分**不合成一个神秘的总分**

- **测验轨**：场景训练答对多少（`training.grade` 的判卷）。
- **对局轨**：方差缩减后的 bb/100。

它们量的是两件事（知不知道 vs 打得怎么样），合成权重没有依据。所以这里
**各给各的分 + 各自的样本量**，合成只在两边都够样本时给，且规则写在明面上。
样本不够就**明说不够**——一个建立在 200 手上的「评级」比没有评级更有害。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .cards import card_to_str, cards_to_str
from .deck import stacked_deck
from .equity import exact_equity
from .evaluator import evaluate
from .history import action_records
from .metrics import bb_per_100
from .state import HandConfig, HandState

__all__ = [
    "AllInSpot", "QuizTrack", "PlayTrack", "Rating",
    "find_allin_showdown", "adjusted_net", "rate",
]

_UNSET = object()
"""「没传」与「传了 None」是两回事：后者表示调用方已经查过、确认没有全下。"""

MIN_QUIZ = 50
"""测验轨至少答这么多题才给分。"""
MIN_HANDS = 2000
"""对局轨至少这么多手才给分。**这个数是下限不是够用**——
两千手的 bb/100 区间仍有 ±20 上下，只是到这儿才谈得上「有个方向」。"""


@dataclass(frozen=True)
class AllInSpot:
    """一手牌里「都推光了、牌还没发完」的那一刻。"""

    board: "tuple[int, ...]"
    """全下时已经翻开的公共牌。"""
    seats: "tuple[int, ...]"
    """还在牌里的座位。"""

    @property
    def cards_left(self) -> int:
        return 5 - len(self.board)


def find_allin_showdown(hand: HandState) -> "AllInSpot | None":
    """找出这手牌是不是「推光了还要发牌」，是的话给出那一刻的公共牌。

    返回 `None` 的情况都不是错误：没人全下、全下时牌已发完、或者只剩一个人
    （别人弃了——那手底池没有运气成分可去）。

    **只认两人**：多人全下要按边池逐层算权益，那是另一套账；算错了会凭空造钱，
    比不调整更糟。多人的那手原样计入，并在 `PlayTrack.skipped_multiway` 里报出来。
    """
    if hand.result is None:
        raise ValueError("这手牌还没打完")

    replayed = _rebuild(hand)
    for record in action_records(hand):
        action = _action_of(record)
        if replayed.to_act != record.seat:
            raise RuntimeError("重放对不上，评级不能建在错位的牌局上")

        board_before = tuple(replayed.board)
        replayed.apply(action)

        # 判据是**公共牌在一步之内从不足五张跳到五张**——那只可能是引擎因为
        # 「没人还能下注了」把剩下的街一次发完，也就是全下摊牌。
        #
        # 前两版都栽在别的判据上：
        # ① 看 apply 之后的 `board`——全发完了，`len(board) < 5` 永远不成立；
        # ② 看 `stacks` 有没有归零——最后那一步 apply 完牌局**已经结算**，
        #    赢家的 stacks 早加上了底池，怎么看都不像全下。
        # 这条判据不碰筹码也不碰行动人，只看牌是怎么发出来的。
        jumped_to_river = len(board_before) < 5 and len(replayed.board) == 5
        contenders = replayed.contenders()
        if jumped_to_river and replayed.is_complete and len(contenders) >= 2:
            return AllInSpot(board=board_before, seats=tuple(contenders))
        if replayed.is_complete:
            break
    return None


def adjusted_net(
    hand: HandState, seat: int, *, spot: "AllInSpot | None | object" = _UNSET
) -> "tuple[float, bool]":
    """这手牌该记多少净额，以及**是否做了调整**。

    没有全下摊牌就原样返回实际净额（`adjusted=False`）。
    有的话按权益重算：`各层边池 × 该层权益 − 自己投进去的钱`。

    `spot` 可以由调用方传进来复用——找全下点要把整手牌重放一遍，
    六个座位各查一次就要重放六遍。
    """
    if hand.result is None:
        raise ValueError("这手牌还没打完")

    actual = float(hand.result.net[seat])
    if spot is _UNSET:
        spot = find_allin_showdown(hand)
    if spot is None or seat not in spot.seats:
        return actual, False

    # **按边池逐层分**。只做两人的话方差几乎降不下来：六人桌上多人全下比两人全下
    # 多一个数量级（实测 3000 手里 223 vs 20），跳过它们等于这个功能白做。
    #
    # 每层底池只在 `eligible` 那几个人之间分，权益也只在他们之间算——
    # 拿全场权益去分边池会让筹码少的人分到他根本够不着的钱。
    shares = _showdown_shares(hand, spot)
    won = sum(pot.amount * shares[pot.eligible].get(seat, 0.0)
              for pot in hand.result.pots)
    contributed = hand.result.contributions[seat] - hand.result.refunds[seat]
    return won - contributed, True


def _showdown_shares(hand: HandState, spot: AllInSpot) -> dict:
    """每一组 `eligible` 各自的权益分配。返回 `{eligible 元组: {座位: 份额}}`。

    **一组只算一次**并缓存：同一层底池会被每个参与者各查一次，每次重抽样的话
    份额加起来不等于 1，调整后的净额就不守恒——那等于凭空造钱或蒸发钱
    （两人版就实测差过 11.6 个筹码）。
    """
    cache: dict = {}
    for pot in hand.result.pots:
        group = pot.eligible
        if group in cache:
            continue
        live = [s for s in group if s in spot.seats]
        if len(live) < 2:
            cache[group] = {live[0]: 1.0} if live else {}
            continue
        cache[group] = _equity_shares(
            {s: hand.hole[s] for s in live}, spot.board
        )
    return cache


def _equity_shares(holes: dict, board) -> dict:
    """多人权益：每人赢下的期望份额（平局按人数均分），加起来正好是 1。

    两人且缺 ≤2 张时走穷举（精确、不到 10ms）；其余抽样。
    **种子由牌派生**：同一手牌重算必须得到同一个数，否则昨天的评级和今天重算的
    对不上，而两次都说自己是对的。
    """
    seats = sorted(holes)
    if len(seats) == 2 and len(board) >= 3:
        low, high = seats
        share = exact_equity(list(holes[low]), list(holes[high]), list(board))
        return {low: share, high: 1.0 - share}

    known = [c for cards in holes.values() for c in cards] + list(board)
    seed = hash((tuple(sorted(known)), len(board))) & 0xFFFFFFFF
    rng = random.Random(seed)
    unseen = [c for c in range(52) if c not in set(known)]
    need = 5 - len(board)

    tally = {s: 0.0 for s in seats}
    samples = 2000
    for _ in range(samples):
        rest = rng.sample(unseen, need) if need else []
        full = list(board) + rest
        scores = {s: evaluate(list(holes[s]) + full) for s in seats}
        best = max(scores.values())
        winners = [s for s in seats if scores[s] == best]
        for s in winners:
            tally[s] += 1.0 / len(winners)
    return {s: tally[s] / samples for s in seats}


@dataclass(frozen=True)
class QuizTrack:
    """测验轨：场景训练答得怎么样。"""

    answered: int
    on_solution: int
    blunders: int

    @property
    def enough(self) -> bool:
        return self.answered >= MIN_QUIZ

    @property
    def accuracy(self) -> "float | None":
        return self.on_solution / self.answered if self.answered else None

    @property
    def blunder_rate(self) -> "float | None":
        return self.blunders / self.answered if self.answered else None

    @property
    def score(self) -> "float | None":
        """0~100。**样本不够就没有分**，不给一个「暂定分」。"""
        if not self.enough or self.accuracy is None:
            return None
        # 明显错误扣双倍：选到解不推荐的动作，比选到次优动作严重得多
        raw = self.accuracy - (self.blunder_rate or 0.0)
        return max(0.0, min(100.0, raw * 100.0))


@dataclass(frozen=True)
class PlayTrack:
    """对局轨：方差缩减后的 bb/100。"""

    hands: int
    adjusted_bb100: float
    raw_bb100: float
    adjusted_hands: int
    """其中做了全下权益调整的手数。"""
    allin_without_hero: int
    """有人全下摊牌、但**英雄没参与**的手数。

    这不是「漏调整」——英雄没进那个池子，那手对他没有运气成分可去。
    单独报出来只是为了让人看懂 `adjusted_hands` 为什么不等于全部全下手数
    （第一版这个字段叫 `skipped_multiway`，读起来像「有一批没处理」，是误导）。
    """
    big_blind: int

    @property
    def enough(self) -> bool:
        return self.hands >= MIN_HANDS

    @property
    def score(self) -> "float | None":
        """把 bb/100 折成 0~100。−20 及以下 0 分，+20 及以上 100 分，中间线性。

        **这条折算是拍的**，只为让两条轨能放在一起看；
        真要比强弱请直接读 `adjusted_bb100`，那才是有单位的量。
        """
        if not self.enough:
            return None
        return max(0.0, min(100.0, (self.adjusted_bb100 + 20.0) / 40.0 * 100.0))


@dataclass(frozen=True)
class Rating:
    quiz: QuizTrack
    play: PlayTrack

    @property
    def score(self) -> "float | None":
        """总评级。**两条轨都够样本才有**，权重各半。

        为什么不在一条够的时候先给个分：那个分会被当成「我的水平」，
        而它其实只反映了一半——知道怎么打和真打得好是两件事。
        """
        quiz, play = self.quiz.score, self.play.score
        if quiz is None or play is None:
            return None
        return (quiz + play) / 2

    @property
    def why(self) -> str:
        if self.score is not None:
            return "两条轨样本都够，评级有效"
        missing = []
        if self.quiz.score is None:
            missing.append(f"测验轨还差 {max(0, MIN_QUIZ - self.quiz.answered)} 题")
        if self.play.score is None:
            missing.append(f"对局轨还差 {max(0, MIN_HANDS - self.play.hands)} 手")
        return "样本不够，先不给评级：" + "、".join(missing)


def rate(*, quiz: QuizTrack, hands, seat: int, big_blind: int) -> Rating:
    """把答题记录与一批打完的牌折成评级。`hands` 是 `HandState` 的序列。"""
    total_adjusted = 0.0
    total_raw = 0
    played = 0
    adjusted_hands = 0
    multiway = 0

    for hand in hands:
        played += 1
        total_raw += hand.result.net[seat]
        # 全下点只找一次：找它要把整手牌重放一遍，别在循环里重复付这个代价
        spot = find_allin_showdown(hand)
        value, did = adjusted_net(hand, seat, spot=spot)
        total_adjusted += value
        if did:
            adjusted_hands += 1
        elif spot is not None:
            multiway += 1

    play = PlayTrack(
        hands=played,
        adjusted_bb100=bb_per_100(total_adjusted, played, big_blind) if played else 0.0,
        raw_bb100=bb_per_100(total_raw, played, big_blind) if played else 0.0,
        adjusted_hands=adjusted_hands,
        allin_without_hero=multiway,
        big_blind=big_blind,
    )
    return Rating(quiz=quiz, play=play)


# ------------------------------------------------------------------ 重放


def _rebuild(hand: HandState) -> HandState:
    config = hand.config
    return HandState(
        HandConfig(
            stacks=config.stacks, button=config.button, big_blind=config.big_blind,
            small_blind=config.small_blind, ante=config.ante,
        ),
        stacked_deck(
            hole={seat: cards_to_str(cards) for seat, cards in enumerate(hand.hole)},
            board="".join(card_to_str(c) for c in hand.board),
            num_seats=config.num_seats,
            button=config.button,
        ),
    )


def _action_of(record):
    from .actions import bet, call, check, fold, raise_to

    return {
        "fold": lambda: fold(), "check": lambda: check(), "call": lambda: call(),
        "bet": lambda: bet(record.to), "raise": lambda: raise_to(record.to),
    }[record.kind]()

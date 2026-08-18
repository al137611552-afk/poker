"""把解出来的翻前范围表接到活的牌局上。

范围表（`preflop_ranges.py`）是按「谁开牌、谁应对」组织的；牌桌上给到的是一个
`HandState`。这里做三件事：**认出局面**、**查表**、**把表里的标签翻译成引擎的动作**。

## 表覆盖哪些局面

| 局面 | 表里有吗 |
|---|---|
| 前面全弃，轮到我第一个开牌 | ✅ 开牌范围 |
| 一个人开牌，中间没人跟，轮到我应对 | ✅ 面对开牌 |
| 我开牌被 3bet，轮到我应对 | ✅ 面对再加注 |
| **跛入局、多人底池、4bet 之后、我已经跟过一手** | ❌ 一律回 `None`，交给兜底策略 |

## 哪张表

随包有两张：六人 100bb 与单挑 200bb（schema 相同）。`PreflopPolicySet` **按桌上人数
分发**——人数没有对应产物就回 `None`，交给兜底。

**深度也要对得上**：范围表是按某个筹码深度解出来的，20bb 的桌子照 100bb 的表打是错的
（那个深度该走推/弃）。所以深度差出**一倍以外**就不查表（`DEPTH_BAND`）。差一倍以内
照用，是因为翻前范围对深度没那么敏感，而「有解可用」比「刚好那个深度」更重要。

**回 `None` 不是缺陷是纪律**：表是按「一个开牌者 + 一个防守者」解出来的（ADR-0004），
拿它去指导多人底池等于拿一份不适用的解冒充精确解，违反 PRD 的「诚实」这条非功能需求。

## 尺度对不上怎么办

表里的加注额是模型的抽象尺度（开到 2.5bb、3bet 到 7.5bb）。真实牌局里对手可能开到
3bb，我们仍然按「这是一个开牌」去查表，只是把动作**按当前合法区间**换算。这是刻意的
近似：动作序列的**类型**比尺度更能决定范围，而尺度不同带来的偏差远小于「没有解可用」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from functools import lru_cache

from .equity_table import equity_vs_range
from .history import action_records
from .positions import position_of
from .preflop_ranges import PRODUCTS, PreflopRangeTable, is_available, load, load_all
from .ranges import NUM_HAND_CLASSES, Range, class_combo_count, class_of
from .realization import RealizationModel, realization_factors
from .state import PREFLOP, HandState

__all__ = [
    "DEPTH_BAND",
    "PreflopSpot",
    "PolicyDecision",
    "PreflopTablePolicy",
    "PreflopPolicySet",
    "effective_depth",
    "parse_label",
]

_CLASSES = range(NUM_HAND_CLASSES)
_COMBOS = tuple(class_combo_count(i) for i in _CLASSES)
_TOTAL_COMBOS = float(sum(_COMBOS))

LIMP = "跟注到1"
"""风格层给怂风格补的跛入动作。解里没有这一支，见 `_tilt`。"""

OPEN = "开牌"
DEFEND = "面对开牌"
VS_RERAISE = "面对再加注"

DEPTH_BAND = (0.5, 2.0)
"""实际深度 / 表的深度，落在这个区间才查表。两端都是「差一倍」。"""


@dataclass(frozen=True)
class PreflopSpot:
    """认出来的局面。`opener` 是开牌者的位置名，开牌局面下就是自己。"""

    kind: str
    hero: str
    opener: str
    reraiser: str | None = None
    """3bet 的是谁。**面对再加注时必须有**——表里的应对是按「谁 3bet 的」分格存的
    （`defense(开牌者, 3bet者).reraise_reply`），少了它就查不到那一格。"""

    @property
    def label(self) -> str:
        if self.kind == OPEN:
            return f"{self.hero} 第一个开牌"
        if self.kind == DEFEND:
            return f"{self.hero} 面对 {self.opener} 开牌"
        return f"{self.hero} 开牌后面对 {self.reraiser} 再加注"


@dataclass(frozen=True)
class PolicyDecision:
    """查表结果：这手牌在这个局面下各动作的概率。"""

    spot: PreflopSpot
    weights: dict[str, float]
    """动作标签 → 概率，已按当前牌类归一化，总和为 1。"""

    def most_likely(self) -> str:
        return max(self.weights, key=self.weights.get)


# ------------------------------------------------------------------ 标签


_RAISE_TO = re.compile(r"加注到([0-9.]+)")


def parse_label(label: str) -> tuple[str, float | None]:
    """把表里的中文标签翻成 (动作类型, 加注到多少大盲)。"""
    if label.startswith("弃"):
        return "fold", None
    if label.startswith("跟注") or label.startswith("过牌"):
        return "call", None
    if label.startswith("全下"):
        return "allin", None
    match = _RAISE_TO.search(label)
    if match:
        return "raise", float(match.group(1))
    raise ValueError(f"无法识别的动作标签: {label!r}")


# ------------------------------------------------------------------ 认局面


def identify(hand: HandState, table_seats: int) -> PreflopSpot | None:
    """认出当前决策点属于哪个局面；表覆盖不到的一律回 None。"""
    if hand.street != PREFLOP or hand.is_complete:
        return None
    config = hand.config
    if config.num_seats != table_seats:
        return None  # 表是按固定人数解的，人数不同就别硬套

    seat = hand.to_act
    hero = position_of(seat, config.button, config.num_seats)
    records = [r for r in action_records(hand) if r.street == PREFLOP]

    raises = [r for r in records if r.kind == "raise"]
    voluntary_calls = [r for r in records if r.kind == "call"]
    mine = [r for r in records if r.seat == seat]

    if not raises:
        # 没人加注：只有「前面全弃」才是表里的开牌局面，有人跛入就不是
        if voluntary_calls or mine:
            return None
        return PreflopSpot(OPEN, hero, hero)

    if len(raises) == 1:
        if mine or voluntary_calls:
            return None  # 我已经动过，或者已经有人冷跟（多人底池，表里没有）
        opener = raises[0].position
        return PreflopSpot(DEFEND, hero, opener)

    if len(raises) == 2:
        opener, reraiser = raises[0], raises[1]
        if opener.seat != seat or voluntary_calls:
            return None
        if [r for r in mine if r.seq > reraiser.seq]:
            return None  # 已经应对过了
        return PreflopSpot(VS_RERAISE, hero, hero, reraiser=reraiser.position)

    return None  # 4bet 之后的局面表里没有


# ------------------------------------------------------------------ 查表


def effective_depth(hand: HandState) -> float:
    """还在牌里的人中最浅的那份**起始**筹码，折成大盲。

    用起始筹码而不是当前剩余：范围表描述的是「这手牌开始时有多深」，
    中途投出去的钱不该让深度看起来变浅。
    """
    config = hand.config
    alive = hand.contenders() or range(config.num_seats)
    return min(config.stacks[seat] for seat in alive) / config.big_blind


class PreflopTablePolicy:
    """按解出来的范围表给出翻前策略。表缺这个局面时回 `None`。"""

    def __init__(self, table: PreflopRangeTable | None = None) -> None:
        self.table = table if table is not None else load()
        self.seats = self.table.num_players
        self.stack_bb = self.table.stack_bb
        # 开牌那一支的标签跟着表里的开牌尺度走，别写死 2.5——换尺度重算的表要能直接用
        self.open_label = f"加注到{float(self.table.table['open_to']):g}"
        self._rankings: dict[tuple, _Ranking] = {}

    @classmethod
    def available(cls) -> bool:
        return is_available()

    def decide(
        self, hand: HandState, *, looseness: float = 1.0, aggression: float = 1.0
    ) -> PolicyDecision | None:
        """给出这手牌的动作分布。`looseness`/`aggression` 是风格层的两个旋钮，
        都以 1.0 表示「照解走」。"""
        spot = identify(hand, self.seats)
        if spot is None:
            return None
        ratio = effective_depth(hand) / self.stack_bb
        if not DEPTH_BAND[0] <= ratio <= DEPTH_BAND[1]:
            return None  # 深度差太多，这张表不适用——交给兜底

        index = class_of(*hand.hole[hand.to_act])
        weights = self._weights(spot, index)
        if weights is None:
            return None
        if looseness != 1.0 or aggression != 1.0:
            weights = self._tilt(spot, index, weights, looseness, aggression)
        return PolicyDecision(spot=spot, weights=weights)

    # -------------------------------------------------------------- 风格层

    def _tilt(
        self,
        spot: PreflopSpot,
        index: int,
        weights: dict[str, float],
        looseness: float,
        aggression: float,
    ) -> dict[str, float]:
        """把解出来的策略按风格拧松/拧紧、拧凶/拧怂。

        **松紧不能用乘法**：解里 100% 弃掉的牌，概率乘多少还是 0，于是「松」这个风格
        永远打不出解以外的牌。这里改成**沿排序移动阈值**——把 169 个牌类按求解器算出的
        逐手 EV 排好（表里没存 EV 的老表退回权益×兑现系数的估分），再把入池频率整体拉到
        目标值。这样既能真的放宽，又保住了解给出的先后顺序。
        """
        fold_label = _fold_label(weights)
        if fold_label is None:
            return weights  # 没有弃牌这个选项（比如能免费过牌），松紧无从谈起

        ranking = self._ranking(spot)
        base_frequency = ranking.frequency
        target = min(1.0, max(0.0, base_frequency * looseness))
        play = ranking.play_probability(index, target)

        others = {label: value for label, value in weights.items() if label != fold_label}

        if spot.kind == OPEN:
            # 开牌局面里，解只给了「开牌或弃牌」两条路——它不含跛入（ADR-0004 的树里
            # 子博弈刻意关掉了跛入）。可是**跟注站在首位是会跛入的**，只按解走会让
            # 一个 aggression=0.15 的风格照样一路开牌，PFR 高得不像话。
            # 所以这里由风格层补上跛入：怂的风格把一部分「该入池」的牌改成平跟。
            # 这是风格层的行为，不是解的一部分——别把它当成求出来的策略。
            raise_share = min(1.0, aggression)
            result = {fold_label: 1.0 - play}
            # 解里从不开牌的牌（权重全 0）归一化会得到 None——那正是被风格放宽纳进来的
            # 那一批，按等分处理即可
            portions = _normalize(others) or {
                label: 1.0 / len(others) for label in others
            }
            for label, portion in portions.items():
                result[label] = play * raise_share * portion
            if raise_share < 1.0:
                result[LIMP] = result.get(LIMP, 0.0) + play * (1.0 - raise_share)
            return result

        share = _aggression_split(others, ranking.average_split, aggression)
        result = {fold_label: 1.0 - play}
        for label, portion in share.items():
            result[label] = play * portion
        return result

    def _ranking(self, spot: PreflopSpot) -> "_Ranking":
        key = (spot.kind, spot.opener, spot.hero, spot.reraiser)
        if key not in self._rankings:
            self._rankings[key] = self._build_ranking(spot)
        return self._rankings[key]

    def _build_ranking(self, spot: PreflopSpot) -> "_Ranking":
        play = [0.0] * NUM_HAND_CLASSES
        split_numerator: dict[str, float] = {}
        for index in _CLASSES:
            weights = self._weights(spot, index)
            if weights is None:
                continue
            fold_label = _fold_label(weights)
            value = 1.0 - weights.get(fold_label, 0.0) if fold_label else 1.0
            play[index] = value
            for label, probability in weights.items():
                if label != fold_label:
                    split_numerator[label] = split_numerator.get(label, 0.0) + probability * _COMBOS[index]
        total = sum(split_numerator.values())
        average = (
            {label: value / total for label, value in split_numerator.items()}
            if total > 0
            else {}
        )
        return _Ranking(play=tuple(play), average_split=average, score=self._score(spot))

    def _score(self, spot: PreflopSpot) -> tuple[float, ...] | None:
        """求解器算出的逐牌类价值：开牌局面用开牌 EV，防守局面用「继续比弃牌好多少」。"""
        if spot.kind == OPEN:
            return self.table.open_ev.get(spot.hero)
        if spot.kind == DEFEND:
            try:
                return self.table.defense(spot.opener, spot.hero).advantage
            except KeyError:
                return None
        return None

    # -------------------------------------------------------------- 内部

    def _weights(self, spot: PreflopSpot, index: int) -> dict[str, float] | None:
        if spot.kind == OPEN:
            try:
                opened = self.table.open_range(spot.hero).weight(index)
            except KeyError:
                return None
            return {self.open_label: opened, "弃牌": 1.0 - opened}

        # 面对开牌查的是「我作为防守者」那一格；面对再加注查的是「我开牌、他 3bet」
        # 那一格里存的应对——**两者的键不一样**，取错了会一路查不到而悄悄退回兜底
        key = (
            (spot.opener, spot.hero)
            if spot.kind == DEFEND
            else (spot.hero, spot.reraiser)
        )
        if key[1] is None:
            return None
        try:
            entry = self.table.defense(*key)
        except KeyError:
            return None

        if spot.kind == DEFEND:
            raw = {label: rng.weight(index) for label, rng in entry.actions.items()}
        else:
            if not entry.reraise_reply:
                return None
            raw = {label: rng.weight(index) for label, rng in entry.reraise_reply.items()}

        return _normalize(raw)

    def defenders_of(self, opener: str) -> tuple[str, ...]:
        return self.table.defenders_of(opener)


@lru_cache(maxsize=1)
def _strength() -> tuple[float, ...]:
    """牌类的强弱打分：对随机手牌的权益 × 兑现系数。

    这个分数决定**解之外的牌按什么顺序被放宽风格纳进来**，所以不能只用生权益——
    那样同花连张永远排在垃圾高张后面，「松」风格会放进 K9o 却放不进 65s，与真人的
    松手完全不像。乘上 `realization.py` 的兑现系数（同花/连张/对子的加成）就把这层
    补回来了，而且用的是项目自己的模型，不另造一套启发式。
    """
    everything = Range.full()
    model = RealizationModel()
    factors = realization_factors(model, in_position=True, spr=model.reference_spr)
    return tuple(
        equity_vs_range(index, everything) * factors[index] for index in _CLASSES
    )


class _Ranking:
    """一个局面下「解有多愿意用这手牌入池」的排序。

    排序与累计组合数在构造时算好：每个决策点都要用它，边用边排会把自对弈拖慢一个量级。
    """

    __slots__ = ("play", "average_split", "frequency", "_offset", "from_solver")

    def __init__(
        self,
        play: tuple[float, ...],
        average_split: dict[str, float],
        score: tuple[float, ...] | None = None,
    ) -> None:
        self.play = play
        self.average_split = average_split
        self.frequency = sum(play[i] * _COMBOS[i] for i in _CLASSES) / _TOTAL_COMBOS

        # 优先按**求解器算出的每手 EV** 排序：它覆盖全部 169 类（包括解里不打的那些），
        # 天然回答了「再放宽一档该先纳进谁」。表里没存 EV 时才退回权益×兑现系数的估分。
        strength = _strength()
        if score is not None:
            order = sorted(_CLASSES, key=lambda i: -score[i])
        else:
            order = sorted(_CLASSES, key=lambda i: (-play[i], -strength[i]))
        offset = [0.0] * NUM_HAND_CLASSES
        cumulative = 0.0
        for index in order:
            offset[index] = cumulative
            cumulative += _COMBOS[index]
        self._offset = tuple(offset)
        self.from_solver = score is not None

    def play_probability(self, index: int, target_frequency: float) -> float:
        """把入池频率整体拉到 `target_frequency` 之后，这手牌该以多大概率入池。

        排在目标频率之内的牌满额入池，跨在边界上的那一类按比例混合，之后的一律不入池。
        """
        budget = target_frequency * _TOTAL_COMBOS - self._offset[index]
        if budget <= 0:
            return 0.0
        return min(1.0, budget / _COMBOS[index])


def _fold_label(weights: dict[str, float]) -> str | None:
    for label in weights:
        if parse_label(label)[0] == "fold":
            return label
    return None


def _aggression_split(
    others: dict[str, float], average: dict[str, float], aggression: float
) -> dict[str, float]:
    """入池之后，钱往「加注」还是「跟注」偏。1.0 表示照解走。"""
    base = others if sum(others.values()) > 1e-9 else average
    if not base:
        return {"跟注": 1.0}
    tilted = {}
    for label, value in base.items():
        kind = parse_label(label)[0]
        tilted[label] = value * (aggression if kind in ("raise", "allin") else 1.0)
    total = sum(tilted.values())
    if total <= 0:
        return {label: 1.0 / len(base) for label in base}
    return {label: value / total for label, value in tilted.items()}


def _normalize(raw: dict[str, float]) -> dict[str, float] | None:
    """把「到达概率 × 动作频率」还原成这手牌的条件策略。

    表里存的是**乘过到达概率**的范围（这样画图与统计才对），所以要除以这手牌到达
    该节点的总权重才是「若我拿着它，该怎么打」。总权重为 0 说明这手牌根本走不到
    这个节点——上一层就没这么打过，此时没有可用的解，回 None 让兜底策略接手。
    """
    total = sum(raw.values())
    if total <= 1e-9:
        return None
    return {label: value / total for label, value in raw.items()}


class PreflopPolicySet:
    """随包的几张范围表凑成一套策略：**按桌上人数分发**。

    六人桌查六人表、单挑查单挑表；没有对应人数的产物就回 `None`，交给兜底策略。
    对外的接口与单张表的 `PreflopTablePolicy` 一样，所以 `bots.Bot` 拿到哪一种都能用。
    """

    def __init__(self, policies: "dict[int, PreflopTablePolicy] | None" = None) -> None:
        if policies is None:
            policies = {
                seats: PreflopTablePolicy(table)
                for seats, table in load_all().items()
            }
        self.policies = policies

    @classmethod
    def available(cls) -> bool:
        return any(path.exists() for path in PRODUCTS)

    def for_seats(self, seats: int) -> PreflopTablePolicy | None:
        return self.policies.get(seats)

    def decide(
        self, hand: HandState, *, looseness: float = 1.0, aggression: float = 1.0
    ) -> PolicyDecision | None:
        policy = self.policies.get(hand.config.num_seats)
        if policy is None:
            return None
        return policy.decide(hand, looseness=looseness, aggression=aggression)

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{seats}人{policy.stack_bb:g}bb" for seats, policy in sorted(self.policies.items())
        )
        return f"PreflopPolicySet({parts or '空'})"

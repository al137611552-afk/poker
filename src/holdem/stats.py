"""牌局统计（FR-7）：口径与 PT4/HM3 对齐。

## 为什么每个指标都是「机会 / 发生」成对的

PT4 那套指标的坑几乎全在**分母**上。「3bet 12%」这个数，分母是「所有手」还是
「面对加注的手」，差出一个数量级；而两边都能算出一个看着合理的百分比。
所以这里每个指标都存成一对计数（`Chance`），**分子分母用同一个判据算出来**，
判据写在各自的注释里。`batch.py` 早就吃过这个亏（"跟在跛入者后面的加注是隔离，
不是开牌"），那条教训在这里被推广成结构。

## 与 `batch.py` 的关系

`batch.py` 里有一份边跑边累计的统计，口径是这份的子集。**两份口径必须收敛成一份**，
否则迟早漂开——但那是下一段的事（改 `batch.py` 要连着它的分片合并一起动）。
在合并之前，这里**不许**悄悄改动那几个既有指标的判据：历史跑出来的数会变得不可比。

## 位置

每个指标都能按位置拆开（`by_position`）。这不是锦上添花：
「开牌率」这类指标**混在一起的平均值会随桌上对手的松紧漂移**——对手越紧，
第一个入池的机会越多落在靠后的位置上，平均开牌率跟着涨，而打法一点没变。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .history import action_records
from .state import PREFLOP

__all__ = ["Chance", "StatLine", "accumulate", "hand_stats"]


@dataclass
class Chance:
    """一个指标：有过多少次机会、其中发生了多少次。

    **没有机会就没有这个指标**——`rate` 返回 `None` 而不是 0。
    「0 次机会」和「有机会但一次没做」是两件完全不同的事，压成 0% 会让
    「这人从没面对过 3bet」读成「这人面对 3bet 从不弃牌」。
    """

    chances: int = 0
    hits: int = 0

    def observe(self, happened: bool) -> None:
        self.chances += 1
        self.hits += int(happened)

    def add(self, other: "Chance") -> None:
        self.chances += other.chances
        self.hits += other.hits

    @property
    def rate(self) -> "float | None":
        return self.hits / self.chances if self.chances else None


@dataclass
class StatLine:
    """一个座位（或一个座位在某个位置上）的一组统计。"""

    hands: int = 0

    # ---- 翻前 ----
    vpip: Chance = field(default_factory=Chance)
    """主动投钱 / 所有手。盲注不算主动（`ActionRecord.is_voluntary`）。"""
    pfr: Chance = field(default_factory=Chance)
    """翻前加注 / 所有手。"""
    rfi: Chance = field(default_factory=Chance)
    """第一个入池就加注 / **前面全弃**的手。分母不是「所有手」。

    跟在跛入者后面的加注是**隔离**，不算 RFI——分子分母同一个判据。
    """
    threebet: Chance = field(default_factory=Chance)
    """面对（且只面对过一次）加注时再加注 / 那种机会。"""
    fold_to_threebet: Chance = field(default_factory=Chance)
    """自己开牌后被 3bet 时弃牌 / 那种机会。**分母是「我开了牌又被 3bet」**，
    不是「所有被 3bet」——没开牌的人被 3bet 是另一回事。"""

    # ---- 翻后 ----
    cbet_flop: Chance = field(default_factory=Chance)
    """翻前进攻方在翻牌下注 / 「翻前进攻方 + 看到翻牌 + 轮到自己时无人先下注」。

    分母里那第三条不能省：被人抢先下注时你**没有**持续下注的机会，
    把它算进分母等于把「没机会」记成「放弃了」。
    """
    fold_to_cbet_flop: Chance = field(default_factory=Chance)
    """面对翻牌持续下注时弃牌 / 面对它的次数。"""
    wtsd: Chance = field(default_factory=Chance)
    """看到翻牌之后走到摊牌 / 看到翻牌。"""
    wsd: Chance = field(default_factory=Chance)
    """摊牌赢下 / 摊牌。"""

    # ---- 攻击性 ----
    postflop_aggressive: int = 0
    """翻后的下注与加注次数。"""
    postflop_calls: int = 0
    """翻后的跟注次数。"""

    def add(self, other: "StatLine") -> None:
        self.hands += other.hands
        for name in _CHANCE_FIELDS:
            getattr(self, name).add(getattr(other, name))
        self.postflop_aggressive += other.postflop_aggressive
        self.postflop_calls += other.postflop_calls

    @property
    def aggression_factor(self) -> "float | None":
        """AF =（下注＋加注）/ 跟注，翻后。

        **一次没跟注过时返回 `None`，不返回无穷大**：那种样本量下这个数没有意义，
        给个 `inf` 只会在报告里排到第一名。
        """
        if not self.postflop_calls:
            return None
        return self.postflop_aggressive / self.postflop_calls


_CHANCE_FIELDS = (
    "vpip", "pfr", "rfi", "threebet", "fold_to_threebet",
    "cbet_flop", "fold_to_cbet_flop", "wtsd", "wsd",
)

_FLOP = PREFLOP + 1


def hand_stats(hand) -> "dict[int, StatLine]":
    """一手牌 → 每个座位的统计。纯逻辑，不碰 IO。"""
    into: dict[int, StatLine] = {}
    accumulate(hand, into)
    return into


def accumulate(hand, into: "dict[int, StatLine]", *, by_position: bool = False) -> None:
    """把一手牌折进 `into`。`by_position=True` 时按 `(座位, 位置)` 分组。

    位置取**翻前**那条记录上的（`ActionRecord.position`）——一手牌里位置不会变，
    但只有翻前每个人必然有记录，翻后弃了牌的人就没有了。
    """
    if hand.result is None:
        # **说不了就说不了**：没打完的牌算不出摊牌类指标，硬算会把「还没到摊牌」
        # 记成「没走到摊牌」，WTSD 会被系统性地压低。宁可让调用方看见这句话。
        raise ValueError("这手牌还没打完，统计不了（`hand.result` 是 None）")

    records = action_records(hand)
    seats = hand.config.num_seats
    positions = _positions(records, seats)

    def line(seat: int) -> StatLine:
        key = (seat, positions.get(seat)) if by_position else seat
        return into.setdefault(key, StatLine())

    pre = _preflop_pass(records, seats)
    post = _flop_pass(records, seats, pre.last_raiser)

    saw_flop = len(hand.board) >= 3
    showdown_seats = hand.result.showdown_scores
    net = hand.result.net

    for seat in range(seats):
        stats = line(seat)
        stats.hands += 1
        stats.vpip.observe(pre.voluntary[seat])
        stats.pfr.observe(pre.raised[seat])
        if pre.open_chance[seat]:
            stats.rfi.observe(pre.opened[seat])
        if pre.threebet_chance[seat]:
            stats.threebet.observe(pre.threebet[seat])
        if pre.faced_threebet[seat]:
            stats.fold_to_threebet.observe(pre.folded_to_threebet[seat])

        if post.cbet_chance[seat]:
            stats.cbet_flop.observe(post.cbet[seat])
        if post.faced_cbet[seat]:
            stats.fold_to_cbet_flop.observe(post.folded_to_cbet[seat])

        reached_flop = saw_flop and not pre.folded_preflop[seat]
        if reached_flop:
            stats.wtsd.observe(seat in showdown_seats)
        if seat in showdown_seats:
            stats.wsd.observe(net[seat] > 0)

        stats.postflop_aggressive += post.aggressive[seat]
        stats.postflop_calls += post.calls[seat]


# ------------------------------------------------------------------ 两遍扫描


@dataclass
class _Preflop:
    voluntary: list
    raised: list
    opened: list
    open_chance: list
    threebet: list
    threebet_chance: list
    faced_threebet: list
    folded_to_threebet: list
    folded_preflop: list
    last_raiser: "int | None"


def _preflop_pass(records, seats: int) -> _Preflop:
    out = _Preflop(*([False] * seats for _ in range(9)), last_raiser=None)
    for name in vars(out):
        if name != "last_raiser":
            setattr(out, name, [False] * seats)

    raises = 0
    entered = False
    opener: "int | None" = None

    for record in records:
        if record.street != PREFLOP:
            continue
        seat = record.seat
        first_in = not entered

        if first_in:
            out.open_chance[seat] = True
        elif raises == 1 and record.to_call > 0:
            out.threebet_chance[seat] = True

        # 我开了牌、现在面对的是 3bet：这一下决定 fold-to-3bet
        if opener == seat and raises >= 2 and record.to_call > 0:
            out.faced_threebet[seat] = True
            if record.kind == "fold":
                out.folded_to_threebet[seat] = True

        if record.kind == "fold":
            out.folded_preflop[seat] = True
        if record.is_voluntary:
            out.voluntary[seat] = True
            entered = True
        if record.kind in ("bet", "raise"):
            out.raised[seat] = True
            if first_in:
                out.opened[seat] = True
                opener = seat
            elif raises == 1:
                out.threebet[seat] = True
            raises += 1
            out.last_raiser = seat

    return out


@dataclass
class _Flop:
    cbet: list
    cbet_chance: list
    faced_cbet: list
    folded_to_cbet: list
    aggressive: list
    calls: list


def _flop_pass(records, seats: int, last_raiser: "int | None") -> _Flop:
    out = _Flop(*([False] * seats for _ in range(6)))
    for name in ("cbet", "cbet_chance", "faced_cbet", "folded_to_cbet"):
        setattr(out, name, [False] * seats)
    out.aggressive = [0] * seats
    out.calls = [0] * seats

    cbet_made = False
    for record in records:
        if record.street == PREFLOP:
            continue
        seat = record.seat

        if record.street == _FLOP:
            if seat == last_raiser and not cbet_made and record.bet_before == 0:
                # 轮到翻前进攻方，且还没人下注——这才是「有机会持续下注」
                out.cbet_chance[seat] = True
                if record.kind == "bet":
                    out.cbet[seat] = True
                    cbet_made = True
            elif cbet_made and record.to_call > 0 and not out.faced_cbet[seat]:
                out.faced_cbet[seat] = True
                if record.kind == "fold":
                    out.folded_to_cbet[seat] = True

        if record.kind in ("bet", "raise"):
            out.aggressive[seat] += 1
        elif record.kind == "call":
            out.calls[seat] += 1

    return out


def _positions(records, seats: int) -> "dict[int, str]":
    out: dict[int, str] = {}
    for record in records:
        if record.street == PREFLOP and record.seat not in out:
            out[record.seat] = record.position
    return out

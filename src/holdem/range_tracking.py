"""从一手打完的牌回推**翻牌时双方的范围**。纯逻辑，不碰 IO。

复盘要问「你这个翻后决策亏了多少」，而任何翻后求解的第一个输入就是**双方的范围**。
范围不是猜的：翻前怎么打的，我们自己的范围表里就有对应的那一格。

```python
setup = flop_ranges(hand)          # 打完的牌局
setup.oop, setup.ip                # 翻后先说话 / 后说话那一方的范围
setup.pot, setup.effective_stack   # 大盲为单位，直接喂给求解器
```

## 只认表里有的线路

| 翻前线路 | 认不认 |
|---|---|
| 有人开牌、一个人跟、其余全弃（单次加注底池） | ✅ |
| 有人开牌、一个人 3bet、开牌者跟（3bet 底池） | ✅ |
| 跛入、多人看翻牌、4bet 之后、开牌尺度对不上表 | ❌ 抛 `NotCovered`，并说清为什么 |

**说不了就说不了**（`NotCovered` 带原因），这跟 `preflop_policy` 回 `None` 是同一条纪律：
拿一份不适用的范围去求解，出来的 EV 损失看着有模有样，其实全错。

## 两条容易搞错的

1. **翻后先说话的是「小盲起数过来第一个还在牌里的人」**，不是座位号小的那个。
   位置弄反，求解出来的两边策略整个对调，而数值看着仍然「像那么回事」。
2. **有效筹码是「底池后面还剩多少」**，不是初始筹码：翻前投进去的钱已经在底池里了。
"""

from __future__ import annotations

from dataclasses import dataclass

from .history import action_records
from .positions import position_of
from .preflop_policy import DEPTH_BAND, effective_depth, parse_label
from .preflop_ranges import PreflopRangeTable, load_all
from .ranges import Range
from .state import PREFLOP, HandState

__all__ = ["FlopRanges", "NotCovered", "flop_ranges"]


class NotCovered(RuntimeError):
    """这手牌的翻前线路我们的表里没有，给不出翻牌范围。"""


@dataclass(frozen=True)
class FlopRanges:
    """翻牌时的局面：双方范围、底池、还剩多深。金额一律是**大盲**。"""

    oop_seat: int
    ip_seat: int
    oop: Range
    ip: Range
    pot: float
    effective_stack: float
    line: str
    """翻前是怎么走到这儿的，写给人看：「单次加注底池」/「3bet 底池」。"""
    aggressor_seat: int = -1
    """翻前最后加注的那个人（单次加注底池＝开牌者，3bet 底池＝3bet 的人）。

    翻后的场景全按这个人分：同一个「翻牌下注」，进攻方打出来是持续下注、
    防守方打出来是领打（donk），是两件完全不同的事。
    """

    def is_aggressor(self, seat: int) -> bool:
        return seat == self.aggressor_seat

    def range_of(self, seat: int) -> Range:
        if seat == self.oop_seat:
            return self.oop
        if seat == self.ip_seat:
            return self.ip
        raise KeyError(f"座位 {seat} 没看到翻牌")

    def player_index(self, seat: int) -> int:
        """求解器口径的编号：0 = OOP、1 = IP。"""
        if seat == self.oop_seat:
            return 0
        if seat == self.ip_seat:
            return 1
        raise KeyError(f"座位 {seat} 没看到翻牌")


def flop_ranges(
    hand: HandState, tables: "dict[int, PreflopRangeTable] | None" = None
) -> FlopRanges:
    """回推翻牌时双方的范围。表里没有这条线路就抛 `NotCovered`，并说清为什么。"""
    config = hand.config
    available = load_all() if tables is None else tables
    table = available.get(config.num_seats)
    if table is None:
        raise NotCovered(f"没有 {config.num_seats} 人桌的翻前范围表")

    depth = effective_depth(hand)
    ratio = depth / table.stack_bb
    if not DEPTH_BAND[0] <= ratio <= DEPTH_BAND[1]:
        raise NotCovered(
            f"这手牌 {depth:.0f}bb 深，表是按 {table.stack_bb:.0f}bb 解的，差得太多"
        )

    records = [r for r in action_records(hand) if r.street == PREFLOP]
    folded = {r.seat for r in records if r.kind == "fold"}
    survivors = [seat for seat in range(config.num_seats) if seat not in folded]
    # 先说「没走到翻牌」再说「人数不对」：翻前就收掉的牌活人只剩一个，
    # 若按人数报错会说成「求解器只解两人」，把「牌局提前结束」误报成覆盖缺口
    if len(hand.board) < 3:
        raise NotCovered("翻前就结束了，没有翻后决策")
    if len(survivors) != 2:
        raise NotCovered(f"翻牌时还有 {len(survivors)} 个人（多人底池），求解器只解两人")

    raises = [r for r in records if r.kind == "raise"]
    calls = [r for r in records if r.kind == "call"]
    if not raises:
        raise NotCovered("没人加注（跛入底池），表里没有这种线路")
    if any(r.seq < raises[0].seq for r in calls):
        raise NotCovered("开牌之前有人跛入，表里没有这种线路")

    name = {seat: position_of(seat, config.button, config.num_seats) for seat in survivors}
    opener = raises[0]
    if opener.seat not in survivors:
        raise NotCovered("开牌的人没看到翻牌，剩下的两个人之间没有表可查")

    if len(raises) == 1:
        if len(calls) != 1:
            raise NotCovered(f"开牌之后有 {len(calls)} 个人跟注，不是标准的单次加注底池")
        caller = calls[0]
        if {opener.seat, caller.seat} != set(survivors):
            raise NotCovered("看到翻牌的两个人对不上开牌者与跟注者")
        aggressor = opener.seat
        opener_range = _open_range(table, name[opener.seat])
        caller_range = _defense_range(table, name[opener.seat], name[caller.seat], "call")
        line = "单次加注底池"
        by_seat = {opener.seat: opener_range, caller.seat: caller_range}
    elif len(raises) == 2:
        reraiser = raises[1]
        if {opener.seat, reraiser.seat} != set(survivors):
            raise NotCovered("看到翻牌的两个人对不上开牌者与 3bet 者")
        if [r for r in calls if r.seq < reraiser.seq]:
            raise NotCovered("3bet 之前有人冷跟，表里没有这种线路")
        if not [r for r in calls if r.seat == opener.seat and r.seq > reraiser.seq]:
            raise NotCovered("开牌者没有跟 3bet，这不是 3bet 底池")
        aggressor = reraiser.seat
        entry = _defense_entry(table, name[opener.seat], name[reraiser.seat])
        by_seat = {
            reraiser.seat: _pick(entry.actions, "raise", f"{name[reraiser.seat]} 的 3bet"),
            opener.seat: _pick(
                entry.reraise_reply or {}, "call", f"{name[opener.seat]} 面对 3bet 的跟注"
            ),
        }
        line = "3bet 底池"
    else:
        raise NotCovered(f"翻前加注了 {len(raises)} 次，4bet 之后的线路表里没有")

    oop_seat = _first_to_act_postflop(survivors, config.button, config.num_seats)
    ip_seat = next(seat for seat in survivors if seat != oop_seat)
    big_blind = config.big_blind
    return FlopRanges(
        oop_seat=oop_seat,
        ip_seat=ip_seat,
        oop=by_seat[oop_seat],
        ip=by_seat[ip_seat],
        pot=_flop_pot(hand) / big_blind,
        effective_stack=_flop_stack(hand, survivors) / big_blind,
        line=line,
        aggressor_seat=aggressor,
    )


# ------------------------------------------------------------------ 内部


def _first_to_act_postflop(survivors: "list[int]", button: int, seats: int) -> int:
    """翻后先说话的人：从小盲（庄家左手）起数，第一个还在牌里的。"""
    for offset in range(1, seats + 1):
        seat = (button + offset) % seats
        if seat in survivors:
            return seat
    raise ValueError("没有人还在牌里")


def _open_range(table: PreflopRangeTable, position: str) -> Range:
    try:
        return table.open_range(position)
    except KeyError:
        raise NotCovered(f"表里没有 {position} 的开牌范围") from None


def _defense_entry(table: PreflopRangeTable, opener: str, defender: str):
    try:
        return table.defense(opener, defender)
    except KeyError:
        raise NotCovered(f"表里没有「{defender} 面对 {opener} 开牌」这一格") from None


def _defense_range(
    table: PreflopRangeTable, opener: str, defender: str, kind: str
) -> Range:
    entry = _defense_entry(table, opener, defender)
    return _pick(entry.actions, kind, f"{defender} 面对 {opener} 开牌的{kind}")


def _pick(actions: "dict[str, Range]", kind: str, what: str) -> Range:
    """按动作**类型**取范围，不按标签——实战的尺度与表里的抽象尺度对不上是常态。"""
    for label, hand_range in actions.items():
        if parse_label(label)[0] == kind:
            if not hand_range:
                raise NotCovered(f"{what}的范围是空的")
            return hand_range
    raise NotCovered(f"表里没有{what}")


def _flop_pot(hand: HandState) -> int:
    """翻牌发出来时底池里有多少（筹码）。"""
    pot = 0
    for event in hand.events:
        if event.kind == "deal_board":
            break
        if event.kind in ("ante", "blind", "call", "bet", "raise"):
            pot += event.amount
        elif event.kind == "refund":
            pot -= event.amount
    return pot


def _flop_stack(hand: HandState, survivors: "list[int]") -> int:
    """翻牌时两人身后还剩多少（取浅的那份）。"""
    spent = [0] * hand.config.num_seats
    for event in hand.events:
        if event.kind == "deal_board":
            break
        if event.kind in ("ante", "blind", "call", "bet", "raise"):
            spent[event.seat] += event.amount
        elif event.kind == "refund":
            spent[event.seat] -= event.amount
    return min(hand.config.stacks[seat] - spent[seat] for seat in survivors)

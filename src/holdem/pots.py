"""底池与边池计算——纯函数，不依赖牌局状态机。

拆成独立模块是因为边池是最容易出错、也最值得单独覆盖测试的一块：
全下金额不同、弃牌者留在池里、未被跟注的下注要退回，这三件事互相纠缠。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pot:
    """一个（主池或边池）。`eligible` 是有资格赢下它的座位号，已排序。"""

    amount: int
    eligible: tuple[int, ...]


def refund_uncalled(contributions: list[int]) -> tuple[list[int], list[int]]:
    """退回未被跟注的下注。

    最高投入者超出「第二高投入者」的部分没有任何人跟，必须退还。
    注意第二高要算上已弃牌的玩家——他们的钱同样构成了对手的跟注额。

    返回 (调整后的投入, 每个座位收到的退款)。
    """
    refunds = [0] * len(contributions)
    if not contributions:
        return list(contributions), refunds

    ordered = sorted(contributions, reverse=True)
    highest = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0
    if highest <= second:
        return list(contributions), refunds

    # 只有一个人能处在最高位；若并列则 highest == second，上面已返回
    seat = contributions.index(highest)
    refunds[seat] = highest - second
    adjusted = list(contributions)
    adjusted[seat] = second
    return adjusted, refunds


def build_pots(contributions: list[int], folded: list[bool]) -> list[Pot]:
    """按全下档位切分主池与边池。

    `contributions` 为每个座位本手投入的总额（应先经 `refund_uncalled` 处理）。
    `folded` 标记谁已弃牌——弃牌者的钱留在池中，但没有赢下它的资格。
    """
    if len(contributions) != len(folded):
        raise ValueError("contributions 与 folded 长度必须一致")

    levels = sorted({c for c in contributions if c > 0})
    pots: list[Pot] = []
    previous = 0
    carry = 0  # 该档位无人有资格（全是弃牌者的钱）时暂存，并入相邻的池

    for level in levels:
        amount = sum(min(c, level) - min(c, previous) for c in contributions)
        previous = level
        if amount <= 0:
            continue
        eligible = tuple(
            seat
            for seat, c in enumerate(contributions)
            if c >= level and not folded[seat]
        )
        if not eligible:
            carry += amount
            continue
        amount += carry
        carry = 0
        # 与上一个池资格相同则合并，避免产生一串无意义的碎池
        if pots and pots[-1].eligible == eligible:
            pots[-1] = Pot(pots[-1].amount + amount, eligible)
        else:
            pots.append(Pot(amount, eligible))

    if carry:
        if not pots:
            raise ValueError("所有投入都无人有资格赢取，牌局状态非法")
        pots[-1] = Pot(pots[-1].amount + carry, pots[-1].eligible)

    return pots


def award(
    pots: list[Pot],
    scores: dict[int, int],
    first_seat_left_of_button: int,
    num_seats: int,
) -> list[int]:
    """按牌力分配每个池，返回每个座位赢得的筹码。

    `scores` 只需包含摊牌玩家的分数（越大越强）。平分时余数按扑克惯例
    从庄家左手第一个有资格的座位开始逐个多分一枚。
    """
    payouts = [0] * num_seats
    order = [(first_seat_left_of_button + i) % num_seats for i in range(num_seats)]

    for pot in pots:
        contenders = [s for s in pot.eligible if s in scores]
        if not contenders:
            raise ValueError(f"底池 {pot} 没有可分配的赢家")
        best = max(scores[s] for s in contenders)
        winners = [s for s in contenders if scores[s] == best]

        share, remainder = divmod(pot.amount, len(winners))
        for seat in winners:
            payouts[seat] += share
        for seat in order:
            if remainder <= 0:
                break
            if seat in winners:
                payouts[seat] += 1
                remainder -= 1

    return payouts

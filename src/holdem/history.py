"""从牌局流水还原每个决策点的上下文。

`HandState.events` 只记了「谁做了什么、付了多少」，但统计与复盘要的是决策**当时**的
局面：底池多大、面对多少下注、还剩几个人能行动、自己在什么位置。这里把两者补齐。

纯逻辑，不碰 IO。M2 的所有统计指标（VPIP / PFR / 3-bet / 持续下注 …）都从这里的
`ActionRecord` 推导，避免每个指标各自解析一遍事件流。
"""

from __future__ import annotations

from dataclasses import dataclass

from .positions import position_of
from .state import HandState

_PLAYER_ACTIONS = ("fold", "check", "call", "bet", "raise")


@dataclass(frozen=True)
class ActionRecord:
    """一个决策点。所有金额都是行动**之前**的局面，除了 amount 与 to。"""

    seq: int
    street: int
    seat: int
    position: str
    kind: str
    amount: int
    """本次付出的筹码增量。"""
    to: int
    """本次动作之后，此人本街的投入总额。"""
    pot_before: int
    bet_before: int
    """行动前本街的最高投入额。"""
    to_call: int
    stack_before: int
    actors_before: int
    """行动前仍有筹码可行动的未弃牌人数（含自己）。"""
    is_voluntary: bool
    """翻前是否属于「主动投钱」——盲注不算，用于 VPIP。"""


def action_records(hand: HandState) -> list[ActionRecord]:
    cfg = hand.config
    n = cfg.num_seats
    stacks = list(cfg.stacks)
    committed_street = [0] * n
    folded = [False] * n
    pot = 0
    current_bet = 0
    in_ante_phase = cfg.ante > 0
    records: list[ActionRecord] = []
    seq = 0

    for event in hand.events:
        kind = event.kind

        if kind == "ante":
            stacks[event.seat] -= event.amount
            pot += event.amount
            continue

        if kind == "blind":
            if in_ante_phase:
                # 引擎在前注之后重置本街投入，这里保持一致
                committed_street = [0] * n
                in_ante_phase = False
            stacks[event.seat] -= event.amount
            pot += event.amount
            committed_street[event.seat] += event.amount
            current_bet = max(current_bet, committed_street[event.seat])
            continue

        if kind == "deal_board":
            committed_street = [0] * n
            current_bet = 0
            continue

        if kind == "refund":
            stacks[event.seat] += event.amount
            pot -= event.amount
            continue

        if kind not in _PLAYER_ACTIONS:
            continue

        seat = event.seat
        actors_before = sum(
            1 for s in range(n) if not folded[s] and stacks[s] > 0
        )
        records.append(
            ActionRecord(
                seq=seq,
                street=event.street,
                seat=seat,
                position=position_of(seat, cfg.button, n),
                kind=kind,
                amount=event.amount,
                to=event.to,
                pot_before=pot,
                bet_before=current_bet,
                to_call=max(0, current_bet - committed_street[seat]),
                stack_before=stacks[seat],
                actors_before=actors_before,
                is_voluntary=kind in ("call", "bet", "raise"),
            )
        )
        seq += 1

        if kind == "fold":
            folded[seat] = True
        elif kind in ("call", "bet", "raise"):
            stacks[seat] -= event.amount
            pot += event.amount
            committed_street[seat] += event.amount
            current_bet = max(current_bet, committed_street[seat])

    return records

"""玩家动作与合法动作集的表示。

下注额一律采用「加注到（raise-to）」语义，即 `to` 表示该玩家本街投入的**目标总额**，
而不是本次追加的增量。求解器、PHH 牌谱和多数扑克软件都用这个口径，
统一之后可以避免在边界（尤其是全下与短加注）上反复换算出错。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    to: int = 0
    """BET / RAISE 时表示本街投入的目标总额；其余动作忽略。"""

    def __str__(self) -> str:
        if self.kind in (ActionKind.BET, ActionKind.RAISE):
            return f"{self.kind.value} to {self.to}"
        return self.kind.value


def fold() -> Action:
    return Action(ActionKind.FOLD)


def check() -> Action:
    return Action(ActionKind.CHECK)


def call() -> Action:
    return Action(ActionKind.CALL)


def bet(to: int) -> Action:
    return Action(ActionKind.BET, to)


def raise_to(to: int) -> Action:
    return Action(ActionKind.RAISE, to)


@dataclass(frozen=True)
class LegalActions:
    """当前行动玩家可以做什么。金额均为本街的目标总额。"""

    seat: int
    can_fold: bool
    can_check: bool
    can_call: bool
    call_to: int
    call_cost: int
    can_raise: bool
    is_opening_bet: bool
    """True 表示本街尚无人下注，加注在术语上应称为「下注」。"""
    min_raise_to: int
    max_raise_to: int

    def contains(self, action: Action) -> bool:
        if action.kind is ActionKind.FOLD:
            return self.can_fold
        if action.kind is ActionKind.CHECK:
            return self.can_check
        if action.kind is ActionKind.CALL:
            return self.can_call
        if action.kind in (ActionKind.BET, ActionKind.RAISE):
            if not self.can_raise:
                return False
            if action.kind is ActionKind.BET and not self.is_opening_bet:
                return False
            if action.kind is ActionKind.RAISE and self.is_opening_bet:
                return False
            # 全下允许低于最小加注额
            if action.to == self.max_raise_to:
                return action.to > self.call_to
            return self.min_raise_to <= action.to <= self.max_raise_to
        return False

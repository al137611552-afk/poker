"""Slumbot 协议 ↔ 我们的引擎，**纯逻辑**（不联网，可单测）。

Slumbot 只给一个动作串（`b200c/kb200c/kk/b400f`）加上我们自己的底牌与公共牌。要让
`holdem.bots.Bot` 直接坐上去打，就得把这串东西**回放成一份 `HandState`**——回放完，
bot 拿到的局面与它在本地自对弈时看到的一模一样，一行策略代码都不用改。

## 协议要点（全部是实测所得，别照直觉猜）

- **下注额是「本街」的目标总额**，与我们引擎的 raise-to 口径正好一致。
  实测样本 `b200c/kb200c/kb400c/kb1600c` 输掉 2400 筹码 ＝ 200+200+400+1600，
  若是「本手总额」这四段就该单调递增。
- **`client_pos == 1` 表示我们是按钮/小盲**（翻前先说话）；0 表示我们是大盲、它先说话。
- **过牌要发 `k`**，发 `c` 会被判 Illegal call；`c` 只用于跟注。
- 盲注 50/100、筹码 20000（200bb），双方等深，所以不会有边池。

## 我们看不见对手的底牌

回放时对手那两张用**剩下的牌随机填**。bot 的决策只用到自己的底牌与公共牌
（`monte_carlo_equity` 是自采样的），所以填什么都不影响它的动作；但也因此，
**摊牌终局的胜负不能由我们的引擎判**——那要用 Slumbot 回的 `winnings`。
`check_result` 就是守这条线的：能核对的账全核对，核对不了的老实说核对不了。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from holdem.actions import Action, ActionKind, bet, call, check, fold, raise_to
from holdem.deck import stacked_deck
from holdem.state import HandConfig, HandState

__all__ = [
    "BIG_BLIND",
    "SMALL_BLIND",
    "STACK",
    "HandView",
    "tokenize",
    "iter_tokens",
    "facing_bet",
    "build_state",
    "to_incr",
    "check_result",
]

BIG_BLIND = 100
SMALL_BLIND = 50
STACK = 20_000
"""Slumbot 固定 50/100 盲注、20000 筹码（200bb）。这三个数不可配，是它那边定死的。"""

BUTTON_SEAT = 0
"""回放时一律把按钮放在座位 0——单挑里按钮就是小盲、翻前先说话，与引擎口径一致。"""

_TOKEN = re.compile(r"b\d+|[cfk]")


# ------------------------------------------------------------------ 动作串


def tokenize(street: str) -> list[str]:
    """把一条街的动作串拆成 `['b200', 'c']` 这样的记号。"""
    return _TOKEN.findall(street)


def iter_tokens(action: str) -> list[str]:
    """整手的记号，按顺序。街的分隔符 `/` 直接丢掉——引擎自己会推进街。"""
    return _TOKEN.findall(action)


def facing_bet(action: str) -> bool:
    """轮到我们时是不是面对一个下注（决定能不能弃牌、该发 `c` 还是 `k`）。"""
    tokens = tokenize(action.split("/")[-1])
    return bool(tokens) and tokens[-1].startswith("b")


# ------------------------------------------------------------------ 一手牌


@dataclass(frozen=True)
class HandView:
    """Slumbot 回给我们的一手牌的公开信息。"""

    hole: tuple[str, ...]
    board: tuple[str, ...]
    action: str
    client_pos: int
    """1 = 我们是按钮（翻前先说话），0 = 我们是大盲。"""
    winnings: int | None = None

    @classmethod
    def from_body(cls, body: dict) -> "HandView":
        return cls(
            hole=tuple(body.get("hole_cards") or ()),
            board=tuple(body.get("board") or ()),
            action=body.get("action") or "",
            client_pos=int(body["client_pos"]),
            winnings=body.get("winnings"),
        )

    @property
    def our_seat(self) -> int:
        return BUTTON_SEAT if self.client_pos == 1 else 1 - BUTTON_SEAT

    @property
    def their_seat(self) -> int:
        return 1 - self.our_seat

    @property
    def is_over(self) -> bool:
        return self.winnings is not None


def build_state(view: HandView, *, rest_seed: int = 0) -> HandState:
    """把动作串回放成一份 `HandState`，回放到「轮到我们说话」为止。

    对手的底牌与还没发出来的公共牌用剩下的牌按 `rest_seed` 填。
    """
    if len(view.hole) != 2:
        raise ValueError(f"底牌必须是两张，收到 {view.hole}")
    config = HandConfig(
        stacks=(STACK, STACK),
        button=BUTTON_SEAT,
        big_blind=BIG_BLIND,
        small_blind=SMALL_BLIND,
    )
    deck = stacked_deck(
        {view.our_seat: "".join(view.hole)},
        board="".join(view.board),
        num_seats=2,
        button=BUTTON_SEAT,
        rest_seed=rest_seed,
    )
    hand = HandState(config, deck)
    for index, token in enumerate(iter_tokens(view.action)):
        if hand.is_complete:
            raise ValueError(f"动作串比牌局长：第 {index} 个记号 {token!r} 已经无处可施")
        hand.apply(_from_token(token, hand))
    return hand


def _from_token(token: str, hand: HandState) -> Action:
    if token == "f":
        return fold()
    if token == "k":
        return check()
    if token == "c":
        return call()
    amount = int(token[1:])
    # 本街还没人下注时，术语上叫「下注」而不是「加注」——引擎会按这个校验合法性
    return bet(amount) if hand.legal_actions().is_opening_bet else raise_to(amount)


def to_incr(action: Action) -> str:
    """把我们 bot 的动作翻成 Slumbot 的记号。"""
    if action.kind is ActionKind.FOLD:
        return "f"
    if action.kind is ActionKind.CHECK:
        return "k"
    if action.kind is ActionKind.CALL:
        return "c"
    if action.kind in (ActionKind.BET, ActionKind.RAISE):
        return f"b{action.to}"
    raise ValueError(f"翻不了的动作: {action}")


# ------------------------------------------------------------------ 对账


def check_result(view: HandView, *, rest_seed: int = 0) -> str | None:
    """拿 Slumbot 报的 `winnings` 核对我们对这手牌的理解；对得上回 `None`。

    这是整条链路唯一的外部真值：动作串解析错、金额口径错、位置认反，都会在这里露馅。
    分两种情况：

    - **有人弃牌**：胜负由下注决定，我们能算出准确的净得失，逐筹码核对。
    - **摊牌**：我们看不见对手的牌，判不了胜负；但双方投入必然相等，
      所以净得失只可能是 ±自己的投入或平分（0）。核对这个。
    """
    if view.winnings is None:
        return "这手还没结束，没有可核对的结果"
    hand = build_state(view, rest_seed=rest_seed)
    if not hand.is_complete:
        return "动作串没把牌局走完，无法核对"

    contribution = hand.result.contributions[view.our_seat]
    if iter_tokens(view.action)[-1] == "f":
        expected = hand.result.net[view.our_seat]
        if expected != view.winnings:
            return f"弃牌终局对不上：我们算 {expected}，Slumbot 说 {view.winnings}"
        return None

    if abs(view.winnings) not in (0, contribution):
        return (
            f"摊牌终局对不上：我们投入 {contribution}，"
            f"净得失只该是 ±{contribution} 或 0，Slumbot 说 {view.winnings}"
        )
    return None

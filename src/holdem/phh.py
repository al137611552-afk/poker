"""PHH（Poker Hand History）牌谱的读写。

PHH 是 TOML 文本格式，由多伦多大学计算机扑克研究组提出，PokerKit 原生支持。
选它而不是自造格式，是为了让牌谱能被外部工具读取、也能读进外部牌谱。

## 两个必须记住的坑

1. **玩家编号不是座位号**。PHH 用 p1..pN，其中 p1 是小盲、pN 是按钮。
   本模块的 `phh_player_order()` 负责与引擎的座位号互转。
2. **单挑是特例**。参考实现（PokerKit）在两人局会把盲注数组反向套用：
   写入 `blinds_or_straddles = [5, 10]` 时，实际是 p1 押 10、p2 押 5，
   由 p2（按钮/小盲）先行动。因此单挑时 p1 是**大盲**，p2 才是按钮。
   这一条是实测出来的，不要凭直觉改。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field

from .actions import bet, call, check, fold, raise_to
from .cards import card_to_str, cards_from_str, cards_to_str
from .deck import stacked_deck
from .state import COMPLETE, HandConfig, HandState

VARIANT = "NT"  # No-limit Texas hold'em


def phh_player_order(num_seats: int, button: int) -> list[int]:
    """返回 PHH 玩家编号 p1..pN 依次对应的引擎座位号。"""
    if num_seats < 2:
        raise ValueError("至少两个座位")
    if num_seats == 2:
        # 单挑：p1 是大盲，p2 是按钮/小盲（见模块文档）
        return [(button + 1) % 2, button % 2]
    return [(button + 1 + i) % num_seats for i in range(num_seats)]


def button_for_player_order(num_seats: int) -> int:
    """当 PHH 玩家编号直接映射为座位号 0..N-1 时，按钮所在的座位。"""
    return 1 if num_seats == 2 else num_seats - 1


# ------------------------------------------------------------------ 写


def _toml_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _toml_array(values: list) -> str:
    parts = []
    for value in values:
        if isinstance(value, str):
            parts.append(_toml_string(value))
        elif isinstance(value, bool):
            parts.append("true" if value else "false")
        else:
            parts.append(str(value))
    return "[" + ", ".join(parts) + "]"


def to_phh(
    hand: HandState,
    *,
    players: list[str] | None = None,
    hand_number: int | None = None,
    extra: dict[str, object] | None = None,
) -> str:
    """把一手已结束（或进行中）的牌局序列化为 PHH 文本。

    `players` 按引擎座位号给出玩家名，会自动转成 PHH 的玩家顺序。
    """
    cfg = hand.config
    n = cfg.num_seats
    order = phh_player_order(n, cfg.button)
    seat_to_player = {seat: index + 1 for index, seat in enumerate(order)}

    lines = [
        f"variant = {_toml_string(VARIANT)}",
        f"antes = {_toml_array([cfg.ante] * n)}",
        f"blinds_or_straddles = {_toml_array([cfg.sb, cfg.big_blind] + [0] * (n - 2))}",
        f"min_bet = {cfg.big_blind}",
        f"starting_stacks = {_toml_array([cfg.stacks[seat] for seat in order])}",
    ]

    if hand_number is not None:
        lines.append(f"hand = {hand_number}")
    if players is not None:
        if len(players) != n:
            raise ValueError("players 的长度必须等于座位数")
        lines.append(f"players = {_toml_array([players[seat] for seat in order])}")
    if hand.is_complete:
        lines.append(
            f"finishing_stacks = {_toml_array([hand.stacks[seat] for seat in order])}"
        )
    for key, value in (extra or {}).items():
        rendered = (
            _toml_string(value)
            if isinstance(value, str)
            else _toml_array(value)
            if isinstance(value, list)
            else str(value)
        )
        lines.append(f"{key} = {rendered}")

    actions: list[str] = []
    for seat in order:
        cards = cards_to_str(hand.hole[seat]) if hand.hole[seat] else "????"
        actions.append(f"d dh p{seat_to_player[seat]} {cards}")

    for event in hand.events:
        player = f"p{seat_to_player[event.seat]}" if event.seat >= 0 else None
        if event.kind == "deal_board":
            actions.append(f"d db {cards_to_str(event.cards)}")
        elif event.kind == "fold":
            actions.append(f"{player} f")
        elif event.kind in ("check", "call"):
            actions.append(f"{player} cc")
        elif event.kind in ("bet", "raise"):
            actions.append(f"{player} cbr {event.to}")
        elif event.kind == "showdown":
            actions.append(f"{player} sm {cards_to_str(event.cards)}")

    lines.append("actions = [")
    lines.extend(f"  {_toml_string(a)}," for a in actions)
    lines.append("]")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ 读


@dataclass
class PhhHand:
    """解析后的 PHH 牌谱。玩家 p1..pN 直接对应座位 0..N-1。"""

    num_seats: int
    button: int
    small_blind: int
    big_blind: int
    ante: int
    starting_stacks: tuple[int, ...]
    hole: dict[int, list[int]]
    board: list[int]
    actions: list[str]
    finishing_stacks: tuple[int, ...] | None = None
    players: list[str] | None = None
    hand_number: int | None = None
    unknown_hole_cards: bool = False
    raw: dict = field(default_factory=dict)


def parse_phh(text: str) -> PhhHand:
    data = tomllib.loads(text)

    variant = data.get("variant")
    if variant != VARIANT:
        raise ValueError(f"只支持无限注德州扑克（NT），收到 {variant!r}")

    stacks = [int(s) for s in data["starting_stacks"]]
    n = len(stacks)
    blinds = [int(b) for b in data["blinds_or_straddles"]]
    if len(blinds) != n:
        raise ValueError("blinds_or_straddles 长度必须等于玩家数")
    antes = [int(a) for a in data.get("antes", [0] * n)]
    if len(set(antes)) > 1:
        raise ValueError("引擎目前只支持所有人相同的前注")

    hole: dict[int, list[int]] = {}
    board: list[int] = []
    unknown = False
    player_actions: list[str] = []

    for raw_action in data["actions"]:
        parts = raw_action.split()
        if parts[0] == "d":
            if parts[1] == "dh":
                seat = _player_index(parts[2])
                cards_text = "".join(parts[3:])
                if "?" in cards_text:
                    unknown = True
                else:
                    hole[seat] = cards_from_str(cards_text)
            elif parts[1] == "db":
                board.extend(cards_from_str("".join(parts[2:])))
            else:
                raise ValueError(f"无法识别的发牌动作: {raw_action!r}")
        else:
            player_actions.append(raw_action)

    finishing = data.get("finishing_stacks")
    return PhhHand(
        num_seats=n,
        button=button_for_player_order(n),
        small_blind=blinds[0],
        big_blind=blinds[1],
        ante=antes[0] if antes else 0,
        starting_stacks=tuple(stacks),
        hole=hole,
        board=board,
        actions=player_actions,
        finishing_stacks=tuple(int(s) for s in finishing) if finishing else None,
        players=list(data["players"]) if "players" in data else None,
        hand_number=data.get("hand"),
        unknown_hole_cards=unknown,
        raw=data,
    )


def _player_index(token: str) -> int:
    if not token.startswith("p") or not token[1:].isdigit():
        raise ValueError(f"无法识别的玩家标记: {token!r}")
    return int(token[1:]) - 1


def replay(phh: PhhHand) -> HandState:
    """把解析后的牌谱重放成引擎状态，顺带验证动作序列自洽。"""
    hole_text = {seat: cards_to_str(cards) for seat, cards in phh.hole.items()}
    if not hole_text:
        raise ValueError("牌谱中没有任何已知底牌，无法重放")
    board_text = "".join(card_to_str(c) for c in phh.board)

    deck = stacked_deck(
        hole=hole_text,
        board=board_text,
        num_seats=phh.num_seats,
        button=phh.button,
    )
    config = HandConfig(
        stacks=phh.starting_stacks,
        button=phh.button,
        big_blind=phh.big_blind,
        small_blind=phh.small_blind,
        ante=phh.ante,
    )
    hand = HandState(config, deck)

    for raw_action in phh.actions:
        parts = raw_action.split()
        seat = _player_index(parts[0])
        verb = parts[1]
        if verb == "sm":  # 亮牌/弃掉，不改变筹码流
            continue
        if hand.is_complete:
            raise ValueError(f"牌谱在牌局结束后仍有动作: {raw_action!r}")
        if hand.to_act != seat:
            raise ValueError(
                f"牌谱动作 {raw_action!r} 与行动顺序不符：当前应由座位 {hand.to_act} 行动"
            )
        if verb == "f":
            hand.apply(fold())
        elif verb == "cc":
            legal = hand.legal_actions()
            hand.apply(call() if legal.can_call else check())
        elif verb == "cbr":
            amount = int(parts[2])
            legal = hand.legal_actions()
            hand.apply(bet(amount) if legal.is_opening_bet else raise_to(amount))
        else:
            raise ValueError(f"无法识别的动作: {raw_action!r}")

    if phh.finishing_stacks is not None and hand.street == COMPLETE:
        if tuple(hand.stacks) != phh.finishing_stacks:
            raise ValueError(
                f"重放结果与牌谱记录的最终筹码不符: {hand.stacks} != {list(phh.finishing_stacks)}"
            )
    return hand


def loads(text: str) -> HandState:
    """PHH 文本 -> 重放后的引擎状态。"""
    return replay(parse_phh(text))

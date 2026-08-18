"""求解请求 → TexasSolver 的命令文件。**纯逻辑，不碰进程也不碰磁盘。**

TexasSolver 的输入是一串命令（`set_pot 60` / `set_board Qs,Jh,2h` / `build_tree` …），
输出是一个 JSON 策略树。这里只负责把一个 `SolveRequest` 渲染成那串命令——渲染是纯函数，
所以格式对不对可以脱离二进制单测，而格式错了是**最贵的一类错**：求解器不会温柔地报错，
它会直接 abort。

## 实测钉死的三条格式（别照文档猜，文档没写）

1. **范围不能用 `+` / `-` 记法**。`set_range_ip 99+,AJs+` 会让求解器抛
   `std::runtime_error: format not recognize` 然后 **SIGABRT**（退出码 134，什么都不输出）。
   所以这里一律展开成逐个牌类：`99,TT,JJ,...`，权重写成 `JTs:0.5`。
2. **金额可以是小数**：`set_pot 12.5` / `set_effective_stack 87.5` 实测可用，
   **但求解器算出来的下注额会取整到整数单位**（底池 5.5、下注 33% 应是 1.815，
   树里给的是 `BET 2.000000`）。所以命令文件里的单位不是大盲，而是 **1/`scale` 大盲**
   （默认 1/10）：放大之后再取整，粒度就是 0.1bb。对外的接口一律还是大盲，
   解回来的金额由 `backend` 除回去。
3. **下注尺度是底池百分比**，按「谁,哪条街,什么动作,尺度…」给，两边分别设。
   **同一格只能发一条命令，多个尺度写在同一行用逗号隔开**——发第二条是覆盖不是追加，
   而且不报错：树悄悄只剩最后那个尺度，直到复盘时发现「实战打的尺度树里没有」
   才暴露出来（实测踩过）。

## 单位

`pot` 与 `effective_stack` 都是**大盲**，且 `effective_stack` 指**底池后面还剩多少**
（不是初始筹码）。求解器只关心两者的比例（SPR），单位一致即可。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from holdem.cards import card_to_str
from holdem.ranges import Range, class_name

__all__ = ["BetSizes", "SolveRequest", "FLOP", "TURN", "RIVER", "format_range"]

FLOP, TURN, RIVER = "flop", "turn", "river"
_STREETS = (FLOP, TURN, RIVER)


def format_range(hand_range: Range) -> str:
    """把范围写成求解器认识的样子：逐个牌类，权重小于 1 的带上 `:权重`。

    **不能用 `+` 记法**——求解器解析不了，而且是直接 abort（见模块说明）。
    """
    if not hand_range:
        raise ValueError("范围是空的，没法求解")
    parts = []
    for index in sorted(hand_range.weights):
        weight = hand_range.weights[index]
        name = class_name(index)
        parts.append(name if weight >= 0.9995 else f"{name}:{round(weight, 4):g}")
    return ",".join(parts)


def format_board(board: "tuple[int, ...]") -> str:
    return ",".join(card_to_str(card) for card in board)


@dataclass(frozen=True)
class BetSizes:
    """下注尺度，单位是**底池百分比**，两边对称。

    尺度越多树越大、解得越慢（一条街多给一个尺度，树大致翻一倍）。复盘打分不需要很细的
    尺度网格——但**我们自己实战打出的那个尺度必须在树里**，否则那个动作在解里根本不存在，
    也就没法给它打分。`with_size` 就是干这个的。
    """

    flop: tuple[float, ...] = (33.0, 75.0)
    turn: tuple[float, ...] = (66.0,)
    river: tuple[float, ...] = (75.0,)
    reraise: tuple[float, ...] = (60.0,)
    """再加注的尺度（各条街通用）。"""
    donk: tuple[float, ...] = ()
    """没位置的一方在非首轮主动下注（donk bet）。默认不开——它让树明显变大。"""
    allin: bool = True

    def __post_init__(self) -> None:
        for name in ("flop", "turn", "river", "reraise", "donk"):
            sizes = getattr(self, name)
            if any(size <= 0 for size in sizes):
                raise ValueError(f"{name} 的尺度必须为正: {sizes}")
        if not (self.flop and self.turn and self.river):
            raise ValueError("每条街至少要有一个下注尺度")

    def street(self, name: str) -> tuple[float, ...]:
        return {FLOP: self.flop, TURN: self.turn, RIVER: self.river}[name]

    def with_size(self, street: str, percent: float) -> "BetSizes":
        """把一个实战打出的尺度并进这条街（已经有相近的就不重复加）。

        「相近」按 2 个百分点算：实战里 33% 与 34% 的底池下注是同一回事，
        为它多长一层树不值得。
        """
        if street not in _STREETS:
            raise ValueError(f"没有这条街: {street}")
        existing = self.street(street)
        if any(abs(size - percent) <= 2.0 for size in existing):
            return self
        merged = tuple(sorted(existing + (round(percent, 1),)))
        return replace(self, **{street: merged})

    def commands(self) -> list[str]:
        """渲染成 `set_bet_sizes` 命令。

        **同一个（谁,哪条街,什么动作）只能发一条命令，多个尺度用逗号并排**——
        发第二条不是追加，是**覆盖**（实测：33/63.6/75 各发一条，树里只剩最后那个 75%）。
        这个错不报任何异常，只是树悄悄变小，然后实战打出的尺度「在树里找不到」。
        """
        lines = []
        for player in ("oop", "ip"):
            for street in _STREETS:
                lines.append(_sizes_command(player, street, "bet", self.street(street)))
                if self.reraise:
                    lines.append(_sizes_command(player, street, "raise", self.reraise))
                if player == "oop" and self.donk and street != FLOP:
                    lines.append(_sizes_command("oop", street, "donk", self.donk))
                if self.allin:
                    lines.append(f"set_bet_sizes {player},{street},allin")
        return lines


def _sizes_command(player: str, street: str, kind: str, sizes: "tuple[float, ...]") -> str:
    return f"set_bet_sizes {player},{street},{kind}," + ",".join(f"{size:g}" for size in sizes)


@dataclass(frozen=True)
class SolveRequest:
    """一个两人翻后局面。**这就是缓存的键**——同一个请求解出来的东西一模一样。

    `oop_range` / `ip_range` 是翻牌时双方的范围；谁是 OOP 由调用方定
    （我们的口径：**0 = 翻后先说话的那个人 = OOP**）。
    """

    board: tuple[int, ...]
    pot: float
    effective_stack: float
    oop_range: Range
    ip_range: Range
    bet_sizes: BetSizes = BetSizes()
    accuracy: float = 0.5
    """收敛门槛：可利用度占底池的百分比。求解器的报告口径就是它。"""
    max_iterations: int = 200
    allin_threshold: float = 0.67
    use_isomorphism: bool = True
    dump_rounds: int = 1
    """dump 几层。**1 = 只导根节点那一层**（几十 KB）；导整棵树动辄几十 MB。"""
    scale: float = 10.0
    """命令文件里的金额单位是 **1/scale 大盲**。

    求解器把算出来的下注额**取整到整数单位**，直接用大盲当单位就只有 1bb 的粒度
    （5.5 的底池打 33%＝1.815，树里会变成 2.0，也就是 36.4%）。放大 10 倍之后粒度是
    0.1bb。**这只是命令文件里的口径**——`SolveRequest` 的字段与解回来的金额都是大盲。
    """

    def __post_init__(self) -> None:
        if not 3 <= len(self.board) <= 5:
            raise ValueError(f"公共牌要 3–5 张，收到 {len(self.board)}")
        if len(set(self.board)) != len(self.board):
            raise ValueError("公共牌有重复")
        if self.pot <= 0:
            raise ValueError("底池必须为正")
        if self.effective_stack <= 0:
            raise ValueError("有效筹码必须为正")
        if self.accuracy <= 0:
            raise ValueError("精度门槛必须为正")
        if self.max_iterations < 1:
            raise ValueError("迭代次数至少为 1")
        if self.dump_rounds < 1:
            raise ValueError("至少要导一层")
        if self.scale <= 0:
            raise ValueError("单位放大倍数必须为正")

    @property
    def rounding(self) -> float:
        """求解器把下注额取整之后，金额最多偏这么多（大盲）。"""
        return 0.5 / self.scale

    @property
    def spr(self) -> float:
        return self.effective_stack / self.pot

    @property
    def street(self) -> str:
        return {3: FLOP, 4: TURN, 5: RIVER}[len(self.board)]

    # ---------------------------------------------------------- 渲染

    def commands(self, dump_path: str, *, threads: int = 2, print_interval: int = 10) -> str:
        """渲染成求解器的输入文件。

        `threads` 与 `dump_path` **不进指纹**：它们不改变解，只改变跑法。
        """
        lines = [
            f"set_pot {self.pot * self.scale:g}",
            f"set_effective_stack {self.effective_stack * self.scale:g}",
            f"set_board {format_board(self.board)}",
            f"set_range_ip {format_range(self.ip_range)}",
            f"set_range_oop {format_range(self.oop_range)}",
            *self.bet_sizes.commands(),
            f"set_allin_threshold {self.allin_threshold:g}",
            "build_tree",
            f"set_thread_num {threads}",
            f"set_accuracy {self.accuracy:g}",
            f"set_max_iteration {self.max_iterations}",
            f"set_print_interval {print_interval}",
            f"set_use_isomorphism {1 if self.use_isomorphism else 0}",
            "start_solve",
            f"set_dump_rounds {self.dump_rounds}",
            f"dump_result {dump_path}",
        ]
        return "\n".join(lines) + "\n"

    def fingerprint(self) -> str:
        """内容指纹，用作缓存键。跑法（线程数、输出路径）不算在内。"""
        payload = self.commands("<dump>", threads=0, print_interval=0)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

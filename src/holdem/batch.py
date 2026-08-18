"""批量自对弈：一桌 bot 连打上万手，量出 bb/100 与翻前统计（FR-4）。

对战一手是**纯逻辑**（引擎 + bot 都不碰 IO），所以批量对局也做成纯函数：给一份
`MatchConfig`，回一份 `BatchResult`。落库、打印、开进程池都在外面（`scripts/play_batch.py`），
这样一万手的统计可以在单测里跑一百手验完。

```python
result = run_batch(MatchConfig(styles=("solved", "tag", "lag", "nit", "station", "maniac"),
                               hands=10_000, seed=7))
print(result.report())
```

## 口径

- **每手重置筹码**：现金局的评估口径。上一手输光不影响下一手，所以 bb/100 衡量的是
  策略本身，而不是一次破产的运气。
- **按钮每手右移一位**：六手一圈，每个座位在每个位置上的手数完全相同。位置带来的
  系统性差异因此互相抵消，剩下的才是风格的差异。
- **bb/100 带置信区间**：单手盈亏的方差极大（一次 100bb 的全下抵得上几百手小池），
  不给区间的 bb/100 是没有意义的数字。算法在 `metrics.py`——全项目只有那一处定义。
  想比出两个 bot 的强弱，看的是**区间重不重叠**，不是点估计谁高。
- **翻前统计与主流软件对齐**：VPIP（主动投钱）、PFR（主动加注）、3bet（面对开牌再加注）。
  盲注不算主动投钱，大盲过牌也不算。

## 加速

三条路，效果依次递减：

1. **分片并行**（`shard`）：把手数切成几段，每段一个进程，最后 `merge`。段之间只差
   种子与起始按钮位，所以合并结果与顺序跑的口径完全一致。
2. **降蒙特卡洛采样**（`MatchConfig.samples`）：bot 翻后要估权益，默认每次 160 个样本。
   压到 40 能快两三倍，代价是翻后决策更噪——**量强弱时别调它**，喂统计数据时可以。
3. 减少手数——但那是拿置信区间换时间，通常得不偿失。

统计口径与 `history.action_records` 共用同一套定义，M2 的 HUD 因此不会与这里的数字打架。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from random import Random

from .bots import STYLES, Bot, BotStyle, play_out
from .deck import deck_from_seed
from .history import action_records
from .metrics import bb_per_100, bb_per_100_interval
from .state import PREFLOP, HandConfig, HandState

__all__ = [
    "MatchConfig",
    "SeatStats",
    "BatchResult",
    "run_batch",
    "shard",
    "merge",
]


# ------------------------------------------------------------------ 配置


@dataclass(frozen=True)
class MatchConfig:
    """一场批量对局的全部参数。可哈希、可 pickle，能直接发给子进程。"""

    styles: tuple[str, ...]
    """每个座位的风格名（`bots.STYLES` 的键）。座位数就是它的长度。"""
    hands: int = 1000
    big_blind: int = 100
    """以筹码点计的大盲。取 100 是为了 2.5bb 这类尺度不用取整。"""
    small_blind: int = 50
    ante: int = 0
    start_stack: int = 10_000
    """每手开局的筹码，默认 100bb。"""
    seed: int = 0
    first_button: int = 0
    samples: int | None = None
    """覆盖所有风格的蒙特卡洛采样数；`None` 表示各用各的默认值。"""

    def __post_init__(self) -> None:
        if not 2 <= len(self.styles) <= 9:
            raise ValueError(f"座位数要在 2..9 之间，收到 {len(self.styles)}")
        unknown = [name for name in self.styles if name not in STYLES]
        if unknown:
            raise ValueError(f"未知风格: {unknown}，可选 {sorted(STYLES)}")
        if self.hands < 1:
            raise ValueError("至少要打一手")
        if self.big_blind <= 0 or self.small_blind < 0 or self.ante < 0:
            raise ValueError("盲注或前注不合法")
        if self.start_stack < 2 * self.big_blind:
            raise ValueError("起始筹码至少两个大盲")
        if not 0 <= self.first_button < len(self.styles):
            raise ValueError(f"按钮位越界: {self.first_button}")
        if self.samples is not None and self.samples < 1:
            raise ValueError("采样数至少为 1")

    @property
    def num_seats(self) -> int:
        return len(self.styles)

    def resolved_styles(self) -> tuple[BotStyle, ...]:
        """把风格名换成风格对象，顺带套上采样数的覆盖。"""
        return tuple(
            STYLES[name] if self.samples is None else replace(STYLES[name], samples=self.samples)
            for name in self.styles
        )


# ------------------------------------------------------------------ 统计


@dataclass
class SeatStats:
    """一个座位的累计统计。可加：分片跑完直接相加即可。"""

    seat: int
    style: str
    hands: int = 0
    net: int = 0
    """净盈亏，筹码点。"""
    net_squares: float = 0.0
    """单手盈亏的平方和——只为算方差，不然置信区间没法在分片之间合并。"""
    vpip_hands: int = 0
    pfr_hands: int = 0
    open_chances: int = 0
    """轮到自己时前面的人全弃了的手数——「第一个入池」的机会。"""
    open_hands: int = 0
    threebet_chances: int = 0
    """面对（且只面对过一次）加注的手数。"""
    threebet_hands: int = 0
    flops: int = 0
    showdowns: int = 0
    showdown_wins: int = 0
    table_decisions: int = 0
    """照范围表走的翻前决策数。"""
    fallback_decisions: int = 0

    def add(self, other: "SeatStats") -> None:
        if (self.seat, self.style) != (other.seat, other.style):
            raise ValueError("只能合并同一个座位、同一种风格的统计")
        for name, value in vars(other).items():
            if name in ("seat", "style"):
                continue
            setattr(self, name, getattr(self, name) + value)

    # ---------------------------------------------------------- 派生指标

    @staticmethod
    def _ratio(part: int, whole: int) -> float:
        return part / whole if whole else 0.0

    @property
    def vpip(self) -> float:
        return self._ratio(self.vpip_hands, self.hands)

    @property
    def pfr(self) -> float:
        return self._ratio(self.pfr_hands, self.hands)

    @property
    def open_rate(self) -> float:
        """有机会第一个入池时真开牌的比例，和范围表的开牌频率是同一个口径。

        **是所有位置混在一起的平均值**：桌上对手越紧，第一个入池的机会就越多地落在
        靠后的位置上，这个数会跟着涨。要与范围表逐格对照，得按位置拆开（M2 的活）。
        """
        return self._ratio(self.open_hands, self.open_chances)

    @property
    def threebet(self) -> float:
        return self._ratio(self.threebet_hands, self.threebet_chances)

    @property
    def wtsd(self) -> float:
        """看到翻牌之后走到摊牌的比例。"""
        return self._ratio(self.showdowns, self.flops)

    @property
    def won_at_showdown(self) -> float:
        return self._ratio(self.showdown_wins, self.showdowns)

    @property
    def solve_coverage(self) -> float:
        """翻前决策里有多大比例是照解走的（其余落在规则兜底上）。"""
        return self._ratio(
            self.table_decisions, self.table_decisions + self.fallback_decisions
        )

    def bb_per_100(self, big_blind: int) -> float:
        return bb_per_100(self.net, self.hands, big_blind)

    def bb_per_100_interval(self, big_blind: int) -> float:
        return bb_per_100_interval(self.net, self.net_squares, self.hands, big_blind)


@dataclass
class BatchResult:
    config: MatchConfig
    seats: list[SeatStats]
    hands: int = 0
    actions: int = 0
    showdown_hands: int = 0

    def seat(self, index: int) -> SeatStats:
        return self.seats[index]

    def add(self, other: "BatchResult") -> None:
        if self.config.styles != other.config.styles:
            raise ValueError("只能合并同一桌（同样的座位风格）的结果")
        if self.config.big_blind != other.config.big_blind:
            raise ValueError("盲注不同的两批对局不能合并")
        self.hands += other.hands
        self.actions += other.actions
        self.showdown_hands += other.showdown_hands
        for mine, theirs in zip(self.seats, other.seats):
            mine.add(theirs)

    @property
    def showdown_rate(self) -> float:
        return self.showdown_hands / self.hands if self.hands else 0.0

    def is_zero_sum(self) -> bool:
        """不抽水的桌子，所有人的净盈亏之和必须是 0。"""
        return sum(seat.net for seat in self.seats) == 0

    def report(self) -> str:
        """一行一个座位的对局报告。"""
        bb = self.config.big_blind
        lines = [
            f"{self.hands:,} 手 · {self.config.num_seats} 人 · "
            f"起始 {self.config.start_stack / bb:g}bb · 摊牌 {self.showdown_rate:.1%}",
            "座位 风格         bb/100 (95%)      VPIP   PFR  开牌  3bet  WTSD  W$SD  照解",
        ]
        for seat in self.seats:
            value = seat.bb_per_100(bb)
            margin = seat.bb_per_100_interval(bb)
            # 风格名是中文，一个字占两格——按显示宽度补空格，别用 %-4s（它按字数算）
            label = STYLES[seat.style].label
            lines.append(
                f"{seat.seat:>2d}   {label}{' ' * max(0, 8 - 2 * len(label))}"
                f"{value:>9.2f} ±{margin:>6.2f}"
                f"{seat.vpip:>8.1%}{seat.pfr:>6.1%}{seat.open_rate:>6.1%}"
                f"{seat.threebet:>6.1%}{seat.wtsd:>6.1%}{seat.won_at_showdown:>6.1%}"
                f"{seat.solve_coverage:>6.1%}"
            )
        return "\n".join(lines)


# ------------------------------------------------------------------ 对局


def run_batch(config: MatchConfig, *, on_hand=None) -> BatchResult:
    """打完 `config.hands` 手，回一份统计。

    `on_hand` 可传 `callable(index, HandState)`，每打完一手调用一次——落库、导牌谱、
    画进度条都挂在这里，本函数自己不碰 IO。
    """
    seats = config.num_seats
    master = Random(config.seed)
    styles = config.resolved_styles()
    # 每个 bot 一条独立的随机流，种子由主流派生：整场对局只由 config.seed 一个数决定
    bots = {
        seat: Bot(styles[seat], seed=master.randrange(1 << 30)) for seat in range(seats)
    }
    stats = [SeatStats(seat=s, style=config.styles[s]) for s in range(seats)]
    result = BatchResult(config=config, seats=stats)
    stacks = tuple(config.start_stack for _ in range(seats))

    for index in range(config.hands):
        hand = HandState(
            HandConfig(
                stacks=stacks,
                button=(config.first_button + index) % seats,
                big_blind=config.big_blind,
                small_blind=config.small_blind,
                ante=config.ante,
            ),
            deck_from_seed(master.randrange(1 << 30)),
        )
        play_out(hand, bots)
        result.actions += _collect(hand, stats)
        result.hands += 1
        if hand.result.went_to_showdown:
            result.showdown_hands += 1
        if on_hand is not None:
            on_hand(index, hand)

    for seat, bot in bots.items():
        stats[seat].table_decisions = bot.table_hits
        stats[seat].fallback_decisions = bot.fallback_hits
    return result


def _collect(hand: HandState, stats: list[SeatStats]) -> int:
    """把一手牌折进统计，返回这手牌的动作数。"""
    records = action_records(hand)
    seats = hand.config.num_seats
    big_blind = hand.config.big_blind

    voluntary = [False] * seats
    raised = [False] * seats
    opened = [False] * seats
    open_chance = [False] * seats
    threebet = [False] * seats
    threebet_chance = [False] * seats
    folded_preflop = [False] * seats
    raises = 0
    entered = False
    """前面有没有人主动入池（跛入也算）。「第一个入池」的机会只在没人进过时才有。"""

    for record in records:
        seat = record.seat
        if record.street != PREFLOP:
            continue
        # 前面的人全弃了：现在这一下就是「第一个入池」的机会，与范围表的开牌口径一致。
        # 分子分母必须用同一个判据——跟在跛入者后面的加注是「隔离」，不是开牌。
        first_in = not entered
        if first_in:
            open_chance[seat] = True
        elif raises == 1 and record.to_call > 0:
            threebet_chance[seat] = True

        if record.kind == "fold":
            folded_preflop[seat] = True
        if record.is_voluntary:
            voluntary[seat] = True
            entered = True
        if record.kind in ("bet", "raise"):
            raised[seat] = True
            if first_in:
                opened[seat] = True
            elif raises == 1:
                threebet[seat] = True
            raises += 1

    saw_flop = len(hand.board) >= 3
    net = hand.result.net
    showdown_seats = hand.result.showdown_scores

    for seat in range(seats):
        entry = stats[seat]
        entry.hands += 1
        entry.net += net[seat]
        entry.net_squares += float(net[seat]) * net[seat]
        entry.vpip_hands += voluntary[seat]
        entry.pfr_hands += raised[seat]
        entry.open_chances += open_chance[seat]
        entry.open_hands += opened[seat]
        entry.threebet_chances += threebet_chance[seat]
        entry.threebet_hands += threebet[seat]
        if saw_flop and not folded_preflop[seat]:
            entry.flops += 1
        if seat in showdown_seats:
            entry.showdowns += 1
            if net[seat] > 0:
                entry.showdown_wins += 1

    # 大盲的盲注不是主动投钱，但翻前无人加注时他连一条记录都没有——上面按记录统计，
    # 天然不会把他算进 VPIP，这里不用再修正。
    return len(records)


# ------------------------------------------------------------------ 分片与合并


def shard(config: MatchConfig, parts: int) -> list[MatchConfig]:
    """把一场对局切成 `parts` 段，每段可以独立跑（进程池、机器都行）。

    段与段之间只差**种子**与**起始按钮位**：按钮位接着上一段往下排，所以合起来的
    位置分布与一口气跑完全一样。手数不够分时段数自动缩小。
    """
    if parts < 1:
        raise ValueError("至少要分一段")
    parts = min(parts, config.hands)
    base, extra = divmod(config.hands, parts)
    shards = []
    played = 0
    for index in range(parts):
        count = base + (1 if index < extra else 0)
        shards.append(
            replace(
                config,
                hands=count,
                seed=config.seed + 1_000_003 * (index + 1),
                first_button=(config.first_button + played) % config.num_seats,
            )
        )
        played += count
    return shards


def merge(results: "list[BatchResult] | tuple[BatchResult, ...]") -> BatchResult:
    """把分片结果合成一份。方差用平方和合并，所以置信区间与顺序跑的口径一致。"""
    if not results:
        raise ValueError("没有结果可合并")
    first, *rest = results
    total = BatchResult(
        config=replace(first.config, hands=sum(r.config.hands for r in results)),
        seats=[replace(seat) for seat in first.seats],
        hands=first.hands,
        actions=first.actions,
        showdown_hands=first.showdown_hands,
    )
    for other in rest:
        total.add(other)
    return total

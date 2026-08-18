"""跟 Slumbot 打一场，量出带置信区间的 bb/100（FR-6）。

Slumbot 是 ADR-0002 定下的**外部标尺**：能免费接口调用、强度有据（2018 年 ACPC 冠军级）
的单挑对手。这里把它接成一场正经对局——我们的 bot 坐一边，打 N 手，报 bb/100 ± 区间。

```python
stats = play_match(Session(), Bot("solved").act, hands=2000)
print(stats.report())
```

## 策略接口

`strategy` 就是一个 `callable(HandState) -> Action`，`holdem.bots.Bot.act` 直接符合。
换任何别的策略（新 bot、纯规则、人来点）都不用碰这里——协议翻译在 `protocol.py`，
它交给策略的是一份和本地自对弈时**完全一样**的 `HandState`。

## 三条必须写在结论旁边的话

1. **单挑 200bb**，不是我们首攻的六人 100bb。位置与深度都不同，这个数字衡量的是
   「我们的策略在单挑深筹码下的强度」，别直接当六人桌的水平。
2. **对手不是 GTO**，赢 Slumbot 不等于接近均衡，只等于打得过它。
   我们这边翻前照单挑 200bb 范围表打（`preflop_ranges_hu_200bb.json`，整树精确解），
   **翻后仍是规则启发式**——所以这个数字是「解 + 规则翻后」这一整套的强度，
   不是翻前解本身的强度。
3. **方差极大**：单挑 200bb 一手就能输赢两百个大盲。2000 手的 95% 区间通常还有
   ±30bb/100 上下——**看区间是否含 0**，别看点估计的正负。想把区间缩一半要四倍手数。

## 每一手都对账

Slumbot 回的 `winnings` 是外部真值。每手结束都拿它核对一次我们对这手牌的理解
（`protocol.check_result`）：动作串解析错、位置认反、金额口径错，都会当场露馅而不是
悄悄变成一个「看着挺像」的 bb/100。对不上的手会被记下来，报告里显式列出。

## 出错的手不算数

网络抖动或非法动作会让一手作废：这时换一条会话重开，**那一手不计入统计**。
（老教训：出错重打时把半手的观察重复计进去，20 手能采出 28 个样本。）
"""

from __future__ import annotations

from dataclasses import dataclass, field

from holdem.metrics import bb_per_100, bb_per_100_interval

from .client import Session, SlumbotError
from .protocol import BIG_BLIND, HandView, build_state, check_result, to_incr

__all__ = ["MatchStats", "play_hand", "play_match"]

MAX_ACTIONS = 40
"""一手牌里我们最多说这么多次话；超了就是死循环，宁可报错也别空转。"""


@dataclass
class MatchStats:
    """一场对局的累计结果。bb/100 与区间的算法与批量自对弈共用一套（`holdem.metrics`）。"""

    hands: int = 0
    net: int = 0
    """净盈亏，筹码。Slumbot 的盲注是 50/100。"""
    net_squares: float = 0.0
    hands_as_button: int = 0
    net_as_button: int = 0
    hands_as_big_blind: int = 0
    net_as_big_blind: int = 0
    showdowns: int = 0
    aborted: int = 0
    """出错作废的手数，不计入上面的统计。"""
    mismatches: list[str] = field(default_factory=list)
    """与 Slumbot 的结果对不上的手；有一条都要当回事。"""
    table_decisions: int = 0
    fallback_decisions: int = 0

    def add(self, other: "MatchStats") -> None:
        """把另一场（另一条会话）的结果并进来。

        方差靠「和 + 平方和」合并，所以几条会话并行打完直接相加，
        置信区间与一条会话打同样多手是同一个口径。
        """
        for name, value in vars(other).items():
            if name == "mismatches":
                self.mismatches.extend(value)
            else:
                setattr(self, name, getattr(self, name) + value)

    def add_hand(self, view: HandView) -> None:
        if view.winnings is None:
            raise ValueError("这手还没结束，不能计入统计")
        won = int(view.winnings)
        self.hands += 1
        self.net += won
        self.net_squares += float(won) * won
        if view.client_pos == 1:
            self.hands_as_button += 1
            self.net_as_button += won
        else:
            self.hands_as_big_blind += 1
            self.net_as_big_blind += won
        if not view.action.endswith("f"):
            self.showdowns += 1

    # ---------------------------------------------------------- 派生指标

    @property
    def bb100(self) -> float:
        return bb_per_100(self.net, self.hands, BIG_BLIND)

    @property
    def interval(self) -> float:
        return bb_per_100_interval(self.net, self.net_squares, self.hands, BIG_BLIND)

    @property
    def beats_slumbot(self) -> bool | None:
        """置信区间整个在 0 以上才算「赢了」；跨 0 就是没测出来，回 `None`。"""
        if self.hands < 2:
            return None
        if self.bb100 - self.interval > 0:
            return True
        if self.bb100 + self.interval < 0:
            return False
        return None

    @property
    def solve_coverage(self) -> float:
        total = self.table_decisions + self.fallback_decisions
        return self.table_decisions / total if total else 0.0

    def report(self) -> str:
        verdict = {
            True: "区间整个在 0 以上：这一档确实赢它",
            False: "区间整个在 0 以下：这一档确实输它",
            None: "区间跨 0：手数还不够，分不出胜负",
        }[self.beats_slumbot]
        lines = [
            f"对 Slumbot {self.hands:,} 手（单挑 200bb，作废 {self.aborted} 手）",
            f"  bb/100   {self.bb100:+.2f} ± {self.interval:.2f}（95%）  → {verdict}",
            f"  按钮位   {bb_per_100(self.net_as_button, self.hands_as_button, BIG_BLIND):+.2f}"
            f"（{self.hands_as_button:,} 手）",
            f"  大盲位   {bb_per_100(self.net_as_big_blind, self.hands_as_big_blind, BIG_BLIND):+.2f}"
            f"（{self.hands_as_big_blind:,} 手）",
            f"  摊牌率   {self.showdowns / self.hands:.1%}" if self.hands else "",
            f"  照解走   {self.solve_coverage:.1%} 的翻前决策"
            f"（其余落在规则兜底上——4bet 之后的局面表里没有）",
        ]
        if self.mismatches:
            lines.append(f"  ⚠ 有 {len(self.mismatches)} 手与 Slumbot 的结果对不上：")
            lines.extend(f"      {text}" for text in self.mismatches[:5])
        return "\n".join(line for line in lines if line)


def play_hand(session: Session, strategy, *, rest_seed: int = 0) -> HandView:
    """打一手，回这手的终局视图。中途出错就把异常抛出去，由调用方作废这一手。"""
    view = HandView.from_body(session.new_hand())
    for _ in range(MAX_ACTIONS):
        if view.is_over:
            return view
        hand = build_state(view, rest_seed=rest_seed)
        body = session.act(to_incr(strategy(hand)))
        view = _updated(view, body)
    raise RuntimeError(f"一手牌说了 {MAX_ACTIONS} 次话还没结束：{view.action!r}")


def _updated(previous: HandView, body: dict) -> HandView:
    """用新回包更新视图。回包里没带的字段（底牌、位置）沿用上一版。"""
    return HandView(
        hole=tuple(body.get("hole_cards") or previous.hole),
        board=tuple(body.get("board") or previous.board),
        action=body.get("action") or previous.action,
        client_pos=int(body.get("client_pos", previous.client_pos)),
        winnings=body.get("winnings"),
    )


def play_match(
    session: Session,
    strategy,
    *,
    hands: int,
    on_hand=None,
    max_errors: int | None = None,
    bot=None,
) -> MatchStats:
    """打 `hands` 手完整的牌，回统计。

    `on_hand(index, view, stats)` 每打完一手调用一次（打印进度、存盘都挂这儿）。
    `bot` 给一个 `holdem.bots.Bot` 的话，顺带把它的「照解/兜底」计数抄进统计。
    """
    if hands < 1:
        raise ValueError("至少要打一手")
    stats = MatchStats()
    budget = max_errors if max_errors is not None else max(20, hands // 20)

    while stats.hands < hands:
        try:
            view = play_hand(session, strategy, rest_seed=stats.hands)
        except (SlumbotError, OSError, KeyError, ValueError) as exc:
            stats.aborted += 1
            if stats.aborted > budget:
                raise RuntimeError(
                    f"作废了 {stats.aborted} 手，最后一个错误：{type(exc).__name__}: {exc}"
                ) from exc
            session.reset()
            continue

        problem = check_result(view, rest_seed=stats.hands)
        if problem:
            stats.mismatches.append(f"第 {stats.hands + 1} 手：{problem}（{view.action}）")
        stats.add_hand(view)
        if bot is not None:
            # 每手都抄一次，断点存下来的快照才带得上覆盖率（是赋值不是累加，抄多少次都一样）
            stats.table_decisions = bot.table_hits
            stats.fallback_decisions = bot.fallback_hits
        if on_hand is not None:
            on_hand(stats.hands, view, stats)

    return stats

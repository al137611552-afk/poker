"""量各风格的统计画像，对照真人参考区间（FR-11 的验收标准）。

PRD 对 FR-11 的验收写的是「全 bot 对打统计落在合理区间」——不是 bb/100 最大。

## 分两类指标，量法不同（2026-08-24 实测修正）

这个脚本最初对所有指标都用**全桌同风格**自对弈，理由是「那才是这个风格本身」。
**那个判断只对一半指标成立**：

| 类别 | 指标 | 换对手池会不会变 |
|---|---|---|
| 自身指标 | VPIP / PFR / RFI / 3bet | 几乎不变（tag 全桌 18.1%，对 5 个 station 17.9%）|
| 交互指标 | WTSD / W$SD / AF | **变得很厉害**（同一个 tag：30.9% → 67.3%）|

所以交互指标必须在一个**标准对手池**里量，否则量到的是「这种桌子长什么样」，
不是「这个玩家长什么样」。真人 HUD 上的数也正是这么来的——一个玩家在正常桌里的统计。

标准池取 **5 个 tag**（默认风格、也最接近真实牌桌的主体）。待测风格坐 1 个座位。

    python3 scripts/style_profile.py --hands 8000
    python3 scripts/style_profile.py --self-play      # 全桌同风格，看极端场景
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from holdem.batch import MatchConfig, run_batch  # noqa: E402
from holdem.stats import StatLine, accumulate  # noqa: E402

# 六人桌真人参考区间（PT4/HM3 常见口径）。**上下界是给人看的对照，不是硬门**——
# 风格本来就该有差异：nit 的 VPIP 低于 TAG 区间是对的，不是 bug。
#
# 两条读数纪律：
# 1. **单次结果有 ±0.5pp 级的种子波动**（实测 tag 的 WTSD 四个种子：31.3/31.8/32.2/32.4）。
#    贴着边界的「出界」别当真，换两个种子再看。
# 2. **这些区间是常识值、不是权威数据**。为了让 bot 精确落进去而调参是本末倒置——
#    调出来的是对这张表过拟合的参数，不是更像真人的打法。
REFERENCE = {
    "vpip": (0.18, 0.28, "入池率"),
    "pfr": (0.13, 0.22, "翻前加注"),
    "wtsd": (0.24, 0.32, "看到翻牌后走到摊牌"),
    "wsd": (0.48, 0.58, "摊牌赢下"),
}
AF_RANGE = (1.5, 3.5)


REFERENCE_POOL = "tag"
"""标准对手池的风格。交互指标（WTSD/W$SD/AF）必须在固定池里量才可比。"""


def profile(style: str, hands: int, seed: int, *, self_play: bool = False) -> StatLine:
    """量一个风格的画像。

    默认：待测风格坐 **1 个座位**，其余 5 个是标准池（`tag`），只统计那个座位。
    `self_play=True`：全桌同风格——那量的是「这种桌子长什么样」，
    交互指标会跑偏，只在想看极端场景时用。
    """
    styles = (style,) * 6 if self_play else (style,) + (REFERENCE_POOL,) * 5
    lines: dict = {}
    run_batch(MatchConfig(styles=styles, hands=hands, seed=seed),
              on_hand=lambda index, hand: accumulate(
                  hand, lines, key_of=lambda seat: "hero" if seat == 0 else "pool"))
    if self_play:
        total = StatLine()
        for line in lines.values():
            total.add(line)
        return total
    return lines.get("hero", StatLine())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--styles", default="tag,lag,nit,station,maniac")
    ap.add_argument("--self-play", action="store_true",
                    help="全桌同风格（交互指标会跑偏，只看极端场景时用）")
    args = ap.parse_args()

    pool = "全桌同风格（交互指标会跑偏）" if args.self_play else f"1 个待测 + 5 个 {REFERENCE_POOL}"
    print(f"每种风格 {args.hands} 手，种子 {args.seed}，对手池：{pool}\n")
    header = f"{'风格':<10}{'VPIP':>8}{'PFR':>8}{'WTSD':>8}{'W$SD':>8}{'AF':>7}{'RFI':>8}{'3bet':>8}"
    print(header)
    print("-" * len(header))

    rows = {}
    for style in args.styles.split(","):
        line = profile(style, args.hands, args.seed, self_play=args.self_play)
        rows[style] = line
        af = line.aggression_factor
        print(f"{style:<10}{_pct(line.vpip.rate):>8}{_pct(line.pfr.rate):>8}"
              f"{_pct(line.wtsd.rate):>8}{_pct(line.wsd.rate):>8}"
              f"{(f'{af:.2f}' if af else '—'):>7}"
              f"{_pct(line.rfi.rate):>8}{_pct(line.threebet.rate):>8}")

    print(f"\n真人参考区间（六人桌）：", end="")
    print("、".join(f"{name} {lo:.0%}–{hi:.0%}" for _, (lo, hi, name) in REFERENCE.items()))
    print(f"AF {AF_RANGE[0]}–{AF_RANGE[1]}")

    # 只对 tag 判「落没落在区间里」：它是默认风格，也是 PRD 说的那个「合理区间」的对象。
    # 其余风格**本来就该出界**——nit 不紧、maniac 不疯才是 bug。
    print("\ntag（默认风格）逐项对照：")
    line = rows.get("tag")
    if line is None:
        return 0
    for key, (lo, hi, name) in REFERENCE.items():
        rate = getattr(line, key).rate
        verdict = "✓" if rate is not None and lo <= rate <= hi else "✗ 出界"
        print(f"  {name:<18}{_pct(rate):>8}   参考 {lo:.0%}–{hi:.0%}   {verdict}")
    af = line.aggression_factor
    ok = af is not None and AF_RANGE[0] <= af <= AF_RANGE[1]
    print(f"  {'攻击性 AF':<18}{(f'{af:.2f}' if af else '—'):>8}   "
          f"参考 {AF_RANGE[0]}–{AF_RANGE[1]}   {'✓' if ok else '✗ 出界'}")
    return 0


def _pct(value: "float | None") -> str:
    return "—" if value is None else f"{value:.1%}"


if __name__ == "__main__":
    raise SystemExit(main())

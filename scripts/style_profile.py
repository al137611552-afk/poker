"""量各风格的统计画像，对照真人参考区间（FR-11 的验收标准）。

PRD 对 FR-11 的验收写的是「全 bot 对打统计落在合理区间」——不是 bb/100 最大。
**统计画像依赖对手池**：同一个 bot 在跟注站堆里和在紧手堆里，WTSD 能差一倍。
所以这里量的是**全桌同风格**的自对弈，那才是「这个风格本身长什么样」。

    python3 scripts/style_profile.py --hands 8000
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


def profile(style: str, hands: int, seed: int) -> StatLine:
    lines: dict = {}
    run_batch(MatchConfig(styles=(style,) * 6, hands=hands, seed=seed),
              on_hand=lambda index, hand: accumulate(hand, lines))
    total = StatLine()
    for line in lines.values():
        total.add(line)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--styles", default="tag,lag,nit,station,maniac")
    args = ap.parse_args()

    print(f"每种风格全桌自对弈 {args.hands} 手，种子 {args.seed}\n")
    header = f"{'风格':<10}{'VPIP':>8}{'PFR':>8}{'WTSD':>8}{'W$SD':>8}{'AF':>7}{'RFI':>8}{'3bet':>8}"
    print(header)
    print("-" * len(header))

    rows = {}
    for style in args.styles.split(","):
        line = profile(style, args.hands, args.seed)
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

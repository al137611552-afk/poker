"""扫翻后收缩参数（ADR-0008 的遗留：那几个数至今是拍的）。

每组参数打一场**同种子**的对局：座位 0/2/4 用该组参数（范围感知），
1/3/5 用随机口径当固定基线。比的是范围组的 bb/100。

**同种子是关键**：牌一样，差异才只来自决策。不同种子跑出来的差异会被
单手盈亏的巨大方差淹掉——那正是这类校准最容易得出假结论的地方。

    python3 scripts/calibrate_narrowing.py --hands 8000 --sweep bet
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from holdem.batch import MatchConfig, run_batch, shard  # noqa: E402
from holdem.postflop_ranges import KeepProfile  # noqa: E402

AWARE = (0, 2, 4)
BB = 10

SWEEPS = {
    # 1.00 那一组是**关键对照，不是补齐网格**：它等于「只用翻前范围、逐街完全不收缩」。
    # 如果它不比别的差，那说明这个模块的价值全在「用范围代替随机牌」，
    # 逐街收缩纯属白干——那是要改设计的结论，不是调参数的结论。
    "bet": [KeepProfile(bet=v) for v in (0.35, 0.55, 0.80, 1.00)],
    "call": [KeepProfile(call=v) for v in (0.55, 0.65, 0.75, 0.85, 0.95)],
    "check": [KeepProfile(check=v) for v in (0.80, 0.90, 1.00)],
}


def _run(config: MatchConfig):
    return run_batch(config)


def measure(profile: KeepProfile, hands: int, seed: int, workers: int):
    config = MatchConfig(styles=("tag",) * 6, hands=hands, seed=seed,
                         range_aware_seats=AWARE, keep_profile=profile)
    parts = shard(config, workers)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run, parts))
    else:
        results = [run_batch(parts[0])]
    merged = results[0]
    for other in results[1:]:
        for a, b in zip(merged.seats, other.seats):
            a.add(b)
        merged.hands += other.hands

    aware = [s for s in merged.seats if s.seat in AWARE]
    net = sum(s.net for s in aware)
    played = sum(s.hands for s in aware)
    used = sum(s.range_decisions for s in aware)
    missed = sum(s.range_missed for s in aware)
    wtsd = sum(s.showdowns for s in aware) / max(1, sum(s.flops for s in aware))
    return {
        "bb100": net / played / BB * 100 if played else 0.0,
        "half": sum(s.bb_per_100_interval(BB) for s in aware) / len(aware),
        "coverage": used / (used + missed) if used + missed else 0.0,
        "wtsd": wtsd,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sweep", choices=sorted(SWEEPS), default="bet")
    args = ap.parse_args()

    profiles = SWEEPS[args.sweep]
    print(f"扫 {args.sweep}：{len(profiles)} 组 × {args.hands} 手，种子 {args.seed}（各组同种子）\n")
    print(f"{'取值':>6}{'bb/100':>10}{'±95%(单座)':>12}{'WTSD':>8}{'覆盖率':>8}{'耗时':>7}")

    rows = []
    for profile in profiles:
        started = time.perf_counter()
        out = measure(profile, args.hands, args.seed, args.workers)
        value = getattr(profile, args.sweep)
        rows.append((value, out))
        print(f"{value:>6.2f}{out['bb100']:>10.1f}{out['half']:>12.1f}"
              f"{out['wtsd']:>8.1%}{out['coverage']:>8.0%}"
              f"{time.perf_counter() - started:>6.0f}s")

    best = max(rows, key=lambda r: r[1]["bb100"])
    spread = best[1]["bb100"] - min(r[1]["bb100"] for r in rows)
    noise = sum(r[1]["half"] for r in rows) / len(rows)
    print(f"\n最好的是 {args.sweep}={best[0]:.2f}（{best[1]['bb100']:.1f} bb/100）")
    print(f"组间极差 {spread:.1f} bb/100，单座 95% 半宽约 {noise:.1f}")
    if spread < noise:
        # **这句必须打出来**：极差比噪声还小的时候，"最好的那一组"只是噪声的最高点
        print("→ **极差小于噪声，这次扫不出结论**：别据此改默认值，加手数或换更细的判据再来")
    else:
        print("→ 极差大过噪声，方向可信；但仍建议换一个种子复跑一次再定")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

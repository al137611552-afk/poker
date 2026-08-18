"""一键让一桌 bot 打一万手（FR-4）。

```bash
python3 scripts/play_batch.py                          # 六种风格各一个座位，1000 手
python3 scripts/play_batch.py --hands 10000 --workers 2
python3 scripts/play_batch.py --seats solved,solved,tag,tag,nit,nit --seed 7
python3 scripts/play_batch.py --hands 2000 --db /tmp/session.sqlite   # 顺带落库喂统计
```

对局本身是纯逻辑（`holdem.batch`），这里只管三件 IO 的事：**开进程池**、**打印**、
**落库**。所以想在别处复用（服务端点一下就跑一万手），直接调 `holdem.batch.run_batch`。

## 加速

`--workers N` 把手数切成 N 段丢进 N 个进程，最后按平方和合并——结果口径与顺序跑
一模一样（段之间只差种子与起始按钮位）。本机双核，`--workers 2` 大约省一半时间。
`--samples` 能再快两三倍，但那是拿翻后决策质量换的，**量强弱时别动它**。

`--db` 会把每一手写进 SQLite（M2 的统计从那里取数）。落库是 IO，为了不让几个进程
争同一个文件，给了 `--db` 就退回单进程。
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from holdem.batch import BatchResult, MatchConfig, merge, run_batch, shard  # noqa: E402
from holdem.bots import STYLES  # noqa: E402
from holdem.store import HandStore  # noqa: E402

DEFAULT_SEATS = "solved,tag,lag,nit,station,maniac"


def _run_shard(config: MatchConfig) -> BatchResult:
    """进程池里的活儿。必须是模块级函数，否则 pickle 不了。"""
    return run_batch(config)


def run_parallel(config: MatchConfig, workers: int, *, quiet: bool = False) -> BatchResult:
    if workers <= 1:
        return run_batch(config, on_hand=None if quiet else _ticker(config.hands))
    parts = shard(config, workers)
    with ProcessPoolExecutor(max_workers=len(parts)) as pool:
        return merge(list(pool.map(_run_shard, parts)))


def _ticker(total: int):
    """每 5% 打一个进度点，长跑时好知道它还活着。"""
    step = max(total // 20, 1)

    def on_hand(index: int, hand) -> None:
        if (index + 1) % step == 0:
            done = index + 1
            print(f"  已打 {done:,}/{total:,} 手（{done / total:.0%}）", flush=True)

    return on_hand


def run_with_store(config: MatchConfig, path: Path, *, quiet: bool = False) -> BatchResult:
    """单进程跑，同时把每一手写进 SQLite。"""
    players = [f"{seat}-{STYLES[name].label}" for seat, name in enumerate(config.styles)]
    tick = None if quiet else _ticker(config.hands)
    with HandStore(path) as store:
        session_id = store.create_session(
            f"批量自对弈 {config.num_seats} 人",
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            ante=config.ante,
            notes=f"seed={config.seed} styles={','.join(config.styles)}",
        )

        def on_hand(index: int, hand) -> None:
            store.save_hand(hand, session_id=session_id, players=players, hand_no=index + 1)
            if tick is not None:
                tick(index, hand)

        result = run_batch(config, on_hand=on_hand)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批量 bot 自对弈")
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seats", default=DEFAULT_SEATS, help=f"逗号分隔的风格名，可选 {sorted(STYLES)}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="并行进程数（给了 --db 则强制 1）")
    parser.add_argument("--samples", type=int, default=None, help="覆盖翻后蒙特卡洛采样数")
    parser.add_argument("--stack", type=float, default=100.0, help="起始筹码（大盲）")
    parser.add_argument("--ante", type=float, default=0.0, help="每人前注（大盲）")
    parser.add_argument("--db", type=Path, default=None, help="把每一手写进这个 SQLite 文件")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    big_blind = 100
    config = MatchConfig(
        styles=tuple(name.strip() for name in args.seats.split(",") if name.strip()),
        hands=args.hands,
        big_blind=big_blind,
        small_blind=big_blind // 2,
        ante=int(round(args.ante * big_blind)),
        start_stack=int(round(args.stack * big_blind)),
        seed=args.seed,
        samples=args.samples,
    )

    started = time.perf_counter()
    if args.db is not None:
        if args.workers > 1:
            print("落库时退回单进程（几个进程写同一个 SQLite 会打架）", flush=True)
        result = run_with_store(config, args.db, quiet=args.quiet)
    else:
        result = run_parallel(config, args.workers, quiet=args.quiet)
    elapsed = time.perf_counter() - started

    print()
    print(result.report())
    print()
    print(f"用时 {elapsed:.1f}s，{result.hands / elapsed:,.0f} 手/秒"
          f"（{args.workers} 进程，平均 {result.actions / result.hands:.1f} 动作/手）")
    if not result.is_zero_sum():
        print("⚠ 净盈亏之和不为零——引擎的筹码守恒被破坏了，别信这批数字")
        return 1
    widest = max(seat.bb_per_100_interval(config.big_blind) for seat in result.seats)
    if widest > 20:
        print(
            f"⚠ 最宽的置信区间还有 ±{widest:.0f} bb/100，手数不足以分出强弱"
            f"（要缩一半区间得打四倍手数）"
        )
    if args.db is not None:
        print(f"牌谱已写入 {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

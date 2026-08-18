"""拿我们的 bot 跟 Slumbot 打，跑出带置信区间的 bb/100（FR-6）。

```bash
python3 scripts/play_slumbot.py --hands 200            # 先小跑一段看链路通不通
python3 scripts/play_slumbot.py --hands 3000 --workers 4 --out slumbot_match.json
python3 scripts/play_slumbot.py --report slumbot_match.json    # 只看结果，不联网
```

**慢在网络**：一手要三次往返，单条会话约 30 手/分钟（三条约 100 手/分钟）。`--workers N`
开 N 条**独立会话**并行打（线程即可，等的是网络不是 CPU），N 条的结果按平方和合并，
口径与单条一样。别把 N 开大：那是别人免费提供的服务，四条已经够用。

每条会话每 `--checkpoint` 手存一次断点（并行时各存各的 `*.partN.json`，合并后删掉），
几十分钟的活儿中途断了不至于全丢。

Slumbot 是 ADR-0002 挑出来的**唯一**能接口调用的强对手（单挑、50/100 盲注、200bb）。
对局逻辑在 `holdem_slumbot.match`，这里只管命令行、进度与存盘。

## 读结果的三条纪律

1. **看区间，不看点估计**。单挑 200bb 的方差极大，几百手的 ±100bb/100 说明不了任何事。
2. **这是单挑 200bb 的强度**，不是六人 100bb 的水平；范围表覆盖不到单挑，
   大部分翻前决策会落在规则兜底上（报告里的「照解走 X%」就是这个）。
3. **对账不能有失败**。每一手都拿 Slumbot 报的 `winnings` 核对过；报告里出现「对不上」
   就说明协议翻译有 bug，那批数字直接作废，先修 bug。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from holdem.bots import STYLES, Bot  # noqa: E402
from holdem_slumbot.client import Session  # noqa: E402
from holdem_slumbot.match import MatchStats, play_match  # noqa: E402


def write(path: Path, stats: MatchStats, style: str, seconds: float) -> None:
    document = {
        "source": "slumbot",
        "style": style,
        "seconds": round(seconds, 1),
        "big_blind": 100,
        "stack_bb": 200,
        "bb_per_100": round(stats.bb100, 3),
        "interval_95": round(stats.interval, 3),
        "stats": asdict(stats),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")


def report_file(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    stats = MatchStats(**document["stats"])
    print(f"风格「{STYLES[document['style']].label}」，用时 {document['seconds'] / 60:.1f} 分钟")
    print(stats.report())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跟 Slumbot 打一场，量 bb/100")
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--style", default="solved", help=f"我们这边用哪种风格，可选 {sorted(STYLES)}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("slumbot_match.json"))
    parser.add_argument("--checkpoint", type=int, default=50, help="每多少手存一次盘")
    parser.add_argument("--pause", type=float, default=0.05, help="每次请求之间歇多久（秒）")
    parser.add_argument("--workers", type=int, default=1, help="并行的独立会话数（1..4）")
    parser.add_argument("--report", type=Path, help="只报告已有的结果文件，不联网")
    args = parser.parse_args(argv)

    if args.report:
        report_file(args.report)
        return 0
    if args.style not in STYLES:
        parser.error(f"未知风格 {args.style}，可选 {sorted(STYLES)}")

    if not 1 <= args.workers <= 4:
        parser.error("并行会话数请保持在 1..4——那是别人免费提供的服务")

    started = time.perf_counter()
    lock = threading.Lock()
    done = [0]
    requests = [0]

    def partial_path(index: int) -> Path:
        """每条会话自己的断点文件。几十分钟的活儿，中途崩了不能全丢。"""
        return args.out.with_name(f"{args.out.stem}.part{index}{args.out.suffix}")

    def run_one(index: int, hands: int, parts: int) -> MatchStats:
        """一条独立会话。每条自带一个 bot，随机流互不干扰。"""
        bot = Bot(args.style, seed=args.seed + index)
        session = Session(pause=args.pause)
        target = args.out if parts == 1 else partial_path(index)

        def on_hand(played: int, view, stats: MatchStats) -> None:
            if played % args.checkpoint == 0:
                write(target, stats, args.style, time.perf_counter() - started)
            with lock:
                done[0] += 1
                total = done[0]
            if total % args.checkpoint == 0:
                print(f"  {total:,}/{args.hands:,} 手（作废 {stats.aborted}）", flush=True)

        result = play_match(session, bot.act, hands=hands, on_hand=on_hand, bot=bot)
        write(target, result, args.style, time.perf_counter() - started)
        with lock:
            requests[0] += session.requests
        return result

    share, extra = divmod(args.hands, args.workers)
    quotas = [share + (1 if i < extra else 0) for i in range(args.workers)]
    quotas = [q for q in quotas if q > 0]
    if len(quotas) == 1:
        stats = run_one(0, quotas[0], 1)
    else:
        stats = MatchStats()
        with ThreadPoolExecutor(max_workers=len(quotas)) as pool:
            jobs = [(index, hands, len(quotas)) for index, hands in enumerate(quotas)]
            for part in pool.map(lambda job: run_one(*job), jobs):
                stats.add(part)

    seconds = time.perf_counter() - started
    write(args.out, stats, args.style, seconds)
    for index in range(len(quotas) if len(quotas) > 1 else 0):
        partial_path(index).unlink(missing_ok=True)  # 合并好了，断点文件就不留了

    print()
    print(f"风格「{STYLES[args.style].label}」，用时 {seconds / 60:.1f} 分钟"
          f"（{stats.hands / seconds * 60:.0f} 手/分钟，{requests[0]} 次请求、"
          f"{len(quotas)} 条会话）")
    print(stats.report())
    print(f"\n结果写入 {args.out}")
    if stats.mismatches:
        print("⚠ 有对不上的手：协议翻译有 bug，这批数字不作数")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

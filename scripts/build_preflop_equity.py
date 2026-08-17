"""预计算 169×169 的翻前对局权益表。

    python3 scripts/build_preflop_equity.py --samples 10000

为什么不精确枚举：单个对局要遍历 C(48,5)=1,712,304 个牌面（本机约 16 秒），
14,365 个对局约需 64 小时。所以走蒙特卡洛，并用 `equity.exact_equity` 抽样校验。

抽样方式：每个样本随机抽一对**不冲突**的具体组合，再抽公共牌。这样同时处理了两件事——
花色配置的平均（例如 AKs 与 QQ 是否共花色），以及共牌导致的不可能对局（例如 AA 与 AKs
不能共用同一张 A）。直接固定代表组合会在这两处系统性偏差。

同类对同类的权益按对称性恒为 0.5，直接写死，不浪费样本。
"""

from __future__ import annotations

import argparse
import array
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from holdem.cards import FULL_DECK  # noqa: E402
from holdem.evaluator import evaluate  # noqa: E402
from holdem.ranges import NUM_HAND_CLASSES, class_combos, class_name  # noqa: E402

MAGIC = b"PFEQ1"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "src" / "holdem" / "data" / "preflop_equity.bin"
DEFAULT_SAMPLES = 10_000


def matchup_equity(index_a: int, index_b: int, samples: int, seed: int) -> float:
    """`index_a` 对 `index_b` 的权益估计。平分计半。"""
    rng = random.Random(seed)
    combos_a = class_combos(index_a)
    combos_b = class_combos(index_b)
    deck_cache: dict[tuple, list[int]] = {}
    choice, sample = rng.choice, rng.sample
    total = 0.0

    for _ in range(samples):
        for _attempt in range(64):
            hole_a = choice(combos_a)
            hole_b = choice(combos_b)
            key = (hole_a, hole_b)
            deck = deck_cache.get(key)
            if deck is None:
                used = {*hole_a, *hole_b}
                if len(used) != 4:  # 共牌，这对组合不可能同时出现
                    deck_cache[key] = []
                    continue
                deck = [c for c in FULL_DECK if c not in used]
                deck_cache[key] = deck
            if deck:
                break
        else:
            raise RuntimeError(
                f"{class_name(index_a)} vs {class_name(index_b)}：抽不到不冲突的组合"
            )

        board = sample(deck, 5)
        score_a = evaluate([*hole_a, *board])
        score_b = evaluate([*hole_b, *board])
        if score_a > score_b:
            total += 1.0
        elif score_a == score_b:
            total += 0.5

    return total / samples


def _worker(job):
    index_a, index_b, samples, seed = job
    return index_a, index_b, matchup_equity(index_a, index_b, samples, seed)


def build(samples: int, seed: int, workers: int) -> array.array:
    table = array.array("f", [0.0]) * (NUM_HAND_CLASSES * NUM_HAND_CLASSES)
    for index in range(NUM_HAND_CLASSES):
        table[index * NUM_HAND_CLASSES + index] = 0.5  # 同类对同类，对称性使然

    jobs = [
        (a, b, samples, seed + a * NUM_HAND_CLASSES + b)
        for a in range(NUM_HAND_CLASSES)
        for b in range(a + 1, NUM_HAND_CLASSES)
    ]
    started = time.perf_counter()
    done = 0

    with Pool(workers) as pool:
        for index_a, index_b, equity in pool.imap_unordered(_worker, jobs, chunksize=16):
            table[index_a * NUM_HAND_CLASSES + index_b] = equity
            table[index_b * NUM_HAND_CLASSES + index_a] = 1.0 - equity
            done += 1
            if done % 500 == 0:
                elapsed = time.perf_counter() - started
                rate = done / elapsed
                print(
                    f"  {done:,}/{len(jobs):,} 个对局  "
                    f"{elapsed:5.0f}s  预计剩余 {(len(jobs) - done) / rate:5.0f}s",
                    flush=True,
                )
    return table


def write_table(path: Path, table: array.array, samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = b"%s %d %d\n" % (MAGIC, NUM_HAND_CLASSES, samples)
    payload = array.array("f", table)
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(header + payload.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="预计算翻前对局权益表")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"样本数 {args.samples:,}/对局 · {args.workers} 进程 · 输出 {args.out}")
    table = build(args.samples, args.seed, args.workers)
    write_table(args.out, table, args.samples)
    size = args.out.stat().st_size
    print(f"完成：{args.out}（{size:,} 字节）")


if __name__ == "__main__":
    main()

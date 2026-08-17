"""离线生成六人桌翻前范围表。

解一次整桌要十几分钟（按位置拆成一串两人子博弈，反复扫），所以和翻前权益表一样
**离线算、随包分发**：

```bash
python3 scripts/build_preflop_ranges.py                 # 默认六人 100bb
python3 scripts/build_preflop_ranges.py --sweeps 10     # 扫更多轮
python3 scripts/build_preflop_ranges.py --players 3 --out /tmp/three.json
```

产物是 JSON（几十 KB，人能读、diff 得动），由 `holdem.preflop_ranges` 读取。
里面**连参数一起存**：兑现模型的系数、开牌尺度、扫了几轮、最后一轮还在动多少。
参数没记下来的范围表是没法复现也没法审计的——将来校准了兑现系数，得能一眼看出
手上这张表是哪套参数算的。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from holdem.preflop_chain import (  # noqa: E402
    TableConfig,
    TableSolution,
    defender_advantage,
    solve_table,
)
from holdem.realization import RealizationModel  # noqa: E402

FORMAT = "PFRANGE1"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "src" / "holdem" / "data" / "preflop_ranges_6max_100bb.json"


def build_document(solution: TableSolution, model: RealizationModel, seconds: float) -> dict:
    config = solution.config
    spots = {}
    for seat in config.openers:
        spot = solution.spot(seat)
        defenses = {}
        for defender, sub in spot.defenses.items():
            root = sub.tree.root
            entry = {
                "actions": {
                    action.label: sub.action_range(root, index).to_text()
                    for index, action in enumerate(root.actions)
                },
                "frequencies": {
                    action.label: round(sub.action_frequency(root, index), 4)
                    for index, action in enumerate(root.actions)
                },
                "exploitability": round(sub.exploitability, 6),
                # 逐牌类的「继续比弃牌好多少」，风格层放宽范围时按它排序
                "advantage": [round(v, 5) for v in defender_advantage(sub)],
            }
            # 开牌者面对 3bet 的应对：根节点第一个加注分支之下就是他的决策
            for index, action in enumerate(root.actions):
                if action.is_raise:
                    child = root.children[index]
                    if not child.is_terminal:
                        entry["vs_reraise"] = {
                            "facing": action.label,
                            "actions": {
                                reply.label: sub.action_range(child, reply_index).to_text()
                                for reply_index, reply in enumerate(child.actions)
                            },
                            "frequencies": {
                                reply.label: round(sub.action_frequency(child, reply_index), 4)
                                for reply_index, reply in enumerate(child.actions)
                            },
                        }
                    break
            defenses[config.position_name(defender)] = entry

        spots[spot.name] = {
            "open": spot.open_range.to_text(),
            "open_frequency": round(spot.open_frequency, 4),
            # 逐牌类的开牌 EV：风格层放宽开牌范围时按它排序
            "open_ev": [round(v, 5) for v in spot.open_hand_ev],
            "fold_value": round(spot.fold_value, 5),
            "defenses": defenses,
        }

    return {
        "format": FORMAT,
        "table": asdict(config),
        "model": asdict(model),
        "sweeps": solution.sweeps,
        "max_change": round(solution.max_change, 6),
        "seconds": round(seconds, 1),
        "spots": spots,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成翻前范围表")
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--stack", type=float, default=100.0, help="有效筹码（大盲）")
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=200, help="每个子博弈的 CFR+ 迭代数")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = TableConfig(num_players=args.players, effective_stack=args.stack)
    model = RealizationModel()

    started = time.time()
    solution = solve_table(
        config,
        model=model,
        sweeps=args.sweeps,
        inner_iterations=args.iterations,
        progress=None if args.quiet else lambda text: print(text, flush=True),
    )
    seconds = time.time() - started

    document = build_document(solution, model, seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(solution.summary())
    print(f"\n最后一轮开牌频率变动 {solution.max_change:.4f}，用时 {seconds / 60:.1f} 分钟")
    print(f"写入 {args.out}（{args.out.stat().st_size / 1024:.0f} KB）")
    if solution.max_change > 0.02:
        print("⚠ 还在明显摆动，建议加大 --sweeps 再跑一次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""离线生成翻前范围表：六人桌一张、单挑一张。

```bash
python3 scripts/build_preflop_ranges.py                 # 六人 100bb（约 23 分钟）
python3 scripts/build_preflop_ranges.py --sweeps 10     # 扫更多轮
python3 scripts/build_preflop_ranges.py --players 3 --out /tmp/three.json
python3 scripts/build_preflop_ranges.py --players 2 --stack 200   # 单挑 200bb（约 2 分钟）
```

产物是 JSON（几十 KB，人能读、diff 得动），由 `holdem.preflop_ranges` 读取。
里面**连参数一起存**：兑现模型的系数、开牌尺度、扫了几轮、最后一轮还在动多少。
参数没记下来的范围表是没法复现也没法审计的——将来校准了兑现系数，得能一眼看出
手上这张表是哪套参数算的。

## 两条路子不一样，别混

- **三人及以上**：整树跑不动（六人 62 万节点），按位置拆成一串两人子博弈再链式合成
  （ADR-0004）。这条路的解是「迭代最佳应对 + CFR+ 平滑」，判据是范围还变不变。
- **单挑**：整棵树只有几十个节点，**直接精确解**（ADR-0003）。所以单挑产物里存的是
  **可利用度**——它是自证的，比什么外部对照都硬。别把单挑也塞进链式求解，那是自找的近似。

## 单挑为什么默认关掉跛入

Slumbot 实测 895 次首行动**全部是开牌到 2bb，一次跛入都没有**。带跛入的树里，
「开牌频率」与不带跛入的树比的是两件不同的事（校准脚本也是这么处理的）。
所以单挑产物按「开牌或弃牌」解，动作序列的形状与六人桌那张表完全一致，
`preflop_policy` 一行都不用改。
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
from holdem.preflop_solver import PreflopSolution, action_advantage, solve_preflop  # noqa: E402
from holdem.preflop_tree import PreflopConfig, build_tree  # noqa: E402
from holdem.positions import position_names  # noqa: E402
from holdem.realization import RealizationModel  # noqa: E402

FORMAT = "PFRANGE1"
DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "holdem" / "data"
DEFAULT_OUT = DATA_DIR / "preflop_ranges_6max_100bb.json"
HEADSUP_OUT = DATA_DIR / "preflop_ranges_hu_200bb.json"


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
                # 他跟注之后身后挤压的概率（ADR-0004 补的那条简化）；身后没人时是 0
                "squeeze": round(spot.squeezes.get(defender, 0.0), 4),
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


def _node_strategy(solution: PreflopSolution, node) -> dict:
    """一个决策点上的动作范围与频率，按产物的写法。"""
    return {
        "actions": {
            action.label: solution.action_range(node, index).to_text()
            for index, action in enumerate(node.actions)
        },
        "frequencies": {
            action.label: round(solution.action_frequency(node, index), 4)
            for index, action in enumerate(node.actions)
        },
    }


def build_headsup_document(
    config: PreflopConfig, model: RealizationModel, iterations: int, seconds_holder: list
) -> tuple[dict, PreflopSolution]:
    """解一棵完整的单挑树，写成与六人桌同一套 schema 的产物。

    单挑不需要链式合成：`open_ev`、`fold_value`、防守者的 `advantage` 全部直接来自
    同一个解，所以这张表内部是自洽的（六人桌那张是十几盘子博弈拼起来的）。
    """
    tree = build_tree(config)
    root = tree.root
    open_index = next(i for i, a in enumerate(root.actions) if a.is_raise)
    fold_index = next(i for i, a in enumerate(root.actions) if a.kind == "fold")
    defense = root.children[open_index]
    if defense.is_terminal:
        raise SystemExit("开牌之后没有防守节点——检查树的参数")

    started = time.time()
    solution = solve_preflop(
        config,
        model=model,
        branch_nodes=(defense.node_id,),
        iterations=iterations,
        tolerance=1e-3,
        check_every=max(iterations // 8, 1),
    )
    seconds_holder.append(time.time() - started)

    branches = solution.node_branches[root.node_id]
    entry = _node_strategy(solution, defense)
    entry["exploitability"] = round(solution.exploitability, 6)
    entry["squeeze"] = 0.0  # 单挑没有身后的人
    entry["advantage"] = [round(v, 5) for v in action_advantage(solution, defense)]
    # 开牌者面对 3bet 的应对
    for index, action in enumerate(defense.actions):
        if action.is_raise:
            child = defense.children[index]
            if not child.is_terminal:
                entry["vs_reraise"] = {"facing": action.label, **_node_strategy(solution, child)}
            break

    names = position_names(2)
    document = {
        "format": FORMAT,
        "table": asdict(config),
        "model": asdict(model),
        "sweeps": 1,
        "max_change": 0.0,
        "exploitability": round(solution.exploitability, 6),
        "iterations": solution.iterations,
        "seconds": round(seconds_holder[-1], 1),
        "spots": {
            names[0]: {
                "open": solution.action_range(root, open_index).to_text(),
                "open_frequency": round(solution.action_frequency(root, open_index), 4),
                "open_ev": [round(v, 5) for v in branches[open_index].hand_ev(0)],
                "fold_value": round(branches[fold_index].hand_ev(0)[0], 5),
                "defenses": {names[1]: entry},
            }
        },
    }
    return document, solution


def _report_headsup(document: dict, solution: PreflopSolution) -> None:
    names = position_names(2)
    spot = document["spots"][names[0]]
    defense = spot["defenses"][names[1]]
    print(f"\n{names[0]} 开牌 {100 * spot['open_frequency']:.1f}%")
    print("  " + names[1] + " 应对：" + "  ".join(
        f"{label} {100 * value:.1f}%" for label, value in defense["frequencies"].items()
    ))
    if "vs_reraise" in defense:
        print(f"  {names[0]} 面对 {defense['vs_reraise']['facing']}：" + "  ".join(
            f"{label} {100 * value:.1f}%"
            for label, value in defense["vs_reraise"]["frequencies"].items()
        ))
    print(
        f"\n可利用度 {solution.exploitability:.5f} 大盲/手"
        f"（{solution.iterations} 次迭代）"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成翻前范围表")
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--stack", type=float, default=100.0, help="有效筹码（大盲）")
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument(
        "--iterations", type=int, default=200,
        help="CFR+ 迭代数（六人桌是每个子博弈的，单挑是整树的；单挑建议 2000）",
    )
    parser.add_argument("--out", type=Path, default=None, help="默认按人数选产物路径")
    parser.add_argument("--open-to", type=float, default=2.5, help="开牌尺度（大盲）")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.players == 2:
        return _build_headsup(args)

    config = TableConfig(
        num_players=args.players, effective_stack=args.stack, open_to=args.open_to
    )
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
    out = args.out or DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(solution.summary())
    print(f"\n最后一轮开牌频率变动 {solution.max_change:.4f}，用时 {seconds / 60:.1f} 分钟")
    print(f"写入 {out}（{out.stat().st_size / 1024:.0f} KB）")
    if solution.max_change > 0.02:
        print("⚠ 还在明显摆动，建议加大 --sweeps 再跑一次")
    return 0


def _build_headsup(args) -> int:
    """单挑：整树精确解。默认 200bb——Slumbot 就是这个深度（ADR-0002）。"""
    stack = args.stack if args.stack != 100.0 else 200.0
    config = PreflopConfig(
        num_players=2,
        effective_stack=stack,
        open_to=args.open_to,
        allow_limp=False,
    )
    model = RealizationModel()
    if not args.quiet:
        print(f"解单挑 {stack:g}bb 整树（开牌到 {args.open_to:g}bb，不含跛入）…", flush=True)

    seconds: list[float] = []
    document, solution = build_headsup_document(config, model, args.iterations, seconds)

    out = args.out or HEADSUP_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.quiet:
        _report_headsup(document, solution)
    print(f"用时 {seconds[-1]:.0f} 秒，写入 {out}（{out.stat().st_size / 1024:.0f} KB）")
    if solution.exploitability > 0.01:
        print("⚠ 可利用度偏高，建议加大 --iterations 再跑一次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

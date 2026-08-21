"""漏洞报告（FR-10）：一批牌打下来，钱主要从哪类局面漏出去。

```bash
# 打 300 手，抽 8 个翻后局面真解，看 0 号座位（照解风格）的漏洞
python3 scripts/leak_report.py --hands 300 --seat 0 --max-solves 8

python3 scripts/leak_report.py --hands 1000 --max-solves 20 --json /tmp/leaks.json
python3 scripts/leak_report.py --hands 300 --plan-only    # 不求解，只看能复盘多少手
```

**求解是这里唯一贵的一步**：一个翻牌局面几分钟到十几分钟，产物上百 MB
（要算跨街 EV 就得导满三条街，见 ADR-0005）。所以：

- `--max-solves` 是**硬预算**，默认小；抽样按种子打乱后取前 N 手，不是只看开头几手。
- 解完打完分**立刻删掉 dump**（`--keep-cache` 可留）。留着几十个上百 MB 的解只为了
  报告里那一个数，不划算。
- `--plan-only` 完全不求解：先看看这批牌有多少能复盘、卡在哪些线路上，再决定要不要花
  几小时去解。**先量能量什么，再花钱。**

## 报告怎么读

主表的「每 100 手」口径是**每 100 手可复盘的牌**（抽到的那些）。真实牌局里还有跛入、
多人底池、4bet 之后这些复盘不了的手，所以末尾另给一行按覆盖率外推到全部牌局的估计
——**那是估计，不是测量**。

排序按总漏损，不按平均：平均值最大的往往是一年遇不上几次的局面（见 `leaks.py`）。

**先看「排行可信度」那一行再看排行**。EV 损失是拿实战动作跟解比出来的差，解要是没收敛，
差里就掺着求解器自己的残差；报告把每个解的可利用度折成大盲（`可利用度% × 底池`）当噪声底，
跟平均每点漏损比。比值小于 3 就别照着排行去改打法——**先加 `--iterations` 重跑**。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from holdem.batch import MatchConfig, run_batch  # noqa: E402
from holdem.range_tracking import NotCovered  # noqa: E402
from holdem_solver import (  # noqa: E402
    BetSizes,
    SolverNotInstalled,
    TexasSolver,
    plan_review,
    score_plan,
)
from holdem_solver.leaks import build_report  # noqa: E402

DEFAULT_SEATS = "solved,tag,lag,nit,station,maniac"
LEAN_SIZES = BetSizes(flop=(33.0,), turn=(66.0,), river=(75.0,), reraise=(60.0,))
"""**尺度给得克制**：一条街多一个尺度，树大致翻一倍，解得慢、产物也大一倍。
实战真正打出的那个尺度会由 `plan_review` 自动并进去，所以这里只留一个基准尺度。"""


def play_hands(config: MatchConfig) -> list:
    hands = []
    run_batch(config, on_hand=lambda index, hand: hands.append(hand))
    return hands


def make_plans(hands, seat: int, *, accuracy: float, iterations: int):
    """先把计划全做出来（纯逻辑，不花钱），顺带统计复盘不了的原因。"""
    plans, uncovered = [], []
    for hand in hands:
        try:
            plan = plan_review(
                hand,
                seat,
                bet_sizes=LEAN_SIZES,
                accuracy=accuracy,
                max_iterations=iterations,
            )
        except NotCovered as reason:
            uncovered.append(str(reason))
            continue
        if plan.points:
            plans.append(plan)
        else:
            uncovered.append("这手牌翻后没轮到他做决策")
    return plans, uncovered


def solve_and_score(plans, solver: TexasSolver, *, timeout: float, keep_cache: bool):
    """真解并打分。**可利用度要一路带回去**——它决定这份报告可不可信（见 `leaks.py`）。"""
    results, exploitability = [], []
    for index, plan in enumerate(plans, start=1):
        started = time.perf_counter()
        report = solver.solve(plan.request, timeout=timeout)
        results.append(score_plan(plan, report.root))
        exploitability.append(report.exploitability)
        cache = solver.cache_dir / f"{report.fingerprint}.json"
        size = cache.stat().st_size / 1e6 if cache.is_file() else 0.0
        if not keep_cache and cache.is_file():
            cache.unlink()
        accuracy = plan.request.accuracy
        if report.exploitability is None:
            verdict = "可利用度：求解器没报"
        else:
            gap = report.exploitability
            verdict = f"可利用度 {gap:.3g}%" + ("" if gap <= accuracy else f"（**没收敛到 {accuracy:g}%**）")
        print(
            f"  [{index}/{len(plans)}] {len(plan.points)} 个决策点，"
            f"{time.perf_counter() - started:.0f}s，dump {size:.0f}MB"
            f"{'（已缓存）' if report.cached else ''}，{verdict}",
            flush=True,
        )
    return results, exploitability


def render(report, *, plan_count: int, dealt: int) -> str:
    lines = [
        f"看了 {dealt} 手，其中 {plan_count} 手可复盘"
        f"（{plan_count / dealt:.0%}），本次求解了 {report.hands} 手",
        f"打上分的决策点 {report.scored_spots} 个，覆盖率 {report.coverage:.0%}",
        "",
        f"{'场景':<34}{'次数':>6}{'总漏损':>10}{'平均':>9}{'每100手':>10}{'占比':>8}"
        f"{'A/B/C':>10}{'可信漏损':>10}",
    ]
    for leak in report.leaks:
        lines.append(
            f"{leak.scenario.title:<34}{leak.spots:>6}"
            f"{leak.total_loss:>10.2f}{leak.mean_loss:>9.2f}"
            f"{report.per_100_hands(leak.total_loss):>10.1f}"
            f"{report.share(leak):>8.0%}"
            f"{'/'.join(str(n) for n in leak.grades):>10}"
            f"{leak.confident_loss:>10.2f}"
        )
    lines.append("")
    lines.append(
        "  A/B/C＝这一格里三档置信度各多少个点；可信漏损＝只把 A 档那些点加起来。"
    )
    lines.append(
        "  **两个漏损差得远，说明这行的名次是 B/C 档撑起来的**——那不是打法漏洞，是噪声。"
    )
    lines.append("")
    lines.append(
        f"合计漏损 {report.total_loss:.2f}bb，"
        f"{report.per_100_hands(report.total_loss):.1f} bb/100（每 100 手**可复盘的**牌）"
    )
    if plan_count and dealt:
        extrapolated = report.per_100_hands(report.total_loss) * plan_count / dealt
        lines.append(
            f"按覆盖率外推到全部牌局约 {extrapolated:.1f} bb/100——**这是外推，不是测量**"
        )
    lines.extend(_convergence_lines(report))
    if report.skipped:
        lines.append("")
        lines.append("打不了分的决策点：")
        for reason, count in report.skipped:
            lines.append(f"  {count:>4} × {reason}")
    if report.uncovered_hands:
        lines.append("")
        lines.append("复盘不了的手牌：")
        for reason, count in report.uncovered_hands:
            lines.append(f"  {count:>4} × {reason}")
    return "\n".join(lines)


def _convergence_json(report):
    """收敛度也要进 JSON——**没有它，一份 leaks.json 事后没人判得出可不可信**。"""
    gaps = report.convergence
    document = {
        "noise_floor_bb": None
        if report.noise_floor is None
        else round(report.noise_floor, 4),
        "mean_loss_bb": round(report.mean_loss, 4),
        "mean_solver_gap_bb": None
        if report.mean_solver_gap is None
        else round(report.mean_solver_gap, 4),
        "signal_to_noise": None
        if report.signal_to_noise is None
        else round(report.signal_to_noise, 3),
        "ranking_trustworthy": report.ranking_trustworthy(),
    }
    if gaps is not None:
        document.update(
            solves=gaps.solves,
            accuracy_percent=gaps.accuracy,
            exploitability_percent=[round(value, 4) for value in gaps.exploitability],
            unmeasured=gaps.unmeasured,
            unconverged=gaps.unconverged,
        )
    return document


def _convergence_lines(report) -> list:
    """把「这批解收敛了没有」写成人话。**这几行比排行本身更该先读。**"""
    gaps = report.convergence
    lines = ["", "解的收敛度："]
    if gaps is not None and gaps.exploitability:
        lines.append(
            f"  可利用度 中位 {gaps.median:.3g}% / 最差 {gaps.worst:.3g}%（%底池）"
            + (f"，门槛 {gaps.accuracy:g}%" if gaps.accuracy is not None else "")
        )
        if gaps.unconverged:
            lines.append(f"  {gaps.unconverged}/{gaps.solves} 个解**没收敛到门槛**")
    if gaps is None:
        lines.append("  求解器一个可利用度都没报（当未知，不当收敛）")
    elif gaps.unmeasured:
        lines.append(f"  {gaps.unmeasured}/{gaps.solves} 个解量不到可利用度（当未知，不当收敛）")
    if report.mean_solver_gap is not None:
        lines.append(
            f"  打分那 {len(report.solver_gaps)} 个点上，解自己平均还差 "
            f"{report.mean_solver_gap:.3f}bb（`DecisionScore.gap`）"
        )
    ratio = report.signal_to_noise
    if ratio is None:
        lines.append("  排行可信度：**判不了**（两路残差都没量到）")
        return lines
    lines.append(
        f"  噪声底 {report.noise_floor:.3f}bb/点，平均每点漏损 {report.mean_loss:.3f}bb，"
        f"信噪比 {ratio:.1f}"
    )
    if report.ranking_trustworthy():
        lines.append("  排行可信度：够用（漏损明显大过解自己的残差）")
    else:
        lines.append(
            "  排行可信度：**不够**——漏损跟解的残差一个量级，"
            "排在前面的多半只是「这格出现得多」。加 `--iterations` 重跑再看排行"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="按场景聚合 EV 损失（FR-10）")
    parser.add_argument("--hands", type=int, default=300, help="打多少手（默认 300）")
    parser.add_argument("--seats", default=DEFAULT_SEATS)
    parser.add_argument("--seat", type=int, default=0, help="给谁做复盘（默认 0 号）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stack", type=float, default=100.0, help="起始筹码（大盲）")
    parser.add_argument(
        "--max-solves", type=int, default=8, help="最多真解多少个局面（硬预算）"
    )
    parser.add_argument("--accuracy", type=float, default=1.0, help="收敛门槛（%%底池）")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--keep-cache", action="store_true", help="留着解出来的 dump")
    parser.add_argument("--plan-only", action="store_true", help="只做计划，不求解")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    big_blind = 100
    config = MatchConfig(
        styles=tuple(name.strip() for name in args.seats.split(",") if name.strip()),
        hands=args.hands,
        big_blind=big_blind,
        small_blind=big_blind // 2,
        start_stack=int(round(args.stack * big_blind)),
        seed=args.seed,
    )
    if not 0 <= args.seat < len(config.styles):
        print(f"没有 {args.seat} 号座位（这桌 {len(config.styles)} 个人）")
        return 2

    print(f"打 {args.hands} 手…", flush=True)
    hands = play_hands(config)
    plans, uncovered = make_plans(
        hands, args.seat, accuracy=args.accuracy, iterations=args.iterations
    )
    print(
        f"可复盘 {len(plans)} 手（{len(plans) / len(hands):.0%}），"
        f"待打分决策点 {sum(len(plan.points) for plan in plans)} 个",
        flush=True,
    )
    if args.plan_only:
        for reason, count in sorted(
            {r: uncovered.count(r) for r in set(uncovered)}.items(),
            key=lambda item: -item[1],
        ):
            print(f"  {count:>4} × {reason}")
        return 0

    random.Random(args.seed).shuffle(plans)
    sample = plans[: args.max_solves]
    if not sample:
        print("没有可复盘的手牌，先加大 --hands")
        return 1
    try:
        solver = TexasSolver(threads=args.threads)
    except SolverNotInstalled as exc:
        print(f"{exc}")
        return 2

    print(f"求解 {len(sample)} 个局面（这一步是唯一贵的）…", flush=True)
    started = time.perf_counter()
    results, exploitability = solve_and_score(
        sample, solver, timeout=args.timeout, keep_cache=args.keep_cache
    )
    elapsed = time.perf_counter() - started

    report = build_report(
        results,
        hands=len(sample),
        uncovered=uncovered,
        exploitability=exploitability,
        accuracy=args.accuracy,
    )
    print()
    print(render(report, plan_count=len(plans), dealt=len(hands)))
    print()
    print(f"求解用时 {elapsed / 60:.1f} 分钟，平均 {elapsed / len(sample):.0f}s/局面")
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "hands_dealt": len(hands),
                    "hands_reviewable": len(plans),
                    "hands_solved": report.hands,
                    "scored_spots": report.scored_spots,
                    "coverage": report.coverage,
                    "total_loss_bb": report.total_loss,
                    "convergence": _convergence_json(report),
                    "leaks": [
                        {
                            "scenario": leak.scenario.title,
                            "street": leak.scenario.street,
                            "role": leak.scenario.role,
                            "position": leak.scenario.position,
                            "facing": leak.scenario.facing,
                            "spots": leak.spots,
                            "total_loss_bb": round(leak.total_loss, 4),
                            "mean_loss_bb": round(leak.mean_loss, 4),
                            "off_range_spots": leak.off_range_spots,
                            # 置信度分布也要进 JSON：只落 total_loss 的话，
                            # 事后拿这份文件排名的人**没法知道哪几行是噪声撑起来的**
                            # （同 convergence 那条的理由）。
                            "grades": {
                                "A": leak.grades[0],
                                "B": leak.grades[1],
                                "C": leak.grades[2],
                            },
                            "confident_loss_bb": round(leak.confident_loss, 4),
                        }
                        for leak in report.leaks
                    ],
                    "skipped": [
                        {"reason": reason, "count": count}
                        for reason, count in report.skipped
                    ],
                    "uncovered_hands": [
                        {"reason": reason, "count": count}
                        for reason, count in report.uncovered_hands
                    ],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"报告已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

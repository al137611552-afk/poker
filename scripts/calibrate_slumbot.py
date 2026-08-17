"""拿 Slumbot 的实测频率校准翻后兑现模型（ADR-0003 的校准计划，第一档）。

`realization.py` 的参数是**未经校准的假设**。这个脚本用 ADR-0002 定下的外部标尺
——Slumbot 免费 HTTP API——采集它的**聚合动作频率**，供我们对照自己的解。

```bash
python3 scripts/calibrate_slumbot.py --hands 2000 --out /tmp/slumbot.json
python3 scripts/calibrate_slumbot.py --report /tmp/slumbot.json     # 只看已采集的结果
```

## 它能校准什么、不能校准什么

**能**：聚合频率。三个格子——

| 格子 | 怎么采 |
|---|---|
| 它在按钮位首个行动 | 我们当大盲，什么都不用做，直接看它开不开牌、开多大 |
| 它在大盲面对我们开牌 | 我们当按钮固定开到 2.5bb，看它弃/跟/3bet |
| 它面对我们的 3bet | 它开牌后我们固定 3bet 到三倍，看它弃/跟/4bet |

**不能**：逐手范围。**除非摊牌，否则看不到它的底牌**，所以没法知道「它用哪些牌 3bet」。
想校准范围的构成只能走 ADR-0003 的第二档（用 TexasSolver 实解反推兑现系数）。

## 三条必须写在结论旁边的限制

1. **Slumbot 是单挑 200bb**（盲注 50/100、筹码 20000），我们要的是六人 100bb。
   深度与人数都不同，只能当量级校验。
2. **它不是 GTO**，是 2018 年 ACPC 冠军级的近似解。对齐它 = 向 Slumbot 看齐，不是向真理看齐。
3. **我们的探针策略是固定的**（永远开牌、永远 3bet）。这不影响测量：Slumbot 策略是静态的，
   它在给定动作序列下的频率就是它自己的频率。但反过来说，我们**测不到**它面对不同
   开牌尺度时的反应——只测到我们打出的那一种。

## 协议（实测所得，别照文档猜）

- `POST /api/new_hand {"token": 上一手的token}`，`POST /api/act {"token":…, "incr": 动作}`。
- `client_pos`：**0 = 我们是大盲**（它先说话），**1 = 我们是按钮/小盲**。带 token 开新手时
  位置逐手轮换。
- 动作串形如 `b200b500c/`：`b<数字>` 是**加注到的本手总额（筹码）**，`c` 兼表跟注与过牌，
  `f` 弃牌，`/` 分街。
- 尺度非法时回的是 `{"old_action":…, "error_msg": "Bet size too small"}`，注意字段名是
  `error_msg` 不是 `error_message`。
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

BASE = "https://slumbot.com/api"
BIG_BLIND = 100
"""Slumbot 的盲注是 50/100，筹码 20000（＝200bb）。"""

SPOTS = ("它按钮首行动", "它大盲面对开牌", "它面对3bet")


class SlumbotError(RuntimeError):
    pass


class Session:
    """一条会话：token 串起连续的手牌，位置逐手轮换。"""

    def __init__(self, timeout: float = 20.0, pause: float = 0.05) -> None:
        self.token: str | None = None
        self.timeout = timeout
        self.pause = pause

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{BASE}/{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read())
        if "error_msg" in body:
            raise SlumbotError(f"{body['error_msg']}（此前动作 {body.get('old_action')!r}）")
        if "token" in body:
            self.token = body["token"]
        time.sleep(self.pause)
        return body

    def new_hand(self) -> dict:
        return self._post("new_hand", {"token": self.token} if self.token else {})

    def act(self, incr: str) -> dict:
        return self._post("act", {"token": self.token, "incr": incr})


# ------------------------------------------------------------------ 动作串


_BET = re.compile(r"b(\d+)")


def tokenize(street: str) -> list[str]:
    """把一条街的动作串拆成 `['b200', 'c']` 这样的记号。"""
    tokens: list[str] = []
    index = 0
    while index < len(street):
        char = street[index]
        if char == "b":
            end = index + 1
            while end < len(street) and street[end].isdigit():
                end += 1
            tokens.append(street[index:end])
            index = end
        elif char in "cfk":
            tokens.append(char)
            index += 1
        else:
            index += 1
    return tokens


def facing_bet(action: str) -> bool:
    """轮到我们时是不是面对一个下注（决定能不能弃牌）。"""
    tokens = tokenize(action.split("/")[-1])
    return bool(tokens) and tokens[-1].startswith("b")


def classify(prefix: str, action_after: str) -> tuple[str, int | None]:
    """对手**在 `prefix` 之后**做了什么：弃牌 / 跟注（含过牌）/ 加注到多少。

    `prefix` 必须包含我们自己刚打出的那一步——否则会把自己的加注当成对手的应对，
    读出「它面对开牌 100% 3bet」这种鬼话（第一版就是这么错的）。
    """
    added = action_after[len(prefix) :]
    if added.startswith("f"):
        return "弃牌", None
    if added.startswith("c") or added.startswith("k"):
        return "跟注", None
    match = _BET.match(added)
    if match:
        return "加注", int(match.group(1))
    return "其他", None


# ------------------------------------------------------------------ 采集


def new_counts() -> dict:
    return {
        spot: {"弃牌": 0, "跟注": 0, "加注": 0, "尺度": Counter()} for spot in SPOTS
    }


def record(counts: dict, spot: str, move: str, size: int | None) -> None:
    bucket = counts[spot]
    bucket[move] = bucket.get(move, 0) + 1
    if size is not None:
        bucket["尺度"][size] += 1


def play_hand(session: Session, open_to: int, three_bet_multiple: float) -> list[tuple[str, str, int | None]]:
    """打一手，返回这手观察到的数据点。

    **打完整手才算数**：中途出错就整手作废。第一版边打边记，一出错重打就把同一手的
    观察重复计进去了（20 手采出 28 个样本）。
    """
    observed: list[tuple[str, str, int | None]] = []
    hand = session.new_hand()
    action = hand.get("action", "")

    if hand["client_pos"] == 1:
        # 我们是按钮：固定开到 open_to，看大盲怎么应
        ours = f"b{open_to}"
        hand = session.act(ours)
        move, size = classify(action + ours, hand["action"])
        observed.append(("它大盲面对开牌", move, size))
    else:
        # 我们是大盲：它先说话，白看一次它的首行动
        move, size = classify("", action)
        observed.append(("它按钮首行动", move, size))
        if move == "加注" and size is not None and size <= 4 * BIG_BLIND:
            ours = f"b{int(size * three_bet_multiple)}"
            hand = session.act(ours)
            move, size = classify(action + ours, hand["action"])
            observed.append(("它面对3bet", move, size))

    # 数据已经拿到，尽快结束这手：面对下注就弃，没人下注就过牌
    # （过牌是 "k"，"c" 只表示跟注——发错会被判 Illegal call）
    for _ in range(8):
        if "winnings" in hand:
            break
        hand = session.act("f" if facing_bet(hand.get("action", "")) else "k")
    return observed


def collect(hands: int, out: Path, open_to: int, three_bet_multiple: float, checkpoint: int) -> dict:
    session = Session()
    counts = new_counts()
    played = 0
    errors = 0
    started = time.time()

    while played < hands:
        try:
            for spot, move, size in play_hand(session, open_to, three_bet_multiple):
                record(counts, spot, move, size)
            played += 1
        except (SlumbotError, urllib.error.URLError, TimeoutError, KeyError) as exc:
            errors += 1
            if errors <= 3:
                print(f"  [第 {errors} 个错误] {type(exc).__name__}: {exc}", flush=True)
            if errors > max(20, hands // 20):
                raise RuntimeError(f"错误太多（{errors} 次），最后一个：{exc}") from exc
            session.token = None  # 出错就换一条会话重开
            time.sleep(1.0)
            continue

        if played % checkpoint == 0:
            write(out, counts, played, errors, time.time() - started, open_to, three_bet_multiple)
            print(f"已打 {played}/{hands} 手，用时 {(time.time() - started) / 60:.1f} 分钟", flush=True)

    write(out, counts, played, errors, time.time() - started, open_to, three_bet_multiple)
    return counts


def write(out: Path, counts: dict, hands: int, errors: int, seconds: float, open_to: int, multiple: float) -> None:
    document = {
        "source": "slumbot",
        "hands": hands,
        "errors": errors,
        "seconds": round(seconds, 1),
        "probe": {"open_to_chips": open_to, "three_bet_multiple": multiple, "big_blind": BIG_BLIND},
        "spots": {
            spot: {
                "弃牌": bucket["弃牌"],
                "跟注": bucket["跟注"],
                "加注": bucket["加注"],
                "尺度": dict(bucket["尺度"]),
            }
            for spot, bucket in counts.items()
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ 报告


def report(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    probe = document["probe"]
    print(
        f"Slumbot 实测 {document['hands']} 手"
        f"（探针：开牌到 {probe['open_to_chips'] / probe['big_blind']:.1f}bb、"
        f"3bet 三倍；出错 {document['errors']} 次，用时 {document['seconds'] / 60:.1f} 分钟）\n"
    )
    for spot, bucket in document["spots"].items():
        total = bucket["弃牌"] + bucket["跟注"] + bucket["加注"]
        if not total:
            continue
        parts = " ".join(
            f"{name} {100 * bucket[name] / total:5.1f}%" for name in ("弃牌", "跟注", "加注")
        )
        sizes = sorted(
            ((int(size), n) for size, n in bucket["尺度"].items()), key=lambda x: -x[1]
        )[:3]
        sizes_text = "、".join(
            f"{size / probe['big_blind']:.1f}bb×{n}" for size, n in sizes
        )
        margin = 100 * (0.25 / total) ** 0.5  # 最坏情况的标准误
        print(f"{spot:14s} n={total:5d}  {parts}   (±{margin:.1f}pp)   常见尺度：{sizes_text}")


# ------------------------------------------------------------------ 对照


def _predict(model, *, stack_bb: float, open_to_bb: float, iterations: int) -> dict:
    """用我们的模型解一局单挑，取出与 Slumbot 三个格子对应的频率。"""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from holdem.preflop_solver import solve_preflop
    from holdem.preflop_tree import PreflopConfig

    # 关掉跛入是为了**可比**：实测里 Slumbot 从不跛入（首行动只有开牌与弃牌），
    # 我们的模型默认允许跛入，带着这个动作去比「开牌频率」等于比两件不同的事
    config = PreflopConfig(
        num_players=2, effective_stack=stack_bb, open_to=open_to_bb, allow_limp=False
    )
    solution = solve_preflop(config, model=model, iterations=iterations, tolerance=2e-3)
    root = solution.tree.root

    def split(node) -> dict:
        totals = {"弃牌": 0.0, "跟注": 0.0, "加注": 0.0}
        for index, action in enumerate(node.actions):
            frequency = solution.action_frequency(node, index)
            key = "弃牌" if action.kind == "fold" else ("跟注" if action.kind == "call" else "加注")
            totals[key] += frequency
        return totals

    raise_index = next(i for i, a in enumerate(root.actions) if a.is_raise)
    facing_open = root.children[raise_index]
    defender_raise = next(
        (i for i, a in enumerate(facing_open.actions) if a.is_raise), None
    )
    result = {"它按钮首行动": split(root), "它大盲面对开牌": split(facing_open)}
    if defender_raise is not None and not facing_open.children[defender_raise].is_terminal:
        result["它面对3bet"] = split(facing_open.children[defender_raise])
    return result


def compare(path: Path, iterations: int, min_samples: int = 100) -> None:
    """把实测频率与我们的模型在一小片参数网格上对照。

    **不自动改默认值**：参数怎么定是有后果的决定，这里只给出对照与建议。
    """
    from holdem.realization import RealizationModel

    document = json.loads(path.read_text(encoding="utf-8"))
    probe = document["probe"]
    observed = {}
    for spot, bucket in document["spots"].items():
        total = bucket["弃牌"] + bucket["跟注"] + bucket["加注"]
        if total >= min_samples:
            observed[spot] = {k: bucket[k] / total for k in ("弃牌", "跟注", "加注")}

    # Slumbot 自己开到 2bb，我们探针开到 2.5bb——两个格子要分别解
    open_sizes = {
        "它按钮首行动": 2.0,
        "它面对3bet": 2.0,
        "它大盲面对开牌": probe["open_to_chips"] / probe["big_blind"],
    }
    stack = 200.0

    grid = [
        RealizationModel(sharpening=s, out_of_position=o)
        for s in (0.2, 0.35, 0.5, 0.7)
        for o in (0.88, 0.93, 0.98)
    ]

    print(f"实测 {document['hands']} 手 vs 我们的模型（单挑 {stack:.0f}bb）\n")
    cache: dict[tuple, dict] = {}
    scored = []
    for model in grid:
        errors = []
        predicted = {}
        for spot, observed_freq in observed.items():
            key = (model.key(), open_sizes[spot])
            if key not in cache:
                cache[key] = _predict(
                    model, stack_bb=stack, open_to_bb=open_sizes[spot], iterations=iterations
                )
            got = cache[key].get(spot)
            if not got:
                continue
            predicted[spot] = got
            errors.extend((got[k] - observed_freq[k]) ** 2 for k in observed_freq)
        if not errors:
            raise SystemExit(
                f"没有样本量达到 {min_samples} 的格子——先多采一些手，"
                f"或用 --min-samples 放低门槛（样本少时结论不可靠）"
            )
        score = (sum(errors) / len(errors)) ** 0.5
        scored.append((score, model, predicted))
        print(
            f"γ=1+{model.sharpening:.2f} OOP={model.out_of_position:.2f} → 均方根偏差 {100 * score:.1f}pp",
            flush=True,
        )

    scored.sort(key=lambda item: item[0])
    score, model, predicted = scored[0]
    print(f"\n最接近的一组：γ=1+{model.sharpening:.2f}、OOP={model.out_of_position:.2f}"
          f"（当前默认是 γ=1+0.35、OOP=0.93），均方根偏差 {100 * score:.1f}pp\n")
    for spot, observed_freq in observed.items():
        if spot not in predicted:
            continue
        line = "  ".join(
            f"{k} 实测{100 * observed_freq[k]:5.1f}% / 模型{100 * predicted[spot][k]:5.1f}%"
            for k in ("弃牌", "跟注", "加注")
        )
        print(f"{spot:14s} {line}")
    print(
        "\n提醒：Slumbot 是单挑 200bb 且非 GTO，这只是量级校验；"
        "改默认参数前请确认这个方向也适用于六人 100bb。"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用 Slumbot 采集聚合频率")
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("slumbot_frequencies.json"))
    parser.add_argument("--open-to", type=int, default=250, help="我们开牌到多少筹码（2.5bb）")
    parser.add_argument("--three-bet", type=float, default=3.0, help="3bet 是它开牌额的几倍")
    parser.add_argument("--checkpoint", type=int, default=100)
    parser.add_argument("--report", type=Path, help="只报告已采集的文件，不联网")
    parser.add_argument("--compare", type=Path, help="把已采集的文件与我们的模型对照，不联网")
    parser.add_argument("--iterations", type=int, default=300, help="对照时每次求解的迭代数")
    parser.add_argument("--min-samples", type=int, default=100, help="样本量低于此值的格子不参与对照")
    args = parser.parse_args(argv)

    if args.report:
        report(args.report)
        return 0
    if args.compare:
        compare(args.compare, args.iterations, args.min_samples)
        return 0

    collect(args.hands, args.out, args.open_to, args.three_bet, args.checkpoint)
    report(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""蒙特卡洛权益估算。

给定自己的底牌与已知公共牌，估计对上 N 个随机对手时的胜率（平分按份计）。
这是一块会被反复复用的基础件：占位规则 bot 现在用它做决策，M4 的多人桌代理也要用它
配合范围跟踪来估权益。

纯逻辑：随机数由调用方注入的 `rng` 提供，样本数固定时结果可复现。
"""

from __future__ import annotations

import random
from itertools import combinations

from .cards import FULL_DECK
from .evaluator import evaluate

BOARD_SIZE = 5


def monte_carlo_equity(
    hole: object,
    board: object = (),
    num_opponents: int = 1,
    *,
    samples: int = 200,
    rng: random.Random | None = None,
) -> float:
    """返回 0..1 的权益估计。平分底池按 1/并列人数 计入。

    对手底牌按均匀随机抽取——也就是「对手范围 = 任意两张」。这是刻意的下限假设：
    真实对手的范围更强，所以这个数字偏乐观，用它做决策阈值时要留余量。
    范围感知的权益估算留到 M4 的多人桌代理。
    """
    hole = list(hole)  # type: ignore[arg-type]
    board = list(board)  # type: ignore[arg-type]
    if len(hole) != 2:
        raise ValueError("底牌必须是两张")
    if len(board) > BOARD_SIZE:
        raise ValueError("公共牌不能超过五张")
    if num_opponents < 1:
        raise ValueError("至少要有一个对手")
    if samples < 1:
        raise ValueError("样本数必须为正")

    rng = rng or random.Random()
    known = set(hole) | set(board)
    if len(known) != len(hole) + len(board):
        raise ValueError("已知的牌中存在重复")

    unseen = [c for c in FULL_DECK if c not in known]
    needed_board = BOARD_SIZE - len(board)
    draw_count = needed_board + 2 * num_opponents
    if draw_count > len(unseen):
        raise ValueError("剩余的牌不足以完成抽样")

    total = 0.0
    for _ in range(samples):
        drawn = rng.sample(unseen, draw_count)
        full_board = board + drawn[:needed_board]
        my_score = evaluate(hole + full_board)

        best_opponent = -1
        ties = 0
        cursor = needed_board
        for _ in range(num_opponents):
            opponent = drawn[cursor : cursor + 2]
            cursor += 2
            score = evaluate(opponent + full_board)
            if score > best_opponent:
                best_opponent = score
                ties = 1
            elif score == best_opponent:
                ties += 1

        if my_score > best_opponent:
            total += 1.0
        elif my_score == best_opponent:
            total += 1.0 / (ties + 1)

    return total / samples


def equity_vs_range(
    hole: object,
    board: object,
    opponent_combos: object,
    *,
    weights: object = None,
    samples: int = 200,
    rng: "random.Random | None" = None,
) -> "float | None":
    """对手手牌**从给定范围里抽**，而不是从整副牌均匀抽（FR-11）。

    `monte_carlo_equity` 的假设是「对手范围＝任意两张」，那是刻意的下限：
    真实对手的范围更强，所以那个数偏乐观。有了翻后范围跟踪
    （`postflop_ranges`）之后，这里可以用真正的范围来估。

    - `opponent_combos`：`(牌, 牌)` 的序列；与我方底牌或公共牌**撞牌的会被跳过**。
    - `weights`：与 combos 一一对应的权重；不给就等权。
    - 返回 `None`：**范围里一手可用的牌都不剩**（全撞牌了）。这不是 0% 权益，
      是「算不了」——返回 0 会让 bot 以为自己必输而弃掉一手好牌。

    只支持**单个**对手：多人底池里各家范围互相依赖，用单人范围逐个近似会
    高估自己的赢面，那种近似不如老老实实用随机牌的下限假设。
    """
    hole = list(hole)  # type: ignore[arg-type]
    board = list(board)  # type: ignore[arg-type]
    if len(hole) != 2:
        raise ValueError("底牌必须是两张")
    if samples < 1:
        raise ValueError("样本数必须为正")

    known = set(hole) | set(board)
    combos = list(opponent_combos)  # type: ignore[arg-type]
    values = list(weights) if weights is not None else [1.0] * len(combos)
    if len(values) != len(combos):
        raise ValueError("权重数量与组合数量不一致")

    live: list = []
    live_weights: list = []
    for combo, weight in zip(combos, values):
        if weight <= 0:
            continue
        if combo[0] in known or combo[1] in known:
            continue        # 撞牌：他不可能拿着我手里或牌面上的牌
        live.append(list(combo))
        live_weights.append(weight)
    if not live:
        return None

    rng = rng or random.Random()
    needed_board = BOARD_SIZE - len(board)
    total = 0.0
    for _ in range(samples):
        opponent = rng.choices(live, weights=live_weights, k=1)[0]
        unseen = [c for c in FULL_DECK if c not in known and c not in opponent]
        if needed_board > len(unseen):
            raise ValueError("剩余的牌不足以完成抽样")
        full_board = board + (rng.sample(unseen, needed_board) if needed_board else [])

        mine = evaluate(hole + full_board)
        theirs = evaluate(opponent + full_board)
        if mine > theirs:
            total += 1.0
        elif mine == theirs:
            total += 0.5
    return total / samples


def exact_equity(
    hole_a: object,
    hole_b: object,
    board: object = (),
) -> float:
    """两副**具体**底牌的精确对局权益，穷举全部可能的公共牌。

    这是全项目的权益基准值——蒙特卡洛估计的正确性以它为准。

    代价随缺牌数急剧上升：翻前（缺 5 张）要枚举 C(48,5)=1,712,304 个牌面，
    单核约 16 秒；翻牌后（缺 2 张）不到一毫秒。翻前的整表预计算不能走这条路，
    见 `scripts/` 下的预计算工具。
    """
    hole_a = list(hole_a)  # type: ignore[arg-type]
    hole_b = list(hole_b)  # type: ignore[arg-type]
    board = list(board)  # type: ignore[arg-type]
    if len(hole_a) != 2 or len(hole_b) != 2:
        raise ValueError("两边的底牌都必须是两张")
    if len(board) > BOARD_SIZE:
        raise ValueError("公共牌不能超过五张")

    known = hole_a + hole_b + board
    if len(set(known)) != len(known):
        raise ValueError("已知的牌中存在重复")

    unseen = [c for c in FULL_DECK if c not in set(known)]
    needed = BOARD_SIZE - len(board)

    wins = ties = total = 0
    for extra in combinations(unseen, needed):
        full_board = board + list(extra)
        score_a = evaluate(hole_a + full_board)
        score_b = evaluate(hole_b + full_board)
        total += 1
        if score_a > score_b:
            wins += 1
        elif score_a == score_b:
            ties += 1

    return (wins + ties / 2) / total

"""预计算的翻前对局权益表。

169×169 的浮点表，由 `scripts/build_preflop_equity.py` 生成，随包分发（约 112 KB）。
读取只用标准库的 `array`，保持 `holdem` 包零依赖。

数值是蒙特卡洛估计（默认每个对局 10,000 个样本），不是精确值；精确基准见
`equity.exact_equity`，两者的一致性由测试守着。同类对同类恒为 0.5（对称性）。
"""

from __future__ import annotations

import array
import sys
from functools import lru_cache
from pathlib import Path

from .ranges import NUM_HAND_CLASSES, Range, class_combo_count, class_combos

MAGIC = b"PFEQ1"
DATA_PATH = Path(__file__).parent / "data" / "preflop_equity.bin"


class EquityTableMissing(RuntimeError):
    """权益表尚未生成。"""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"找不到翻前权益表 {path}；"
            f"先运行 python3 scripts/build_preflop_equity.py 生成"
        )


@lru_cache(maxsize=1)
def _load(path_text: str) -> tuple[array.array, int]:
    path = Path(path_text)
    if not path.exists():
        raise EquityTableMissing(path)
    raw = path.read_bytes()
    newline = raw.index(b"\n")
    magic, classes_text, samples_text = raw[:newline].split()
    if magic != MAGIC:
        raise ValueError(f"权益表格式不对：{magic!r}")
    classes = int(classes_text)
    if classes != NUM_HAND_CLASSES:
        raise ValueError(f"权益表的牌类数不匹配：{classes}")
    values = array.array("f")
    values.frombytes(raw[newline + 1 :])
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != NUM_HAND_CLASSES * NUM_HAND_CLASSES:
        raise ValueError(f"权益表长度不对：{len(values)}")
    return values, int(samples_text)


def is_available(path: Path = DATA_PATH) -> bool:
    return path.exists()


def sample_count(path: Path = DATA_PATH) -> int:
    """生成该表时每个对局用了多少样本，用于判断精度。"""
    return _load(str(path))[1]


def preflop_equity(index_a: int, index_b: int, path: Path = DATA_PATH) -> float:
    """牌类 `index_a` 对 `index_b` 的翻前全下权益（平分计半）。"""
    values, _ = _load(str(path))
    return values[index_a * NUM_HAND_CLASSES + index_b]


def equity_matrix(path: Path = DATA_PATH) -> array.array:
    """整张表，供需要批量运算的调用方直接索引 `i * 169 + j`。"""
    return _load(str(path))[0]


# ------------------------------------------------------------------ 共牌权重


@lru_cache(maxsize=1)
def removal_weights() -> tuple[float, ...]:
    """`weights[i * 169 + j]` = 我方持有牌类 i 时，对手仍可能持有的牌类 j 的组合数。

    这就是「共牌效应」：我拿着 AA，对手能拿到的 AK 组合就少了一半。翻前求解若忽略这一项，
    推弃门槛会系统性偏移，与公开的纳什表对不上。数值精确可算，与蒙特卡洛无关。
    """
    combos = [class_combos(i) for i in range(NUM_HAND_CLASSES)]
    weights = [0.0] * (NUM_HAND_CLASSES * NUM_HAND_CLASSES)
    for i in range(NUM_HAND_CLASSES):
        mine = combos[i]
        for j in range(NUM_HAND_CLASSES):
            total = 0
            for hero in mine:
                hero_cards = set(hero)
                total += sum(
                    1 for villain in combos[j] if hero_cards.isdisjoint(villain)
                )
            weights[i * NUM_HAND_CLASSES + j] = total / len(mine)
    return tuple(weights)


def equity_vs_range(index: int, opponent: Range, path: Path = DATA_PATH) -> float:
    """牌类 `index` 对上一个范围的权益，按组合数与共牌效应加权。

    对手范围为空时返回 0.0——调用方应先判断范围非空，这里不猜。
    """
    values, _ = _load(str(path))
    weights = removal_weights()
    numerator = 0.0
    denominator = 0.0
    for other, weight in opponent.weights.items():
        share = weight * weights[index * NUM_HAND_CLASSES + other]
        if share <= 0:
            continue
        numerator += share * values[index * NUM_HAND_CLASSES + other]
        denominator += share
    return numerator / denominator if denominator else 0.0


def range_vs_range_equity(hero: Range, villain: Range, path: Path = DATA_PATH) -> float:
    """范围对范围的权益。复盘里展示「你的范围对他的范围」用的就是它。"""
    numerator = 0.0
    denominator = 0.0
    weights = removal_weights()
    for index, weight in hero.weights.items():
        share = weight * class_combo_count(index)
        available = sum(
            other_weight * weights[index * NUM_HAND_CLASSES + other]
            for other, other_weight in villain.weights.items()
        )
        if share <= 0 or available <= 0:
            continue
        numerator += share * available * equity_vs_range(index, villain, path)
        denominator += share * available
    return numerator / denominator if denominator else 0.0

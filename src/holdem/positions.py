"""位置命名。

统计与复盘几乎每个指标都要按位置拆分，所以位置名必须有唯一口径。
表以「相对按钮的偏移」为索引：偏移 0 是按钮，1 是小盲，2 是大盲，其余向枪口位延伸。
"""

from __future__ import annotations

# 索引 = 相对按钮的偏移量
_TABLES: dict[int, tuple[str, ...]] = {
    2: ("BTN", "BB"),
    3: ("BTN", "SB", "BB"),
    4: ("BTN", "SB", "BB", "CO"),
    5: ("BTN", "SB", "BB", "UTG", "CO"),
    6: ("BTN", "SB", "BB", "UTG", "HJ", "CO"),
    7: ("BTN", "SB", "BB", "UTG", "UTG+1", "HJ", "CO"),
    8: ("BTN", "SB", "BB", "UTG", "UTG+1", "MP", "HJ", "CO"),
    9: ("BTN", "SB", "BB", "UTG", "UTG+1", "UTG+2", "MP", "HJ", "CO"),
}


def position_names(num_seats: int) -> tuple[str, ...]:
    """按「相对按钮偏移」顺序返回位置名。单挑时按钮即小盲，记作 BTN。"""
    try:
        return _TABLES[num_seats]
    except KeyError:
        raise ValueError(f"不支持的座位数: {num_seats}") from None


def position_of(seat: int, button: int, num_seats: int) -> str:
    offset = (seat - button) % num_seats
    return position_names(num_seats)[offset]


def is_in_position_preflop(seat: int, button: int, num_seats: int) -> bool:
    """翻前是否属于后位（劫机位及之后）。用于粗粒度的位置分组统计。"""
    return position_of(seat, button, num_seats) in ("BTN", "CO", "HJ")

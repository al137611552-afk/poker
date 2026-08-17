"""读取离线生成的翻前范围表。

表由 `scripts/build_preflop_ranges.py` 算出来（解一次要十几分钟，见 `preflop_chain.py`），
以 JSON 随包分发。这里只负责读取与查询，**不做任何求解**——运行时要的是查表的速度，
不是重算的能力。

```python
table = preflop_ranges.load()
table.open_range("CO")                 # CO 的开牌范围
table.defense("CO", "BB").action("加注到7.5")   # BB 面对 CO 开牌的 3bet 范围
```

表里连**参数一起存**：兑现模型的系数、开牌尺度、扫了几轮。范围表是一堆数字，脱离
参数无法审计——将来兑现系数校准过了（ADR-0003），得能一眼看出手上这张是哪套算的。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .ranges import Range

__all__ = [
    "DefenseEntry",
    "PreflopRangeTable",
    "RangeTableMissing",
    "is_available",
    "load",
]

FORMAT = "PFRANGE1"
DATA_PATH = Path(__file__).parent / "data" / "preflop_ranges_6max_100bb.json"


class RangeTableMissing(RuntimeError):
    """范围表尚未生成。"""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"找不到翻前范围表 {path}；"
            f"先运行 python3 scripts/build_preflop_ranges.py 生成（约十几分钟）"
        )


@dataclass(frozen=True)
class DefenseEntry:
    """一个防守者面对某个位置开牌时的策略。"""

    opener: str
    defender: str
    actions: dict[str, Range]
    """动作标签 → 范围。标签就是树里的中文标签，如「弃牌」「跟注到2.5」「加注到7.5」。"""
    frequencies: dict[str, float]
    """动作标签 → 频率（按组合数加权，且只统计走得到这里的牌）。"""
    exploitability: float
    reraise_reply: dict[str, Range] | None = None
    """开牌者面对这个防守者再加注时的应对；没有再加注分支时是 None。"""
    reraise_reply_frequencies: dict[str, float] | None = None
    facing_reraise: str | None = None
    """上面那组应对是在面对哪个加注尺度。"""

    def action(self, label: str) -> Range:
        try:
            return self.actions[label]
        except KeyError:
            raise KeyError(
                f"{self.defender} 面对 {self.opener} 开牌没有「{label}」这个动作；"
                f"有的是：{'、'.join(self.actions)}"
            ) from None

    def frequency(self, label: str) -> float:
        return self.frequencies.get(label, 0.0)

    @property
    def fold_frequency(self) -> float:
        return self.frequency("弃牌")


@dataclass(frozen=True)
class PreflopRangeTable:
    table: dict
    """生成这张表用的整桌参数（人数、筹码、开牌尺度……）。"""
    model: dict
    """生成这张表用的权益兑现参数。"""
    sweeps: int
    max_change: float
    opens: dict[str, Range]
    open_frequencies: dict[str, float]
    defenses: dict[tuple[str, str], DefenseEntry]

    @property
    def positions(self) -> tuple[str, ...]:
        """有开牌范围的位置，按行动顺序。"""
        return tuple(self.opens)

    def open_range(self, position: str) -> Range:
        try:
            return self.opens[position]
        except KeyError:
            raise KeyError(
                f"没有 {position} 的开牌范围；表里有：{'、'.join(self.opens)}"
            ) from None

    def open_frequency(self, position: str) -> float:
        return self.open_frequencies[position]

    def defense(self, opener: str, defender: str) -> DefenseEntry:
        try:
            return self.defenses[(opener, defender)]
        except KeyError:
            raise KeyError(f"表里没有「{defender} 面对 {opener} 开牌」这一格") from None

    def defenders_of(self, opener: str) -> tuple[str, ...]:
        return tuple(defender for open_, defender in self.defenses if open_ == opener)


def is_available(path: Path = DATA_PATH) -> bool:
    return path.exists()


def load(path: Path = DATA_PATH) -> PreflopRangeTable:
    return _load(str(path))


@lru_cache(maxsize=4)
def _load(path_text: str) -> PreflopRangeTable:
    path = Path(path_text)
    if not path.exists():
        raise RangeTableMissing(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != FORMAT:
        raise ValueError(f"范围表格式不对：{document.get('format')!r}")

    opens: dict[str, Range] = {}
    frequencies: dict[str, float] = {}
    defenses: dict[tuple[str, str], DefenseEntry] = {}
    for opener, spot in document["spots"].items():
        opens[opener] = Range.parse(spot["open"])
        frequencies[opener] = float(spot["open_frequency"])
        for defender, entry in spot["defenses"].items():
            reply = entry.get("vs_reraise")
            defenses[(opener, defender)] = DefenseEntry(
                opener=opener,
                defender=defender,
                actions={
                    label: Range.parse(text) for label, text in entry["actions"].items()
                },
                frequencies={
                    label: float(value) for label, value in entry["frequencies"].items()
                },
                exploitability=float(entry["exploitability"]),
                reraise_reply=(
                    {label: Range.parse(text) for label, text in reply["actions"].items()}
                    if reply
                    else None
                ),
                reraise_reply_frequencies=(
                    {label: float(value) for label, value in reply["frequencies"].items()}
                    if reply
                    else None
                ),
                facing_reraise=reply["facing"] if reply else None,
            )

    return PreflopRangeTable(
        table=document["table"],
        model=document["model"],
        sweeps=int(document["sweeps"]),
        max_change=float(document["max_change"]),
        opens=opens,
        open_frequencies=frequencies,
        defenses=defenses,
    )

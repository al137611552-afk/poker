"""接 Slumbot 的那一层：外部强度标尺（ADR-0002 / FR-6）。

三块，界线是「碰不碰网络」：

| 模块 | 干什么 | 联网 |
|---|---|---|
| `protocol` | 动作串 ↔ `HandState` 的翻译，以及拿 `winnings` 对账 | 否，纯逻辑 |
| `client` | HTTP 会话（`new_hand` / `act`） | **是，全项目仅此一处** |
| `match` | 对局循环 + bb/100 与置信区间 | 否，靠注入的会话 |

命令行入口在 `scripts/play_slumbot.py`（跑基线）与 `scripts/calibrate_slumbot.py`
（采聚合频率校准兑现模型）。两者共用同一套客户端，差别只在策略。
"""

from .client import Session, SlumbotError
from .match import MatchStats, play_hand, play_match
from .protocol import BIG_BLIND, SMALL_BLIND, STACK, HandView, build_state, to_incr

__all__ = [
    "Session",
    "SlumbotError",
    "MatchStats",
    "play_hand",
    "play_match",
    "HandView",
    "build_state",
    "to_incr",
    "BIG_BLIND",
    "SMALL_BLIND",
    "STACK",
]

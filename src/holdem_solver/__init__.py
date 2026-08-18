"""接 TexasSolver：翻后求解的适配层（M3 / FR-9 的地基）。

四块，界线是「碰不碰进程与磁盘」：

| 模块 | 干什么 | 碰 IO |
|---|---|---|
| `request` | 局面 → 求解器的命令文件 | 否，纯逻辑 |
| `result` | 策略树 JSON → `SolvedNode` | 否，纯逻辑 |
| `evaluate` | 在解出来的树上算 EV 与 EV 损失 | 否，纯逻辑 |
| `review` | 打完的一手牌 → 逐个翻后决策点的 EV 损失 | 否，纯逻辑 |
| `backend` | 跑二进制、按指纹缓存 | **是，仅此一处** |

## 为什么是 CPU 版

立项时定的是「TexasSolver **GPU** 版，CPU 作降级」。实测推翻了这条：GPU 版是
**Windows-only、只有 GUI、闭源**，**没有任何 headless / 命令行 / API 接口**，
我们没法自动调用它。能自动化的只有 AGPL 的 CPU 版（console + C FFI，跨平台）。
详见 ADR-0005。

由此推出一条硬约束：**一个翻牌局面要几分钟 ⇒ 实战中现解不可行**。
所以这一层是给**复盘**用的（打完再算，几分钟完全够），不是给实时对战用的。

## 它给得了什么、给不了什么

**给得了**：均衡策略（逐具体组合）、收敛到的可利用度。
**给不了**：**EV**——dump 里一个 ev 字段都没有。所以「你这个动作亏了多少」由 `evaluate`
自己在解出来的树上算（固定英雄的两张牌，对手按解走）。别拿「你打了低频动作」冒充 EV 损失。
"""

from .backend import (
    SolveFailed,
    SolveReport,
    SolverNotInstalled,
    TexasSolver,
    default_cache_dir,
    find_solver_home,
)
from .evaluate import DecisionScore, Spot, evaluate_actions, hand_ev, score_decision
from .request import BetSizes, SolveRequest, format_range
from .review import (
    DecisionPoint,
    LineNotInTree,
    ReviewPlan,
    ReviewResult,
    ScoredDecision,
    Step,
    plan_review,
    score_plan,
)
from .result import SolvedAction, SolvedNode, parse_action, parse_result

__all__ = [
    "BetSizes",
    "DecisionPoint",
    "DecisionScore",
    "LineNotInTree",
    "ReviewPlan",
    "ReviewResult",
    "ScoredDecision",
    "Step",
    "Spot",
    "evaluate_actions",
    "hand_ev",
    "score_decision",
    "SolveRequest",
    "SolveReport",
    "SolvedAction",
    "SolvedNode",
    "SolverNotInstalled",
    "SolveFailed",
    "TexasSolver",
    "default_cache_dir",
    "find_solver_home",
    "format_range",
    "parse_action",
    "parse_result",
    "plan_review",
    "score_plan",
]

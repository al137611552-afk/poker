"""胜率口径：bb/100 与它的置信区间。**全项目只有这一处定义。**

三个地方要报「赢了多少」——批量自对弈（`batch.py`）、Slumbot 基线（`holdem_slumbot`）、
牌谱库的玩家汇总（`store.py`）。它们的算法必须逐位一致：两处口径不同的胜率数字比
没有数字更糟，因为没人会怀疑它。

放在这里而不是 `batch.py`，是为了让 `store.py` 用得上又不必把 bot 与权益估算一起拖进来。

## 为什么必须报区间

扑克单手盈亏的方差极大——单挑 200bb 一手就能输赢两百个大盲。「我们赢 5bb/100」这句话
脱离区间没有任何信息量：几百手的 95% 区间通常有上百 bb/100 宽，正负都在噪声里。
**比强弱看的是区间重不重叠**，不是点估计谁高。

区间只需要「和、平方和、手数」三个数，所以分片并行跑完**直接相加**即可合并，
不必留着每一手的流水。
"""

from __future__ import annotations

import math

__all__ = ["CI95", "bb_per_100", "bb_per_100_interval"]

CI95 = 1.959964
"""95% 置信区间的正态分位数。单手盈亏远非正态，但上万手的**均值**足够接近。"""


def bb_per_100(net: int, hands: int, big_blind: int) -> float:
    """每百手赢取的大盲数。手数为零时返回 0。"""
    if not hands or not big_blind:
        return 0.0
    return 100.0 * net / big_blind / hands


def bb_per_100_interval(
    net: int, net_squares: float, hands: int, big_blind: int
) -> float:
    """bb/100 的 95% 置信半宽；手数不足两手时是 `inf`——没法谈区间。

    `net_squares` 是**单手盈亏的平方和**。样本方差用 n−1 做分母，所以小样本上
    「四倍手数把区间缩一半」只是近似，差着零点几个百分点。
    """
    if hands < 2 or not big_blind:
        return float("inf") if hands < 2 else 0.0
    mean = net / hands
    variance = (net_squares - hands * mean * mean) / (hands - 1)
    if variance <= 0:
        return 0.0
    standard_error = math.sqrt(variance / hands)
    return 100.0 * CI95 * standard_error / big_blind

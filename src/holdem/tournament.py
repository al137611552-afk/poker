"""锦标赛结构与 ICM（FR-13）。纯计算，不碰 IO。

## ICM 在回答什么

现金桌里筹码就是钱，赢一个筹码就是赢一块钱。**锦标赛不是**：奖金是按名次分的，
所以第 10001 个筹码远不如第 1 个值钱。ICM（独立筹码模型）把「我有多少筹码」
换算成「我期望能拿多少奖金」。

这个换算不是学术趣味——它直接改打法。泡沫圈上一个**筹码 EV 为正**的全下，
ICM 口径下常常是**负的**：赢了多拿一点点奖金，输了直接归零。
不做这个换算，短筹码在钱圈附近的建议全是错的。

## 用的是 Malmuth-Harville

「拿第一名的概率 = 自己筹码 / 全场筹码」，然后把冠军拿走、在剩下的人里递归算第二名。

这个模型**假定筹码量就是全部信息**——不看位置、不看技术、不看盲注水平。
它是业界标准，但那是「大家都这么算」，不是「它对」。**别拿它当真值**：
两个筹码相同的人，会打的那个实际价值更高，ICM 看不出来。

## 只算到有奖名次

第四名和第八名都是 0 奖金，没必要区分。名次分布只递归到奖金结构的长度——
九人桌算前三名是 9×8×7 次递归，瞬间出结果；要是傻算到底就是 9! 次。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["BlindLevel", "Structure", "icm_equity", "finish_distribution"]


@dataclass(frozen=True)
class BlindLevel:
    """一个盲注级别。"""

    small_blind: int
    big_blind: int
    ante: int = 0
    minutes: int = 15

    @property
    def cost_per_orbit(self) -> int:
        """一圈的固定成本（不含跟注）。判断「筹码还剩几圈」用的就是它。"""
        return self.small_blind + self.big_blind + self.ante


@dataclass(frozen=True)
class Structure:
    """一个锦标赛的结构：起始筹码、盲注表、奖金分配。"""

    name: str
    starting_stack: int
    levels: "tuple[BlindLevel, ...]"
    payouts: "tuple[float, ...]"
    """按名次的奖金**占奖池的比例**，从第一名起。加起来应当是 1。"""
    entrants: int = 9

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("盲注表不能为空")
        if not self.payouts:
            raise ValueError("奖金结构不能为空")
        if len(self.payouts) > self.entrants:
            raise ValueError("发奖名次比参赛人数还多")
        total = sum(self.payouts)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"奖金比例加起来应当是 1，现在是 {total:.4f}")

    def level_at(self, minute: float) -> BlindLevel:
        """第几分钟处在哪个级别。超出盲注表就停在最后一级。"""
        elapsed = 0.0
        for level in self.levels:
            elapsed += level.minutes
            if minute < elapsed:
                return level
        return self.levels[-1]

    def stack_in_bb(self, chips: int, minute: float) -> float:
        return chips / self.level_at(minute).big_blind

    @property
    def in_the_money(self) -> int:
        """有奖名次数。"""
        return len(self.payouts)


def finish_distribution(
    stacks: "dict[str, int]", places: int
) -> "dict[str, tuple[float, ...]]":
    """每个人拿第 1..places 名的概率（Malmuth-Harville）。

    **筹码为 0 的人直接排除**——他已经出局了，把他留在分母里会稀释所有人的概率。
    """
    live = {name: chips for name, chips in stacks.items() if chips > 0}
    if not live:
        raise ValueError("没有还有筹码的人")
    places = min(places, len(live))

    result = {name: [0.0] * places for name in live}
    _walk(live, places, 1.0, [], result)
    return {name: tuple(values) for name, values in result.items()}


def _walk(live: dict, places: int, weight: float, taken: list, result: dict) -> None:
    """递归：谁拿下这一名，然后在剩下的人里继续。"""
    depth = len(taken)
    if depth >= places or not live:
        return
    total = sum(live.values())
    if total <= 0:
        return
    for name, chips in live.items():
        share = weight * chips / total
        result[name][depth] += share
        if depth + 1 < places:
            rest = {other: value for other, value in live.items() if other != name}
            _walk(rest, places, share, taken + [name], result)


def icm_equity(
    stacks: "dict[str, int]", payouts: "tuple[float, ...]", prize_pool: float = 1.0
) -> "dict[str, float]":
    """每个人的奖金期望。

    `payouts` 是**比例**，`prize_pool` 是奖池总额（默认 1，那样返回的就是比例）。

    **守恒的是「活人还能分的那部分奖池」，不一定是全部奖池。** 三人赛只剩两人时，
    第三名已经落定、那份奖金不在这个计算里，两人分的是前两名那 80%。
    这是 ICM 的正确语义，不是缺陷——但很容易断错（写测试时就断错过一次）。

    出局的人（筹码 0）在结果里**值为 0.0**：那表示「不在此计算范围内」，
    **不是「他没拿到奖金」**——他多半已经锁定了某个名次的奖金。
    保留这个键只是为了让调用方不必到处 `.get()`。
    """
    distribution = finish_distribution(stacks, len(payouts))
    out = {}
    for name, chances in distribution.items():
        out[name] = sum(
            chance * payouts[place] * prize_pool
            for place, chance in enumerate(chances)
        )
    # 出局的人（筹码 0）没有期望，但要在结果里出现——少一个键会让调用方 KeyError
    for name, chips in stacks.items():
        out.setdefault(name, 0.0)
    return out


def icm_pressure(
    stacks: "dict[str, int]", payouts: "tuple[float, ...]", hero: str,
    prize_pool: float = 1.0,
) -> "tuple[float, float]":
    """英雄**赢下一手翻倍**与**输光出局**分别值多少奖金变化。

    返回 `(赢的收益, 输的损失)`，两个都是正数。

    泡沫圈的本质就在这两个数的**不对称**上：赢了多拿一点，输了失去一切。
    筹码 EV 只看筹码，看不到这个不对称——所以它在钱圈附近会给出错的建议。
    """
    if hero not in stacks or stacks[hero] <= 0:
        raise ValueError(f"{hero} 不在牌局里")

    now = icm_equity(stacks, payouts, prize_pool)[hero]

    doubled = dict(stacks)
    doubled[hero] = stacks[hero] * 2
    won = icm_equity(doubled, payouts, prize_pool)[hero]

    busted = dict(stacks)
    busted[hero] = 0
    lost = icm_equity(busted, payouts, prize_pool).get(hero, 0.0)

    return won - now, now - lost


# ------------------------------------------------------------------ 常见结构


TURBO_SNG_9 = Structure(
    name="9 人 turbo SNG",
    starting_stack=1500,
    levels=(
        BlindLevel(10, 20, minutes=6),
        BlindLevel(15, 30, minutes=6),
        BlindLevel(25, 50, minutes=6),
        BlindLevel(50, 100, minutes=6),
        BlindLevel(75, 150, ante=15, minutes=6),
        BlindLevel(100, 200, ante=25, minutes=6),
    ),
    payouts=(0.50, 0.30, 0.20),
    entrants=9,
)
"""最常见的 SNG 结构，用来跑通链路；真要练某个赛事请自己填 `Structure`。"""

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


# ------------------------------------------------------------------ ICM 口径的推弃收益


def icm_push_fold_payoffs(
    stacks: "dict[str, int]", payouts: "tuple[float, ...]", *,
    hero: str, villain: str, big_blind: int, ante: int = 0, prize_pool: float = 1.0,
):
    """把推弃的七个终局折成 **ICM 收益**，喂给 `pushfold.solve_push_fold`。

    `hero` 是小盲、`villain` 是大盲，两人之间推弃；桌上其他人这手牌不参与，
    只交前注。

    ## 为什么值得做

    筹码口径下「+EV 就该推」，ICM 口径下常常不该：泡沫圈上赢了多拿一点点，
    输了直接归零。**两个口径的答案会相反**，而现金桌那套直觉在这里是错的。

    ## 七个数与牌类无关

    终局只有「弃 / 偷到 / 摊牌赢 / 摊牌输」这几种，牌类只影响权益。
    所以 ICM 在这里只算七次，CFR 内部一次都不用重算——不然每步每类都要递归一遍。

    ## 单位

    返回的收益单位是**奖金**（`prize_pool` 给什么就是什么）。
    所以解出来的可利用度也是奖金单位，**别拿它跟筹码口径的 bb/手比大小**。

    ## ⚠ 已知偏差：这是**单手** ICM，会系统性偏紧

    它只问「这一手推还是不推」，**假定不打就没事**。真实的锦标赛里不打也在流血——
    盲注每圈都要交，5bb 的筹码熬不过几圈。

    实测这个偏差有多大：三人都有奖时，短筹码的推范围从筹码口径的 71.6% 掉到
    **11.7%**——那是「反正保住第三名」的逻辑推到极端。照这个打会被盲注吃死。

    严肃工具（HoldemResources 那类）用的是**未来博弈**模型：把后续几手的盲注压力
    一并算进去。我们没有那个。**所以这里的输出适合用来看「方向」
    （ICM 比筹码口径紧多少），不适合直接当作开牌表照抄。**
    """
    from .pushfold import Payoffs

    if hero not in stacks or villain not in stacks:
        raise ValueError("英雄或对手不在筹码表里")
    if stacks[hero] <= 0 or stacks[villain] <= 0:
        raise ValueError("推弃的双方都得还有筹码")

    others = [name for name in stacks if name not in (hero, villain)]
    dead = ante * len(others)
    """其他人交的前注——这手牌他们赢不到，所以那部分钱归本手的赢家。"""

    hero_blind = big_blind // 2 + ante
    villain_blind = big_blind + ante
    effective = min(stacks[hero] - ante, stacks[villain] - ante)

    def board(hero_delta: int, villain_delta: int) -> dict:
        out = dict(stacks)
        out[hero] = stacks[hero] + hero_delta
        out[villain] = stacks[villain] + villain_delta
        for name in others:
            out[name] = stacks[name] - ante
        return out

    def icm_of(who: str, table: dict) -> float:
        return icm_equity(table, payouts, prize_pool).get(who, 0.0)

    now = board(0, 0)
    now_hero, now_villain = icm_of(hero, now), icm_of(villain, now)

    # 小盲弃：他的盲注与前注归大盲，其他人的前注也归大盲
    folded = board(-hero_blind, +hero_blind + dead)
    # 小盲全下、大盲弃：大盲的盲注与前注归小盲
    stolen = board(+villain_blind + dead, -villain_blind)
    # 摊牌：赢家拿走对方投进去的（有效筹码 + 他的前注）以及其他人的死钱。
    # **写成最简形式**——第一版把盲注加加减减地摊在式子里，等价但看不懂，
    # 我自己都得验算一遍才敢确定它对。
    won = board(+effective + ante + dead, -effective - ante)
    lost = board(-effective - ante, +effective + ante + dead)

    return Payoffs(
        sb_fold=icm_of(hero, folded) - now_hero,
        sb_steal=icm_of(hero, stolen) - now_hero,
        sb_win=icm_of(hero, won) - now_hero,
        sb_lose=icm_of(hero, lost) - now_hero,
        bb_fold=icm_of(villain, stolen) - now_villain,
        bb_win=icm_of(villain, lost) - now_villain,
        bb_lose=icm_of(villain, won) - now_villain,
    )

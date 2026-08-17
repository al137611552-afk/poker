"""随机自对弈压测：验证规则不变量，并测出引擎的实际吞吐。

    python3 scripts/soak.py 200000

不变量：筹码守恒、筹码非负、净盈亏归零、发牌不重复、牌局必然收敛。
吞吐数字用于判断「后台自对弈填充数据库」这一功能需不需要把热路径下沉到 Rust。
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from holdem.actions import bet, call, check, fold, raise_to  # noqa: E402
from holdem.deck import deck_from_seed  # noqa: E402
from holdem.phh import loads as phh_loads  # noqa: E402
from holdem.phh import phh_player_order, to_phh  # noqa: E402
from holdem.state import HandConfig, HandState  # noqa: E402

STACK_CHOICES = [2, 7, 25, 100, 400, 1000, 5000]
PHH_CHECK_EVERY = 50  # 每隔这么多手做一次牌谱往返验证


def check_phh_round_trip(hand: HandState) -> None:
    replayed = phh_loads(to_phh(hand))
    order = phh_player_order(hand.config.num_seats, hand.config.button)
    for phh_index, seat in enumerate(order):
        if replayed.stacks[phh_index] != hand.stacks[seat]:
            raise AssertionError(
                f"牌谱往返后筹码不一致: {replayed.stacks} vs {hand.stacks}"
            )


def random_action(hand, rng):
    legal = hand.legal_actions()
    choices = []
    if legal.can_fold:
        choices.append(fold())
    if legal.can_check:
        choices.append(check())
    if legal.can_call:
        choices.append(call())
    if legal.can_raise:
        sizes = {legal.min_raise_to, legal.max_raise_to}
        mid = (legal.min_raise_to + legal.max_raise_to) // 2
        if legal.min_raise_to <= mid <= legal.max_raise_to:
            sizes.add(mid)
        for size in sizes:
            choices.append(bet(size) if legal.is_opening_bet else raise_to(size))
    if not choices:
        raise AssertionError(f"座位 {legal.seat} 无合法动作: {legal}")
    return rng.choice(choices)


def run(num_hands: int, seed: int = 20260817) -> None:
    rng = random.Random(seed)
    started = time.perf_counter()
    showdowns = 0
    total_actions = 0

    for index in range(num_hands):
        num_seats = rng.randint(2, 9)
        stacks = tuple(rng.choice(STACK_CHOICES) for _ in range(num_seats))
        config = HandConfig(
            stacks=stacks,
            button=rng.randrange(num_seats),
            big_blind=10,
            small_blind=5,
            ante=rng.choice([0, 0, 0, 1]),
        )
        hand = HandState(config, deck_from_seed(rng.randrange(1 << 30)))

        steps = 0
        while not hand.is_complete:
            hand.apply(random_action(hand, rng))
            steps += 1
            if steps > 400:
                raise AssertionError(f"第 {index} 手未收敛")
        total_actions += steps

        assert sum(hand.stacks) == sum(stacks), f"第 {index} 手筹码不守恒"
        assert all(s >= 0 for s in hand.stacks), f"第 {index} 手出现负筹码"
        assert sum(hand.result.net) == 0, f"第 {index} 手净盈亏不为零"
        cards = list(hand.board) + [c for h in hand.hole for c in h]
        assert len(set(cards)) == len(cards), f"第 {index} 手出现重复的牌"
        if hand.result.went_to_showdown:
            showdowns += 1
        if index % PHH_CHECK_EVERY == 0:
            check_phh_round_trip(hand)

    elapsed = time.perf_counter() - started
    print(f"手数        {num_hands:,}")
    print(f"耗时        {elapsed:.1f}s")
    print(f"吞吐        {num_hands / elapsed:,.0f} 手/秒（单核 Python）")
    print(f"平均动作数  {total_actions / num_hands:.1f}")
    print(f"摊牌比例    {showdowns / num_hands:.1%}")
    print(f"牌谱往返    每 {PHH_CHECK_EVERY} 手抽验一次，共 {num_hands // PHH_CHECK_EVERY + 1} 手")
    print("全部不变量通过")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)

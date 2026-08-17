"""PHH 牌谱读写测试。

三层验证：
1. 引擎自洽——写出去再读回来，结果必须一致；
2. 外部互认——用 PokerKit 读我们写的文件，最终筹码必须对上（装了才跑）；
3. 反向读入——手写的 PHH 片段能被正确重放。
"""

import random

import pytest

from holdem.actions import bet, call, check, fold, raise_to
from holdem.deck import deck_from_seed, stacked_deck
from holdem.phh import (
    button_for_player_order,
    loads,
    parse_phh,
    phh_player_order,
    to_phh,
)
from holdem.state import HandConfig, HandState

def _play(stacks, button, actions, deck=None, bb=10, sb=5, ante=0):
    config = HandConfig(
        stacks=tuple(stacks), button=button, big_blind=bb, small_blind=sb, ante=ante
    )
    hand = HandState(config, deck if deck is not None else deck_from_seed(7))
    for action in actions:
        hand.apply(action)
    return hand


# ------------------------------------------------------------------ 玩家编号映射


def test_player_order_six_max():
    # 按钮在座位 0 时，小盲是座位 1，PHH 的 p1 就是它
    assert phh_player_order(6, button=0) == [1, 2, 3, 4, 5, 0]
    assert phh_player_order(6, button=3) == [4, 5, 0, 1, 2, 3]


def test_player_order_heads_up_is_reversed():
    # 单挑特例：p1 是大盲，p2 才是按钮/小盲
    assert phh_player_order(2, button=0) == [1, 0]
    assert phh_player_order(2, button=1) == [0, 1]


def test_button_for_player_order_round_trips():
    for n in range(2, 10):
        button = button_for_player_order(n)
        order = phh_player_order(n, button)
        assert order == list(range(n)), "读回时玩家编号应直接对应座位号"


# ------------------------------------------------------------------ 写出的内容


def test_written_fields():
    hand = _play([1000] * 6, button=0, actions=[fold()] * 5)
    text = to_phh(hand, players=[f"座位{i}" for i in range(6)], hand_number=3)
    phh = parse_phh(text)
    assert phh.num_seats == 6
    assert phh.small_blind == 5 and phh.big_blind == 10
    assert phh.hand_number == 3
    assert phh.finishing_stacks is not None
    assert '"d dh p1 ' in text
    assert phh.players[0] == "座位1", "p1 应是小盲（座位 1）"


def test_actions_use_raise_to_amounts():
    hand = _play([1000] * 3, button=0, actions=[raise_to(30), fold(), fold()])
    text = to_phh(hand)
    assert "p3 cbr 30" in text, "三人桌的按钮是 p3，加注额应写目标总额"


# ------------------------------------------------------------------ 往返


def _assert_round_trip(hand):
    text = to_phh(hand)
    replayed = loads(text)
    order = phh_player_order(hand.config.num_seats, hand.config.button)
    assert replayed.is_complete == hand.is_complete
    for phh_index, seat in enumerate(order):
        assert replayed.stacks[phh_index] == hand.stacks[seat]
        assert replayed.hole[phh_index] == hand.hole[seat]
    assert replayed.board == hand.board
    assert replayed.result.went_to_showdown == hand.result.went_to_showdown


def test_round_trip_fold_around():
    _assert_round_trip(_play([1000] * 6, button=0, actions=[fold()] * 5))


def test_round_trip_heads_up_showdown():
    deck = stacked_deck(
        hole={0: "AsAd", 1: "KsKd"}, board="2c3d4h5s7c", num_seats=2, button=0
    )
    hand = _play([500, 500], button=0, actions=[raise_to(500), call()], deck=deck)
    _assert_round_trip(hand)


def test_round_trip_multi_street():
    # 四人桌、按钮在座位 2：小盲 3、大盲 0，翻前由座位 1 先行动
    hand = _play(
        [1000] * 4,
        button=2,
        actions=[
            call(), call(), call(), check(),                 # 翻前：1,2,3,0
            check(), bet(30), fold(), call(), fold(),        # 翻牌：3,0,1,2,3
            check(), check(),                                # 转牌：0,2
            bet(100), raise_to(300), call(),                 # 河牌：0,2,0
        ],
    )
    assert hand.is_complete
    assert hand.result.went_to_showdown
    _assert_round_trip(hand)


def test_round_trip_with_antes_and_side_pots():
    # 三个不同的短筹码全下，制造两层边池
    hand = _play(
        [1000, 300, 100, 40],
        button=0,
        actions=[
            raise_to(39),   # 座位 3 全下（枪口位，前注扣掉 1）
            raise_to(200),  # 座位 0
            call(),         # 座位 1
            call(),         # 座位 2 全下
            check(), check(),  # 翻牌
            check(), check(),  # 转牌
            check(), check(),  # 河牌
        ],
        ante=1,
    )
    assert hand.is_complete
    assert len(hand.result.pots) >= 3, "应产生主池加两层边池"
    _assert_round_trip(hand)


def test_round_trip_random_hands():
    rng = random.Random(4242)
    for _ in range(400):
        n = rng.randint(2, 9)
        stacks = [rng.choice([25, 100, 400, 1000]) for _ in range(n)]
        config = HandConfig(
            stacks=tuple(stacks),
            button=rng.randrange(n),
            big_blind=10,
            small_blind=5,
            ante=rng.choice([0, 0, 1]),
        )
        hand = HandState(config, deck_from_seed(rng.randrange(1 << 30)))
        while not hand.is_complete:
            legal = hand.legal_actions()
            choices = []
            if legal.can_check:
                choices.append(check())
            if legal.can_call:
                choices.append(call())
            if legal.can_fold:
                choices.append(fold())
            if legal.can_raise:
                size = rng.choice([legal.min_raise_to, legal.max_raise_to])
                choices.append(bet(size) if legal.is_opening_bet else raise_to(size))
            hand.apply(rng.choice(choices))
        _assert_round_trip(hand)


# ------------------------------------------------------------------ 读入外部牌谱


def test_reads_hand_written_by_hand():
    text = """
variant = "NT"
antes = [0, 0, 0]
blinds_or_straddles = [5, 10, 0]
min_bet = 10
starting_stacks = [1000, 1000, 1000]
actions = [
  "d dh p1 AcKc",
  "d dh p2 7h6h",
  "d dh p3 2c2d",
  "p3 cbr 30",
  "p1 f",
  "p2 cc",
  "d db AhKh2s",
  "p2 cc",
  "p3 cc",
  "d db 9d",
  "p2 cbr 40",
  "p3 f",
]
"""
    hand = loads(text)
    assert hand.is_complete
    assert not hand.result.went_to_showdown
    # p2（座位 1）赢下底池：翻前 30+30+5 弃掉的小盲，翻后无人跟注的 40 退回
    assert sum(hand.result.net) == 0
    assert hand.result.net[1] > 0


def test_rejects_action_out_of_turn():
    text = """
variant = "NT"
antes = [0, 0]
blinds_or_straddles = [5, 10]
min_bet = 10
starting_stacks = [1000, 1000]
actions = ["d dh p1 AcKc", "d dh p2 7h6h", "p1 f"]
"""
    # 单挑时先行动的是 p2（按钮/小盲），p1 先动属于顺序错误
    with pytest.raises(ValueError, match="行动顺序"):
        loads(text)


def test_rejects_finishing_stack_mismatch():
    hand = _play([1000] * 6, button=0, actions=[fold()] * 5)
    text = to_phh(hand).replace("finishing_stacks = [", "finishing_stacks = [999, ", 1)
    with pytest.raises(ValueError, match="最终筹码"):
        loads(text)


def test_rejects_non_holdem_variant():
    with pytest.raises(ValueError, match="NT"):
        parse_phh('variant = "FT"\nstarting_stacks = [1,1]\n'
                  'blinds_or_straddles = [1,2]\nmin_bet = 2\nactions = []\nantes = [0,0]\n')


# ------------------------------------------------------------------ 外部互认


def test_pokerkit_reads_our_files():
    """验收标准：我们写的牌谱能被 PokerKit 读回，且最终筹码一致。"""
    pk = pytest.importorskip("pokerkit", reason="未安装 pokerkit，跳过外部互认测试")

    rng = random.Random(2026)
    checked = 0
    for _ in range(120):
        n = rng.randint(2, 9)
        stacks = [rng.choice([100, 400, 1000]) for _ in range(n)]
        config = HandConfig(
            stacks=tuple(stacks), button=rng.randrange(n), big_blind=10, small_blind=5
        )
        hand = HandState(config, deck_from_seed(rng.randrange(1 << 30)))
        while not hand.is_complete:
            legal = hand.legal_actions()
            choices = []
            if legal.can_check:
                choices.append(check())
            if legal.can_call:
                choices.append(call())
            if legal.can_fold:
                choices.append(fold())
            if legal.can_raise:
                size = rng.choice([legal.min_raise_to, legal.max_raise_to])
                choices.append(bet(size) if legal.is_opening_bet else raise_to(size))
            hand.apply(rng.choice(choices))

        text = to_phh(hand)
        history = pk.HandHistory.loads(text)
        state = None
        for state in history:
            pass
        order = phh_player_order(n, config.button)
        expected = [hand.stacks[seat] for seat in order]
        actual = list(state.stacks)

        assert sum(actual) == sum(expected), f"筹码总量不一致\n{text}"

        # 平分底池的零头分配规则两边不同：我们按赌场惯例从庄家左手起依次多分一枚，
        # PokerKit 用自己的 divmod 策略。除零头外必须逐一对上。
        tolerance = sum(
            pot.amount % len(winners)
            for pot in hand.result.pots
            if (
                winners := [
                    s
                    for s in pot.eligible
                    if hand.result.showdown_scores.get(s)
                    == max(
                        hand.result.showdown_scores.get(e, -1) for e in pot.eligible
                    )
                    and s in hand.result.showdown_scores
                ]
            )
            and len(winners) > 1
        )
        for index, (got, want) in enumerate(zip(actual, expected)):
            assert abs(got - want) <= tolerance, (
                f"p{index + 1} 筹码不一致（零头容差 {tolerance}）：{got} != {want}\n{text}"
            )
        checked += 1

    assert checked == 120

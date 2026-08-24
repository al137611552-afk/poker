"""水平评级（FR-14）。

重点测两件**错了不会报错**的事：
1. 调整后的账**守恒**（不守恒＝凭空造钱或蒸发钱，而评级照样给得出来）；
2. 样本不够时**不给分**（一个建在 200 手上的「评级」比没有评级更有害）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from holdem.actions import call, check, fold, raise_to  # noqa: E402
from holdem.deck import deck_from_seed  # noqa: E402
from holdem.rating import (  # noqa: E402
    MIN_HANDS, MIN_QUIZ, PlayTrack, QuizTrack, Rating,
    adjusted_net, find_allin_showdown, rate,
)
from holdem.state import HandConfig, HandState  # noqa: E402


def _finish(hand):
    while not hand.is_complete:
        hand.apply(fold() if hand.legal_actions().can_fold else check())
    return hand


def _heads_up_allin(stacks=(1000,) * 6, seed=3):
    hand = HandState(
        HandConfig(stacks=stacks, button=0, big_blind=10, small_blind=5),
        deck_from_seed(seed),
    )
    hand.apply(raise_to(stacks[3]))
    hand.apply(fold())
    hand.apply(fold())
    hand.apply(call())
    return _finish(hand)


def _three_way_allin():
    hand = HandState(
        HandConfig(stacks=(1000, 600, 1000, 1000, 1000, 1000), button=0,
                   big_blind=10, small_blind=5),
        deck_from_seed(9),
    )
    hand.apply(raise_to(1000))
    hand.apply(fold())
    hand.apply(fold())
    hand.apply(call())
    hand.apply(call())
    return _finish(hand)


# ------------------------------------------------------------------ 找全下点


def test_an_all_in_before_the_river_is_found():
    """判据是**公共牌一步跳到五张**——一旦没人能再下注，引擎会把剩下的街一次发完。

    别拿 `stacks` 判：最后那一步 apply 完牌局**已经结算**，赢家的筹码早加上了底池，
    怎么看都不像全下（第一版就栽在这儿，跑三千手一次都没触发）。
    """
    spot = find_allin_showdown(_heads_up_allin())
    assert spot is not None
    assert len(spot.seats) == 2 and spot.cards_left == 5


def test_a_hand_that_ends_by_folding_has_no_all_in_spot():
    """别人全弃、一个人收池——那手没有运气成分可去。"""
    hand = HandState(
        HandConfig(stacks=(1000,) * 6, button=0, big_blind=10, small_blind=5),
        deck_from_seed(5),
    )
    hand.apply(raise_to(30))
    for _ in range(5):
        if not hand.is_complete:
            hand.apply(fold())
    _finish(hand)
    assert find_allin_showdown(hand) is None


def test_an_unfinished_hand_is_refused():
    hand = HandState(
        HandConfig(stacks=(1000,) * 6, button=0, big_blind=10, small_blind=5),
        deck_from_seed(5),
    )
    hand.apply(raise_to(30))
    with pytest.raises(ValueError, match="还没打完"):
        find_allin_showdown(hand)


# ------------------------------------------------------------------ 调整后的账要守恒


def test_the_adjusted_table_still_sums_to_zero():
    """**不守恒＝凭空造钱**，而评级照样给得出来，所以这条必须钉死。

    第一版对两个座位各算一次蒙特卡洛，两次抽样不互补，实测差了 11.6 个筹码。
    """
    for hand in (_heads_up_allin(), _three_way_allin()):
        spot = find_allin_showdown(hand)
        total = sum(adjusted_net(hand, seat, spot=spot)[0] for seat in range(6))
        assert total == pytest.approx(0.0, abs=1e-6)


def test_multiway_all_ins_are_split_by_side_pot():
    """短筹码只能赢他够得着的那层——拿全场权益去分边池会让他分到够不着的钱。"""
    hand = _three_way_allin()
    spot = find_allin_showdown(hand)
    assert len(spot.seats) == 3
    assert len(hand.result.pots) >= 2, "筹码不等，必然分层"
    adjusted = {s: adjusted_net(hand, s, spot=spot) for s in spot.seats}
    assert all(did for _, did in adjusted.values()), "三个人都该被调整"


def test_the_same_hand_always_gets_the_same_number():
    """重算必须一致，否则「昨天的评级」和「今天重算的」对不上，而两次都说自己对。"""
    hand = _heads_up_allin()
    first = adjusted_net(hand, 0)
    assert adjusted_net(hand, 0) == first


def test_a_hand_without_an_all_in_keeps_its_actual_net():
    hand = HandState(
        HandConfig(stacks=(1000,) * 6, button=0, big_blind=10, small_blind=5),
        deck_from_seed(5),
    )
    hand.apply(raise_to(30))
    for _ in range(5):
        if not hand.is_complete:
            hand.apply(fold())
    _finish(hand)
    value, did = adjusted_net(hand, 3)
    assert did is False and value == float(hand.result.net[3])


# ------------------------------------------------------------------ 样本不够就不给分


def test_no_score_until_there_is_enough_quiz():
    small = QuizTrack(answered=MIN_QUIZ - 1, on_solution=MIN_QUIZ - 1, blunders=0)
    assert small.score is None, "全答对也不行——样本不够就是不够"
    enough = QuizTrack(answered=MIN_QUIZ, on_solution=MIN_QUIZ, blunders=0)
    assert enough.score == pytest.approx(100.0)


def test_blunders_cost_double():
    """选到解不推荐的动作，比选到次优动作严重得多。"""
    clean = QuizTrack(answered=100, on_solution=70, blunders=0)
    sloppy = QuizTrack(answered=100, on_solution=70, blunders=20)
    assert clean.score > sloppy.score


def test_no_overall_rating_until_both_tracks_are_ready():
    """一条轨够了也不先给分：那个分会被当成「我的水平」，而它只反映了一半。"""
    play = PlayTrack(hands=MIN_HANDS, adjusted_bb100=0.0, raw_bb100=0.0,
                     adjusted_hands=10, allin_without_hero=3, big_blind=10)
    rating = Rating(quiz=QuizTrack(answered=5, on_solution=5, blunders=0), play=play)
    assert rating.score is None
    assert "测验轨还差" in rating.why

    ready = Rating(quiz=QuizTrack(answered=MIN_QUIZ, on_solution=40, blunders=2), play=play)
    assert ready.score is not None and "评级有效" in ready.why


def test_the_play_track_reports_both_numbers():
    """原始与调整后都要给——差得远本身就是信息（说明那批牌运气成分很大）。"""
    hands = [_heads_up_allin(seed=s) for s in (3, 4, 5)]
    track = rate(quiz=QuizTrack(0, 0, 0), hands=hands, seat=3, big_blind=10).play
    assert track.hands == 3
    assert track.adjusted_hands >= 1
    assert track.raw_bb100 != track.adjusted_bb100
    assert track.score is None, "三手牌当然不够给分"

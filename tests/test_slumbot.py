"""Slumbot 接入层的测试（FR-6）——**全部离线**。

联网的只有 `holdem_slumbot.client`，这里一概不碰它：协议翻译本来就是纯逻辑，
对局循环靠一个按剧本回放的假会话驱动。这样测试既不依赖别人的服务，也能把
「出错怎么办」「对不上怎么办」这类分支逼出来。

剧本里的动作串是**实测采到的真样本**，不是编的。
"""

import pytest

from holdem.actions import bet, call, check, fold, raise_to
from holdem_slumbot.client import SlumbotError
from holdem_slumbot.match import MatchStats, play_match
from holdem_slumbot.protocol import (
    BIG_BLIND,
    HandView,
    build_state,
    check_result,
    facing_bet,
    iter_tokens,
    to_incr,
    tokenize,
)

HOLE = ("As", "Kd")


def view(action, *, client_pos=0, board=(), winnings=None, hole=HOLE):
    return HandView(
        hole=hole, board=tuple(board), action=action, client_pos=client_pos, winnings=winnings
    )


# ------------------------------------------------------------------ 动作串


def test_tokenize_splits_bets_from_letters():
    assert tokenize("b200c") == ["b200", "c"]
    assert tokenize("kb1600c") == ["k", "b1600", "c"]
    assert tokenize("") == []


def test_iter_tokens_drops_the_street_separators():
    assert iter_tokens("b200c/kb200c/kk/kf") == [
        "b200", "c", "k", "b200", "c", "k", "k", "k", "f",
    ]


def test_facing_bet_looks_only_at_the_current_street():
    assert facing_bet("b200c/kb400")
    assert not facing_bet("b200c/kb400c")
    assert not facing_bet("b200c/k"), "上一街的下注不算数"
    assert not facing_bet("")


# ------------------------------------------------------------------ 回放


def test_client_pos_decides_which_seat_is_ours():
    assert view("", client_pos=1).our_seat == 0, "1 = 我们是按钮（翻前先说话）"
    assert view("", client_pos=0).our_seat == 1, "0 = 我们是大盲"


def test_replay_puts_us_on_the_clock():
    """它开到 2bb，轮到我们（大盲）说话。"""
    hand = build_state(view("b200", client_pos=0))
    assert hand.to_act == 1
    assert hand.legal_actions().call_to == 200
    assert hand.hole[1] == build_state(view("b200", client_pos=0)).hole[1], "回放要可重复"


def test_bet_sizes_are_per_street_totals():
    """`b200c/kb400c` ＝ 翻前 200 + 翻牌 400，一共 600。

    这是实测定下的口径：若把数字当成「本手总额」，这里会算成 400，
    整场对局的 bb/100 就会系统性地错。
    """
    # 我们是大盲：翻前跟到 200，翻牌先说话下注到 400
    hand = build_state(view("b200c/b400", client_pos=0, board=("Th", "8s", "6s")))
    assert hand.committed_street[1] == 400, "本街投了 400"
    assert hand.committed_total[1] == 600, "两条街合计 600"


def test_replay_deals_the_board_we_were_told_about():
    hand = build_state(view("b200c/", client_pos=0, board=("Th", "8s", "6s")))
    from holdem.cards import card_to_str

    assert [card_to_str(card) for card in hand.board] == ["Th", "8s", "6s"]


def test_our_hole_cards_land_in_our_seat():
    from holdem.cards import card_to_str

    hand = build_state(view("b200", client_pos=1, hole=("7c", "2d")))
    assert sorted(card_to_str(c) for c in hand.hole[0]) == ["2d", "7c"]


def test_an_over_long_action_string_is_caught():
    with pytest.raises(ValueError, match="比牌局长"):
        build_state(view("b200fc", client_pos=1))


def test_hole_cards_must_be_two():
    with pytest.raises(ValueError, match="两张"):
        build_state(view("", client_pos=1, hole=("As",)))


# ------------------------------------------------------------------ 出招翻译


def test_every_action_kind_translates():
    assert to_incr(fold()) == "f"
    assert to_incr(check()) == "k", "过牌是 k，发 c 会被判 Illegal call"
    assert to_incr(call()) == "c"
    assert to_incr(bet(400)) == "b400"
    assert to_incr(raise_to(750)) == "b750"


# ------------------------------------------------------------------ 对账


def test_a_fold_ending_reconciles_to_the_chip():
    """我们开到 2bb 它弃牌：净赚一个大盲，一分不差。"""
    assert check_result(view("b200f", client_pos=1, winnings=100)) is None


def test_a_wrong_fold_payout_is_caught():
    problem = check_result(view("b200f", client_pos=1, winnings=200))
    assert problem and "弃牌终局对不上" in problem


def test_a_showdown_reconciles_by_the_amount_at_risk():
    """摊牌看不见对手的牌，但双方投入相等——净得失只可能是 ±投入或平分。"""
    hand = view(
        "b200c/kk/kk/kk", client_pos=0, board=("9c", "4s", "3c", "Qs", "6d"), winnings=200
    )
    assert check_result(hand) is None
    assert check_result(
        view("b200c/kk/kk/kk", client_pos=0, board=("9c", "4s", "3c", "Qs", "6d"), winnings=0)
    ) is None, "平分底池"
    problem = check_result(
        view("b200c/kk/kk/kk", client_pos=0, board=("9c", "4s", "3c", "Qs", "6d"), winnings=350)
    )
    assert problem and "摊牌终局对不上" in problem


def test_a_long_real_hand_reconciles():
    """实测样本：四条街打到河牌，输掉 200+200+400+1600。"""
    assert (
        check_result(
            view(
                "b200c/kb200c/kb400c/kb1600c",
                client_pos=0,
                board=("Th", "8s", "6s", "7h", "Jd"),
                winnings=-2400,
            )
        )
        is None
    )


def test_an_unfinished_hand_has_nothing_to_reconcile():
    assert "还没结束" in check_result(view("b200", client_pos=0))


# ------------------------------------------------------------------ 对局循环


class FakeSession:
    """按剧本回放的假会话。每手一份剧本：第一帧是 new_hand 的回包，其后每 act 一帧。"""

    def __init__(self, script):
        self.script = list(script)
        self.frames = iter(())
        self.sent: list[str] = []
        self.resets = 0
        self.requests = 0

    def new_hand(self):
        if not self.script:
            raise AssertionError("剧本用完了，说明对局循环打得比预期多")
        self.frames = iter(self.script.pop(0))
        return self._next()

    def act(self, incr):
        self.sent.append(incr)
        return self._next()

    def _next(self):
        self.requests += 1
        frame = next(self.frames)
        if isinstance(frame, Exception):
            raise frame
        return frame

    def reset(self):
        self.resets += 1


def frame(action, *, client_pos, board=(), winnings=None, hole=HOLE):
    body = {
        "action": action,
        "client_pos": client_pos,
        "hole_cards": list(hole),
        "board": list(board),
    }
    if winnings is not None:
        body["winnings"] = winnings
    return body


ALWAYS_FOLD = lambda hand: fold()  # noqa: E731


def test_a_match_counts_completed_hands_only():
    session = FakeSession(
        [
            # 我们是大盲，它开到 2bb，我们弃牌 → 输掉 1bb
            [frame("b200", client_pos=0), frame("b200f", client_pos=0, winnings=-100)],
            [frame("b200", client_pos=0), frame("b200f", client_pos=0, winnings=-100)],
        ]
    )
    stats = play_match(session, ALWAYS_FOLD, hands=2)
    assert stats.hands == 2
    assert session.sent == ["f", "f"]
    assert stats.net == -200
    assert stats.bb100 == pytest.approx(-100.0), "每手输 1bb ＝ −100bb/100"
    assert stats.mismatches == []


def test_positions_are_split_out():
    session = FakeSession(
        [
            [frame("b200", client_pos=0), frame("b200f", client_pos=0, winnings=-100)],
            [frame("", client_pos=1), frame("f", client_pos=1, winnings=-50)],
        ]
    )
    stats = play_match(session, ALWAYS_FOLD, hands=2)
    assert (stats.hands_as_big_blind, stats.net_as_big_blind) == (1, -100)
    assert (stats.hands_as_button, stats.net_as_button) == (1, -50), "按钮位只亏小盲"


def test_a_broken_hand_is_thrown_away_and_replayed():
    """出错的那一手整手作废，换会话重来——绝不能把半手的结果算进统计。"""
    session = FakeSession(
        [
            [frame("b200", client_pos=0), SlumbotError("Illegal action")],
            [frame("b200", client_pos=0), frame("b200f", client_pos=0, winnings=-100)],
        ]
    )
    stats = play_match(session, ALWAYS_FOLD, hands=1)
    assert (stats.hands, stats.aborted, session.resets) == (1, 1, 1)
    assert stats.net == -100, "只算那一手打完的"


def test_too_many_errors_stop_the_match():
    session = FakeSession([[frame("b200", client_pos=0), SlumbotError("boom")]] * 5)
    with pytest.raises(RuntimeError, match="作废了"):
        play_match(session, ALWAYS_FOLD, hands=2, max_errors=2)


def test_a_result_that_cannot_be_true_is_recorded():
    session = FakeSession(
        [[frame("b200", client_pos=0), frame("b200f", client_pos=0, winnings=-999)]]
    )
    stats = play_match(session, ALWAYS_FOLD, hands=1)
    assert len(stats.mismatches) == 1 and "对不上" in stats.mismatches[0]


def test_on_hand_is_called_once_per_hand():
    session = FakeSession(
        [[frame("b200", client_pos=0), frame("b200f", client_pos=0, winnings=-100)]] * 3
    )
    seen = []
    play_match(session, ALWAYS_FOLD, hands=3, on_hand=lambda n, v, s: seen.append(n))
    assert seen == [1, 2, 3]


def test_at_least_one_hand():
    with pytest.raises(ValueError, match="至少要打一手"):
        play_match(FakeSession([]), ALWAYS_FOLD, hands=0)


# ------------------------------------------------------------------ 统计


def _stats(*wins, client_pos=0):
    stats = MatchStats()
    for won in wins:
        stats.add_hand(view("b200f", client_pos=client_pos, winnings=won))
    return stats


def test_bb_per_100_uses_slumbots_blind():
    assert _stats(100, -100, 300, -100).bb100 == pytest.approx(
        100.0 * 200 / BIG_BLIND / 4
    )


def test_a_verdict_needs_the_interval_to_clear_zero():
    assert _stats(*([100] * 50)).beats_slumbot is True, "每手都赢 1bb，区间不含 0"
    assert _stats(*([100, -100] * 25)).beats_slumbot is None, "输赢各半，分不出"
    assert _stats(*([-100] * 50)).beats_slumbot is False
    assert MatchStats().beats_slumbot is None, "一手没打，谈不上胜负"


def test_sessions_merge_by_sums_of_squares():
    """并行的几条会话合起来，与一条会话打同样多手是同一个口径。"""
    left, right = _stats(100, -300), _stats(200, 400)
    together = _stats(100, -300, 200, 400)
    left.add(right)
    assert left.hands == together.hands
    assert left.net == together.net
    assert left.bb100 == pytest.approx(together.bb100)
    assert left.interval == pytest.approx(together.interval)


def test_merging_keeps_every_mismatch():
    left, right = MatchStats(), MatchStats()
    left.mismatches.append("左边这手")
    right.mismatches.append("右边这手")
    left.add(right)
    assert left.mismatches == ["左边这手", "右边这手"]


def test_an_unfinished_hand_cannot_be_counted():
    with pytest.raises(ValueError, match="还没结束"):
        MatchStats().add_hand(view("b200", client_pos=0))


def test_the_report_says_what_it_measured():
    text = _stats(*([100] * 50)).report()
    assert "对 Slumbot 50 手" in text
    assert "bb/100" in text and "按钮位" in text and "大盲位" in text

"""把打完的一手牌接到求解器上的测试（FR-9 的接线）。

这一层自己不算 EV、也不解牌——它只做「接线」，所以测试盯的是**接歪了会怎样**：

1. **哪几个点该打分**：只有英雄自己的翻后决策，弃牌也算（弃牌也是个决策）。
2. **实战尺度必须并进树里**：不并进去，那个动作在解里根本不存在，也就没法打分。
3. **说不了要说清原因**，而且**一个点说不了不能让整手牌作废**——尺度对不上、
   dump 层数不够，都只影响那一个点。

求解本身不在这里测（`test_solver.py` 管那个）：这里的树是手搭的小树，
范围也小到能用纸笔核对每条路值多少。
"""

import pytest

from holdem import preflop_ranges
from holdem.actions import bet, call, check, fold, raise_to
from holdem.cards import card_from_str, card_to_str, cards_from_str
from holdem.deck import deck_from_seed
from holdem.range_tracking import FlopRanges, NotCovered
from holdem.ranges import Range, class_combos
from holdem.state import FLOP, TURN, HandConfig, HandState
from holdem_solver.request import BetSizes, SolveRequest
from holdem_solver.result import SolvedAction, SolvedNode
from holdem_solver.review import (
    DecisionPoint,
    LineNotInTree,
    ReviewPlan,
    Step,
    plan_review,
    resolve_line,
    score_plan,
)

BIG_BLIND = 100

# 座位 0 = 按钮（9s5d）、座位 2 = 大盲（3c4h）、翻牌 8c 7d Kc
OPEN_FOLDS = [fold(), fold(), fold()]
SINGLE_RAISED = [*OPEN_FOLDS, raise_to(250), fold(), call()]

needs_table = pytest.mark.skipif(
    not preflop_ranges.is_available(), reason="翻前范围表尚未生成"
)


def play(actions, *, seats=6, button=0, stack_bb=100.0):
    config = HandConfig(
        stacks=[int(stack_bb * BIG_BLIND)] * seats,
        button=button,
        big_blind=BIG_BLIND,
        small_blind=BIG_BLIND // 2,
    )
    hand = HandState(config, deck_from_seed(11))
    for action in actions:
        hand.apply(action)
    return hand


def hand_cards(text: str) -> "tuple[int, int]":
    return (card_from_str(text[:2]), card_from_str(text[2:]))


# ------------------------------------------------------------------ 规划


@needs_table
def test_only_the_hero_own_decisions_get_scored():
    """要打分的是英雄自己的决策，对手打了什么只是走到那儿的路。"""
    hand = play([*SINGLE_RAISED, check(), bet(350), call()])
    plan = plan_review(hand, hero_seat=0)

    assert [point.taken.kind for point in plan.points] == ["bet"]
    point = plan.points[0]
    assert point.hero == 1, "按钮有位置，求解器口径里是 1"
    assert point.hero_cards == hand_cards("9s5d")
    assert point.prefix == (Step(kind="check", seat=2, street=FLOP),)
    assert point.taken.amount == pytest.approx(3.5), "金额是大盲、是「本街投到多少」"


@needs_table
def test_folding_is_a_decision_too():
    """弃牌也要打分——「该不该弃」正是复盘最常问的那个问题。"""
    hand = play([*SINGLE_RAISED, bet(350), fold()])
    plan = plan_review(hand, hero_seat=0)
    assert [point.taken.kind for point in plan.points] == ["fold"]


@needs_table
def test_the_size_we_actually_bet_gets_added_to_the_tree():
    """实战打出的尺度必须并进树里，否则那个动作在解里根本不存在。

    350 打进 550 的底池是 63.6%——默认的 33/75 都不算「相近」，得单独长一层。
    """
    hand = play([*SINGLE_RAISED, check(), bet(350), call()])
    plan = plan_review(hand, hero_seat=0)
    assert any(abs(size - 63.6) < 0.1 for size in plan.request.bet_sizes.flop), (
        plan.request.bet_sizes.flop
    )
    assert 33.0 in plan.request.bet_sizes.flop, "默认的尺度不该被挤掉"


@needs_table
def test_a_size_close_to_a_default_does_not_grow_the_tree():
    """185 打进 550 是 33.6%，与默认的 33% 是同一回事——不为它多长一层。"""
    hand = play([*SINGLE_RAISED, check(), bet(185), call()])
    plan = plan_review(hand, hero_seat=0)
    assert plan.request.bet_sizes.flop == BetSizes().flop


@needs_table
def test_the_request_comes_from_the_flop_situation():
    """要解的是**翻牌**局面：三张公共牌、翻牌时的底池与身后筹码、双方翻前范围。"""
    hand = play([*SINGLE_RAISED, check(), bet(350), call()])
    plan = plan_review(hand, hero_seat=0)
    assert plan.request.board == tuple(cards_from_str("8c7dKc"))
    assert plan.request.pot == pytest.approx(5.5)
    assert plan.request.effective_stack == pytest.approx(97.5)
    assert plan.request.oop_range == plan.setup.oop
    assert plan.request.ip_range == plan.setup.ip


@needs_table
def test_the_turn_card_is_part_of_the_line():
    """转牌是路上的一步（求解器的树里就是个发牌节点），不是决策点。"""
    hand = play([*SINGLE_RAISED, check(), check(), check(), bet(300), call()])
    plan = plan_review(hand, hero_seat=0)
    turn_point = plan.points[-1]
    assert turn_point.street == TURN
    assert any(step.is_deal for step in turn_point.prefix), "转牌那张牌要在路上"
    assert [step.kind for step in turn_point.prefix] == ["check", "check", "deal", "check"]


@needs_table
def test_a_seat_that_never_saw_the_flop_has_nothing_to_review():
    hand = play([*SINGLE_RAISED, check(), bet(350), call()])
    with pytest.raises(NotCovered, match="没看到翻牌"):
        plan_review(hand, hero_seat=4)


@needs_table
def test_an_uncovered_preflop_line_says_so():
    """翻前线路表里没有，就没有范围可用——这时候**不能**硬凑一个局面去解。"""
    hand = play([call(), call(), call(), call(), call(), check()])
    with pytest.raises(NotCovered):
        plan_review(hand, hero_seat=0)


# ------------------------------------------------------------------ 手搭的小样本

BOARD = tuple(cards_from_str("Qs7h2c9d3s"))
POT = 6.0
STACK = 10.0
VILLAIN = Range.parse("QQ, 77, 22")
HERO = hand_cards("KhKd")
"""样本用**河牌**局面：对手全是暗三条、英雄拿 KK，摊牌胜负确定，每条路都能纸笔核对。

（复盘真正要解的是翻牌局面，但那要枚举 45×44 种跑马，EV 就成了「算出来的数」而不是
「纸笔核得出的数」——接线对不对该在胜负确定的样本上验。）
"""


def action(label, kind, amount=None):
    return SolvedAction(label=label, kind=kind, amount=amount)


def node(player, actions, children, strategy=None):
    return SolvedNode(
        kind="action",
        player=player,
        actions=tuple(actions),
        strategy=strategy or {},
        children=children,
    )


def spread(hand_range, weights):
    """给范围里**每一个**组合都写上同一份策略。

    只给一两个组合写策略，等于其余组合到不了那个节点——对手范围会凭空缩水，
    而 EV 仍然「看着挺像」。
    """
    strategy = {}
    for index in hand_range.classes():
        for card_a, card_b in class_combos(index):
            if card_a in BOARD or card_b in BOARD:
                continue
            strategy[card_to_str(card_a) + card_to_str(card_b)] = weights
    return strategy


def sample_tree(*, villain_calls=True):
    """OOP 过牌 → IP 过牌或下注 3 → OOP 跟或弃。**只有这一条街**。"""
    reply = node(
        0,
        [action("FOLD", "fold"), action("CALL", "call", 3.0)],
        {},
        strategy=spread(VILLAIN, (0.0, 1.0) if villain_calls else (1.0, 0.0)),
    )
    ip = node(
        1,
        [action("CHECK", "check"), action("BET 3.0", "bet", 3.0)],
        {"BET 3.0": reply},
        strategy={"KhKd": (0.5, 0.5)},
    )
    return node(
        0,
        [action("CHECK", "check"), action("BET 2.0", "bet", 2.0)],
        {"CHECK": ip},
        strategy=spread(VILLAIN, (1.0, 0.0)),
    )


def sample_plan(points, *, ip="KK"):
    setup = FlopRanges(
        oop_seat=2,
        ip_seat=0,
        oop=VILLAIN,
        ip=Range.parse(ip),
        pot=POT,
        effective_stack=STACK,
        line="单次加注底池",
    )
    request = SolveRequest(
        board=BOARD,
        pot=POT,
        effective_stack=STACK,
        oop_range=setup.oop,
        ip_range=setup.ip,
        bet_sizes=BetSizes(flop=(50.0,), turn=(50.0,), river=(50.0,)),
    )
    return ReviewPlan(request=request, setup=setup, hero_seat=0, points=tuple(points))


def ip_point(taken, *, hero_cards=HERO):
    return DecisionPoint(
        street=FLOP,
        seat=0,
        hero=1,
        hero_cards=hero_cards,
        prefix=(Step(kind="check", seat=2, street=FLOP),),
        taken=taken,
    )


# ------------------------------------------------------------------ 走线


def test_the_line_is_translated_into_the_labels_the_tree_uses():
    """实战的一步 → 树里的标签。标签的写法（`BET 3.0` 还是 `BET 3.000000`）由树说了算。"""
    labels = resolve_line(sample_tree(), (Step(kind="check", seat=2, street=FLOP),))
    assert labels == ("CHECK",)


def test_a_size_that_is_not_in_the_tree_is_reported_not_rounded():
    """实战打了 7bb、树里只有 3bb：**说不了就说不了**，不许悄悄换成最近的那个。"""
    prefix = (
        Step(kind="check", seat=2, street=FLOP),
        Step(kind="bet", seat=0, amount=7.0, street=FLOP),
    )
    with pytest.raises(LineNotInTree, match="7.00bb"):
        resolve_line(sample_tree(), prefix)


def test_a_truncated_dump_says_it_is_truncated():
    """树只导了翻牌一层，实战却走到了跟注之后——这条路在树里没有下文。"""
    prefix = (
        Step(kind="check", seat=2, street=FLOP),
        Step(kind="bet", seat=0, amount=3.0, street=FLOP),
        Step(kind="call", seat=2, street=FLOP),
    )
    with pytest.raises(LineNotInTree, match="没有子节点"):
        resolve_line(sample_tree(), prefix)


def test_a_deal_where_the_tree_expects_an_action_is_caught():
    prefix = (Step(kind="deal", card=card_from_str("9d"), street=TURN),)
    with pytest.raises(LineNotInTree, match="轮到有人说话"):
        resolve_line(sample_tree(), prefix)


# ------------------------------------------------------------------ 打分


def test_the_loss_is_the_gap_to_the_best_line():
    """KK 面对全是暗三条的范围：过牌 0、下注 3 白扔 3 个大盲，所以这一注亏 3。"""
    point = ip_point(Step(kind="bet", seat=0, amount=3.0, street=FLOP))
    result = score_plan(sample_plan([point]), sample_tree())

    (scored,) = result.decisions
    assert scored.label == "BET 3.0"
    assert scored.score.evs["CHECK"] == pytest.approx(0.0, abs=1e-9)
    assert scored.score.evs["BET 3.0"] == pytest.approx(-3.0, abs=1e-9)
    assert scored.loss == pytest.approx(3.0, abs=1e-9)
    assert result.total_loss == pytest.approx(3.0, abs=1e-9)
    assert result.worst is scored


def test_betting_is_right_when_the_villain_folds():
    """同一手牌、同一个动作，对手改成必弃就该是最优——EV 损失量的是局面，不是动作本身。"""
    point = ip_point(Step(kind="bet", seat=0, amount=3.0, street=FLOP))
    result = score_plan(sample_plan([point]), sample_tree(villain_calls=False))
    (scored,) = result.decisions
    assert scored.score.evs["BET 3.0"] == pytest.approx(POT, abs=1e-9), "对手弃牌，底池归他"
    assert scored.loss == pytest.approx(0.0, abs=1e-9)


def test_one_unscorable_point_does_not_sink_the_whole_hand():
    """尺度不在树里的那个点跳过并说明原因，别的点照样有分。"""
    good = ip_point(Step(kind="bet", seat=0, amount=3.0, street=FLOP))
    bad = ip_point(Step(kind="bet", seat=0, amount=7.0, street=FLOP))
    result = score_plan(sample_plan([good, bad]), sample_tree())

    assert len(result.scored) == 1
    assert result.decisions[1].score is None
    assert "尺度" in result.decisions[1].skipped
    assert result.total_loss == pytest.approx(3.0, abs=1e-9), "跳过的点不进总账"
    assert len(result.skipped_reasons()) == 1


def test_a_hand_outside_the_assumed_range_is_flagged():
    """风格层会打出表外的牌。EV 照样算得出来，但解在那个点上没给这手牌频率——要标出来。"""
    point = ip_point(
        Step(kind="bet", seat=0, amount=3.0, street=FLOP), hero_cards=hand_cards("8s8d")
    )
    result = score_plan(sample_plan([point]), sample_tree())
    (scored,) = result.decisions
    assert scored.in_range is False
    assert scored.score is not None, "牌不在范围里不妨碍算 EV——英雄的牌是固定的"
    assert scored.score.strategy == {}


def test_an_in_range_hand_is_not_flagged():
    point = ip_point(Step(kind="bet", seat=0, amount=3.0, street=FLOP))
    result = score_plan(sample_plan([point]), sample_tree())
    assert result.decisions[0].in_range is True
    assert result.decisions[0].score.strategy == {"CHECK": 0.5, "BET 3.0": 0.5}


# ------------------------------------------------------------------ 真跑（慢）


@pytest.mark.slow
@needs_table
def test_a_real_hand_gets_scored_end_to_end():
    """引擎打一手 → 规划 → **真解一次** → 逐点打分，整条链路走通。

    范围缩到几手、筹码压到 20bb，是为了把这条慢测控制在几分钟内：翻牌局面要导满三层
    （`dump_rounds=3`，见模块说明），树深一点产物就是几十 MB。**接线与口径不受影响**
    ——尺度、位置、底池、英雄的牌全部照实战来。
    """
    from dataclasses import replace

    from holdem_solver import TexasSolver

    if not TexasSolver.available():
        pytest.skip("没装 TexasSolver（TEXAS_SOLVER_HOME）")

    hand = play([*SINGLE_RAISED, check(), bet(350), call()])
    plan = plan_review(hand, hero_seat=0)
    assert plan.points, "按钮在翻牌上打了一注，总得有个点要打分"

    oop, ip = Range.parse("KQs, 99, 87s"), Range.parse("AKo, TT, 95o")
    assert plan.request.dump_rounds == 3, "翻牌局面要导满三条街，不然 EV 积不出来"
    plan = replace(
        plan,
        setup=replace(plan.setup, oop=oop, ip=ip, effective_stack=20.0),
        request=replace(
            plan.request,
            oop_range=oop,
            ip_range=ip,
            effective_stack=20.0,
            accuracy=2.0,
            max_iterations=20,
            use_isomorphism=False,
        ),
    )
    report = TexasSolver(threads=2).solve(plan.request, timeout=1800)
    result = score_plan(plan, report.root)

    (scored,) = result.decisions
    assert scored.skipped is None, scored.skipped
    assert scored.label.startswith("BET"), "实战打的那一注得在树里找得到"
    assert scored.score.evs["CHECK"] == pytest.approx(
        scored.score.evs["CHECK"]
    ), "过牌这条路要算得出来"
    assert scored.loss >= -1e-9, "EV 损失不可能是负的（最优就是 0）"
    assert scored.loss <= plan.request.pot + plan.request.effective_stack
    assert result.total_loss == pytest.approx(scored.loss)

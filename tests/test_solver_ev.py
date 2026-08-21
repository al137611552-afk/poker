"""在解出来的树上算 EV 的测试（FR-9 的地基）。

求解器的产物里没有 EV，这一层是我们自己算的——**所以它必须能被独立验证**，
不能「看着挺像」。这里守三道：

1. **手算对账**：样本是一个刻意造出来的河牌局面——OOP 全是暗三条、IP 全是一对，
   摊牌胜负完全确定，于是每条路的 EV 都能用纸笔算出准确值。
2. **口径硬约束**：弃牌的 EV 恒等于「−已经投进去的钱」；任何动作的 EV 都落在
   `[−已投入, 底池−已投入]` 之间。
3. **与求解器互证**：解自己离最优的差（`gap`）应当与求解器报的可利用度同一量级。
   我们算错了，这个差会明显鼓起来。

发牌节点用手搭的小树验——真求解器的多街产物动辄几百 KB，不适合当样本。
"""

import json
from pathlib import Path

import pytest

from holdem.cards import card_from_str, cards_from_str
from holdem.ranges import Range, class_combos
from holdem_solver.backend import TexasSolver
from holdem_solver.evaluate import Spot, hand_ev, score_decision
from holdem_solver.request import SolveRequest
from holdem_solver.result import SolvedAction, SolvedNode, parse_result

RIVER = Path(__file__).parent / "data" / "texassolver_river.json"
BOARD = tuple(cards_from_str("Qs7h2c9d3s"))
POT = 6.0


@pytest.fixture(scope="module")
def river():
    return parse_result(json.loads(RIVER.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def spot():
    """OOP 三种暗三条、IP 三种一对——IP 在这个牌面上**一手都赢不了**。"""
    return Spot(
        board=BOARD,
        pot=POT,
        effective_stack=9.0,
        oop_range=Range.parse("QQ, 77, 22"),
        ip_range=Range.parse("AA, KK, AKo"),
    )


def hand(text: str) -> tuple[int, int]:
    return (card_from_str(text[:2]), card_from_str(text[2:]))


# ------------------------------------------------------------------ 手算对账


def test_a_hand_that_never_wins_is_priced_to_the_chip(river, spot):
    """IP 拿 AA 面对 3 的下注（底池 6）：弃牌 0、跟注 −3、全下 −9，一分不差。

    对手全是暗三条，摊牌必输，所以这三个数都能用纸笔算出来：跟注就是白扔 3，
    全下就是白扔 9，弃牌一分不亏（这个局面开始时他还没投过钱）。
    """
    score = score_decision(
        spot, river, line=("BET 3.000000",), hero=1, hero_cards=hand("AhAd"), taken="CALL"
    )
    assert score.evs["FOLD"] == pytest.approx(0.0, abs=1e-9)
    assert score.evs["CALL"] == pytest.approx(-3.0, abs=1e-9)
    assert score.evs["RAISE 9.000000"] == pytest.approx(-9.0, abs=1e-9)
    assert score.best == "FOLD"
    assert score.loss() == pytest.approx(3.0, abs=1e-9), "这一跟就是三个大盲"


def test_the_solver_folds_that_hand_too(river, spot):
    """交叉印证：我们算出「弃牌最好」，解给这手牌的频率也该压在弃牌上。"""
    score = score_decision(
        spot, river, line=("BET 3.000000",), hero=1, hero_cards=hand("AhAd")
    )
    assert score.strategy["FOLD"] > 0.9


def test_the_nuts_are_worth_the_pot(river, spot):
    """OOP 拿 QQ 永远赢：每条路都值一个底池（对手弃牌也好、跟注也好，钱都归他）。"""
    score = score_decision(spot, river, line=(), hero=0, hero_cards=hand("QhQd"))
    for label, ev in score.evs.items():
        assert ev == pytest.approx(POT, abs=0.1), label
    assert hand_ev(spot, river, 0, hand("QhQd")) == pytest.approx(POT, abs=0.1)


def test_folding_costs_exactly_what_you_already_put_in(river, spot):
    """OOP 下注 3 之后被加注到 9，此时弃牌的 EV 必须**正好**是 −3。

    这条盯的是投入的账：金额是「本街投到多少」的总额，算增量时减错一次，
    这个数就不是 −3 了，而是 −9 或者 0——都还「看着挺像」。
    """
    score = score_decision(
        spot,
        river,
        line=("BET 3.000000", "RAISE 9.000000"),
        hero=0,
        hero_cards=hand("QhQd"),
    )
    assert score.evs["FOLD"] == pytest.approx(-3.0, abs=1e-9)


def test_every_action_stays_inside_the_pot(river, spot):
    """任何一条路都不可能赢超过底池、也不可能亏超过已投入 + 还要投的钱。"""
    for text in ("AhAd", "KhKd", "AhKd"):
        score = score_decision(
            spot, river, line=("BET 3.000000",), hero=1, hero_cards=hand(text)
        )
        for label, ev in score.evs.items():
            assert -9.0 - 1e-9 <= ev <= POT + 3.0 + 1e-9, (text, label, ev)


# ------------------------------------------------------------------ 与求解器互证


def test_the_solved_strategy_is_close_to_optimal_on_our_numbers(river, spot):
    """**这是这套 EV 算法的自证**。

    解是收敛过的（求解器报的可利用度 0.17% 底池 ≈ 0.010bb），所以按解的混合策略打，
    与「挑最好的那条路」之间的差应当同一量级。我们的 EV 要是算错了——把投入算漏、
    把对手范围的到达概率乘错——这个差会明显鼓起来，而不是停在千分之几个大盲。

    门槛按量级定（0.05bb ≈ 求解器那个数的五倍），**不是拿它当基准数字**。
    """
    for hero, line, hand_range in (
        (0, (), spot.oop_range),
        (1, ("BET 3.000000",), spot.ip_range),
    ):
        gaps = []
        for index in hand_range.weights:
            for combo in class_combos(index):
                if combo[0] in BOARD or combo[1] in BOARD:
                    continue
                score = score_decision(spot, river, line=line, hero=hero, hero_cards=combo)
                if score.strategy:
                    gaps.append(score.gap)
        assert gaps, f"玩家 {hero} 一手都没算到"
        assert all(gap >= -1e-9 for gap in gaps), "解不可能比最优还好"
        average = sum(gaps) / len(gaps)
        assert average < 0.05, f"玩家 {hero} 的自证差 {average:.4f}bb 偏大"


# ------------------------------------------------------------------ 打分对象


def test_the_score_reports_loss_against_the_best_line(river, spot):
    score = score_decision(
        spot, river, line=("BET 3.000000",), hero=1, hero_cards=hand("AhAd"), taken="CALL"
    )
    assert score.loss("FOLD") == pytest.approx(0.0, abs=1e-9)
    assert score.loss("RAISE 9.000000") == pytest.approx(9.0, abs=1e-9)
    assert score.solved_ev <= score.evs[score.best] + 1e-9
    with pytest.raises(KeyError, match="没有"):
        score.loss("BET 100")


def test_a_score_without_an_action_cannot_report_a_loss(river, spot):
    score = score_decision(spot, river, line=(), hero=0, hero_cards=hand("QhQd"))
    with pytest.raises(ValueError, match="不知道实际打的是哪个"):
        score.loss()


# ------------------------------------------------------------------ 发牌节点（手搭小树）


def always(hand_range: Range, actions=("CHECK",)) -> dict:
    """给范围里**每一个组合**都写上策略。

    只给一个组合是不够的：查不到策略的手牌会被当成「走不到这里」而权重清零，
    于是对手的范围凭空缩水——手搭样本时最容易漏的就是这一条（这里踩过）。
    """
    from holdem.cards import card_to_str

    weights = tuple([1.0] + [0.0] * (len(actions) - 1))
    return {
        card_to_str(a) + card_to_str(b): weights
        for index in hand_range.weights
        for a, b in class_combos(index)
    }


def leaf(player: int, hand_range: Range) -> SolvedNode:
    """一个「过牌就摊牌」的叶子：过牌之后没有子节点 ＝ 摊牌。"""
    return SolvedNode(
        kind="action",
        player=player,
        actions=(SolvedAction("CHECK", "check", None),),
        strategy=always(hand_range),
        children={},
    )


def turn_tree(cards: dict, hero_range: Range) -> SolvedNode:
    """英雄过牌 → 发一张牌 → 对手过牌 → 摊牌。"""
    return SolvedNode(
        kind="action",
        player=0,
        actions=(SolvedAction("CHECK", "check", None),),
        strategy=always(hero_range),
        children={"CHECK": SolvedNode("chance", None, (), {}, cards, deal_number=len(cards))},
    )


def test_a_runout_is_averaged_over_the_cards_that_can_actually_come():
    """英雄 AA 对 KK：来 K 就输，别的就赢。发牌节点里那些**牌面上已有的牌**要跳过。"""
    board = tuple(cards_from_str("Qs7h2c9d"))
    spot = Spot(
        board=board,
        pot=8.0,
        effective_stack=10.0,
        oop_range=Range.parse("AA"),
        ip_range=Range.parse("KK"),
    )
    villain = Range.parse("KK")
    tree = turn_tree(
        {
            "Kc": leaf(1, villain),  # 对手中三条，英雄输
            "3d": leaf(1, villain),  # 英雄赢
            "Qs": leaf(1, villain),  # 已经在牌面上，不该算进去
        },
        Range.parse("AA"),
    )
    # 两张有效牌，一赢一输 → 一半底池
    assert hand_ev(spot, tree, 0, hand("AhAc")) == pytest.approx(4.0)


def test_placeholder_deals_are_skipped():
    """求解器会把牌面上已有的牌导成空占位符，遍历时按它的形状也要跳过。"""
    board = tuple(cards_from_str("Qs7h2c9d"))
    spot = Spot(
        board=board, pot=8.0, effective_stack=10.0,
        oop_range=Range.parse("AA"), ip_range=Range.parse("KK"),
    )
    empty = SolvedNode(kind="action", player=1, actions=(), strategy={}, children={})
    tree = turn_tree({"3d": leaf(1, Range.parse("KK")), "4d": empty}, Range.parse("AA"))
    assert hand_ev(spot, tree, 0, hand("AhAc")) == pytest.approx(8.0), "只剩 3d 一张有效牌"


def test_a_runout_with_nothing_dealable_is_an_error():
    board = tuple(cards_from_str("Qs7h2c9d"))
    spot = Spot(
        board=board, pot=8.0, effective_stack=10.0,
        oop_range=Range.parse("AA"), ip_range=Range.parse("KK"),
    )
    with pytest.raises(ValueError, match="dump 的层数不够"):
        hand_ev(spot, turn_tree({}, Range.parse("AA")), 0, hand("AhAc"))


# ------------------------------------------------------------------ 说不了的就报错


def test_a_truncated_dump_refuses_to_guess():
    """只导了一层的翻牌产物算不了跨街 EV——必须明确报错，不能拿半棵树糊弄。"""
    flop = parse_result(
        json.loads((Path(__file__).parent / "data" / "texassolver_flop.json").read_text(encoding="utf-8"))
    )
    spot = Spot(
        board=tuple(cards_from_str("Qs7h2c")),
        pot=6.0,
        effective_stack=12.0,
        oop_range=Range.parse("QQ, 77, AKo"),
        ip_range=Range.parse("AA, KK, JTs"),
    )
    with pytest.raises(ValueError, match="层数不够"):
        hand_ev(spot, flop, 0, hand("QhQd"))


def test_scoring_the_wrong_players_node_is_an_error(river, spot):
    with pytest.raises(ValueError, match="不是"):
        score_decision(spot, river, line=(), hero=1, hero_cards=hand("AhAd"))


def test_an_action_that_is_not_in_the_tree_is_an_error(river, spot):
    with pytest.raises(KeyError, match="没有"):
        score_decision(spot, river, line=("BET 4.000000",), hero=1, hero_cards=hand("AhAd"))


def test_hero_cards_must_not_clash_with_the_board(river, spot):
    with pytest.raises(ValueError, match="撞"):
        score_decision(spot, river, line=(), hero=0, hero_cards=hand("QsQd"))


def test_an_empty_villain_range_is_an_error(river):
    """对手范围被牌面挡光了就没法算期望——报错，别返回一个 0。"""
    # 牌面上有 Qs，英雄手里 QhQd —— 对手的 QQ 只剩一张 Qc，配不成对子
    blocked = Spot(
        board=BOARD,
        pot=POT,
        effective_stack=9.0,
        oop_range=Range.parse("QQ"),
        ip_range=Range.parse("QQ"),
    )
    with pytest.raises(ValueError, match="一手都不剩"):
        score_decision(blocked, river, line=(), hero=0, hero_cards=hand("QhQd"))


# ------------------------------------------------------------------ 真跑（慢）


@pytest.mark.slow
def test_cross_street_ev_on_a_real_solve():
    """**跨街那条路只有真产物验得了**：手搭的小树造不出「全下之后跑马」与
    「某一层展开、某几支没展开」并存的局面，而那正是最容易算错的地方。

    这里同时验三件事：跨街走得通、全下跑马不报错、我们的 EV 与求解器自己报的
    可利用度对得上（解的自证差应当明显小于它）。
    """
    from holdem_solver import BetSizes, SolveRequest, TexasSolver

    if not TexasSolver.available():
        pytest.skip("没装 TexasSolver（TEXAS_SOLVER_HOME）")

    request = SolveRequest(
        board=tuple(cards_from_str("Qs7h2c9d")),
        pot=6.0,
        effective_stack=30.0,
        oop_range=Range.parse("QQ, 77, AQo, KJs"),
        ip_range=Range.parse("AA, KK, AJs, T8s"),
        bet_sizes=BetSizes(flop=(50.0,), turn=(50.0,), river=(50.0,), reraise=(60.0,), allin=True),
        accuracy=1.0,
        max_iterations=30,
        dump_rounds=2,
        use_isomorphism=False,
    )
    solver = TexasSolver(threads=2)
    report = solver.solve(request)
    spot = Spot.from_request(request)

    # **标签是树里的原始键（求解器单位），`amount` 才是大盲**：命令文件用 1/10 大盲
    # （`SolveRequest.scale`），所以 3.0bb 的那一注在树里叫 `BET 30.000000`。
    # 生产路径不受影响——`review._match` 按「动作类型 + 大盲金额」找，再取它的 label。
    score = score_decision(
        spot, report.root, line=("BET 30.000000",), hero=1, hero_cards=hand("AhAc")
    )
    assert score.evs["FOLD"] == pytest.approx(0.0, abs=1e-9)
    assert score.best == "CALL", "AA 面对小注该跟——对手范围里一半是他打得过的"
    budget = report.exploitability * request.pot / 100.0
    assert 0.0 <= score.gap <= budget + 1e-9, (
        f"解的自证差 {score.gap:.4f}bb 超过了求解器自己报的可利用度 {budget:.4f}bb"
    )


# ------------------------------------------------------------------ 直接问求解器要 EV（慢）


@pytest.mark.slow
@pytest.mark.skipif(not TexasSolver.supports_evs(),
                    reason="求解器不认 dump_evs（要按 docs/solver-build 自己编，官方预编译包没有）")
def test_solver_evs_match_the_hand_computed_spot(tmp_path):
    """`dump_evs`（ADR-0006 的自建补丁）吐出来的 EV 必须对上纸笔。

    局面还是那个「结果完全确定」的河牌：OOP 全是暗三条、IP 全是一对，IP 一手都赢不了。
    底池 6、有效筹码 9，双方各已投 3。于是每个数都能手算：

    | 谁 · 在哪 | 求解器口径 | 加回自己已投的 3 | 手算 |
    |---|---|---|---|
    | OOP 在根（必赢底池） | +3 | 6 | 赢下整个底池 |
    | IP 在根（必输、还没再投） | −3 | 0 | 一分不亏地弃掉 |
    | IP 面对全下 · FOLD | −3 | 0 | 同上 |
    | IP 面对全下 · CALL | −12 | −9 | 白扔九个大盲 |

    **口径差就是「自己已投进底池的那一份」**，不是可调的系数。不翻译直接用，
    每个数都会差这一截，而且看上去完全合理——这正是 ADR-0006 立的规矩：
    先拿硬基准对账，对不上别调系数糊过去。
    """
    solver = TexasSolver(cache_dir=tmp_path / "cache", threads=2)
    request = SolveRequest(
        board=BOARD,
        pot=POT,
        effective_stack=9.0,
        oop_range=Range.parse("QQ, 77, 22"),
        ip_range=Range.parse("AA, KK, AKo"),
        accuracy=0.1,
        max_iterations=200,
    )
    own_commit = POT / 2          # 双方各投了半个底池

    at_root_oop = solver.solve_evs(request, (), player=0)
    for hand, evs in at_root_oop.items():
        for label, ev in evs.items():
            assert ev + own_commit == pytest.approx(POT, abs=1e-6), (hand, label)

    at_root_ip = solver.solve_evs(request, (), player=1)
    for hand, evs in at_root_ip.items():
        for label, ev in evs.items():
            assert ev + own_commit == pytest.approx(0.0, abs=1e-6), (hand, label)

    # 找到 OOP 的全下标签（金额由求解器取整决定，别写死）
    bet = next(l for l in next(iter(at_root_oop.values())) if l.startswith("BET"))
    facing = solver.solve_evs(request, (bet,), player=1)
    for hand, evs in facing.items():
        assert evs["FOLD"] + own_commit == pytest.approx(0.0, abs=1e-6), hand
        assert evs["CALL"] + own_commit == pytest.approx(-9.0, abs=1e-6), hand

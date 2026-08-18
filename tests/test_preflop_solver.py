"""翻前树 CFR+ 求解器的测试。

**最重要的一条是退化交叉验证**：把树退化成「全下或弃牌」，解必须与 `pushfold` 那套独立
实现对上。它一次性验证了树、终局收益、共牌处理与 CFR+ 实现——四处只要有一处错，
总频率和 EV 就对不上。其余检查是可利用度自证与结构性质，同样不依赖外部数据。
"""

import pytest

from holdem import equity_table
from holdem.preflop_solver import SqueezeRisk, solve_preflop
from holdem.preflop_tree import PreflopConfig, build_tree
from holdem.pushfold import solve_push_fold
from holdem.ranges import NUM_HAND_CLASSES, class_from_name
from holdem.realization import RealizationModel

pytestmark = pytest.mark.skipif(
    not equity_table.is_available(),
    reason="翻前权益表尚未生成，先跑 scripts/build_preflop_equity.py",
)

PUSH_FOLD = PreflopConfig(
    effective_stack=10.0,
    open_to=10.0,
    reraise_multiples=(),
    allow_limp=False,
    jam_from_level=99,
)
"""退化成推/弃的配置：开牌尺度就是全部筹码，不许跛入。"""

TRIMMED = PreflopConfig(allow_limp=False, reraise_multiples=(3.0,))
"""裁剪过的 100bb 单挑树：不许跛入、只到 3bet，5 个决策节点，够验性质又跑得快。"""


@pytest.fixture(scope="module")
def push_fold_solution():
    return solve_preflop(PUSH_FOLD, iterations=600, tolerance=1e-3)


@pytest.fixture(scope="module")
def reference():
    return solve_push_fold(10.0)


@pytest.fixture(scope="module")
def trimmed():
    return solve_preflop(TRIMMED, iterations=200, tolerance=1e-3, check_every=50)


# ------------------------------------------------------------------ 退化交叉验证


def test_degenerate_tree_matches_the_push_fold_solver(push_fold_solution, reference):
    """总频率与 EV 必须对上另一套独立实现。"""
    push = push_fold_solution.action_range(push_fold_solution.tree.root, 1)
    call = push_fold_solution.action_range(push_fold_solution.tree.root.children[1], 1)

    assert push.percent() == pytest.approx(reference.push_percent, abs=0.015), (
        f"全下范围 {push.percent():.1%}，推弃求解器给的是 {reference.push_percent:.1%}"
    )
    assert call.percent() == pytest.approx(reference.call_percent, abs=0.015), (
        f"跟注范围 {call.percent():.1%}，推弃求解器给的是 {reference.call_percent:.1%}"
    )
    assert push_fold_solution.player_ev[0] == pytest.approx(
        reference.small_blind_ev, abs=2e-3
    ), "小盲每手期望对不上"


def test_degenerate_tree_agrees_hand_by_hand(push_fold_solution, reference):
    """逐手方向一致：对方几乎必推的牌，我们这边也得站在推的一侧。

    不比逐手频率的绝对值——均衡里无差异的牌可以任意混合，两套实现落在不同角上很正常。
    """
    push = push_fold_solution.action_range(push_fold_solution.tree.root, 1)
    for index in range(NUM_HAND_CLASSES):
        expected = reference.push.weight(index)
        if expected > 0.98:
            assert push.weight(index) > 0.5, f"牌类 {index} 该全下却没有"
        elif expected < 0.02:
            assert push.weight(index) < 0.5, f"牌类 {index} 不该全下"


def test_degenerate_tree_is_an_equilibrium(push_fold_solution):
    assert push_fold_solution.exploitability < 2e-3


# ------------------------------------------------------------------ 自证


def test_trimmed_tree_converges(trimmed):
    assert trimmed.exploitability < 0.01, (
        f"可利用度 {trimmed.exploitability:.4f} bb/手，没收敛到均衡附近"
    )


def test_expected_values_are_zero_sum(trimmed):
    assert sum(trimmed.player_ev) == pytest.approx(0.0, abs=1e-9)


def test_button_has_the_edge_heads_up(trimmed):
    """单挑里按钮（＝小盲）有位置优势，均衡下应当是赢家。"""
    assert trimmed.player_ev[0] > 0


def test_strategies_are_probability_distributions(trimmed):
    for node in trimmed.tree.decisions:
        strategy = trimmed.strategy_at(node)
        assert len(strategy) == NUM_HAND_CLASSES
        for row in strategy:
            assert len(row) == len(node.actions)
            assert sum(row) == pytest.approx(1.0, abs=1e-9)
            assert all(0.0 <= value <= 1.0 for value in row)


# ------------------------------------------------------------------ 到达概率


def test_root_range_is_everything(trimmed):
    assert trimmed.arriving_range(trimmed.tree.root).percent() == pytest.approx(1.0)


def test_arriving_range_narrows_with_depth(trimmed):
    """开牌之后再面对 3bet 的，只能是开过牌的那部分。"""
    root = trimmed.tree.root
    opened = root.children[1]
    facing_three_bet = opened.children[2]
    open_range = trimmed.action_range(root, 1)
    assert trimmed.arriving_range(facing_three_bet).percent() == pytest.approx(
        open_range.percent(), abs=1e-3
    )


def test_action_range_excludes_hands_that_never_got_there(trimmed):
    """72o 从不开牌，就不该出现在「面对 3bet」的任何一个动作范围里。

    这是第一版的真实缺陷：只看策略表会把没走到这里的牌一起统计进去，
    于是解出来像是「72o 有四成跟 3bet」。
    """
    trash = class_from_name("72o")
    facing_three_bet = trimmed.tree.root.children[1].children[2]
    for action in range(len(facing_three_bet.actions)):
        assert trimmed.action_range(facing_three_bet, action).weight(trash) < 0.01
    assert trimmed.reaches[facing_three_bet.node_id][trash] < 0.01


def test_action_frequencies_sum_to_one_over_the_arriving_range(trimmed):
    for node in trimmed.tree.decisions:
        total = sum(
            trimmed.action_frequency(node, action) for action in range(len(node.actions))
        )
        assert total == pytest.approx(1.0, abs=1e-6), f"节点 {node.node_id} 的频率不闭合"


# ------------------------------------------------------------------ 结构性质


def test_premium_hands_never_fold(trimmed):
    root = trimmed.tree.root
    for name in ["AA", "KK", "AKs"]:
        index = class_from_name(name)
        assert trimmed.strategy_at(root)[index][0] < 0.01, f"{name} 不该弃牌"


def test_trash_folds_from_the_small_blind(trimmed):
    root = trimmed.tree.root
    for name in ["72o", "32o", "82o"]:
        index = class_from_name(name)
        assert trimmed.strategy_at(root)[index][0] > 0.5, f"{name} 该弃牌"


def test_three_bet_range_is_tighter_than_the_open(trimmed):
    root = trimmed.tree.root
    opened = root.children[1]
    open_percent = trimmed.action_range(root, 1).percent()
    three_bet_percent = trimmed.action_range(opened, 2).percent()
    assert three_bet_percent < open_percent


# ------------------------------------------------------------------ 入参


def test_multiway_is_not_supported_yet():
    with pytest.raises(NotImplementedError, match="单挑"):
        solve_preflop(PreflopConfig(num_players=6))


def test_iterations_must_be_positive():
    with pytest.raises(ValueError, match="迭代次数"):
        solve_preflop(PUSH_FOLD, iterations=0)


# ------------------------------------------------------------------ 慢测


@pytest.mark.slow
def test_full_tree_converges():
    """完整的 100bb 单挑树（含跛入、开到 4bet）——比裁剪树慢一个量级，单独跑。"""
    solution = solve_preflop(iterations=1500, tolerance=1e-3, check_every=100)
    assert solution.exploitability < 4e-3, (
        f"可利用度 {solution.exploitability:.5f} bb/手；收敛大致是 O(1/t)，"
        f"1500 次迭代实测约 0.0026"
    )
    assert sum(solution.player_ev) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.slow
def test_sharpening_makes_the_big_blind_defend_tighter():
    """终局模型确实在驱动解：γ 越大，弱牌越难兑现权益，大盲防守就该收紧。

    这条把「模型参数 → 解的形状」的因果链锁住了；没有它，模型层可能整个是死代码。
    """
    flat = solve_preflop(
        TRIMMED, model=RealizationModel(sharpening=0.0), iterations=200, check_every=200
    )
    sharp = solve_preflop(
        TRIMMED, model=RealizationModel(sharpening=0.8), iterations=200, check_every=200
    )
    flat_fold = flat.action_frequency(flat.tree.root.children[1], 0)
    sharp_fold = sharp.action_frequency(sharp.tree.root.children[1], 0)
    assert sharp_fold > flat_fold + 0.02, (
        f"大盲弃牌率 {flat_fold:.1%} → {sharp_fold:.1%}，锐化系数没起到作用"
    )


# ------------------------------------------------------------------ 第三方接管终局


def _main_line_terminal():
    """「开牌 → 跟注」那个进翻牌的终局。六人桌的挤压就挂在这种终局上。"""
    root = build_tree(TRIMMED).root
    open_index = next(i for i, a in enumerate(root.actions) if a.is_raise)
    reply = root.children[open_index]
    call_index = next(i for i, a in enumerate(reply.actions) if a.kind == "call")
    terminal = reply.children[call_index]
    assert terminal.is_terminal
    return reply, call_index, terminal.node_id


def test_zero_probability_changes_nothing(trimmed):
    """概率为 0 时必须逐位复现原来的解——混合公式的退化行为。"""
    _, _, terminal = _main_line_terminal()
    risk = SqueezeRisk(0.0, ((0.0,) * NUM_HAND_CLASSES, (-99.0,) * NUM_HAND_CLASSES))
    same = solve_preflop(
        TRIMMED,
        squeeze={terminal: risk},
        iterations=200,
        tolerance=1e-3,
        check_every=50,
    )
    assert same.player_ev == pytest.approx(trimmed.player_ev)


def test_a_punishing_third_party_scares_the_caller_away(trimmed):
    """跟注之后有人来收钱，跟注就该变少——这是挤压建模的作用机制。"""
    reply, call_index, terminal = _main_line_terminal()
    risk = SqueezeRisk(
        0.5, ((0.0,) * NUM_HAND_CLASSES, (-8.0,) * NUM_HAND_CLASSES)
    )
    scared = solve_preflop(
        TRIMMED,
        squeeze={terminal: risk},
        iterations=200,
        tolerance=1e-3,
        check_every=50,
    )
    before = trimmed.action_frequency(reply, call_index)
    after = scared.action_frequency(reply, call_index)
    assert after < before - 0.05, f"跟注 {before:.1%} → {after:.1%}，没被吓退"


def test_third_party_cannot_be_hung_on_a_decision_node():
    """挂错节点要当场报错——挂到决策点上会被无声忽略，那种 bug 最难查。"""
    root = build_tree(TRIMMED).root
    risk = SqueezeRisk(0.5, ((0.0,) * NUM_HAND_CLASSES,) * 2)
    with pytest.raises(ValueError, match="不是终局"):
        solve_preflop(TRIMMED, squeeze={root.node_id: risk}, iterations=1)


def test_third_party_arguments_are_checked():
    with pytest.raises(ValueError, match="接管概率"):
        SqueezeRisk(1.5, ((0.0,) * NUM_HAND_CLASSES,) * 2)
    with pytest.raises(ValueError, match="169"):
        SqueezeRisk(0.5, ((0.0, 0.0), (0.0, 0.0)))

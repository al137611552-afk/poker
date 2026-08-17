"""翻前抽象树的测试。

这棵树把 `state.py` 的下注轮逻辑重写了一遍（原因见模块 docstring），所以这里守的第一件事
就是**筹码守恒**与**盲注/说话顺序与真引擎同口径**——重写的部分一旦跑偏，后面求出来的
范围会错得很隐蔽。
"""

import pytest

from holdem.preflop_tree import PreflopConfig, TerminalNode, build_tree

# ------------------------------------------------------------------ 遍历工具


def walk(node):
    """深度优先枚举全部节点。"""
    yield node
    if not node.is_terminal:
        for child in node.children:
            yield from walk(child)


def paths(node, prefix=()):
    """枚举 (动作标签路径, 终局) 对，测试里用来定位具体线路。"""
    if node.is_terminal:
        yield prefix, node
        return
    for action, child in zip(node.actions, node.children):
        yield from paths(child, prefix + (f"{node.player}:{action.label}",))


def terminal_at(tree, *labels):
    wanted = tuple(labels)
    for path, node in paths(tree.root):
        if path == wanted:
            return node
    raise AssertionError(f"树里没有这条线路: {wanted}")


@pytest.fixture(scope="module")
def heads_up():
    return build_tree()


@pytest.fixture(scope="module")
def six_max():
    # 完整六人树有六十万个节点，测试用裁剪过的配置——测的是规则，不是规模
    return build_tree(
        PreflopConfig(num_players=6, allow_limp=False, reraise_multiples=(3.0,), jam_from_level=3)
    )


# ------------------------------------------------------------------ 守恒


def test_every_terminal_conserves_chips(heads_up):
    """底池等于各人投入之和，投入不超过有效筹码。"""
    stack = heads_up.config.effective_stack
    for node in heads_up.terminals:
        assert node.pot == pytest.approx(sum(node.contributions))
        for contribution in node.contributions:
            assert 0 <= contribution <= stack + 1e-9


def test_fold_payoffs_sum_to_zero(heads_up):
    for node in heads_up.terminals:
        if node.kind == "fold":
            assert sum(node.fold_payoffs()) == pytest.approx(0.0)


def test_fold_payoffs_only_defined_for_fold_terminals(heads_up):
    showdown = next(n for n in heads_up.terminals if n.kind != "fold")
    with pytest.raises(ValueError):
        showdown.fold_payoffs()


def test_ante_is_dead_money_not_a_bet():
    """前注进底池、计入净得失，但不参与「跟到多少」。"""
    tree = build_tree(PreflopConfig(ante=0.125))
    node = terminal_at(tree, "0:弃牌")
    assert node.contributions == pytest.approx((0.625, 1.125))
    assert node.fold_payoffs() == pytest.approx((-0.625, 0.625))


# ------------------------------------------------------------------ 与真引擎同口径


def test_heads_up_button_posts_small_blind_and_acts_first(heads_up):
    """单挑特例：按钮就是小盲，翻前先说话——与 state.py 一致。"""
    assert heads_up.root.player == 0
    node = terminal_at(heads_up, "0:弃牌")
    assert node.contributions == pytest.approx((0.5, 1.0))
    assert node.alive == (1,)


def test_six_max_blinds_and_first_actor(six_max):
    assert six_max.root.player == 3, "六人桌枪口位（按钮偏移 3）先说话"
    everyone_folds = tuple(f"{p}:弃牌" for p in (3, 4, 5, 0, 1))
    node = terminal_at(six_max, *everyone_folds)
    assert node.contributions == pytest.approx((0.0, 0.5, 1.0, 0.0, 0.0, 0.0))
    assert node.alive == (2,), "无人跟注时大盲收走盲注"


def test_button_is_last_to_act_postflop(heads_up):
    flop = terminal_at(heads_up, "0:加注到2.5", "1:跟注到2.5")
    assert flop.kind == "flop"
    assert flop.in_position == 0, "单挑翻后是大盲先说、按钮最后"


def test_in_position_is_the_latest_alive_seat(six_max):
    flop = terminal_at(six_max, "3:弃牌", "4:弃牌", "5:加注到2.5", "0:跟注到2.5", "1:弃牌", "2:弃牌")
    assert flop.alive == (0, 5)
    assert flop.in_position == 0, "按钮翻后最后说话，位置优于任何前位"


# ------------------------------------------------------------------ 动作集


def test_no_fold_action_when_checking_is_free(heads_up):
    for node in heads_up.decisions:
        kinds = [action.kind for action in node.actions]
        checks = [a for a in node.actions if a.kind == "call" and a.to_amount == 0]
        if checks:
            assert "fold" not in kinds


def test_big_blind_gets_the_option_after_a_limp(heads_up):
    limp = heads_up.root.children[heads_up.root.actions.index(
        next(a for a in heads_up.root.actions if a.label == "跟注到1")
    )]
    assert not limp.is_terminal, "跛入之后大盲还有选择权，不能直接进翻牌"
    assert limp.player == 1
    assert [a.label for a in limp.actions] == ["过牌", "加注到3.5"]


def test_raise_ladder_amounts(heads_up):
    """开牌 2.5 → 3bet ×3 → 4bet ×2.2 → 梯子用尽只剩全下。"""
    root = heads_up.root
    assert [a.label for a in root.actions] == ["弃牌", "跟注到1", "加注到2.5"]

    open_node = root.children[2]
    assert [a.label for a in open_node.actions] == ["弃牌", "跟注到2.5", "加注到7.5", "全下"]

    three_bet = open_node.children[2]
    assert [a.label for a in three_bet.actions] == ["弃牌", "跟注到7.5", "加注到16.5", "全下"]

    four_bet = three_bet.children[2]
    assert [a.label for a in four_bet.actions] == ["弃牌", "跟注到16.5", "全下"]


def test_limpers_push_the_open_size_up(six_max):
    """每多一个跛入者，开牌尺度抬高 1bb。"""
    tree = build_tree(PreflopConfig(num_players=6, reraise_multiples=(3.0,)))
    after_limp = tree.root.children[tree.root.actions.index(
        next(a for a in tree.root.actions if a.label == "跟注到1")
    )]
    assert any(a.label == "加注到3.5" for a in after_limp.actions)


def test_limp_can_be_switched_off(six_max):
    assert all(
        not (action.kind == "call" and node.raise_level == 0 and action.to_amount > 0)
        for node in six_max.decisions
        for action in node.actions
    ), "关掉跛入之后，无人加注的局面不该有「只跟一个大盲」"


def test_no_raise_when_nobody_can_call(heads_up):
    """对手已经全下，再加注没有意义，树里不放这种动作。"""
    all_in_line = terminal_at(heads_up, "0:加注到2.5", "1:全下", "0:跟注到100")
    assert all_in_line.kind == "showdown"
    facing_jam = heads_up.root.children[2].children[3]
    assert [a.kind for a in facing_jam.actions] == ["fold", "call"]


def test_raise_amounts_are_strictly_increasing_and_capped(heads_up):
    stack = heads_up.config.effective_stack
    for node in heads_up.decisions:
        amounts = [a.to_amount for a in node.actions if a.is_raise]
        assert amounts == sorted(set(amounts)), f"节点 {node.node_id} 的加注额有重复或乱序"
        assert all(amount <= stack for amount in amounts)
        assert node.actions, "决策节点不能没有动作"


# ------------------------------------------------------------------ 终局分类


def test_terminal_kinds(heads_up):
    assert terminal_at(heads_up, "0:弃牌").kind == "fold"
    assert terminal_at(heads_up, "0:加注到2.5", "1:跟注到2.5").kind == "flop"
    assert terminal_at(heads_up, "0:加注到2.5", "1:全下", "0:跟注到100").kind == "showdown"


def test_showdown_only_when_everyone_alive_is_all_in(heads_up):
    stack = heads_up.config.effective_stack
    for node in heads_up.terminals:
        if node.kind == "showdown":
            assert all(node.contributions[p] >= stack for p in node.alive)
        if node.kind == "flop":
            assert len(node.alive) >= 2
            assert any(node.contributions[p] < stack for p in node.alive)


def test_all_nodes_are_registered(heads_up):
    counted = list(walk(heads_up.root))
    assert len(counted) == heads_up.size
    ids = sorted(node.node_id for node in counted)
    assert ids == list(range(len(counted))), "节点编号必须连续且唯一"
    assert sum(isinstance(node, TerminalNode) for node in counted) == len(heads_up.terminals)


# ------------------------------------------------------------------ 退化成推/弃


def test_degenerates_into_push_fold():
    """把开牌尺度设成有效筹码、关掉跛入，就得到经典的推/弃博弈。

    这条退化关系是后面拿 `pushfold.solve_push_fold` 交叉验证求解器的前提。
    """
    tree = build_tree(
        PreflopConfig(
            effective_stack=10.0,
            open_to=10.0,
            reraise_multiples=(),
            allow_limp=False,
            jam_from_level=99,
        )
    )
    assert len(tree.decisions) == 2 and len(tree.terminals) == 3
    assert [a.label for a in tree.root.actions] == ["弃牌", "全下"]
    facing = tree.root.children[1]
    assert [a.label for a in facing.actions] == ["弃牌", "跟注到10"]
    assert terminal_at(tree, "0:全下", "1:跟注到10").kind == "showdown"
    assert terminal_at(tree, "0:全下", "1:弃牌").fold_payoffs() == pytest.approx((1.0, -1.0))


# ------------------------------------------------------------------ 入参


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_players": 1}, "两个玩家"),
        ({"effective_stack": 0}, "有效筹码"),
        ({"big_blind": 0}, "盲注"),
        ({"ante": -1}, "前注"),
        ({"effective_stack": 0.5}, "不足一个大盲"),
    ],
)
def test_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PreflopConfig(**kwargs)

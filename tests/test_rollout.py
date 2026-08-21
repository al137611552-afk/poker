"""把上一街的解滚成下一街范围的测试（方案 B / ADR-0005 的接缝）。

这一层最容易出的错不是算错乘法，而是**把损失藏起来**：求解器的范围只认 169 个牌类，
滚回去必然要按牌类聚合，同花听牌会被抹平。所以这里除了验权重算得对，还专门守两件事：

1. 一个牌类里被单独摘出去的那支（典型就是同花听牌）**必须被点名**——哪怕它已经被打成 0，
   统计里也要留着它的位置，否则「类内一致」会是个假象。
2. 滚出来的范围**只能说牌类**：喂具体组合会让求解器直接 abort（`range str AhKs len not
   valid`），这是实测过的坑，得有回归守着。
"""

import json
from pathlib import Path

import pytest

from holdem.cards import cards_from_str
from holdem.ranges import Range, class_from_name, class_name
from holdem.state import FLOP
from holdem_solver.request import SolveRequest, format_range
from holdem_solver.result import SolvedAction, SolvedNode, parse_result
from holdem_solver.review import LineNotInTree, Step
from holdem_solver.rollout import roll_forward

CHECK = SolvedAction("CHECK", "check", None)
CALL = SolvedAction("CALL", "call", None)
FOLD = SolvedAction("FOLD", "fold", None)
BET3 = SolvedAction("BET 3.000000", "bet", 3.0)


def leaf(player=0):
    return SolvedNode(kind="action", player=player, actions=(), strategy={}, children={})


def node(player, actions, strategy, children):
    return SolvedNode(
        kind="action", player=player, actions=actions, strategy=strategy, children=children
    )


def chance(cards):
    return SolvedNode(
        kind="chance", player=None, actions=(), strategy={}, children=cards,
        deal_number=len(cards),
    )


def request(oop, ip, *, board="Qs7h2h", pot=6.0, stack=9.0):
    return SolveRequest(
        board=tuple(cards_from_str(board)),
        oop_range=Range.parse(oop),
        ip_range=Range.parse(ip),
        pot=pot,
        effective_stack=stack,
    )


def check(seat=0):
    return Step(kind="check", seat=seat, street=FLOP)


# ---------------------------------------------------------------- 权重怎么滚


def test_the_weight_is_the_product_of_the_frequencies_along_the_line():
    """滚出来的权重＝输入权重 × 沿途每一步走这条线的频率，逐组合各算各的。"""
    strategy = {"AhKh": (0.25, 0.75), "AsKs": (0.5, 0.5), "AcKc": (1.0, 0.0), "AdKd": (1.0, 0.0)}
    tree = node(0, (CHECK, BET3), strategy, {"CHECK": leaf(1)})

    rolled = roll_forward(tree, (check(),), request=request("AKs", "QQ"))

    assert rolled.oop.combos == pytest.approx(
        {"AhKh": 0.25, "AsKs": 0.5, "AcKc": 1.0, "AdKd": 1.0}
    )
    assert rolled.ip.combos["QcQd"] == 1.0, "没轮到 IP 说话，它的范围不该动"


def test_only_the_player_to_act_has_their_range_rolled():
    tree = node(1, (CHECK, BET3), {"QcQd": (0.5, 0.5)}, {"CHECK": leaf(0)})
    rolled = roll_forward(tree, (check(seat=3),), request=request("AKs", "QQ"))

    assert rolled.ip.combos["QcQd"] == pytest.approx(0.5)
    assert rolled.oop.combos["AhKh"] == 1.0


def test_a_combo_the_solver_never_saw_drops_out():
    """解里查不到的组合是「压根走不到这儿」，不是「保持原样」。"""
    tree = node(0, (CHECK,), {"AhKh": (1.0,)}, {"CHECK": leaf(1)})
    rolled = roll_forward(tree, (check(),), request=request("AKs", "QQ"))

    assert set(rolled.oop.combos) == {"AhKh"}


# ------------------------------------------------------- 聚合损失要看得见


def test_a_class_split_by_suit_is_flagged():
    """`Qs7h2h` 上 `AhKh` 是同花听牌：解让它下注、别的三支过牌。

    聚合之后四支拿同一个权重——这正是方案 B 传不过去的东西，必须被点名。
    """
    strategy = {
        "AhKh": (0.0, 1.0),   # 同花听牌：全下注
        "AsKs": (1.0, 0.0),
        "AcKc": (1.0, 0.0),
        "AdKd": (1.0, 0.0),
    }
    tree = node(0, (CHECK, BET3), strategy, {"CHECK": leaf(1)})

    rolled = roll_forward(tree, (check(),), request=request("AKs", "QQ"))
    aks = class_from_name("AKs")

    assert rolled.oop.combos["AhKh"] == 0.0, "被打成 0 的组合要留在统计里"
    assert rolled.oop.spread[aks] == pytest.approx(1.0)
    assert aks in rolled.oop.flagged()
    assert [class_name(i) for i in rolled.oop.flagged()] == ["AKs"]


def test_a_class_the_solver_treats_alike_is_not_flagged():
    strategy = {combo: (0.6, 0.4) for combo in ("AhKh", "AsKs", "AcKc", "AdKd")}
    tree = node(0, (CHECK, BET3), strategy, {"CHECK": leaf(1)})

    rolled = roll_forward(tree, (check(),), request=request("AKs", "QQ"))

    assert rolled.oop.spread[class_from_name("AKs")] == pytest.approx(0.0)
    assert rolled.oop.flagged() == ()


def test_a_class_that_loses_one_combo_gets_lighter():
    """四支里死掉一支，这个牌类就该轻四分之一——均值要把 0 算进去。"""
    strategy = {
        "AhKh": (0.0, 1.0),
        "AsKs": (1.0, 0.0),
        "AcKc": (1.0, 0.0),
        "AdKd": (1.0, 0.0),
    }
    tree = node(
        0, (CHECK, BET3), {**strategy, "QcQd": (1.0, 0.0), "QcQh": (1.0, 0.0), "QdQh": (1.0, 0.0)},
        {"CHECK": leaf(1)},
    )

    rolled = roll_forward(tree, (check(),), request=request("AKs,QQ", "JJ"))
    weights = rolled.oop.hand_range.weights

    assert weights[class_from_name("QQ")] == pytest.approx(1.0)
    assert weights[class_from_name("AKs")] == pytest.approx(0.75)


# ------------------------------------------------------------ 街的边界


def test_the_dealt_card_leaves_both_ranges():
    """转牌发出来那张牌，谁手里有就删谁——牌只有一张。"""
    tree = node(
        0, (CHECK,), {"AhKh": (1.0,), "AsKs": (1.0,), "AcKc": (1.0,), "AdKd": (1.0,)},
        {"CHECK": chance({"Kh": leaf(0), "9d": leaf(0)})},
    )
    steps = (check(), Step(kind="deal", card=cards_from_str("Kh")[0], street=FLOP))

    rolled = roll_forward(tree, steps, request=request("AKs", "AKs"))

    assert "AhKh" not in rolled.oop.combos
    assert "AhKh" not in rolled.ip.combos, "对手手里也不可能有这张牌"
    assert len(rolled.board) == 4


def test_the_pot_and_stacks_advance_at_the_street_boundary():
    """本街投进去的钱进底池、从筹码里扣——下一街的树要用推进后的数。"""
    turn = chance({"9d": leaf(0)})
    ip_node = node(1, (FOLD, CALL), {"QcQd": (0.0, 1.0)}, {"CALL": turn})
    tree = node(0, (CHECK, BET3), {"AhKh": (0.0, 1.0)}, {"BET 3.000000": ip_node})
    steps = (
        Step(kind="bet", seat=0, amount=3.0, street=FLOP),
        Step(kind="call", seat=3, street=FLOP),
        Step(kind="deal", card=cards_from_str("9d")[0], street=FLOP),
    )

    rolled = roll_forward(tree, steps, request=request("AKs", "QQ", pot=6.0, stack=9.0))

    assert rolled.pot == pytest.approx(12.0), "6 + 3 + 3"
    assert rolled.effective_stack == pytest.approx(6.0), "9 − 3"


def test_a_line_with_a_fold_has_no_next_street():
    ip_node = node(1, (FOLD, CALL), {"QcQd": (1.0, 0.0)}, {"FOLD": leaf(0)})
    tree = node(0, (CHECK, BET3), {"AhKh": (0.0, 1.0)}, {"BET 3.000000": ip_node})
    steps = (
        Step(kind="bet", seat=0, amount=3.0, street=FLOP),
        Step(kind="fold", seat=3, street=FLOP),
    )

    with pytest.raises(LineNotInTree, match="弃牌"):
        roll_forward(tree, steps, request=request("AKs", "QQ"))


def test_a_line_that_runs_off_the_dump_says_so():
    """dump 层数不够时要说清楚，别假装滚到了。"""
    tree = node(0, (CHECK,), {"AhKh": (1.0,)}, {})

    with pytest.raises(LineNotInTree, match="dump"):
        roll_forward(tree, (check(),), request=request("AKs", "QQ"))


# ------------------------------------------------- 喂得回求解器才算滚成功


def test_the_rolled_range_only_ever_speaks_classes():
    """回归：范围里出现具体组合会让求解器 abort（`range str AhKs len not valid`）。"""
    strategy = {"AhKh": (0.0, 1.0), "AsKs": (1.0, 0.0), "AcKc": (1.0, 0.0), "AdKd": (0.5, 0.5)}
    tree = node(0, (CHECK, BET3), strategy, {"CHECK": leaf(1)})

    rolled = roll_forward(tree, (check(),), request=request("AKs", "QQ"))
    rendered = format_range(rolled.oop.hand_range)

    for token in rendered.split(","):
        name = token.split(":")[0]
        assert 2 <= len(name) <= 3, f"{name} 不是牌类写法，求解器会直接崩"


def test_to_request_carries_the_spot_to_the_next_street():
    tree = node(
        0, (CHECK,), {"AhKh": (1.0,), "AsKs": (1.0,), "AcKc": (1.0,), "AdKd": (1.0,)},
        {"CHECK": chance({"9d": leaf(0)})},
    )
    steps = (check(), Step(kind="deal", card=cards_from_str("9d")[0], street=FLOP))
    flop = request("AKs", "QQ")

    rolled = roll_forward(tree, steps, request=flop)
    turn = rolled.to_request(flop)

    assert turn.street == "turn"
    assert len(turn.board) == 4
    assert turn.pot == pytest.approx(flop.pot)
    assert turn.bet_sizes == flop.bet_sizes, "跑法参数默认沿用上一街"
    assert turn.oop_range.weights == rolled.oop.hand_range.weights


def test_dump_rounds_can_be_overridden_per_street():
    tree = node(0, (CHECK,), {"AhKh": (1.0,)}, {"CHECK": chance({"9d": leaf(0)})})
    steps = (check(), Step(kind="deal", card=cards_from_str("9d")[0], street=FLOP))
    flop = request("AKs", "QQ")

    rolled = roll_forward(tree, steps, request=flop)

    assert rolled.to_request(flop, dump_rounds=2).dump_rounds == 2


# ------------------------------------------------------------ 真样本


def test_the_real_sample_rolls_forward():
    """拿真求解器跑出来的样本走一遍：滚完只会更轻，不会凭空变重。"""
    document = json.loads(
        (Path(__file__).parent / "data" / "texassolver_flop.json").read_text(encoding="utf-8")
    )
    root = parse_result(document)
    flop = request("QQ,77,22", "AA,KK,AKo", board="Qs7h2c", pot=6.0, stack=9.0)

    rolled = roll_forward(root, (check(seat=2),), request=flop)

    assert rolled.oop.combos, "行动方的范围不该整个消失"
    assert all(0.0 <= w <= 1.0 for w in rolled.oop.combos.values())
    assert rolled.ip.combos == {
        key: 1.0 for key in rolled.ip.combos
    }, "没轮到 IP，权重原样"
    assert format_range(rolled.oop.hand_range), "滚出来的范围要能渲染成命令"

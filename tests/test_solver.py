"""TexasSolver 适配层的测试。

**大头是离线的**：命令渲染与策略解析都是纯逻辑，用一份**真求解器跑出来的样本**
（`tests/data/texassolver_flop.json`）当基准——手写的假样本挡不住格式漂移，而格式漂移
在这里的代价是求解器直接 abort 或者我们把两边认反。

真跑一次的测试标了 `slow`，而且没装求解器就跳过（`TEXAS_SOLVER_HOME`）。
"""

import json
from pathlib import Path

import pytest

from holdem.cards import card_from_str, cards_from_str
from holdem.ranges import Range, class_from_name
from holdem_solver.backend import (
    CACHE_FORMAT,
    SolveReport,
    SolverNotInstalled,
    TexasSolver,
    _explain,
    find_solver_home,
)
from holdem_solver.request import BetSizes, SolveRequest, format_board, format_range
from holdem_solver.result import parse_action, parse_result

FIXTURE = Path(__file__).parent / "data" / "texassolver_flop.json"


def a_request(**overrides) -> SolveRequest:
    base = dict(
        board=tuple(cards_from_str("QsJh2h")),
        pot=6.0,
        effective_stack=20.0,
        oop_range=Range.parse("TT+, AQs+"),
        ip_range=Range.parse("99+, AJs+"),
        accuracy=1.0,
        max_iterations=10,
    )
    base.update(overrides)
    return SolveRequest(**base)


# ------------------------------------------------------------------ 范围与牌面


def test_ranges_are_written_out_class_by_class():
    text = format_range(Range.parse("QQ+, AKs"))
    assert text == "AA,AKs,KK,QQ"


def test_ranges_never_use_plus_notation():
    """**这条是防 abort 的**：求解器不认 `99+`，它会抛异常然后直接 SIGABRT。

    我们的记法里 `+` 到处都是，一不留神就原样送进去了——所以展开是必须的，
    而且要在这里钉死。
    """
    text = format_range(Range.parse("99+"))
    assert "+" not in text
    assert text.split(",") == ["AA", "KK", "QQ", "JJ", "TT", "99"]


def test_partial_weights_travel_with_the_range():
    text = format_range(Range({class_from_name("JTs"): 0.5, class_from_name("AA"): 1.0}))
    assert text == "AA,JTs:0.5"


def test_an_empty_range_cannot_be_solved():
    with pytest.raises(ValueError, match="空的"):
        format_range(Range.empty())


def test_board_is_comma_separated():
    assert format_board(tuple(cards_from_str("QsJh2h"))) == "Qs,Jh,2h"


# ------------------------------------------------------------------ 命令渲染


def test_commands_are_in_the_order_the_solver_expects():
    lines = a_request().commands("/tmp/out.json").splitlines()
    order = [lines.index(x) for x in ("build_tree", "start_solve")]
    assert order == sorted(order), "先建树再求解"
    assert lines[-1] == "dump_result /tmp/out.json", "导出必须是最后一步"
    assert lines[0].startswith("set_pot")


def test_fractional_amounts_are_allowed():
    """实测：`set_pot 12.5` 求解器认（不必为它换算成整数筹码）。"""
    lines = (
        a_request(pot=12.5, effective_stack=87.5, scale=1.0)
        .commands("/tmp/out.json")
        .splitlines()
    )
    assert "set_pot 12.5" in lines
    assert "set_effective_stack 87.5" in lines


def test_the_command_file_is_written_in_tenths_of_a_big_blind():
    """**求解器把算出来的下注额取整到整数单位**（底池 5.5 打 33% 会变成 `BET 2`）。

    所以命令文件里放大 10 倍，粒度变成 0.1bb；对外的字段仍然是大盲。
    """
    lines = a_request(pot=5.5, effective_stack=97.5).commands("/tmp/out.json").splitlines()
    assert "set_pot 55" in lines
    assert "set_effective_stack 975" in lines
    assert a_request(pot=5.5).rounding == pytest.approx(0.05)


def test_amounts_come_back_in_big_blinds():
    """放大是命令文件里的事，解回来的金额必须除回大盲——不然 EV 会差十倍。"""
    document = {
        "node_type": "action_node",
        "player": 1,
        "strategy": {"actions": ["CHECK", "BET 35.000000"], "strategy": {}},
        "childrens": {},
    }
    root = parse_result(document, scale=10.0)
    assert root.actions[1].amount == pytest.approx(3.5)
    assert root.actions[1].label == "BET 35.000000", "标签是子节点的键，不能改"


def test_bet_sizes_cover_both_players_and_all_streets():
    lines = BetSizes(flop=(33.0, 75.0), turn=(66.0,), river=(75.0,)).commands()
    assert "set_bet_sizes oop,flop,bet,33,75" in lines
    assert "set_bet_sizes ip,flop,bet,33,75" in lines
    assert "set_bet_sizes oop,river,bet,75" in lines
    assert sum(1 for line in lines if line.endswith(",allin")) == 6, "两人 × 三条街"


def test_all_the_sizes_of_one_spot_go_on_one_line():
    """**同一格发第二条命令是覆盖，不是追加**（实测：树里只剩最后那个尺度，而且不报错）。

    这个错的代价不在求解——它照样解得出来——而在复盘时才暴露：
    实战打出的那个尺度「树里没有」，那个决策就打不了分。
    """
    lines = BetSizes(flop=(33.0, 63.6, 75.0)).commands()
    flop_bets = [line for line in lines if line.startswith("set_bet_sizes oop,flop,bet")]
    assert flop_bets == ["set_bet_sizes oop,flop,bet,33,63.6,75"]


def test_a_real_bet_size_can_be_folded_in():
    """实战打了 40% 底池，就得让树里有 40%——不然那个动作在解里不存在，没法打分。"""
    sizes = BetSizes(flop=(33.0, 75.0))
    assert sizes.with_size("flop", 40.0).flop == (33.0, 40.0, 75.0)
    assert sizes.with_size("flop", 34.0).flop == (33.0, 75.0), "差 1 个点算同一个尺度"
    with pytest.raises(ValueError, match="没有这条街"):
        sizes.with_size("preflop", 40.0)


def test_donk_sizes_only_apply_off_the_flop():
    lines = BetSizes(donk=(30.0,)).commands()
    assert "set_bet_sizes oop,turn,donk,30" in lines
    assert not any("flop,donk" in line for line in lines), "翻牌圈先说话就是普通下注"


# ------------------------------------------------------------------ 指纹


def test_the_same_spot_has_the_same_fingerprint():
    assert a_request().fingerprint() == a_request().fingerprint()


def test_anything_that_changes_the_answer_changes_the_fingerprint():
    base = a_request().fingerprint()
    assert a_request(pot=7.0).fingerprint() != base
    assert a_request(oop_range=Range.parse("TT+")).fingerprint() != base
    assert a_request(accuracy=0.2).fingerprint() != base
    assert a_request(bet_sizes=BetSizes(flop=(50.0,))).fingerprint() != base


def test_how_we_run_it_does_not_change_the_fingerprint():
    """线程数与输出路径不改变解，所以不能进缓存键——否则换台机器缓存全废。"""
    request = a_request()
    assert request.commands("/a.json", threads=2) != request.commands("/b.json", threads=8)
    assert request.fingerprint() == request.fingerprint()


def test_requests_are_validated():
    with pytest.raises(ValueError, match="3–5 张"):
        a_request(board=tuple(cards_from_str("QsJh")))
    with pytest.raises(ValueError, match="重复"):
        a_request(board=tuple(cards_from_str("QsQs2h")))
    with pytest.raises(ValueError, match="底池"):
        a_request(pot=0)
    with pytest.raises(ValueError, match="有效筹码"):
        a_request(effective_stack=-1)


def test_street_and_spr_come_from_the_request():
    assert a_request().street == "flop"
    assert a_request(board=tuple(cards_from_str("QsJh2h5d"))).street == "turn"
    assert a_request(board=tuple(cards_from_str("QsJh2h5d9c"))).street == "river"
    assert a_request(pot=6.0, effective_stack=30.0).spr == 5.0


# ------------------------------------------------------------------ 解析（真样本）


@pytest.fixture(scope="module")
def solved():
    return parse_result(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_the_root_belongs_to_the_out_of_position_player(solved):
    """**求解器管 OOP 叫 player 1**，我们的口径是 0。翻译错了，两边的解就整个对调。

    样本是 OOP=QQ,77,AKo / IP=AA,KK,JTs 跑出来的，所以看根节点上出现的是谁的牌就知道
    翻译对不对——`AhAd` 是 IP 的牌，不该出现在根节点。
    """
    assert solved.player == 0
    assert solved.for_combo(card_from_str("Qh"), card_from_str("Qd")) is not None, "QQ 是 OOP 的"
    assert solved.for_combo(card_from_str("Ah"), card_from_str("Ad")) is None, "AA 是 IP 的"


def test_actions_parse_into_kind_and_amount(solved):
    kinds = [action.kind for action in solved.actions]
    assert kinds[0] == "check"
    assert all(kind in ("check", "bet") for kind in kinds)
    bets = [action.amount for action in solved.actions if action.kind == "bet"]
    assert bets == sorted(bets) and all(amount > 0 for amount in bets)


def test_every_hand_has_one_probability_per_action(solved):
    for combo, weights in solved.strategy.items():
        assert len(weights) == len(solved.actions), combo
        assert sum(weights) == pytest.approx(1.0, abs=1e-3), combo


def test_a_hand_can_be_looked_up_in_either_card_order(solved):
    first, second = card_from_str("Qh"), card_from_str("Qd")
    assert solved.for_combo(first, second) == solved.for_combo(second, first)


def test_hand_classes_average_over_their_combos(solved):
    average = solved.for_class(class_from_name("QQ"))
    assert average is not None and sum(average) == pytest.approx(1.0, abs=1e-3)
    assert solved.for_class(class_from_name("72o")) is None, "不在范围里就没有策略"
    with pytest.raises(ValueError, match="越界"):
        solved.for_class(999)


def test_children_hang_off_the_action_labels(solved):
    assert set(solved.children) == {action.label for action in solved.actions}
    for label, child in solved.children.items():
        assert child.player in (0, 1, None)
    assert len(list(solved.walk())) > len(solved.children), "walk 要走到孙子辈"


def test_the_action_index_helper_finds_by_kind_and_size(solved):
    assert solved.action_index("check") == 0
    bet = solved.actions[solved.action_index("bet", 3.0)]
    assert bet.kind == "bet" and bet.amount == pytest.approx(3.0)
    assert solved.action_index("fold") is None, "根节点没人下注，弃不了"


def test_action_labels_are_parsed_strictly():
    assert parse_action("CHECK").kind == "check"
    assert parse_action("BET 30.000000").amount == 30.0
    assert parse_action("RAISE 102.5").amount == 102.5
    with pytest.raises(ValueError, match="看不懂"):
        parse_action("BET 三十")
    with pytest.raises(ValueError, match="没见过"):
        parse_action("STRADDLE 5")


def test_the_player_numbering_is_flipped_on_the_way_in():
    """直接喂一个手写节点，确认 1→0、0→1 的翻译。"""
    document = {
        "node_type": "action_node",
        "player": 1,
        "strategy": {"actions": ["CHECK"], "strategy": {"AsKh": [1.0]}},
        "childrens": {},
    }
    assert parse_result(document).player == 0
    document["player"] = 0
    assert parse_result(document).player == 1


def test_a_mismatched_strategy_vector_is_caught():
    document = {
        "node_type": "action_node",
        "player": 1,
        "strategy": {"actions": ["CHECK", "BET 3"], "strategy": {"AsKh": [1.0]}},
        "childrens": {},
    }
    with pytest.raises(ValueError, match="动作却有"):
        parse_result(document)


def test_chance_nodes_keep_their_deal_count():
    node = parse_result({"node_type": "chance_node", "deal_number": 45, "childrens": {}})
    assert node.kind == "chance" and node.deal_number == 45 and node.player is None


def test_unknown_node_types_are_refused():
    with pytest.raises(ValueError, match="没见过的节点类型"):
        parse_result({"node_type": "showdown_node"})


# ------------------------------------------------------------------ 后端（不跑二进制）


def test_a_missing_installation_says_how_to_install(monkeypatch):
    monkeypatch.delenv("TEXAS_SOLVER_HOME", raising=False)
    with pytest.raises(SolverNotInstalled, match="releases"):
        find_solver_home()


def test_an_incomplete_installation_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("TEXAS_SOLVER_HOME", raising=False)
    with pytest.raises(SolverNotInstalled, match="不是一个目录"):
        find_solver_home(tmp_path / "nope")

    home = tmp_path / "solver"
    home.mkdir()
    with pytest.raises(SolverNotInstalled, match="console_solver"):
        find_solver_home(home)

    (home / "console_solver").write_text("")
    with pytest.raises(SolverNotInstalled, match="resources"):
        find_solver_home(home)

    (home / "resources").mkdir()
    assert find_solver_home(home) == home


def test_availability_is_a_question_not_an_exception(tmp_path):
    assert TexasSolver.available(tmp_path) is False


def _fake_home(tmp_path):
    home = tmp_path / "solver"
    (home / "resources").mkdir(parents=True)
    (home / "console_solver").write_text("")
    return home


def test_the_cache_round_trips_without_running_anything(tmp_path):
    solver = TexasSolver(_fake_home(tmp_path), cache_dir=tmp_path / "cache")
    request = a_request()
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report = SolveReport(
        root=parse_result(document),
        exploitability=0.42,
        iterations=8,
        seconds=47.0,
        cached=False,
        fingerprint=request.fingerprint(),
    )
    solver._write_cache(request.fingerprint(), request, report, document, "set_pot 6\n")

    again = solver._read_cache(request.fingerprint(), request)
    assert again is not None and again.cached is True
    assert again.exploitability == 0.42 and again.iterations == 8
    assert again.root.player == report.root.player
    assert len(again.root.strategy) == len(report.root.strategy)


def test_the_cache_keeps_the_input_beside_the_answer(tmp_path):
    """脱离输入的解没法审计，也没法复现——命令原文要跟着一起存。"""
    solver = TexasSolver(_fake_home(tmp_path), cache_dir=tmp_path / "cache")
    request = a_request()
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report = SolveReport(parse_result(document), 0.42, 8, 47.0, False, request.fingerprint())
    solver._write_cache(request.fingerprint(), request, report, document, "set_pot 6\n")

    stored = json.loads(solver._cache_path(request.fingerprint()).read_text(encoding="utf-8"))
    assert stored["format"] == CACHE_FORMAT
    assert stored["commands"] == "set_pot 6\n"
    assert stored["accuracy"] == request.accuracy
    assert not list((tmp_path / "cache").glob("*.partial")), "临时文件要改名，不能留下"


def test_a_stale_cache_format_is_ignored(tmp_path):
    solver = TexasSolver(_fake_home(tmp_path), cache_dir=tmp_path / "cache")
    path = solver._cache_path("deadbeef")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": "OLD", "result": {}}), encoding="utf-8")
    assert solver._read_cache("deadbeef", a_request()) is None


def test_an_abort_is_explained_not_just_numbered():
    """退出码 134 光秃秃地报出来没人查得动，要把 what() 与最常见的成因带上。"""
    log = "EXEC FROM FILE\nterminate called...\n  what():  format not recognize\n"
    message = _explain(134, log)
    assert "format not recognize" in message and "99+" in message
    assert "退出码 7" in _explain(7, "")


def test_the_report_knows_whether_it_hit_the_target():
    root = parse_result(json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert SolveReport(root, 0.4, 100, 1.0, False, "x").meets(0.5) is True
    assert SolveReport(root, 4.0, 10, 1.0, False, "x").meets(0.5) is False
    assert SolveReport(root, None, None, 1.0, False, "x").meets(0.5) is False


# ------------------------------------------------------------------ 真跑（慢）


@pytest.mark.slow
@pytest.mark.skipif(not TexasSolver.available(), reason="没装 TexasSolver（TEXAS_SOLVER_HOME）")
def test_a_real_solve_converges_and_caches(tmp_path):
    solver = TexasSolver(cache_dir=tmp_path / "cache", threads=2)
    request = a_request(
        oop_range=Range.parse("QQ, 77"),
        ip_range=Range.parse("AA, KK"),
        bet_sizes=BetSizes(flop=(50.0,), turn=(66.0,), river=(75.0,), allin=False),
        max_iterations=6,
        accuracy=5.0,
    )
    report = solver.solve(request)
    assert report.cached is False
    assert report.exploitability is not None and report.iterations is not None
    assert report.root.player == 0, "根节点是 OOP"
    assert [action.kind for action in report.root.actions][0] == "check"

    again = solver.solve(request)
    assert again.cached is True, "第二次必须走缓存，不能再解一遍"

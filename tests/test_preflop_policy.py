"""翻前查表策略层与风格层的测试。

分两块：**认局面**不需要范围表（纯逻辑，随时能跑），**查表与风格层**需要产物，
表没生成时自动跳过。
"""

import pytest

from holdem.actions import call, fold, raise_to
from holdem.deck import deck_from_seed
from holdem.preflop_policy import (
    DEFEND,
    LIMP,
    OPEN,
    VS_RERAISE,
    PreflopPolicySet,
    PreflopSpot,
    PreflopTablePolicy,
    effective_depth,
    identify,
    parse_label,
)
from holdem.preflop_ranges import is_available
from holdem.ranges import NUM_HAND_CLASSES, class_combo_count, class_from_name
from holdem.state import HandConfig, HandState

BIG_BLIND = 100


def six_max(button: int = 0, seed: int = 1) -> HandState:
    """六人桌；按钮在 0 时，座位号正好等于「相对按钮的偏移」，位置一目了然。"""
    config = HandConfig(
        stacks=[100 * BIG_BLIND] * 6, button=button, big_blind=BIG_BLIND, small_blind=50
    )
    return HandState(config, deck_from_seed(seed))


# ------------------------------------------------------------------ 认局面


def test_first_to_act_is_an_open_spot():
    hand = six_max()
    spot = identify(hand, 6)
    assert spot == PreflopSpot(OPEN, "UTG", "UTG")
    assert "第一个开牌" in spot.label


def test_folded_around_still_counts_as_first_in():
    hand = six_max()
    hand.apply(fold())  # UTG
    hand.apply(fold())  # HJ
    assert identify(hand, 6) == PreflopSpot(OPEN, "CO", "CO")


def test_facing_an_open_is_a_defend_spot():
    hand = six_max()
    hand.apply(raise_to(250))  # UTG 开牌
    spot = identify(hand, 6)
    assert spot == PreflopSpot(DEFEND, "HJ", "UTG")
    assert spot.label == "HJ 面对 UTG 开牌"


def test_opener_facing_a_three_bet():
    hand = six_max()
    hand.apply(raise_to(250))  # UTG 开牌
    hand.apply(raise_to(750))  # HJ 3bet
    hand.apply(fold())  # CO
    hand.apply(fold())  # BTN
    hand.apply(fold())  # SB
    hand.apply(fold())  # BB
    spot = identify(hand, 6)
    assert spot == PreflopSpot(VS_RERAISE, "UTG", "UTG", reraiser="HJ")
    assert spot.label == "UTG 开牌后面对 HJ 再加注"


# ------------------------------------------------------------------ 表覆盖不到的局面


def test_limped_pot_is_not_in_the_table():
    """有人跛入之后，解就不适用了——表是按「一个开牌者」解出来的。"""
    hand = six_max()
    hand.apply(call())  # UTG 跛入
    assert identify(hand, 6) is None


def test_cold_call_makes_it_multiway():
    hand = six_max()
    hand.apply(raise_to(250))  # UTG
    hand.apply(call())  # HJ 冷跟
    assert identify(hand, 6) is None, "多人底池不在表里"


def test_four_bet_is_not_in_the_table():
    hand = six_max()
    hand.apply(raise_to(250))
    hand.apply(raise_to(750))
    hand.apply(fold())
    hand.apply(fold())
    hand.apply(fold())
    hand.apply(fold())
    hand.apply(raise_to(1650))  # UTG 4bet
    assert identify(hand, 6) is None


def test_other_table_sizes_are_not_forced_into_the_table():
    config = HandConfig(stacks=[10000] * 3, button=0, big_blind=BIG_BLIND, small_blind=50)
    hand = HandState(config, deck_from_seed(3))
    assert identify(hand, 6) is None, "表是六人桌解的，三人桌别硬套"


def test_postflop_is_never_a_preflop_spot():
    hand = six_max()
    for _ in range(5):
        hand.apply(fold() if hand.legal_actions().can_fold else call())
    if not hand.is_complete:
        hand.apply(call())
    assert identify(hand, 6) is None


# ------------------------------------------------------------------ 标签


@pytest.mark.parametrize(
    "label, expected",
    [
        ("弃牌", ("fold", None)),
        ("过牌", ("call", None)),
        ("跟注到2.5", ("call", None)),
        ("加注到7.5", ("raise", 7.5)),
        ("加注到16.5", ("raise", 16.5)),
        ("全下", ("allin", None)),
    ],
)
def test_parse_label(label, expected):
    assert parse_label(label) == expected


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="无法识别"):
        parse_label("摸鱼")


# ------------------------------------------------------------------ 查表与风格层


pytestmark_table = pytest.mark.skipif(
    not is_available(), reason="翻前范围表尚未生成，先跑 scripts/build_preflop_ranges.py"
)


@pytest.fixture(scope="module")
def policy():
    if not is_available():
        pytest.skip("翻前范围表尚未生成")
    return PreflopTablePolicy()


@pytestmark_table
def test_table_answers_a_covered_spot(policy):
    hand = six_max()
    decision = policy.decide(hand)
    assert decision is not None
    assert decision.spot.kind == OPEN
    assert sum(decision.weights.values()) == pytest.approx(1.0)


@pytestmark_table
def test_uncovered_spot_returns_none(policy):
    hand = six_max()
    hand.apply(call())  # 跛入
    assert policy.decide(hand) is None


@pytestmark_table
def test_premium_hands_open_and_trash_folds(policy):
    spot = PreflopSpot(OPEN, "UTG", "UTG")
    strong = policy._weights(spot, class_from_name("AA"))
    weak = policy._weights(spot, class_from_name("72o"))
    assert strong["加注到2.5"] > 0.9
    assert weak["弃牌"] > 0.9


@pytestmark_table
def test_looseness_widens_monotonically(policy):
    """松紧旋钮必须单调：越松，入池的牌越多，且不能把已经在打的牌挤出去。"""
    spot = PreflopSpot(OPEN, "UTG", "UTG")
    ranking = policy._ranking(spot)
    previous = -1.0
    for looseness in (0.5, 0.8, 1.0, 1.4, 2.0):
        target = min(1.0, ranking.frequency * looseness)
        played = sum(
            ranking.play_probability(i, target) * class_combo_count(i)
            for i in range(NUM_HAND_CLASSES)
        )
        assert played > previous
        previous = played



@pytestmark_table
def test_loosening_lets_suited_connectors_in(policy):
    """放宽顺序按求解器的逐手 EV 排——同花连张不该排在垃圾高张后面。

    这是换排序依据的原因：按生权益排，65s 到 2.5 倍都进不来，而 Q7o 早就进来了。
    """
    spot = PreflopSpot(OPEN, "BTN", "BTN")
    ranking = policy._ranking(spot)
    if not ranking.from_solver:
        pytest.skip("这张表是旧版，没有存逐手 EV")
    target = min(1.0, ranking.frequency * 1.6)
    suited = ranking.play_probability(class_from_name("76s"), target)
    trash = ranking.play_probability(class_from_name("Q4o"), target)
    assert suited >= trash


@pytestmark_table
def test_aggression_moves_weight_to_raising(policy):
    spot = PreflopSpot(DEFEND, "BB", "BTN")
    index = class_from_name("KTs")
    base = policy._weights(spot, index)
    timid = policy._tilt(spot, index, base, 1.0, 0.4)
    fierce = policy._tilt(spot, index, base, 1.0, 2.5)

    def raising(weights):
        return sum(v for k, v in weights.items() if k.startswith("加注") or k == "全下")

    assert raising(fierce) > raising(timid)


@pytestmark_table
def test_passive_style_limps_instead_of_opening(policy):
    """解里首位只有「开牌或弃牌」，但跟注站在真人里是会跛入的——由风格层补上。"""
    spot = PreflopSpot(OPEN, "BTN", "BTN")
    index = class_from_name("KTs")
    base = policy._weights(spot, index)
    passive = policy._tilt(spot, index, base, 1.0, 0.15)
    assert passive.get(LIMP, 0.0) > 0.5
    assert passive["加注到2.5"] < 0.3


@pytestmark_table
def test_the_reply_to_a_three_bet_comes_from_the_table(policy):
    """「面对再加注」必须真的查到表里那一格。

    表把开牌者的应对存在 `defense(开牌者, 3bet者).reraise_reply` 里——按**谁 3bet 的**
    分格。策略层曾经拿 `defense(自己, 自己)` 去查，那一格永远不存在，于是这整行局面
    一直在悄悄退回兜底规则，而所有测试都是绿的（连「72o 查不到」那条都因为查不到
    而通过）。这个测试盯的就是「查得到」本身。
    """
    from holdem.deck import stacked_deck

    hand = six_max(button=0, seed=4)
    hand = HandState(
        HandConfig(
            stacks=[100 * BIG_BLIND] * 6, button=0, big_blind=BIG_BLIND, small_blind=50
        ),
        stacked_deck({5: "AhAs"}, num_seats=6, button=0, rest_seed=4),
    )
    for action in (fold(), fold(), raise_to(250), raise_to(750), fold(), fold()):
        hand.apply(action)  # UTG弃 HJ弃 CO开 BTN 3bet SB弃 BB弃

    decision = policy.decide(hand)
    assert decision is not None, "CO 拿 AA 面对 3bet，表里必须有这一格"
    assert decision.spot.kind == VS_RERAISE and decision.spot.reraiser == "BTN"
    entry = policy.table.defense("CO", "BTN")
    assert set(decision.weights) == set(entry.reraise_reply)
    assert decision.weights["弃牌"] == 0.0, "AA 不会弃"


@pytestmark_table
def test_hands_that_never_arrive_have_no_solution(policy):
    """72o 从不开牌，所以「开牌被 3bet」这个节点它根本走不到，查表要回 None。"""
    spot = PreflopSpot(VS_RERAISE, "UTG", "UTG", reraiser="HJ")
    assert policy._weights(spot, class_from_name("72o")) is None


# ------------------------------------------------------------------ 接到 bot 上


@pytestmark_table
def test_bot_mostly_plays_from_the_table():
    """自对弈里大多数翻前决策应该走解，而不是兜底规则。"""
    from holdem.bots import Bot, play_out

    bots = {seat: Bot("solved", seed=500 + seat) for seat in range(6)}
    for index in range(30):
        hand = six_max(button=index % 6, seed=index * 13 + 2)
        play_out(hand, bots)
    hits = sum(bot.table_hits for bot in bots.values())
    total = hits + sum(bot.fallback_hits for bot in bots.values())
    assert hits / total > 0.6, f"只有 {hits / total:.0%} 的翻前决策用上了解"


@pytestmark_table
def test_styles_separate_in_the_expected_order():
    """风格必须真的分得开：越松的风格入池率越高。"""
    from holdem.bots import Bot, play_out
    from holdem.history import action_records
    from holdem.state import PREFLOP

    def vpip(style: str) -> float:
        bots = {seat: Bot(style, seed=7000 + seat) for seat in range(6)}
        played = seen = 0
        for index in range(40):
            hand = six_max(button=index % 6, seed=index * 31 + 5)
            play_out(hand, bots)
            records = [r for r in action_records(hand) if r.street == PREFLOP]
            for seat in range(6):
                mine = [r for r in records if r.seat == seat]
                if not mine:
                    continue
                seen += 1
                if any(r.kind in ("call", "bet", "raise") for r in mine):
                    played += 1
        return played / seen

    tight, solved, loose = vpip("nit"), vpip("solved"), vpip("maniac")
    assert tight < solved < loose, f"岩石 {tight:.0%} / 照解 {solved:.0%} / 疯子 {loose:.0%}"


def test_bot_runs_without_the_table():
    """产物缺失时引擎仍要能跑——bot 自动退回规则策略。"""
    from holdem.bots import Bot, play_out

    bots = {seat: Bot("tag", seed=seat, policy=None) for seat in range(6)}
    for bot in bots.values():
        bot.policy = None
    hand = six_max(seed=99)
    play_out(hand, bots)
    assert hand.is_complete
    assert all(bot.table_hits == 0 for bot in bots.values())


def test_table_fold_becomes_a_check_when_checking_is_free():
    """解里的「弃牌」是面对下注时的选择；能免费过牌还弃牌是白送。

    真实牌局里这两件事几乎不会同时出现（能查表的局面都面对着下注），所以直接在
    动作翻译这一层验：给它一个「100% 弃牌」的解，看它在能过牌时会不会照弃。
    """
    from holdem.actions import ActionKind
    from holdem.bots import Bot
    from holdem.preflop_policy import PolicyDecision

    hand = six_max(seed=17)
    hand.apply(call())  # UTG 跛入
    for _ in range(4):
        hand.apply(fold())
    legal = hand.legal_actions()
    assert legal.can_check, "此时大盲可以免费过牌"

    bot = Bot("nit", seed=3, policy=None)
    decision = PolicyDecision(spot=PreflopSpot(OPEN, "BB", "BB"), weights={"弃牌": 1.0})
    assert bot._from_table(hand, legal, decision).kind is ActionKind.CHECK


@pytestmark_table
def test_the_shared_policy_survives_a_thread_stampede():
    """几条线程同时要范围表时，每一条都得拿到表。

    并行跑对局（`play_slumbot.py --workers N`）时每条线程各建一个 bot，都会来要这份共享
    的表。加载要是不加锁，先到的那条会先把「试过了」立起来再去建表，后到的就拿到 `None`
    ——那个 bot 整场退回兜底规则，覆盖率掉一半而且不报错。
    """
    import threading

    from holdem import bots

    with bots._LOCK:
        bots._SHARED, bots._TRIED = None, False

    ready = threading.Barrier(8)
    got: list = []

    def grab():
        ready.wait()
        got.append(bots.shared_policy())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(got) == 8
    assert all(policy is not None for policy in got), "有线程拿到了空策略"
    assert len({id(policy) for policy in got}) == 1, "应当共用同一份，别每条线程各建一份"


# ------------------------------------------------------------------ 哪张表


def heads_up(stack_bb: float = 200.0, seed: int = 2) -> HandState:
    """单挑：按钮即小盲、翻前先说话（与引擎、与 Slumbot 同一口径）。"""
    config = HandConfig(
        stacks=[int(stack_bb * BIG_BLIND)] * 2,
        button=0,
        big_blind=BIG_BLIND,
        small_blind=50,
    )
    return HandState(config, deck_from_seed(seed))


def test_effective_depth_is_the_shallowest_stack():
    config = HandConfig(
        stacks=[10000, 4000, 8000], button=0, big_blind=BIG_BLIND, small_blind=50
    )
    hand = HandState(config, deck_from_seed(1))
    assert effective_depth(hand) == 40.0, "最浅的那份筹码说了算"


@pytestmark_table
def test_the_policy_set_dispatches_by_table_size():
    """六人桌查六人表、单挑查单挑表；没有对应人数的产物就交给兜底。"""
    policies = PreflopPolicySet()
    assert policies.for_seats(6) is not None

    six = six_max()
    assert policies.decide(six) is not None, "六人桌第一个开牌，表里有"

    three = HandState(
        HandConfig(stacks=[10000] * 3, button=0, big_blind=BIG_BLIND, small_blind=50),
        deck_from_seed(1),
    )
    assert policies.decide(three) is None, "三人桌没有产物，该交给兜底"


@pytest.mark.skipif(
    not __import__("holdem.preflop_ranges", fromlist=["HEADSUP_PATH"]).HEADSUP_PATH.exists(),
    reason="单挑范围表尚未生成",
)
def test_heads_up_hands_use_the_heads_up_table():
    policies = PreflopPolicySet()
    hand = heads_up()
    decision = policies.decide(hand)
    assert decision is not None, "单挑按钮位第一个行动，单挑表里有"
    assert decision.spot == PreflopSpot(OPEN, "BTN", "BTN")
    assert set(decision.weights) == {"加注到2.5", "弃牌"}


@pytestmark_table
def test_a_table_that_is_far_off_depth_is_not_used():
    """20bb 的桌子照 100bb 的表打是错的——那个深度该走推/弃。

    差一倍以内照用（翻前范围对深度没那么敏感），差得更多就交给兜底。
    """
    policy = PreflopTablePolicy()
    assert policy.decide(six_max()) is not None, "100bb 正好对上"

    def six_max_at(stack_bb):
        config = HandConfig(
            stacks=[int(stack_bb * BIG_BLIND)] * 6,
            button=0,
            big_blind=BIG_BLIND,
            small_blind=50,
        )
        return HandState(config, deck_from_seed(1))

    assert policy.decide(six_max_at(60)) is not None, "60bb 还在一倍以内"
    assert policy.decide(six_max_at(20)) is None, "20bb 太浅，别硬套"
    assert policy.decide(six_max_at(400)) is None, "400bb 太深，别硬套"


@pytest.mark.skipif(
    not __import__("holdem.preflop_ranges", fromlist=["HEADSUP_PATH"]).HEADSUP_PATH.exists(),
    reason="单挑范围表尚未生成",
)
def test_heads_up_bots_open_at_the_solved_frequency():
    """端到端：单挑自对弈打出来的开牌率要与表里的开牌频率对得上。

    这条串起了整条链——认局面、按人数选表、查表、把标签翻成引擎动作。中间任何一环
    悄悄退回兜底，这个数就会明显偏离（兜底是另一套阈值，不可能正好撞上）。
    """
    from holdem.batch import MatchConfig, run_batch
    from holdem.preflop_ranges import HEADSUP_PATH, load

    solved = load(HEADSUP_PATH).open_frequency("BTN")
    result = run_batch(
        MatchConfig(styles=("solved", "solved"), hands=400, start_stack=20_000, seed=9)
    )
    seat = result.seat(0)
    assert seat.solve_coverage > 0.95, (
        f"只有 {seat.solve_coverage:.0%} 的翻前决策走了解，链路多半断在某一环"
    )
    assert abs(seat.open_rate - solved) < 0.08, (
        f"实测开牌 {seat.open_rate:.1%}，表里是 {solved:.1%}"
    )

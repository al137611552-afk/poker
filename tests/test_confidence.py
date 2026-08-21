"""A/B/C 置信度分级（FR-9）。

这个模块很小，但**它的默认值决定了报告在最该示警的时候是不是干净的**，
所以测试盯的主要不是"算得对不对"，而是那几条**容易被后来者顺手优化掉**的规矩。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from holdem_solver.confidence import Confidence, grade, why  # noqa: E402


def test_a_clean_point_is_a():
    level, reasons = grade(in_range=True, loss=2.0, noise_floor=0.3)
    assert level is Confidence.A and reasons == ()
    assert level.usable is True


def test_unknown_noise_floor_never_reads_as_good():
    """**量不到 ≠ 很好。** 没记收敛度的老报告不该因为「没有坏消息」就拿 A。

    这条是整个模块最容易被改坏的地方：`noise_floor is None` 时直接跳过检查、
    让它落到 A，代码更短、测试也照样绿——然后报告会在最该示警的时候显得最干净。
    """
    level, reasons = grade(in_range=True, loss=2.0, noise_floor=None)
    assert level is Confidence.C
    assert any("量不到噪声底" in r for r in reasons)


def test_a_loss_inside_the_noise_floor_is_c_even_when_everything_else_is_clean():
    """解自己还差 0.5bb 时，一个 0.3bb 的「漏洞」量的是残差，不是打法。"""
    level, _ = grade(in_range=True, loss=0.3, noise_floor=0.5)
    assert level is Confidence.C


def test_a_loss_above_the_noise_floor_survives():
    level, _ = grade(in_range=True, loss=0.6, noise_floor=0.5)
    assert level is Confidence.A


def test_off_range_hands_are_c():
    """英雄打表外牌时，解没给这手牌频率——「最优动作」是拿别人的策略在评判他。"""
    level, reasons = grade(in_range=False, loss=2.0, noise_floor=0.1)
    assert level is Confidence.C
    assert any("不在假设的范围里" in r for r in reasons)


def test_aggregation_damage_is_b_not_c():
    """聚合伤害让结论打折，但不像「表外牌」那样让它失去意义——分开两档才有信息量。"""
    flagged, _ = grade(in_range=True, loss=2.0, noise_floor=0.1, hand_class_flagged=True)
    rolled, _ = grade(in_range=True, loss=2.0, noise_floor=0.1, rolled_streets=1)
    assert flagged is Confidence.B and rolled is Confidence.B
    assert flagged.usable is False, "B 档不能据此改打法"


def test_the_worst_signal_decides_the_grade():
    """置信度不是加权打分，是「最弱的一环有多弱」。"""
    level, reasons = grade(
        in_range=False, loss=2.0, noise_floor=0.1, hand_class_flagged=True, rolled_streets=2
    )
    assert level is Confidence.C
    assert len(reasons) >= 2, "定档取最差的那条，但理由要全给出来"


def test_all_reasons_are_reported_not_just_the_deciding_one():
    """读报告的人需要知道有**几处**不确定，不是只看到最重的那一处。"""
    _, reasons = grade(
        in_range=True, loss=0.1, noise_floor=0.5, hand_class_flagged=True, rolled_streets=1
    )
    assert len(reasons) == 3


def test_unscored_points_still_grade_on_the_other_signals():
    """没打上分（`loss=None`）不代表其它信号也不算——比如表外牌照样是 C。"""
    level, _ = grade(in_range=False, loss=None, noise_floor=0.5)
    assert level is Confidence.C


def test_why_says_something_useful_for_every_grade():
    for kwargs in (
        dict(in_range=True, loss=2.0, noise_floor=0.3),
        dict(in_range=True, loss=2.0, noise_floor=0.3, rolled_streets=1),
        dict(in_range=False, loss=2.0, noise_floor=None),
    ):
        text = why(grade(**kwargs))
        assert text[0] in "ABC" and len(text) > 10, text


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print(f"\n{sum(1 for n in globals() if n.startswith('test_'))}/"
          f"{sum(1 for n in globals() if n.startswith('test_'))} passed")


if __name__ == "__main__":
    _run_all()

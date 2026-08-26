"""解树必须在下一次求解开始前释放（`leak_report.solve_and_score`）。

**这条是 2026-08-25 Windows 真机撞出来的**：`report = solver.solve(...)` 赋值时
右侧先求值，新树建好的那一刻旧树还被 `report` 指着——峰值是**两棵**。
而一棵真实局面的解树 dump 3.2 GB、解析成 Python 对象十几 GB，
两棵就把 32 GB 的机器打爆了（`--max-solves 3` 照样卡死）。

用 `weakref` 直接验「新的一次求解开始时，上一棵还在不在」——
这比量内存可靠：内存数字受 GC 时机与分配器影响，引用在不在是确定的。
"""
import gc
import importlib.util
import sys
import weakref
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("lr", ROOT / "scripts" / "leak_report.py")
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)


class _Tree:
    """假装是一棵大解树。"""


class _Report:
    def __init__(self, tree):
        self.root = tree
        self.exploitability = 0.5
        self.fingerprint = "x"
        self.cached = False


class _Solver:
    """每次 solve 造一棵新树，并记下上一棵还活着没有。"""

    def __init__(self):
        self.cache_dir = Path("/tmp")
        self.alive_when_solving = []
        self._previous = None

    def solve(self, request, timeout=None):
        # **关键时刻**：新的一次求解开始时，上一棵树该已经没了
        self.alive_when_solving.append(
            self._previous is not None and self._previous() is not None
        )
        tree = _Tree()
        self._previous = weakref.ref(tree)
        return _Report(tree)


class _Plan:
    points = ()

    class request:
        accuracy = 1.0


def test_the_tree_is_gone_before_the_next_solve():
    solver = _Solver()
    original = lr.score_plan
    lr.score_plan = lambda plan, root: object()   # 打分不该抓着树不放
    try:
        lr.solve_and_score([_Plan(), _Plan(), _Plan()], solver,
                           timeout=1.0, keep_cache=True)
    finally:
        lr.score_plan = original

    assert solver.alive_when_solving == [False, False, False], (
        "上一棵树还没释放就开始解下一个——峰值会是两棵，"
        "而一棵真实局面就十几 GB"
    )


if __name__ == "__main__":
    test_the_tree_is_gone_before_the_next_solve()
    print("✅ 峰值只有一棵树")

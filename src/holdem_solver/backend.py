"""跑 TexasSolver：**本模块是这一层唯一碰进程与磁盘的地方。**

```python
solver = TexasSolver()                      # 从 TEXAS_SOLVER_HOME 找二进制
report = solver.solve(request)              # 命中缓存就不重算
print(report.exploitability, report.root.actions)
```

## 求解器不随包走

TexasSolver 是 **AGPL-3.0**：自己用没问题，**随我们的包分发不行**（分发就要同协议开源），
拿它做联网服务更不行。所以这里只**调用**，不携带——路径由 `TEXAS_SOLVER_HOME` 给，
装不装、装在哪是用户的事，装不上就明确报错并给出安装说明，绝不悄悄退回近似。

## 缓存是必需品，不是优化

一个翻牌局面解到求解器级精度要**几分钟**（本机双核实测：小树 20 次迭代 48 秒，
可利用度还有 4%）。复盘一手牌要解好几个局面，同一个局面又会在不同手牌里反复出现，
所以按**请求指纹**存盘缓存。缓存键不含线程数与输出路径——它们不改变解。

## 出错就是 SIGABRT

输入格式不对时求解器不会温柔报错，它抛 `std::runtime_error` 然后直接 abort
（退出码 134，什么都不输出）。所以这里把日志里的 `what():` 捞出来塞进异常，
不然只能看到一个光秃秃的 134。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .request import SolveRequest
from .result import SolvedNode, parse_result

__all__ = [
    "SolverNotInstalled",
    "SolveFailed",
    "SolveReport",
    "TexasSolver",
    "find_solver_home",
    "default_cache_dir",
]

CACHE_FORMAT = "TSCACHE1"
HOME_ENV = "TEXAS_SOLVER_HOME"
CACHE_ENV = "HOLDEM_SOLVE_CACHE"

_TOTAL = re.compile(r"Total exploitability\s+([0-9.eE+-]+)\s+precent")
"""求解器把 percent 拼成了 precent，别「修正」它，那样就匹配不上了。"""
_ITER = re.compile(r"Iter:\s*(\d+)")
_WHAT = re.compile(r"what\(\):\s*(.+)")


class SolverNotInstalled(RuntimeError):
    """没找到 TexasSolver。"""

    def __init__(self, hint: str) -> None:
        super().__init__(
            f"{hint}\n"
            f"装法：从 https://github.com/bupticybee/TexasSolver/releases 下载对应平台的包，\n"
            f"解开之后把 {HOME_ENV} 指向那个目录（里面要有 console_solver 与 resources/）。\n"
            f"它是 AGPL-3.0，我们只调用、不随包分发。"
        )


class SolveFailed(RuntimeError):
    """求解器跑挂了。"""


@dataclass(frozen=True)
class SolveReport:
    """一次求解的结果与它的账。"""

    root: SolvedNode
    exploitability: float | None
    """收敛到的可利用度，**占底池的百分比**（求解器自己的口径）。日志里没有就是 `None`。"""
    iterations: int | None
    seconds: float
    cached: bool
    fingerprint: str

    def meets(self, accuracy: float) -> bool:
        """有没有解到请求要求的精度。没到就别把这个解当准数用。"""
        return self.exploitability is not None and self.exploitability <= accuracy


def default_cache_dir() -> Path:
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "holdem-trainer" / "solves"


def find_solver_home(explicit: "str | Path | None" = None) -> Path:
    """定位 TexasSolver 的安装目录。"""
    candidate = explicit or os.environ.get(HOME_ENV)
    if not candidate:
        raise SolverNotInstalled(f"环境变量 {HOME_ENV} 没设。")
    home = Path(candidate).expanduser()
    if not home.is_dir():
        raise SolverNotInstalled(f"{home} 不是一个目录。")
    if not _binary_in(home):
        raise SolverNotInstalled(f"{home} 里没找到 console_solver。")
    if not (home / "resources").is_dir():
        raise SolverNotInstalled(f"{home} 里没有 resources/ 目录，求解器缺牌力字典跑不起来。")
    return home


def _binary_in(home: Path) -> Path | None:
    for name in ("console_solver", "console_solver.exe"):
        path = home / name
        if path.is_file():
            return path
    return None


class TexasSolver:
    """TexasSolver 的适配器。"""

    def __init__(
        self,
        home: "str | Path | None" = None,
        *,
        cache_dir: "str | Path | None" = None,
        threads: int | None = None,
    ) -> None:
        self.home = find_solver_home(home)
        self.binary = _binary_in(self.home)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.threads = threads or max(1, (os.cpu_count() or 2))

    @staticmethod
    def available(home: "str | Path | None" = None) -> bool:
        """装没装。没装的地方（比如 CI）相关测试自动跳过。"""
        try:
            find_solver_home(home)
        except SolverNotInstalled:
            return False
        return True

    # ---------------------------------------------------------- 求解

    def solve(
        self,
        request: SolveRequest,
        *,
        timeout: float = 3600.0,
        refresh: bool = False,
        on_progress=None,
    ) -> SolveReport:
        """解一个局面。命中缓存就直接读，除非 `refresh=True`。"""
        fingerprint = request.fingerprint()
        cached = None if refresh else self._read_cache(fingerprint, request)
        if cached is not None:
            return cached

        with tempfile.TemporaryDirectory(prefix="holdem-solve-") as workspace:
            work = Path(workspace)
            dump = work / "result.json"
            commands = request.commands(str(dump), threads=self.threads)
            script = work / "input.txt"
            script.write_text(commands, encoding="utf-8")

            started = time.perf_counter()
            process = subprocess.run(
                [
                    str(self.binary),
                    "--input_file", str(script),
                    "--resource_dir", str(self.home / "resources"),
                    "--mode", "holdem",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.home),
            )
            seconds = time.perf_counter() - started
            log = (process.stdout or "") + (process.stderr or "")
            if on_progress is not None:
                on_progress(log)

            if process.returncode != 0 or not dump.exists():
                raise SolveFailed(_explain(process.returncode, log))
            document = json.loads(dump.read_text(encoding="utf-8"))

        report = SolveReport(
            root=parse_result(document, scale=request.scale),
            exploitability=_last_float(_TOTAL, log),
            iterations=_last_int(_ITER, log),
            seconds=seconds,
            cached=False,
            fingerprint=fingerprint,
        )
        self._write_cache(fingerprint, request, report, document, commands)
        return report

    @staticmethod
    def supports_evs(home: "str | Path | None" = None) -> bool:
        """这个二进制认不认 `dump_evs`。

        官方预编译包**不认**——那条命令是我们自己打的补丁（ADR-0006、
        `docs/solver-build/0001-dump-evs.patch`）。不查这一条，用预编译包跑
        EV 相关的东西会红成一片，而真正的原因（"你这个二进制里根本没这个命令"）
        埋在求解器日志的一行 `command not recognized` 里。

        判据是在二进制里找命令名本身：比真跑一次求解便宜几个数量级，
        且不依赖求解器的报错文案（那是上游的，随时会变）。
        """
        try:
            binary = _binary_in(find_solver_home(home))
        except Exception:
            return False
        if binary is None:
            return False
        try:
            return b"dump_evs" in binary.read_bytes()
        except OSError:
            return False

    def solve_evs(
        self,
        request: SolveRequest,
        line: "tuple[str, ...]",
        player: int,
        *,
        timeout: float = 3600.0,
    ) -> "dict[str, dict[str, float]]":
        """解一次，然后**直接向求解器要 EV**（`dump_evs`，见 ADR-0006）。

        返回 `{手牌: {动作标签: EV}}`，金额是大盲，动作标签与 `dump_result` 里的一致。

        两件必须说清的事：

        1. **`player` 用我们的编号**（0 = OOP、1 = IP），这里翻译成求解器的编号再传。
           求解器管 OOP 叫 1、IP 叫 0，取错一侧的解看着仍然「像那么回事」。
        2. **EV 是求解器口径**：`最终底池份额 − 从这个局面起的投入 − 自己已投进底池的份额`。
           要换成我们 `evaluate.py` 的口径，得**加回该节点上自己已投入的那部分**。
           实测对上了手算（三个局面五个数，零误差，见 ADR-0006）——
           口径不翻译就用，每个数都会差自己已投的那一截，而且看着完全合理。

        `line` 里的动作标签直接来自 dump，含空格（`"BET 30.000000"`），
        所以命令里用 `|` 分隔，不是空格。
        """
        solver_player = 1 - player          # 我们的 0/1 → 求解器的 1/0
        with tempfile.TemporaryDirectory(prefix="holdem-evs-") as workspace:
            work = Path(workspace)
            evs_path = work / "evs.json"
            commands = request.commands(str(work / "unused.json"), threads=self.threads)
            # 换掉 dump_result：这一趟要的是 EV，不是策略树
            body = [ln for ln in commands.splitlines() if not ln.startswith("dump_result")]
            body.append(f"dump_evs {evs_path} {solver_player} {'|'.join(line)}".rstrip())
            script = work / "input.txt"
            script.write_text("\n".join(body) + "\n", encoding="utf-8")

            process = subprocess.run(
                [
                    str(self.binary),
                    "--input_file", str(script),
                    "--resource_dir", str(self.home / "resources"),
                    "--mode", "holdem",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.home),
            )
            if process.returncode != 0 or not evs_path.exists():
                raise SolveFailed(_explain(process.returncode,
                                           (process.stdout or "") + (process.stderr or "")))
            document = json.loads(evs_path.read_text(encoding="utf-8"))

        actions = document["actions"]
        return {
            hand: {actions[i]: value / request.scale for i, value in enumerate(row)}
            for hand, row in document["evs"].items()
        }

    # ---------------------------------------------------------- 缓存

    def _cache_path(self, fingerprint: str) -> Path:
        return self.cache_dir / f"{fingerprint}.json"

    def _read_cache(self, fingerprint: str, request: SolveRequest) -> SolveReport | None:
        path = self._cache_path(fingerprint)
        if not path.is_file():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("format") != CACHE_FORMAT:
            return None
        return SolveReport(
            root=parse_result(document["result"], scale=request.scale),
            exploitability=document.get("exploitability"),
            iterations=document.get("iterations"),
            seconds=float(document.get("seconds", 0.0)),
            cached=True,
            fingerprint=fingerprint,
        )

    def _write_cache(
        self,
        fingerprint: str,
        request: SolveRequest,
        report: SolveReport,
        document: dict,
        commands: str,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CACHE_FORMAT,
            "fingerprint": fingerprint,
            # 连输入一起存：脱离输入的解没法审计，也没法复现
            "commands": commands,
            "board": [int(card) for card in request.board],
            "pot": request.pot,
            "effective_stack": request.effective_stack,
            "accuracy": request.accuracy,
            "exploitability": report.exploitability,
            "iterations": report.iterations,
            "seconds": round(report.seconds, 2),
            "result": document,
        }
        path = self._cache_path(fingerprint)
        # 先写临时文件再改名：跑了几分钟的解不能被一次中断毁掉
        staging = path.with_suffix(".partial")
        staging.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        staging.replace(path)


def _explain(code: int, log: str) -> str:
    what = _WHAT.search(log)
    detail = what.group(1).strip() if what else "日志里没有更多线索"
    if code == 134:
        return (
            f"求解器 abort 了（退出码 134）：{detail}。"
            f"最常见的原因是范围写成了 `99+` 这种记法——它只认逐个牌类"
        )
    return f"求解器退出码 {code}：{detail}"


def _last_float(pattern: re.Pattern, log: str) -> float | None:
    found = pattern.findall(log)
    return float(found[-1]) if found else None


def _last_int(pattern: re.Pattern, log: str) -> int | None:
    found = pattern.findall(log)
    return int(found[-1]) if found else None

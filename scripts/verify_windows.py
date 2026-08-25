"""Windows 真机自检：能自动验的全自动跑，验不了的明确报出来。

这份脚本是给**在 Windows 本机跑的 agent** 用的：它把「点开界面看一眼」的检查
尽量翻译成**走 HTTP 接口的断言**——接口比像素可靠，也不需要浏览器。

    python scripts/verify_windows.py            # 全部
    python scripts/verify_windows.py --only C   # 只跑某一组
    python scripts/verify_windows.py --json out.json

## 三种结果，含义不同

- `PASS` —— 检查通过。
- `FAIL` —— **代码有问题**，把这一条连同 detail 原样报回来。
- `SKIP` —— 环境不具备（没装求解器、没有旧数据库…）。**SKIP 不是失败**，
  但要说清跳过了什么，否则「全绿」会掩盖「一半没跑」。

## 这份脚本**验不了**什么

浏览器里的布局、颜色、触屏手感、手机上的换行——这些没有真人看不行。
脚本会在最后把它们逐条列出来交给人，**不要因为脚本全绿就以为验完了**。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class Result:
    group: str
    name: str
    status: str
    detail: str = ""


RESULTS: "list[Result]" = []


def record(group, name, status, detail=""):
    RESULTS.append(Result(group, name, status, detail))
    mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭"}[status]
    line = f"  {mark} [{group}] {name}"
    if detail:
        line += f" —— {detail}"
    print(line)


class SkipCheck(Exception):
    """环境不具备，这条验不了。**不是失败。**"""


def check(group, name):
    """把一个断言函数包成一条检查。

    - 正常返回（可带一句说明）＝ PASS
    - `AssertionError` ＝ FAIL
    - `raise SkipCheck(理由)` ＝ SKIP

    **跳过必须用异常显式声明，不能靠返回值区分**：第一版让「返回字符串」表示
    SKIP，而通过的检查也返回字符串当说明——于是十七条全被记成跳过，
    脚本却「没有失败」。两件事压成同一种表示就分不开了。
    """
    def wrap(fn):
        try:
            outcome = fn()
        except SkipCheck as exc:
            record(group, name, "SKIP", str(exc))
        except AssertionError as exc:
            record(group, name, "FAIL", str(exc) or "断言失败")
        except Exception as exc:
            record(group, name, "FAIL", f"{type(exc).__name__}: {exc}")
        else:
            record(group, name, "PASS", outcome or "")
        return fn
    return wrap


# ================================================================== 第 0 组 · 环境


def group_env():
    print("\n第 0 组 · 环境与全回归")

    @check("0", "Python 版本 ≥ 3.11")
    def _():
        assert sys.version_info >= (3, 11), f"当前是 {sys.version.split()[0]}"
        return sys.version.split()[0]

    @check("0", "依赖装齐（含 server extra）")
    def _():
        import fastapi, httpx, pytest, uvicorn  # noqa: F401
        return "fastapi / uvicorn / httpx / pytest 都在"

    @check("0", "预计算数据随包分发")
    def _():
        data = ROOT / "src" / "holdem" / "data"
        need = ["preflop_equity.bin", "preflop_ranges_6max_100bb.json",
                "preflop_ranges_hu_200bb.json"]
        missing = [n for n in need if not (data / n).exists()]
        assert not missing, f"缺文件：{missing}"
        return f"{len(need)} 个文件都在"

    @check("0", "全回归（pytest）")
    def _():
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "--tb=line"],
            cwd=ROOT, capture_output=True, text=True, timeout=1800,
        )
        tail = (proc.stdout or "").strip().splitlines()[-3:]
        assert proc.returncode == 0, "失败：" + " | ".join(tail)
        return " ".join(tail)[-80:]


# ================================================================== A 组 · HUD


def _client():
    from fastapi.testclient import TestClient

    from holdem_server.app import create_app

    return TestClient(create_app(":memory:", bot_delay=0))


def _table_payload():
    return {
        "seats": [{"name": "你", "isHuman": True, "style": "tag"}]
        + [{"name": f"bot{i}", "isHuman": False, "style": "tag"} for i in range(1, 6)],
        "startingStack": 1000, "bigBlind": 10, "smallBlind": 5, "ante": 0, "seed": 42,
    }


def _play(client, count=1):
    for _ in range(count):
        client.post("/api/hand")
        for _ in range(200):
            view = client.get("/api/state").json()
            if not view["inProgress"]:
                break
            if view["waitingForHuman"]:
                client.post("/api/action", json={"kind": "fold"})
            time.sleep(0.002)


def group_hud():
    print("\nA 组 · HUD（FR-8）")

    @check("A", "一手没打时每个座位都有行，且全是「不知道」")
    def _():
        with _client() as c:
            c.post("/api/table", json=_table_payload())
            seats = c.get("/api/hud").json()["seats"]
            assert len(seats) == 6, f"只有 {len(seats)} 个座位"
            rates = [m["rate"] for s in seats for m in s["stats"] if "rate" in m]
            assert all(r is None for r in rates), "没打牌就不该有百分比"
        return "6 个座位，全是 None"

    @check("A", "每个比率指标都带样本量")
    def _():
        with _client() as c:
            c.post("/api/table", json=_table_payload())
            _play(c, 4)
            for seat in c.get("/api/hud").json()["seats"]:
                rates = [m for m in seat["stats"] if "rate" in m]
                assert len(rates) >= 8, f"比率指标只有 {len(rates)} 个"
                for m in rates:
                    assert m["hits"] <= m["chances"], f"{m['key']}: {m['hits']}>{m['chances']}"
                    if m["chances"] == 0:
                        assert m["rate"] is None, f"{m['key']} 没机会却给了 {m['rate']}"
        return "chances/hits 齐全且自洽"

    @check("A", "AF 不是比率，有自己的形状")
    def _():
        with _client() as c:
            c.post("/api/table", json=_table_payload())
            _play(c, 2)
            af = next(m for m in c.get("/api/hud").json()["seats"][0]["stats"]
                      if m["key"] == "aggression_factor")
            assert "rate" not in af and "chances" not in af, f"AF 混进了比率字段：{af}"
            assert "aggressive" in af and "calls" in af
        return "AF 用 value/aggressive/calls"

    @check("A", "HUD 不跟着每次动作广播")
    def _():
        with _client() as c:
            c.post("/api/table", json=_table_payload())
            view = c.get("/api/state").json()
            assert "hud" not in view and "stats" not in view
        return "state 里没有统计"


# ================================================================== C 组 · 求解器


def group_solver():
    print("\nC 组 · 求解器（EV 口径，这台机器最要紧的一组）")

    @check("C", "TEXAS_SOLVER_HOME 已设置且目录存在")
    def _():
        home = os.environ.get("TEXAS_SOLVER_HOME")
        if not home:
            raise SkipCheck("没设 TEXAS_SOLVER_HOME —— C 组整组跳过")
        assert Path(home).is_dir(), f"目录不存在：{home}"
        return home

    @check("C", "求解器认得 dump_evs（自编 console 分支）")
    def _():
        from holdem_solver.backend import TexasSolver

        if not os.environ.get("TEXAS_SOLVER_HOME"):
            raise SkipCheck("没设 TEXAS_SOLVER_HOME")
        assert TexasSolver.supports_evs(), (
            "这个二进制不认 dump_evs —— 官方预编译包没有这条命令，"
            "要按 docs/solver-build/README.md 自己编并打上 0001-dump-evs.patch"
        )
        return "认得"

    @check("C", "慢测全过（含 EV 口径的两道门）")
    def _():
        from holdem_solver.backend import TexasSolver

        if not os.environ.get("TEXAS_SOLVER_HOME") or not TexasSolver.supports_evs():
            raise SkipCheck("求解器不可用（没设 TEXAS_SOLVER_HOME 或不认 dump_evs）")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-m", "slow", "-q", "-p", "no:warnings",
             "--tb=short"],
            cwd=ROOT, capture_output=True, text=True, timeout=3600,
        )
        tail = (proc.stdout or "").strip().splitlines()[-5:]
        assert proc.returncode == 0, "失败：" + " | ".join(tail)
        return " ".join(tail)[-90:]


# ================================================================== E 组 · 场景训练


def group_training():
    print("\nE 组 · 场景训练（FR-12）")

    @check("E", "场景与位置表由引擎给出")
    def _():
        with _client() as c:
            body = c.get("/api/training/scenarios").json()
            assert {"UTG", "BTN", "BB"} <= set(body["positions"])
            assert {k["id"] for k in body["kinds"]} == {"开牌", "面对开牌", "面对再加注"}
        return "三类场景、六个位置"

    @check("E", "训练不需要牌桌")
    def _():
        with _client() as c:
            body = c.post("/api/training/deal",
                          json={"kind": "开牌", "hero": "UTG"})
            assert body.status_code == 200, f"没建桌就发不了题：{body.text[:120]}"
        return "没建桌也能发题"

    @check("E", "非法加注被拒绝且说明原因，不给判词")
    def _():
        with _client() as c:
            c.post("/api/table", json=_table_payload())
            c.post("/api/training/deal",
                   json={"kind": "面对开牌", "hero": "BB", "villain": "BTN"})
            resp = c.post("/api/training/answer", json={"kind": "raise", "to": 11})
            assert resp.status_code == 422, f"竟然接受了：{resp.status_code}"
            assert "不合法" in resp.json()["detail"]
        return "422 + 说明"

    @check("E", "判卷给完整分布")
    def _():
        with _client() as c:
            c.post("/api/table", json=_table_payload())
            c.post("/api/training/deal", json={"kind": "开牌", "hero": "BTN"})
            body = c.post("/api/training/answer", json={"kind": "fold"}).json()
            assert body["graded"] is True
            assert body["best"] in body["weights"]
            assert 0.0 <= body["frequency"] <= 1.0
        return f"分布 {len(body['weights'])} 项"


# ================================================================== F 组 · 复盘


def group_review():
    print("\nF 组 · 复盘（FR-9）")

    @check("F", "没打完的牌不给复盘")
    def _():
        with _client() as c:
            assert c.get("/api/review").status_code == 409
            c.post("/api/table", json=_table_payload())
            assert c.get("/api/review").status_code == 409
        return "两种情况都是 409"

    @check("F", "每个决策点带的是**那一刻**的底池")
    def _():
        # **人类一直弃牌就只有一次决策**，验不到「多次决策底池不同」这条。
        # 所以这里让人类先跟注、把牌打下去，制造出第二次说话的机会。
        with _client() as c:
            c.post("/api/table", json=_table_payload())
            for _ in range(12):
                c.post("/api/hand")
                for _ in range(200):
                    view = c.get("/api/state").json()
                    if not view["inProgress"]:
                        break
                    if view["waitingForHuman"]:
                        legal = view["legal"]
                        kind = "call" if legal.get("canCall") else "check"
                        c.post("/api/action", json={"kind": kind})
                    time.sleep(0.002)
                steps = c.get("/api/review").json()["preflop"]
                if len(steps) >= 2:
                    pots = [s["potBefore"] for s in steps]
                    assert len(set(pots)) > 1, (
                        f"多次决策的底池全一样（{pots}）——重放没生效，"
                        "多半是拿终局数字填的"
                    )
                    return f"两次决策底池 {pots}"
        raise SkipCheck(
            "打了 12 手都没出现「英雄翻前说话两次以上」的局面——"
            "换个 seed 重跑一次多半就有；这条没验到"
        )

    @check("F", "没装求解器时如实说，不留空")
    def _():
        with _client() as c:
            c.post("/api/table", json=_table_payload())
            _play(c, 1)
            postflop = c.get("/api/review").json()["postflop"]
            if postflop["available"]:
                raise SkipCheck("装了求解器，这条不适用")
            assert postflop["why"] and "求解器" in postflop["why"], \
                f"没装却没说清楚：{postflop}"
        return "给出了「没装求解器」的原话"

    @check("F", "复盘跟着最新那手，不是第一手")
    def _():
        with _client() as c:
            c.post("/api/table", json=_table_payload())
            _play(c, 1)
            first = c.get("/api/review").json()["handNo"]
            _play(c, 1)
            second = c.get("/api/review").json()["handNo"]
            assert second > first, f"还停在第 {first} 手"
        return f"第 {first} 手 → 第 {second} 手"


# ================================================================== G 组 · 评级


def group_rating():
    print("\nG 组 · 水平评级（FR-14）")

    @check("G", "答题确实落库了（守着一个曾被吞掉的 bug）")
    def _():
        with _client() as c:
            c.post("/api/table", json=_table_payload())
            c.post("/api/training/deal", json={"kind": "开牌", "hero": "UTG"})
            body = c.post("/api/training/answer", json={"kind": "fold"}).json()
            assert body.get("saved") is True, (
                f"没存进去：{body.get('saveError')} —— 这正是 except:pass 吞掉过的那个 bug"
            )
            assert c.get("/api/rating").json()["quiz"]["answered"] == 1
        return "答一题、测验轨加一"

    @check("G", "样本不够时不给分")
    def _():
        with _client() as c:
            c.post("/api/table", json=_table_payload())
            _play(c, 1)
            body = c.get("/api/rating").json()
            assert body["score"] is None, f"样本这么少却给了 {body['score']} 分"
            assert "还差" in body["why"]
        return body["why"][:40]

    @check("G", "原始与调整后两个数都给")
    def _():
        with _client() as c:
            c.post("/api/table", json=_table_payload())
            _play(c, 3)
            play = c.get("/api/rating").json()["play"]
            assert "rawBb100" in play and "adjustedBb100" in play
        return f"{play['rawBb100']:.1f} → {play['adjustedBb100']:.1f}"

    @check("G", "旧版数据库能升上来（有 hands.sqlite 才验）")
    def _():
        from holdem.store import HandStore

        candidates = [ROOT / "hands.sqlite", Path.cwd() / "hands.sqlite"]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise SkipCheck("本机没有旧的 hands.sqlite")
        with HandStore(found) as store:
            version = store.conn.execute("SELECT version FROM schema_info").fetchone()
            assert version["version"] == 2, f"版本是 {version['version']}"
            hands = store.count_hands()
        return f"{found.name} 升到 v2，旧牌局 {hands} 手还在"


# ================================================================== 真起服务


def group_serve():
    print("\nH 组 · 真起一次服务（TestClient 验不到的那部分）")

    @check("H", "uvicorn 能起来并响应")
    def _():
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        proc = subprocess.Popen(
            [sys.executable, "-m", "holdem_server", "--port", "8123"],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            import urllib.request

            for _ in range(60):
                time.sleep(0.5)
                if proc.poll() is not None:
                    out = proc.stdout.read() if proc.stdout else ""
                    raise AssertionError(f"进程退出了：{out[-400:]}")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8123/", timeout=2) as r:
                        assert r.status == 200
                        body = r.read().decode("utf-8", "replace")
                        assert "德扑训练台" in body, "首页内容不对"
                        return "起来了，首页可访问"
                except AssertionError:
                    raise
                except Exception:
                    continue
            raise AssertionError("30 秒内没起来")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


# ================================================================== 主流程


GROUPS = {
    "0": group_env, "A": group_hud, "C": group_solver,
    "E": group_training, "F": group_review, "G": group_rating, "H": group_serve,
}

MANUAL = [
    ("B", "手机与电脑同一 Wi-Fi，用启动时打印的局域网网址能打开牌桌"),
    ("B", "竖屏下座位、公共牌、操作栏布局正常，按钮 ≥48px 好按"),
    ("B", "加注滑杆与「最小/半池/底池/全下」快捷键在触屏上可用"),
    ("B", "息屏再回来时 WebSocket 能自动重连"),
    ("B", "Windows 防火墙首次运行的弹窗要选「允许」，否则手机连不上"),
    ("A", "座位卡片上那行统计：手数少时是**灰色**或「—」，不是正常颜色的精确百分比"),
    ("A", "点座位那行能弹出完整浮层；点「关闭」能关；**刷新后不该自己弹出来**"),
    ("E", "条形图里**你选的那行名字是高亮的**；底部有「这是频率不是 EV 损失」那句话"),
    ("F", "复盘面板**开着**时打完新一手会自动刷新；底部有「两者不是同一把尺子」那句"),
    ("G", "评级面板顶部在样本不够时显示「还不能评级」，**不是一个数**"),
    ("G", "底部有「折算规则是拍的，真比强弱看调整后 bb/100」那句"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="只跑某几组，如 --only 0AC")
    ap.add_argument("--json", help="把结果写成 JSON")
    args = ap.parse_args()

    wanted = list(args.only.upper()) if args.only else list(GROUPS)
    print("=" * 70)
    print("德扑训练台 · Windows 自检")
    print("=" * 70)
    for key in wanted:
        fn = GROUPS.get(key)
        if fn is None:
            print(f"（没有 {key} 组，跳过）")
            continue
        fn()

    passed = [r for r in RESULTS if r.status == "PASS"]
    failed = [r for r in RESULTS if r.status == "FAIL"]
    skipped = [r for r in RESULTS if r.status == "SKIP"]

    print("\n" + "=" * 70)
    print(f"通过 {len(passed)} · 失败 {len(failed)} · 跳过 {len(skipped)}")
    if failed:
        print("\n失败的（把这几条连同 detail 原样报回来）：")
        for r in failed:
            print(f"  ❌ [{r.group}] {r.name}\n     {r.detail}")
    if skipped:
        print("\n跳过的（不是失败，但要说清楚跳了什么）：")
        for r in skipped:
            print(f"  ⏭ [{r.group}] {r.name} —— {r.detail}")

    print("\n" + "-" * 70)
    print("以下**必须人眼看**，脚本验不了。逐条确认后回报：")
    for group, text in MANUAL:
        print(f"  [ ] [{group}] {text}")
    print("-" * 70)
    print("\n⚠ **脚本全绿不等于验完了** —— 上面那 11 条要人来看。")

    if args.json:
        Path(args.json).write_text(json.dumps(
            [r.__dict__ for r in RESULTS], ensure_ascii=False, indent=2
        ), encoding="utf-8")
        print(f"\n结果已写入 {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

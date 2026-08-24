"""牌谱落库（SQLite）。

这是引擎之外的 IO 层：引擎保持纯逻辑，落库只做「把状态翻译成行」这一件事。

存两份表示是有意为之：
- `hands.phh` 保留 PHH 原文，作为唯一事实来源，可导出、可被外部工具读取；
- 结构化表（`hand_players` / `hand_actions`）是为查询与统计建的索引，随时可从 PHH 重建。

任何指标算错时，先信 PHH 原文，重建结构化表即可。
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .cards import cards_from_str, cards_to_str
from .history import ActionRecord, action_records
from .phh import loads as phh_loads, to_phh
from .positions import position_of
from .metrics import bb_per_100
from .stats import HandFacts, accumulate_facts
from .state import HandState

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY,
    label       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    small_blind INTEGER,
    big_blind   INTEGER,
    ante        INTEGER,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS hands (
    id               INTEGER PRIMARY KEY,
    session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    hand_no          INTEGER NOT NULL,
    played_at        TEXT NOT NULL,
    num_seats        INTEGER NOT NULL,
    button           INTEGER NOT NULL,
    small_blind      INTEGER NOT NULL,
    big_blind        INTEGER NOT NULL,
    ante             INTEGER NOT NULL,
    board            TEXT NOT NULL,
    pot              INTEGER NOT NULL,
    went_to_showdown INTEGER NOT NULL,
    seed             INTEGER,
    phh              TEXT NOT NULL,
    UNIQUE (session_id, hand_no)
);

CREATE TABLE IF NOT EXISTS hand_players (
    hand_id        INTEGER NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    seat           INTEGER NOT NULL,
    player         TEXT NOT NULL,
    position       TEXT NOT NULL,
    starting_stack INTEGER NOT NULL,
    hole           TEXT NOT NULL,
    contributed    INTEGER NOT NULL,
    won            INTEGER NOT NULL,
    net            INTEGER NOT NULL,
    folded         INTEGER NOT NULL,
    all_in         INTEGER NOT NULL,
    showed         INTEGER NOT NULL,
    PRIMARY KEY (hand_id, seat)
);

CREATE TABLE IF NOT EXISTS hand_actions (
    hand_id       INTEGER NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,
    street        INTEGER NOT NULL,
    seat          INTEGER NOT NULL,
    player        TEXT NOT NULL,
    position      TEXT NOT NULL,
    kind          TEXT NOT NULL,
    amount        INTEGER NOT NULL,
    to_amount     INTEGER NOT NULL,
    pot_before    INTEGER NOT NULL,
    bet_before    INTEGER NOT NULL,
    to_call       INTEGER NOT NULL,
    stack_before  INTEGER NOT NULL,
    actors_before INTEGER NOT NULL,
    PRIMARY KEY (hand_id, seq)
);

CREATE TABLE IF NOT EXISTS quiz_answers (
    id           INTEGER PRIMARY KEY,
    answered_at  TEXT NOT NULL,
    kind         TEXT NOT NULL,
    hero         TEXT NOT NULL,
    villain      TEXT,
    hole         TEXT NOT NULL,
    taken        TEXT NOT NULL,
    best         TEXT NOT NULL,
    frequency    REAL NOT NULL,
    on_solution  INTEGER NOT NULL,
    blunder      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_quiz_time ON quiz_answers (answered_at);
CREATE INDEX IF NOT EXISTS idx_hands_session ON hands (session_id, hand_no);
CREATE INDEX IF NOT EXISTS idx_players_name ON hand_players (player);
CREATE INDEX IF NOT EXISTS idx_actions_player ON hand_actions (player, street);
"""


@dataclass(frozen=True)
class PlayerSummary:
    player: str
    hands: int
    net: int
    showdowns: int
    big_blind: int

    @property
    def bb_per_100(self) -> float:
        """每百手赢取的大盲数。口径与批量对局、Slumbot 基线共用（`metrics.py`）。"""
        return bb_per_100(self.net, self.hands, self.big_blind)


class HandStore:
    """牌谱数据库。路径给 ":memory:" 即为内存库，测试常用。"""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        # **允许跨线程**：FastAPI 把同步路由扔进线程池，而这个连接建在主线程，
        # 默认的 `check_same_thread=True` 会直接报错（2026-08-24 踩到，
        # 而且当时被一个 `except: pass` 吞掉了，症状是「测验轨永远是 0」）。
        # 本工具单人本地使用，并发极低，用一把锁把对外的读写包住就够了。
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        row = self.conn.execute("SELECT version FROM schema_info").fetchone()
        if row is None:
            self.conn.execute("INSERT INTO schema_info (version) VALUES (?)", (SCHEMA_VERSION,))
        elif row["version"] < SCHEMA_VERSION:
            # **v1 → v2 是纯增量**（只加了 `quiz_answers` 一张新表），
            # 上面的 `CREATE TABLE IF NOT EXISTS` 已经把它建好了，所以直接改版本号。
            #
            # 将来若有**破坏性**变更（改列、改语义），不能照抄这条路——
            # 那时要写真正的迁移，并且**先备份**。这里之所以敢直接升，
            # 是因为老数据一个字节都没动。
            self.conn.execute("UPDATE schema_info SET version = ?", (SCHEMA_VERSION,))
        elif row["version"] > SCHEMA_VERSION:
            raise RuntimeError(
                f"数据库结构版本比程序还新：文件是 v{row['version']}，"
                f"程序只认到 v{SCHEMA_VERSION}——用新版程序打开它"
            )
        self.conn.commit()

    # ------------------------------------------------------------- 会话

    def create_session(
        self,
        label: str,
        *,
        small_blind: int = 0,
        big_blind: int = 0,
        ante: int = 0,
        notes: str = "",
    ) -> int:
        cursor = self.conn.execute(
            "INSERT INTO sessions (label, created_at, small_blind, big_blind, ante, notes)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                label,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                small_blind,
                big_blind,
                ante,
                notes,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    # ------------------------------------------------------------- 写入

    def save_hand(
        self,
        hand: HandState,
        *,
        session_id: int,
        players: list[str],
        hand_no: int | None = None,
        seed: int | None = None,
    ) -> int:
        """把一手已结束的牌局写入数据库，返回 hand_id。"""
        if not hand.is_complete:
            raise ValueError("只能保存已结束的牌局")
        cfg = hand.config
        n = cfg.num_seats
        if len(players) != n:
            raise ValueError("players 的长度必须等于座位数")

        if hand_no is None:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(hand_no), 0) + 1 AS next FROM hands WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            hand_no = int(row["next"])

        result = hand.result
        assert result is not None
        phh_text = to_phh(hand, players=players, hand_number=hand_no)

        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO hands (session_id, hand_no, played_at, num_seats, button,"
                " small_blind, big_blind, ante, board, pot, went_to_showdown, seed, phh)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    hand_no,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    n,
                    cfg.button,
                    cfg.sb,
                    cfg.big_blind,
                    cfg.ante,
                    cards_to_str(hand.board),
                    sum(result.contributions),
                    int(result.went_to_showdown),
                    seed,
                    phh_text,
                ),
            )
            hand_id = int(cursor.lastrowid)

            self.conn.executemany(
                "INSERT INTO hand_players (hand_id, seat, player, position, starting_stack,"
                " hole, contributed, won, net, folded, all_in, showed)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        hand_id,
                        seat,
                        players[seat],
                        position_of(seat, cfg.button, n),
                        cfg.stacks[seat],
                        cards_to_str(hand.hole[seat]),
                        result.contributions[seat],
                        result.payouts[seat],
                        result.net[seat],
                        int(hand.folded[seat]),
                        int(hand.all_in[seat]),
                        int(seat in result.showdown_scores),
                    )
                    for seat in range(n)
                ],
            )

            self.conn.executemany(
                "INSERT INTO hand_actions (hand_id, seq, street, seat, player, position,"
                " kind, amount, to_amount, pot_before, bet_before, to_call, stack_before,"
                " actors_before) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        hand_id,
                        record.seq,
                        record.street,
                        record.seat,
                        players[record.seat],
                        record.position,
                        record.kind,
                        record.amount,
                        record.to,
                        record.pot_before,
                        record.bet_before,
                        record.to_call,
                        record.stack_before,
                        record.actors_before,
                    )
                    for record in action_records(hand)
                ],
            )

        return hand_id

    # ------------------------------------------------------------- 读取

    def load_phh(self, hand_id: int) -> str:
        row = self.conn.execute("SELECT phh FROM hands WHERE id = ?", (hand_id,)).fetchone()
        if row is None:
            raise KeyError(f"没有 id 为 {hand_id} 的牌局")
        return row["phh"]

    def iter_phh(self, session_id: int | None = None):
        """按手序遍历牌谱原文，用于导出或批量复盘。"""
        if session_id is None:
            rows = self.conn.execute("SELECT phh FROM hands ORDER BY id")
        else:
            rows = self.conn.execute(
                "SELECT phh FROM hands WHERE session_id = ? ORDER BY hand_no",
                (session_id,),
            )
        for row in rows:
            yield row["phh"]

    def count_hands(self, session_id: int | None = None) -> int:
        if session_id is None:
            row = self.conn.execute("SELECT COUNT(*) AS c FROM hands").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS c FROM hands WHERE session_id = ?", (session_id,)
            ).fetchone()
        return int(row["c"])

    def player_summary(self, player: str, session_id: int | None = None) -> PlayerSummary:
        """基础盈亏汇总。完整的 HUD 指标在 M2 阶段实现。"""
        clause = "" if session_id is None else " AND h.session_id = :session"
        row = self.conn.execute(
            "SELECT COUNT(*) AS hands, COALESCE(SUM(p.net), 0) AS net,"
            " COALESCE(SUM(p.showed), 0) AS showdowns,"
            " COALESCE(MAX(h.big_blind), 0) AS big_blind"
            " FROM hand_players p JOIN hands h ON h.id = p.hand_id"
            f" WHERE p.player = :player{clause}",
            {"player": player, "session": session_id},
        ).fetchone()
        return PlayerSummary(
            player=player,
            hands=int(row["hands"]),
            net=int(row["net"]),
            showdowns=int(row["showdowns"]),
            big_blind=int(row["big_blind"]),
        )

    def record_answer(
        self, *, kind: str, hero: str, villain: "str | None", hole: str,
        taken: str, best: str, frequency: float, on_solution: bool, blunder: bool,
    ) -> int:
        """记一道场景训练的答题（FR-14 测验轨的原料）。

        **判卷的结论存下来，而不是只存动作**：判卷依赖范围表，表以后会重算，
        那时旧答题按新表重判会得出不同结论——而用户看到的「我当时答对了」应该
        是当时那张表说的。存结论也让统计不必每次重新查表。
        """
        cursor = self.conn.execute(
            "INSERT INTO quiz_answers (answered_at, kind, hero, villain, hole,"
            " taken, best, frequency, on_solution, blunder)"
            " VALUES (:at, :kind, :hero, :villain, :hole, :taken, :best,"
            " :freq, :on_solution, :blunder)",
            {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kind": kind, "hero": hero, "villain": villain, "hole": hole,
                "taken": taken, "best": best, "freq": float(frequency),
                "on_solution": int(on_solution), "blunder": int(blunder),
            },
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def quiz_track(self, *, kind: "str | None" = None) -> "tuple[int, int, int]":
        """测验轨的三个计数：答了多少、照解走多少、明显错误多少。"""
        clause = "" if kind is None else " WHERE kind = :kind"
        row = self.conn.execute(
            "SELECT COUNT(*) AS answered,"
            " COALESCE(SUM(on_solution), 0) AS on_solution,"
            " COALESCE(SUM(blunder), 0) AS blunders"
            f" FROM quiz_answers{clause}",
            {"kind": kind},
        ).fetchone()
        return int(row["answered"]), int(row["on_solution"]), int(row["blunders"])

    def replay_hands(self, *, session_id: "int | None" = None, limit: "int | None" = None):
        """把落库的牌谱重放成 `HandState`，**最近的在前**。

        存的是 PHH 原文（那是唯一事实来源），所以这里走 `phh.loads` 重放，
        不另存一份解析结果——两份表示迟早会漂。

        **重放不了的那手跳过，不让它带垮整批**：一手牌谱有问题（比如手工改过）
        不该让「看看我的评级」整个失败。跳过多少条由调用方从数量差里看得出来。
        """
        clause = "" if session_id is None else " WHERE session_id = :session"
        sql = ("SELECT phh FROM hands" + clause +
               " ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else ""))
        for row in self.conn.execute(sql, {"session": session_id}).fetchall():
            try:
                yield phh_loads(row["phh"])
            except Exception:
                continue

    def player_stats(
        self, *, session_id: "int | None" = None, players: "tuple[str, ...] | None" = None,
        by_position: bool = False,
    ) -> "dict":
        """从**已落库的牌**算 PT4 口径的统计（FR-7；FR-8 的 HUD 就靠它）。

        按**玩家名**归集，不是按座位——HUD 问的是「这个对手怎么打」，
        而同一个人跨局会换座位。

        口径来自 `stats.py`，**这里一行都不重算**：数据库那条路与内存那条路
        折成同样的 `(记录序列, HandFacts)` 之后走同一个函数。两份实现各算各的，
        是统计口径漂移的头号原因（`batch.py` 那份已经在排队合并了）。

        `players` 给了就只统计这几个人**在他们参与的那些手牌里**的表现——
        注意仍要把整手牌喂进算法：3bet 的分母得知道别人加没加注。
        """
        facts_by_hand, records_by_hand, seat_names = self._hand_material(session_id)
        into: dict = {}
        wanted = set(players) if players else None

        for hand_id, facts in facts_by_hand.items():
            names = seat_names[hand_id]
            if wanted is not None and not (wanted & set(names.values())):
                continue

            def key_of(seat: int, _names=names) -> str:
                return _names.get(seat, f"seat{seat}")

            accumulate_facts(records_by_hand.get(hand_id, ()), facts, into,
                             by_position=by_position, key_of=key_of)

        if wanted is None:
            return into
        return {
            key: line for key, line in into.items()
            if (key[0] if by_position else key) in wanted
        }

    def _hand_material(self, session_id: "int | None"):
        """把一批牌读成算统计要的三样：每手的事实、动作记录、座位→玩家名。"""
        clause = "" if session_id is None else " WHERE h.session_id = :session"
        params = {"session": session_id}

        facts_rows = self.conn.execute(
            "SELECT h.id, h.num_seats, h.board FROM hands h" + clause, params
        ).fetchall()
        players_rows = self.conn.execute(
            "SELECT p.hand_id, p.seat, p.player, p.net, p.showed"
            " FROM hand_players p JOIN hands h ON h.id = p.hand_id" + clause, params
        ).fetchall()
        action_rows = self.conn.execute(
            "SELECT a.* FROM hand_actions a JOIN hands h ON h.id = a.hand_id"
            + clause + " ORDER BY a.hand_id, a.seq", params
        ).fetchall()

        seat_names: dict = {}
        nets: dict = {}
        showdowns: dict = {}
        for row in players_rows:
            hand_id = int(row["hand_id"])
            seat_names.setdefault(hand_id, {})[int(row["seat"])] = row["player"]
            nets.setdefault(hand_id, {})[int(row["seat"])] = int(row["net"])
            if int(row["showed"]):
                showdowns.setdefault(hand_id, set()).add(int(row["seat"]))

        facts = {}
        for row in facts_rows:
            hand_id = int(row["id"])
            # 牌面是**连写**的（`Qs7h2c`），不是空格分隔——用现成的解析器，
            # 别自己数字符：这里第一版就是 `.split()`，于是每手牌都被判成"没看到翻牌"，
            # WTSD 的分母整个塌成 0（对账测试当场逮到）。
            board = cards_from_str(row["board"] or "")
            facts[hand_id] = HandFacts(
                num_seats=int(row["num_seats"]),
                saw_flop=len(board) >= 3,
                showdown_seats=frozenset(showdowns.get(hand_id, ())),
                net=nets.get(hand_id, {}),
            )

        records: dict = {}
        for row in action_rows:
            records.setdefault(int(row["hand_id"]), []).append(_record_from_row(row))
        return facts, records, seat_names

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "HandStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _record_from_row(row) -> ActionRecord:
    """数据库行 → `ActionRecord`。

    `is_voluntary` **不存在表里，也不需要**：它的定义就是 `kind in (call, bet, raise)`
    （盲注根本不是一条动作记录）。存一份等于给同一个事实开第二个真相源。
    """
    return ActionRecord(
        seq=int(row["seq"]),
        street=int(row["street"]),
        seat=int(row["seat"]),
        position=row["position"],
        kind=row["kind"],
        amount=int(row["amount"]),
        to=int(row["to_amount"]),
        pot_before=int(row["pot_before"]),
        bet_before=int(row["bet_before"]),
        to_call=int(row["to_call"]),
        stack_before=int(row["stack_before"]),
        actors_before=int(row["actors_before"]),
        is_voluntary=row["kind"] in ("call", "bet", "raise"),
    )

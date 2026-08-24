"""FastAPI 服务端。

只做三件事：把 HTTP/WebSocket 请求翻译成 `TableSession` 的调用、把状态推给前端、
托管静态页面。所有牌局逻辑都在 `table.py` 与引擎里，这一层不做决策。

bot 的行动由一个后台任务逐步推进（每步之间有间隔），这样前端能一步步看到对手的动作，
而不是整条街瞬间结算完。
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from holdem import training
from holdem.actions import Action, call, check, fold, raise_to
from holdem.bots import STYLES
from holdem.cards import card_to_str
from holdem.store import HandStore

from .table import MAX_SEATS, MIN_SEATS, SeatConfig, TableConfig, TableSession

STATIC_DIR = Path(__file__).parent / "static"
BOT_STEP_DELAY = 0.55  # 秒；纯粹为了让人看清对手的动作


class SeatRequest(BaseModel):
    name: str = Field(min_length=1, max_length=24)
    isHuman: bool = False
    style: str = "tag"


class TableRequest(BaseModel):
    seats: list[SeatRequest] = Field(min_length=MIN_SEATS, max_length=MAX_SEATS)
    startingStack: int = Field(default=1000, gt=0)
    bigBlind: int = Field(default=10, gt=0)
    smallBlind: int = Field(default=5, ge=0)
    ante: int = Field(default=0, ge=0)
    seed: int | None = None


class ActionRequest(BaseModel):
    kind: str
    amount: int | None = None


class TrainingDealRequest(BaseModel):
    kind: str = "开牌"
    hero: str = "UTG"
    villain: str | None = None
    """面对开牌时是开牌者，面对再加注时是 3bet 的人；开牌场景用不到。"""


class TrainingAnswerRequest(BaseModel):
    kind: str
    to: int | None = None


class TrainingManager:
    """场景训练的当前题目（FR-12）。

    **题目存在服务端**：判卷要拿着那手牌去查表，让前端把整个牌局传回来既笨重
    又给了它篡改的机会——练习的意义就是判卷说了算。本工具是单人本地使用，
    一次一道题，不做多题并发。
    """

    def __init__(self) -> None:
        self.current = None

    def deal(self, request: TrainingDealRequest):
        from holdem.preflop_policy import DEFEND, OPEN, VS_RERAISE

        try:
            if request.kind == OPEN:
                self.current = training.deal_open(request.hero)
            elif request.kind == DEFEND:
                self.current = training.deal_defend(request.hero, request.villain or "")
            elif request.kind == VS_RERAISE:
                self.current = training.deal_threebet(request.hero, request.villain or "")
            else:
                raise HTTPException(status_code=422, detail=f"不认识的场景：{request.kind}")
        except ValueError as exc:                       # 位置名写错之类
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:                     # 造题造出了别的场景
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return self.view()

    def require(self):
        if self.current is None:
            raise HTTPException(status_code=409, detail="还没有题目，先发一道")
        return self.current

    def view(self) -> dict:
        spot = self.require()
        hand = spot.hand
        legal = hand.legal_actions()
        big_blind = hand.config.big_blind
        return {
            "label": spot.label,
            "kind": spot.spot.kind,
            "hero": {
                "position": spot.spot.hero,
                "seat": spot.hero_seat,
                "cards": [card_to_str(c) for c in hand.hole[spot.hero_seat]],
            },
            "opener": spot.spot.opener,
            "reraiser": spot.spot.reraiser,
            "pot": hand.pot_size,
            "bigBlind": big_blind,
            # 合法动作原样给出去：前端**不该自己推**加注的上下界，
            # 那是引擎的判断，猜一份出来会让界面允许打不出来的动作
            "legal": {
                "canFold": legal.can_fold,
                "canCheck": legal.can_check,
                "canCall": legal.can_call,
                "callCost": legal.call_cost,
                "callTo": legal.call_to,
                "canRaise": legal.can_raise,
                "minRaiseTo": legal.min_raise_to,
                "maxRaiseTo": legal.max_raise_to,
            },
        }

    def answer(self, request: TrainingAnswerRequest) -> dict:
        spot = self.require()
        action = _to_action(request)
        try:
            verdict = training.grade(spot.hand, action)
        except ValueError as exc:                       # 非法动作
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if verdict is None:
            # **说不了就说不了**：表里没有这一格时不编一个分数出来
            return {"graded": False, "why": "翻前范围表里没有这一格，判不了"}
        return {
            "graded": True,
            "verdict": verdict.verdict,
            "frequency": verdict.frequency,
            "best": verdict.best,
            "taken": verdict.taken,
            "blunder": verdict.blunder,
            "onSolution": verdict.on_solution,
            "weights": verdict.weights,
        }


def _to_action(request: TrainingAnswerRequest) -> Action:
    kind = request.kind
    if kind == "fold":
        return fold()
    if kind == "check":
        return check()
    if kind == "call":
        return call()
    if kind in ("bet", "raise"):
        if request.to is None:
            raise HTTPException(status_code=422, detail="加注要给 to（本街投入的目标总额）")
        return raise_to(request.to)
    raise HTTPException(status_code=422, detail=f"不认识的动作：{kind}")


class TableManager:
    """单张牌桌的持有者。本工具是单人本地使用，暂不做多桌。"""

    def __init__(
        self, db_path: str | Path = "hands.sqlite", bot_delay: float = BOT_STEP_DELAY
    ) -> None:
        self.db_path = str(db_path)
        self.bot_delay = bot_delay
        self.store: HandStore | None = None
        self.session: TableSession | None = None
        self.lock = asyncio.Lock()
        self.clients: set[WebSocket] = set()
        self._driver: asyncio.Task | None = None

    def require_session(self) -> TableSession:
        if self.session is None:
            raise HTTPException(status_code=409, detail="还没有创建牌桌")
        return self.session

    def create(self, request: TableRequest) -> TableSession:
        for seat in request.seats:
            if not seat.isHuman and seat.style not in STYLES:
                raise HTTPException(status_code=422, detail=f"未知风格: {seat.style}")
        try:
            config = TableConfig(
                seats=[
                    SeatConfig(name=s.name, is_human=s.isHuman, style=s.style)
                    for s in request.seats
                ],
                starting_stack=request.startingStack,
                big_blind=request.bigBlind,
                small_blind=request.smallBlind,
                ante=request.ante,
                seed=request.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if self.store is None:
            self.store = HandStore(self.db_path)
        session_id = self.store.create_session(
            label="网页牌桌",
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            ante=config.ante,
        )
        self.session = TableSession(
            config=config, store=self.store, store_session_id=session_id
        )
        return self.session

    async def broadcast(self) -> None:
        if self.session is None:
            return
        payload = self.session.view()
        dead = []
        for client in list(self.clients):
            try:
                await client.send_json(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)

    async def drive_bots(self) -> None:
        """逐步推进 bot 的行动，每步之间留出间隔并广播。"""
        session = self.session
        if session is None:
            return
        while True:
            async with self.lock:
                if session is not self.session:
                    return
                if not session.step_bot():
                    break
            await self.broadcast()
            if self.bot_delay:
                await asyncio.sleep(self.bot_delay)
            else:
                await asyncio.sleep(0)  # 让出事件循环，避免占满
        await self.broadcast()

    def schedule_bots(self) -> None:
        if self._driver is not None and not self._driver.done():
            return
        self._driver = asyncio.create_task(self.drive_bots())

    async def shutdown(self) -> None:
        if self._driver is not None:
            self._driver.cancel()
            with suppress(asyncio.CancelledError):
                await self._driver
        if self.store is not None:
            self.store.close()
            self.store = None


def create_app(
    db_path: str | Path | None = None, bot_delay: float | None = None
) -> FastAPI:
    db_path = db_path or os.environ.get("HOLDEM_DB", "hands.sqlite")
    if bot_delay is None:
        bot_delay = float(os.environ.get("HOLDEM_BOT_DELAY", BOT_STEP_DELAY))
    manager = TableManager(db_path, bot_delay)
    trainer = TrainingManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await manager.shutdown()

    app = FastAPI(title="德扑训练台", version="0.1.0", lifespan=lifespan)
    app.state.manager = manager

    @app.get("/api/styles")
    def list_styles() -> dict:
        return {
            "styles": [
                {"id": style.name, "label": style.label} for style in STYLES.values()
            ]
        }

    @app.post("/api/table")
    async def create_table(request: TableRequest) -> dict:
        async with manager.lock:
            session = manager.create(request)
        await manager.broadcast()
        return session.view()

    @app.get("/api/state")
    async def get_state() -> dict:
        return manager.require_session().view()

    @app.get("/api/training/scenarios")
    def training_scenarios() -> dict:
        """有哪些场景可练。位置表来自引擎，不在前端写死。"""
        from holdem.positions import position_names
        from holdem.preflop_policy import DEFEND, OPEN, VS_RERAISE

        return {
            "positions": list(position_names(6)),
            "kinds": [
                {"id": OPEN, "label": "第一个开牌", "needsVillain": False},
                {"id": DEFEND, "label": "面对开牌", "needsVillain": True,
                 "villainLabel": "开牌者"},
                {"id": VS_RERAISE, "label": "开牌后面对再加注", "needsVillain": True,
                 "villainLabel": "3bet 的人"},
            ],
        }

    @app.post("/api/training/deal")
    def training_deal(request: TrainingDealRequest) -> dict:
        return trainer.deal(request)

    @app.get("/api/training/current")
    def training_current() -> dict:
        return trainer.view()

    @app.post("/api/training/answer")
    def training_answer(request: TrainingAnswerRequest) -> dict:
        return trainer.answer(request)

    @app.get("/api/hud")
    async def get_hud(scope: str = "session") -> dict:
        """牌桌浮层的统计（FR-8）。

        **不塞进 `/api/state` 的广播里**：那条每次动作都推，而统计一手牌才变一次；
        跟着广播走等于把一份基本不动的数据推几十遍，还让每次动作的延迟受它拖累。
        前端在开局与每手结束后拉一次就够。
        """
        if scope not in ("session", "all"):
            raise HTTPException(status_code=422, detail="scope 只能是 session 或 all")
        return manager.require_session().hud(scope=scope)

    @app.post("/api/hand")
    async def start_hand() -> dict:
        async with manager.lock:
            session = manager.require_session()
            try:
                session.start_hand()
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            view = session.view()
        await manager.broadcast()
        manager.schedule_bots()
        return view

    @app.post("/api/action")
    async def act(request: ActionRequest) -> dict:
        async with manager.lock:
            session = manager.require_session()
            try:
                session.apply_human(request.kind, request.amount)
            except (RuntimeError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            view = session.view()
        await manager.broadcast()
        manager.schedule_bots()
        return view

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await ws.accept()
        manager.clients.add(ws)
        try:
            if manager.session is not None:
                await ws.send_json(manager.session.view())
            while True:
                await ws.receive_text()  # 客户端只用来保活
        except WebSocketDisconnect:
            pass
        finally:
            manager.clients.discard(ws)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()

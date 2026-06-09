from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.api import actions, config_api, health, history, sessions, stream
from backend.config import STATE_DB, load_config
from backend.detectors.filesystem_watch import FilesystemWatcher
from backend.detectors.iterm_cache import ItermLocationCache
from backend.detectors.linker import LinkerState, build_sessions
from backend.detectors.tmux_cache import TmuxLocationCache
from backend.models import ClaudeSession
from backend.state import State

log = logging.getLogger("claudewatch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@dataclass
class AppState:
    config: dict[str, Any]
    sessions: dict[int, ClaudeSession] = field(default_factory=dict)
    sessions_started_at: dict[int, Any] = field(default_factory=dict)
    linker_state: LinkerState = field(default_factory=LinkerState)
    fs_watcher: FilesystemWatcher | None = None
    iterm_cache: ItermLocationCache | None = None
    tmux_cache: TmuxLocationCache | None = None
    state: State | None = None
    last_sig: dict[int, str] = field(default_factory=dict)
    sse_queues: set[asyncio.Queue] = field(default_factory=set)

    async def broadcast(self, event: dict) -> None:
        dead = []
        for q in list(self.sse_queues):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for d in dead:
            self.sse_queues.discard(d)


# Fields that change almost every scan tick but aren't worth pushing to clients on their
# own (CPU/mem jitter, ever-incrementing timers). Excluding them from the change signature
# means an otherwise-idle session stops emitting an SSE update every 2s. The frontend keeps
# elapsed/"active Xs ago" counters ticking client-side, and any *real* change (status,
# tokens, task, context, in-flight) still flips the signature and broadcasts.
_VOLATILE_FIELDS = frozenset(
    {"cpu_percent", "memory_mb", "duration_seconds", "current_task_elapsed_seconds", "last_activity_at"}
)


def _significant_signature(dump: dict) -> str:
    return json.dumps(
        {k: v for k, v in dump.items() if k not in _VOLATILE_FIELDS},
        sort_keys=True,
        default=str,
    )


async def _scheduler_loop(s: AppState) -> None:
    interval = float(s.config.get("process_scan_interval_seconds", 2))
    while True:
        try:
            new_sessions = await build_sessions(
                s.config,
                s.linker_state,
                s.fs_watcher,
                iterm_cache=s.iterm_cache,
                tmux_cache=s.tmux_cache,
            )
            prev = s.sessions
            new_map = {x.pid: x for x in new_sessions}

            # Detect new + ended + persist. Only broadcast an update when something
            # meaningful changed (see _significant_signature) — avoids pushing a full
            # snapshot of every session every tick.
            for pid, sess in new_map.items():
                dump = sess.model_dump(mode="json")
                if pid not in prev:
                    await s.broadcast({"event": "session.started", "session": dump})
                    s.sessions_started_at[pid] = sess.started_at
                    s.last_sig[pid] = _significant_signature(dump)
                else:
                    sig = _significant_signature(dump)
                    if sig != s.last_sig.get(pid):
                        await s.broadcast({"event": "session.updated", "session": dump})
                        s.last_sig[pid] = sig
                if s.state:
                    await s.state.upsert_active(sess)

            for pid in list(prev.keys()):
                if pid not in new_map:
                    await s.broadcast({"event": "session.ended", "pid": pid})
                    s.last_sig.pop(pid, None)
                    if s.state:
                        started = s.sessions_started_at.pop(pid, prev[pid].started_at)
                        await s.state.mark_ended(pid, started)

            s.sessions = new_map

            if s.fs_watcher:
                cwds = {x.cwd for x in new_sessions if x.cwd}
                await s.fs_watcher.sync_active_cwds(cwds)
        except Exception as e:  # noqa: BLE001
            log.exception("scheduler iteration failed: %s", e)
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    state = State(STATE_DB)
    await state.init_db()
    await state.prune()
    fs_watcher = FilesystemWatcher(
        retention_minutes=int(cfg.get("file_change_retention_minutes", 10)),
        ignore_patterns=cfg.get("ignore_patterns", []),
    )
    iterm_cache = ItermLocationCache(
        refresh_interval=float(cfg.get("iterm_refresh_interval_seconds", 5)),
    )
    tmux_cache = TmuxLocationCache(
        refresh_interval=float(cfg.get("tmux_refresh_interval_seconds", 5)),
    )
    s = AppState(
        config=cfg,
        state=state,
        fs_watcher=fs_watcher,
        iterm_cache=iterm_cache,
        tmux_cache=tmux_cache,
    )
    app.state.s = s
    await iterm_cache.start()
    await tmux_cache.start()
    task = asyncio.create_task(_scheduler_loop(s))
    log.info("ClaudeWatch backend started on http://127.0.0.1:%d", int(cfg.get("port", 7788)))
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await iterm_cache.stop()
        await tmux_cache.stop()
        await fs_watcher.stop_all()


def create_app() -> FastAPI:
    app = FastAPI(title="ClaudeWatch", version="0.2.0", lifespan=lifespan)

    @app.middleware("http")
    async def _no_cache_static(request, call_next):
        # The dashboard's HTML/JS/CSS are tiny and local; never let the browser serve a
        # stale copy (a cached app.js against a newer index.html breaks the page).
        resp = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static"):
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp

    app.include_router(sessions.router)
    app.include_router(actions.router)
    app.include_router(stream.router)
    app.include_router(health.router)
    app.include_router(history.router)
    app.include_router(config_api.router)

    if FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def index():
        path = FRONTEND_DIR / "index.html"
        if not path.is_file():
            return {"message": "ClaudeWatch backend running. Frontend not yet built."}
        html = path.read_text(encoding="utf-8")
        # Cache-bust static assets with their mtime so a freshly-served index.html can never
        # pair with a stale cached app.js/styles.css — a mismatch silently breaks methods
        # like activeCount()/sessionTitle() and leaves blank stats / missing titles.
        try:
            ver = int(max(
                (FRONTEND_DIR / f).stat().st_mtime
                for f in ("app.js", "styles.css", "index.html")
            ))
        except OSError:
            ver = 0
        html = html.replace("/static/app.js", f"/static/app.js?v={ver}")
        html = html.replace("/static/styles.css", f"/static/styles.css?v={ver}")
        return HTMLResponse(html)

    return app


app = create_app()

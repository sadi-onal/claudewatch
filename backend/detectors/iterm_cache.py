"""Background snapshot of iTerm session locations — AppleScript-only mode.

History: Two prior approaches (per-tick websocket churn, single persistent
websocket) both correlated with iTerm UI freezes that required force-quit. The
Python API path was retired on 2026-05-14 in favor of a single `osascript`
enumeration every `iterm_refresh_interval_seconds` (default 30s). One main-
thread iTerm call every half-minute gives iTerm long quiet periods, and there
are no long-lived websocket dispatcher tasks to leak. See memory note
`iterm-freeze-root-cause` for context.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable

import psutil

from backend.detectors.iterm_applescript import (
    ItermSessionTty,
    list_iterm_sessions_via_applescript,
)

log = logging.getLogger(__name__)


@dataclass
class ItermLocation:
    window_id: int
    tab_id: int | None  # always None on the AppleScript path
    tab_index: int | None
    session_id: str
    tab_title: str
    tty: str | None


def _ancestor_ttys(pid: int, max_depth: int = 12) -> list[str]:
    out: list[str] = []
    try:
        cur = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return out
    for _ in range(max_depth):
        try:
            t = cur.terminal()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if t and t not in out:
            out.append(t)
        try:
            parent = cur.parent()
            if parent is None:
                break
            cur = parent
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
    return out


class ItermLocationCache:
    def __init__(
        self,
        refresh_interval: float = 30.0,
        query_timeout: float = 3.0,
    ) -> None:
        self.refresh_interval = max(5.0, float(refresh_interval))
        self.query_timeout = query_timeout

        self._sessions: list[ItermSessionTty] = []
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        await self._refresh_once()
        self._task = asyncio.create_task(self._run(), name="iterm-cache")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    def get_locations(self, pids: Iterable[int]) -> dict[int, ItermLocation]:
        pids = list(pids)
        if not pids or not self._sessions:
            return {}
        tty_map: dict[str, ItermSessionTty] = {
            s.tty: s for s in self._sessions if s.tty and s.tty != "?"
        }
        if not tty_map:
            return {}
        out: dict[int, ItermLocation] = {}
        for pid in pids:
            for tty in _ancestor_ttys(pid):
                if tty in tty_map:
                    s = tty_map[tty]
                    out[pid] = ItermLocation(
                        window_id=s.window_id,
                        tab_id=None,
                        tab_index=s.tab_index,
                        session_id=s.unique_id,
                        tab_title=s.name,
                        tty=s.tty,
                    )
                    break
        return out

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.refresh_interval)
                return
            except asyncio.TimeoutError:
                pass
            await self._refresh_once()

    async def _refresh_once(self) -> None:
        try:
            self._sessions = await asyncio.to_thread(
                list_iterm_sessions_via_applescript, self.query_timeout
            )
        except Exception as e:  # noqa: BLE001
            log.debug("iterm_cache: applescript refresh failed: %s", e)
            self._sessions = []

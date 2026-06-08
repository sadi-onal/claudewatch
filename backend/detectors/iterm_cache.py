"""Background snapshot of iTerm session locations — AppleScript-only, demand-driven.

History: per-tick websocket churn and a single persistent websocket (Python API)
both correlated with iTerm UI freezes. The Python API was retired on 2026-05-14.
But a *timer-driven* `osascript` enumeration (every N seconds, unconditionally)
still wedged iTerm: the full window→tab→session walk runs on iTerm's main thread,
and when it ran long the 3s subprocess timeout SIGKILL'd osascript mid-flight,
orphaning the in-flight Apple Event. Repeated every cycle, iTerm's main thread
progressively locked up → "not responding". See memory `iterm-freeze-root-cause`.

This version makes enumeration **demand-driven + circuit-broken**:
  * A session's tab/tty never changes, so once a pid is located it is cached for
    its lifetime and never re-enumerated.
  * Enumeration only happens when there is at least one *unlocated* pid (a new
    session). In steady state — all sessions located, or none running — iTerm is
    never touched.
  * Pids that can't be placed (Terminal.app, etc.) are given up on after a few
    attempts so they don't keep triggering enumeration.
  * A circuit breaker opens after a single enumeration failure (hang/timeout) and
    stays open for a cooldown, so a slow/wedged iTerm is never hammered.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import psutil

from backend.detectors.iterm_applescript import (
    ItermQueryError,
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
        query_timeout: float = 4.0,
        failure_threshold: int = 1,
        breaker_cooldown: float = 600.0,
        give_up_after: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.refresh_interval = max(5.0, float(refresh_interval))
        self.query_timeout = query_timeout
        self.failure_threshold = max(1, int(failure_threshold))
        self.breaker_cooldown = float(breaker_cooldown)
        self.give_up_after = max(1, int(give_up_after))
        self._clock = clock

        # Latest enumeration snapshot (transient — only used to match pending pids).
        self._sessions: list[ItermSessionTty] = []
        # Permanent per-pid results (pruned to live pids in get_locations).
        self._located: dict[int, ItermLocation] = {}
        self._attempts: dict[int, int] = {}
        self._gaveup: set[int] = set()
        self._pending: set[int] = set()

        # Circuit breaker
        self._consecutive_failures = 0
        self._breaker_until = 0.0

        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._wake: asyncio.Event | None = None

    # --- lifecycle ---

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="iterm-cache")

    async def stop(self) -> None:
        self._stop.set()
        if self._wake is not None:
            self._wake.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    # --- matching (no iTerm I/O) ---

    def _match(self, pid: int) -> ItermLocation | None:
        tty_map: dict[str, ItermSessionTty] = {
            s.tty: s for s in self._sessions if s.tty and s.tty != "?"
        }
        if not tty_map:
            return None
        for tty in _ancestor_ttys(pid):
            s = tty_map.get(tty)
            if s:
                return ItermLocation(
                    window_id=s.window_id,
                    tab_id=None,
                    tab_index=s.tab_index,
                    session_id=s.unique_id,
                    tab_title=s.name,
                    tty=s.tty,
                )
        return None

    def get_locations(self, pids: Iterable[int]) -> dict[int, ItermLocation]:
        pids = list(pids)
        alive = set(pids)
        # Prune all per-pid bookkeeping to processes that still exist.
        self._located = {p: loc for p, loc in self._located.items() if p in alive}
        self._attempts = {p: n for p, n in self._attempts.items() if p in alive}
        self._gaveup &= alive

        out: dict[int, ItermLocation] = {}
        pending: set[int] = set()
        for pid in pids:
            if pid in self._located:
                out[pid] = self._located[pid]
                continue
            if pid in self._gaveup:
                continue
            loc = self._match(pid)
            if loc is not None:
                self._located[pid] = loc
                out[pid] = loc
            else:
                pending.add(pid)
        self._pending = pending

        # Nudge the background loop to enumerate promptly for new sessions, but only
        # when it's allowed to (breaker closed). Never blocks iTerm from here.
        if pending and self._wake is not None and not self._breaker_open():
            self._wake.set()
        return out

    # --- circuit breaker ---

    def _breaker_open(self) -> bool:
        return self._clock() < self._breaker_until

    def should_enumerate(self) -> bool:
        return bool(self._pending) and not self._breaker_open()

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._breaker_until = self._clock() + self.breaker_cooldown
            log.warning(
                "iterm_cache: enumeration failed; circuit breaker open for %.0fs",
                self.breaker_cooldown,
            )

    def _record_success(self, sessions: list[ItermSessionTty]) -> None:
        self._consecutive_failures = 0
        self._breaker_until = 0.0
        self._sessions = sessions
        for pid in list(self._pending):
            loc = self._match(pid)
            if loc is not None:
                self._located[pid] = loc
            else:
                self._attempts[pid] = self._attempts.get(pid, 0) + 1
                if self._attempts[pid] >= self.give_up_after:
                    self._gaveup.add(pid)
        self._pending = {
            p for p in self._pending if p not in self._located and p not in self._gaveup
        }

    # --- background loop ---

    async def _refresh_once(self) -> None:
        try:
            sessions = await asyncio.to_thread(
                list_iterm_sessions_via_applescript, self.query_timeout
            )
        except ItermQueryError as e:
            log.debug("iterm_cache: enumeration failed: %s", e)
            self._record_failure()
            return
        except Exception as e:  # noqa: BLE001
            log.debug("iterm_cache: unexpected enumeration error: %s", e)
            self._record_failure()
            return
        self._record_success(sessions)

    async def _run(self) -> None:
        assert self._wake is not None
        while not self._stop.is_set():
            if self.should_enumerate():
                await self._refresh_once()
            self._wake.clear()
            stop_task = asyncio.ensure_future(self._stop.wait())
            wake_task = asyncio.ensure_future(self._wake.wait())
            try:
                done, undone = await asyncio.wait(
                    {stop_task, wake_task},
                    timeout=self.refresh_interval,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                stop_task.cancel()
                wake_task.cancel()

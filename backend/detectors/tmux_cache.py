"""Background snapshot of tmux panes, mirrored on the iTerm cache pattern.

tmux polling is far cheaper than iTerm (no UI thread to block), but the same
decoupling is applied for consistency and so that `tmux_refresh_interval_seconds`
from config has the effect it implies.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable

from backend.detectors.tmux_detector import (
    TmuxLocation,
    TmuxPane,
    _descendants,
    list_tmux_panes,
)

log = logging.getLogger(__name__)


class TmuxLocationCache:
    def __init__(self, refresh_interval: float = 5.0) -> None:
        self.refresh_interval = max(1.0, float(refresh_interval))
        self._panes: list[TmuxPane] = []
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        await self._refresh_once()
        self._task = asyncio.create_task(self._run(), name="tmux-cache")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    def get_locations(self, pids: Iterable[int]) -> dict[int, TmuxLocation]:
        pids = list(pids)
        if not pids or not self._panes:
            return {}
        out: dict[int, TmuxLocation] = {}
        pid_set = set(pids)
        for pane in self._panes:
            kids = _descendants(pane.pane_pid)
            for pid in pid_set & kids:
                out[pid] = TmuxLocation(
                    session=pane.session, window=pane.window, pane=pane.pane
                )
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
            self._panes = await asyncio.to_thread(list_tmux_panes)
        except Exception as e:  # noqa: BLE001
            log.debug("tmux_cache: refresh failed: %s", e)
            self._panes = []

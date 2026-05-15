from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import psutil


@dataclass
class TmuxPane:
    session: str
    window: str
    pane: str
    pane_pid: int
    current_command: str
    current_path: str


@dataclass
class TmuxLocation:
    session: str
    window: str
    pane: str


_TMUX_FORMAT = (
    "#{session_name}|#{window_index}|#{pane_index}|"
    "#{pane_pid}|#{pane_current_command}|#{pane_current_path}"
)


def tmux_available() -> bool:
    return shutil.which("tmux") is not None


def list_tmux_panes() -> list[TmuxPane]:
    if not tmux_available():
        return []
    try:
        result = subprocess.run(
            ["tmux", "list-panes", "-a", "-F", _TMUX_FORMAT],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    if result.returncode != 0:
        return []
    panes: list[TmuxPane] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        try:
            panes.append(
                TmuxPane(
                    session=parts[0],
                    window=parts[1],
                    pane=parts[2],
                    pane_pid=int(parts[3]),
                    current_command=parts[4],
                    current_path=parts[5],
                )
            )
        except ValueError:
            continue
    return panes


def _descendants(root_pid: int, max_depth: int = 10) -> set[int]:
    try:
        proc = psutil.Process(root_pid)
    except psutil.NoSuchProcess:
        return set()
    seen: set[int] = {root_pid}
    frontier: list[psutil.Process] = [proc]
    depth = 0
    while frontier and depth < max_depth:
        next_frontier: list[psutil.Process] = []
        for p in frontier:
            try:
                for child in p.children():
                    if child.pid not in seen:
                        seen.add(child.pid)
                        next_frontier.append(child)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        frontier = next_frontier
        depth += 1
    return seen



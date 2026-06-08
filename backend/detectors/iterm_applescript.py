"""AppleScript-based iTerm session enumeration.

Plain-AppleScript fallback for when the iTerm2 Python API is unreachable
(iTerm not running, Python API disabled, websocket churn, etc.). Used by
[iterm_cache.py](iterm_cache.py); the pid → location matching itself lives in
that cache.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

APPLESCRIPT_DIR = Path(__file__).resolve().parent.parent / "applescript"
LIST_SCRIPT = APPLESCRIPT_DIR / "list_iterm_sessions.applescript"


class ItermQueryError(Exception):
    """Enumeration did not complete (timeout, osascript error, or iTerm unreachable).

    Raised — rather than returning [] — so the caller's circuit breaker can tell a real
    failure (iTerm hanging) apart from a genuine empty result (iTerm has no sessions).
    """


@dataclass
class ItermSessionTty:
    window_id: int
    tab_index: int
    tty: str
    unique_id: str
    name: str


def list_iterm_sessions_via_applescript(timeout: float = 4.0) -> list[ItermSessionTty]:
    """Return iTerm sessions enumerated via plain AppleScript.

    Raises ItermQueryError on timeout / osascript failure. On timeout the osascript
    child is stopped with SIGTERM first (so it can tear down its in-flight Apple Event
    to iTerm) and only SIGKILL'd as a last resort — a hard SIGKILL mid-enumeration is
    what previously orphaned Apple Events and wedged iTerm's main thread.
    """
    try:
        proc = subprocess.Popen(
            ["osascript", str(LIST_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as e:
        raise ItermQueryError("osascript not found") from e
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        proc.terminate()  # SIGTERM — let osascript cancel its Apple Event cleanly
        try:
            proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        raise ItermQueryError(f"osascript timed out after {timeout}s") from e
    if proc.returncode != 0:
        raise ItermQueryError(f"osascript returned {proc.returncode}: {stderr.strip()}")
    r_stdout = stdout
    out: list[ItermSessionTty] = []
    for line in r_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 4)
        if len(parts) < 5:
            continue
        try:
            window_id = int(parts[0])
            tab_index = int(parts[1])
        except ValueError:
            continue
        out.append(
            ItermSessionTty(
                window_id=window_id,
                tab_index=tab_index,
                tty=parts[2].strip(),
                unique_id=parts[3].strip(),
                name=parts[4].strip(),
            )
        )
    return out

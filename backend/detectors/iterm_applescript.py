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


@dataclass
class ItermSessionTty:
    window_id: int
    tab_index: int
    tty: str
    unique_id: str
    name: str


def list_iterm_sessions_via_applescript(timeout: float = 3.0) -> list[ItermSessionTty]:
    """Return iTerm sessions enumerated via plain AppleScript. Returns [] on error."""
    try:
        r = subprocess.run(
            ["osascript", str(LIST_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        log.debug("osascript list failed: %s", e)
        return []
    if r.returncode != 0:
        log.debug("osascript returned %d: %s", r.returncode, r.stderr.strip())
        return []
    out: list[ItermSessionTty] = []
    for line in r.stdout.splitlines():
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

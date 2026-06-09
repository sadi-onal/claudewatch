from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend.models import NewSessionRequest

router = APIRouter(prefix="/api")
log = logging.getLogger(__name__)

APPLESCRIPT_DIR = Path(__file__).resolve().parent.parent / "applescript"

# A flag is `--name` or `--name=value`. Names: lowercase + digits + dash, must start lowercase.
SAFE_FLAG_RE = re.compile(r"^--[a-z][a-z0-9-]*(=[A-Za-z0-9._/=:,@-]+)?$")
VALUE_FLAG_NAMES = {
    "--model",
    "--system-prompt-file",
    "--print",
    "--mcp-config",
    "--settings",
    "--add-dir",
    "--permission-mode",
    "--allowed-tools",
    "--disallowed-tools",
    "--resume",
    "--session-id",
    "--agent",
}
UNSAFE_VALUE_CHARS = (";", "&", "|", "`", "$", "\n", "\r", ">", "<", "\\", "\x00")


def _state(request: Request):
    return request.app.state.s


def _check_read_only(request: Request) -> None:
    if request.app.state.s.config.get("read_only"):
        raise HTTPException(403, "server is in read-only mode")


def sanitize_new_session(body: NewSessionRequest) -> tuple[str, list[str]]:
    cwd = Path(body.cwd).expanduser()
    try:
        cwd_resolved = cwd.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise HTTPException(400, f"cwd does not exist: {body.cwd}")
    if not cwd_resolved.is_dir():
        raise HTTPException(400, "cwd is not a directory")
    home = Path.home().resolve()
    if home not in cwd_resolved.parents and cwd_resolved != home:
        raise HTTPException(400, "cwd must be under the user's home directory")

    # Command path
    if body.command == "claude":
        cmd_str = "claude"
    else:
        try:
            cmd_path = Path(body.command).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError):
            raise HTTPException(400, "command not found")
        allowed_prefixes = [home / ".local" / "bin", home / "Library" / "Application Support" / "Claude"]
        if not any(str(cmd_path).startswith(str(p)) for p in allowed_prefixes):
            raise HTTPException(400, "command must live under ~/.local/bin or Claude support dir")
        cmd_str = str(cmd_path)

    i = 0
    out_flags: list[str] = []
    while i < len(body.flags):
        f = body.flags[i]
        if not SAFE_FLAG_RE.match(f):
            raise HTTPException(400, f"unsafe flag: {f}")
        out_flags.append(f)
        # Value-taking flag → swallow next token as value
        flag_name = f.split("=", 1)[0]
        if flag_name in VALUE_FLAG_NAMES and "=" not in f:
            if i + 1 >= len(body.flags):
                raise HTTPException(400, f"flag {f} requires value")
            v = body.flags[i + 1]
            if any(c in v for c in UNSAFE_VALUE_CHARS):
                raise HTTPException(400, f"unsafe flag value for {f}")
            out_flags.append(v)
            i += 2
        else:
            i += 1
    return str(cwd_resolved), [cmd_str, *out_flags]


@router.post("/sessions/new")
async def new_session(body: NewSessionRequest, request: Request):
    _check_read_only(request)
    cwd, argv = sanitize_new_session(body)
    cmd_str = shlex.join(argv)
    script_name = (
        "new_iterm_window.applescript"
        if body.window_type == "new-window"
        else "new_iterm_tab.applescript"
    )
    script_path = APPLESCRIPT_DIR / script_name
    try:
        subprocess.run(
            ["osascript", str(script_path), cwd, cmd_str],
            check=True,
            timeout=10,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        log.error("AppleScript failed: %s", e.stderr)
        raise HTTPException(500, f"AppleScript failed: {e.stderr.strip() or e}")
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "AppleScript timed out")
    return {"success": True, "cwd": cwd, "command": cmd_str}


@router.post("/sessions/{pid}/halt")
async def halt(pid: int, request: Request):
    _check_read_only(request)
    s = _state(request)
    if pid not in s.sessions:
        raise HTTPException(404, "session not found")
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        raise HTTPException(404, "process not running")
    except PermissionError:
        raise HTTPException(403, "permission denied")
    # Wait up to 5s for the process to exit
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return {"success": True, "exited": True}
        await asyncio.sleep(0.2)  # async — don't block the event loop while polling
    return {"success": True, "exited": False, "note": "SIGINT sent; process still running"}


@router.get("/iterm/current")
async def iterm_current():
    """Unique id of the frontmost iTerm session — used by the dashboard's 'find my
    session' action to highlight the tab you're in. A single light AppleScript query
    (no window enumeration), run only on demand."""
    # `window 1` (frontmost iTerm window) rather than `current window`: the latter errors
    # (-1728) when iTerm isn't the active app — which it isn't while you're viewing this
    # dashboard in a browser. window 1 resolves to the last-focused iTerm window/tab.
    script = (
        'tell application "iTerm" to tell window 1 '
        "to tell current tab to tell current session to id"
    )
    try:
        r = await asyncio.to_thread(
            subprocess.run,
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {"current": None}
    sid = r.stdout.strip() if r.returncode == 0 else ""
    return {"current": sid or None}


@router.post("/sessions/{pid}/focus")
async def focus(pid: int, request: Request):
    _check_read_only(request)
    s = _state(request)
    sess = s.sessions.get(pid)
    if not sess:
        raise HTTPException(404, "session not found")
    if sess.location_type == "headless":
        raise HTTPException(400, "headless session has no window to focus")

    try:
        if sess.iterm_tty:
            r = subprocess.run(
                [
                    "osascript",
                    str(APPLESCRIPT_DIR / "focus_by_tty.applescript"),
                    sess.iterm_tty,
                ],
                check=True,
                timeout=5,
                capture_output=True,
                text=True,
            )
            if r.stdout.strip() == "not_found":
                raise HTTPException(404, f"iTerm session for {sess.iterm_tty} not found")
        elif sess.iterm_window_id and sess.iterm_tab_id:
            subprocess.run(
                [
                    "osascript",
                    str(APPLESCRIPT_DIR / "focus_iterm.applescript"),
                    str(sess.iterm_window_id),
                    str(sess.iterm_tab_id),
                ],
                check=True,
                timeout=5,
                capture_output=True,
                text=True,
            )
        if sess.location_type == "tmux" and sess.tmux_session is not None:
            subprocess.run(
                ["tmux", "select-window", "-t", f"{sess.tmux_session}:{sess.tmux_window}"],
                check=True,
                timeout=3,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "tmux",
                    "select-pane",
                    "-t",
                    f"{sess.tmux_session}:{sess.tmux_window}.{sess.tmux_pane}",
                ],
                check=True,
                timeout=3,
                capture_output=True,
                text=True,
            )
    except subprocess.CalledProcessError as e:
        log.error("focus failed: %s", e.stderr)
        raise HTTPException(500, f"focus failed: {e.stderr.strip() or e}")
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "focus action timed out")
    return {"success": True}

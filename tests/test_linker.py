from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import pytest

from backend.config import DEFAULT_CONFIG
from backend.detectors.iterm_cache import ItermLocation
from backend.detectors.linker import LinkerState, build_sessions
from backend.detectors.process_detector import ProcInfo
from backend.detectors.tmux_detector import TmuxLocation

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@dataclass
class FakeItermCache:
    """Stand-in for ItermLocationCache in tests — get_locations returns the fixture map."""

    locations: dict[int, ItermLocation] = field(default_factory=dict)

    def get_locations(self, pids: Iterable[int]) -> dict[int, ItermLocation]:
        pids = set(pids)
        return {p: loc for p, loc in self.locations.items() if p in pids}


@dataclass
class FakeTmuxCache:
    locations: dict[int, TmuxLocation] = field(default_factory=dict)

    def get_locations(self, pids: Iterable[int]) -> dict[int, TmuxLocation]:
        pids = set(pids)
        return {p: loc for p, loc in self.locations.items() if p in pids}


def _proc(
    pid: int,
    cwd: str,
    model: str | None = None,
    session_id: str | None = None,
    env_session_id: str | None = None,
) -> ProcInfo:
    cmdline = ["claude"]
    if model:
        cmdline += ["--model", model]
    if session_id:
        cmdline += ["--resume", session_id]
    return ProcInfo(
        pid=pid,
        ppid=1,
        cwd=cwd,
        started_at=datetime(2026, 5, 12, 10, 0, 0, tzinfo=timezone.utc),
        cpu_percent=0.0,
        memory_mb=200.0,
        cmdline=cmdline,
        cmdline_parsed={"model": model, "permission_mode_flag": None, "session_id": session_id, "extra_flags": []},
        env_session_id=env_session_id,
    )


@pytest.fixture
def isolated_log_dir(tmp_path, monkeypatch):
    """Build a fake ~/.claude/projects layout under tmp_path."""
    log_dir = tmp_path / "projects"
    cwd = "/tmp/fakecwd"  # outside home, but we patch find_logs_for_cwd via log_dir override
    target = log_dir / "-tmp-fakecwd"
    target.mkdir(parents=True)
    src = FIXTURE_DIR / "multi_assistant.jsonl"
    # Copy with a session-uuid filename so the linker can find it by sessionId
    (target / "sess-A.jsonl").write_bytes(src.read_bytes())
    yield log_dir, cwd


async def test_build_sessions_with_log_match(isolated_log_dir):
    log_dir, cwd = isolated_log_dir
    procs = [_proc(pid=9999, cwd=cwd, session_id="sess-A")]
    state = LinkerState()
    state.log_dir = log_dir

    with patch("backend.detectors.linker.scan_claude_processes", return_value=procs):
        sessions = await build_sessions(
            DEFAULT_CONFIG, state, iterm_cache=FakeItermCache(), tmux_cache=FakeTmuxCache()
        )

    assert len(sessions) == 1
    s = sessions[0]
    assert s.pid == 9999
    assert s.cwd == cwd
    assert s.model == "claude-opus-4-7"
    assert s.conversation_id == "sess-A"
    assert s.message_count == 6
    assert s.usage is not None
    assert s.usage.input_tokens == 135  # from multi_assistant fixture
    assert s.usage.output_tokens == 73
    assert s.usage.cache_read_input_tokens == 300
    assert s.tool_calls.total == 4
    assert s.tool_calls.breakdown == {"Edit": 2, "Bash": 2}
    assert s.permission_mode == "auto"
    assert s.thinking_enabled is True
    assert s.usage.cost_estimate_usd is not None  # opus-4-7 priced


async def test_build_sessions_with_tmux_location(isolated_log_dir):
    log_dir, cwd = isolated_log_dir
    procs = [_proc(pid=1234, cwd=cwd)]
    state = LinkerState()
    state.log_dir = log_dir
    tmux_cache = FakeTmuxCache(locations={1234: TmuxLocation(session="main", window="0", pane="1")})

    with patch("backend.detectors.linker.scan_claude_processes", return_value=procs):
        sessions = await build_sessions(
            DEFAULT_CONFIG, state, iterm_cache=FakeItermCache(), tmux_cache=tmux_cache
        )

    s = sessions[0]
    assert s.location_type == "tmux"
    assert s.tmux_session == "main"
    assert s.tmux_pane == "1"


async def test_build_sessions_with_iterm_python_api_location(isolated_log_dir):
    """Python API path: tab_id populated, tab_index/tty are None."""
    log_dir, cwd = isolated_log_dir
    procs = [_proc(pid=5555, cwd=cwd)]
    state = LinkerState()
    state.log_dir = log_dir
    iterm_cache = FakeItermCache(
        locations={
            5555: ItermLocation(
                window_id=42,
                tab_id=7,
                tab_index=None,
                session_id="abc-123",
                tab_title="claude",
                tty=None,
            )
        }
    )

    with patch("backend.detectors.linker.scan_claude_processes", return_value=procs):
        sessions = await build_sessions(
            DEFAULT_CONFIG, state, iterm_cache=iterm_cache, tmux_cache=FakeTmuxCache()
        )

    s = sessions[0]
    assert s.location_type == "iterm"
    assert s.iterm_window_id == 42
    assert s.iterm_tab_id == 7
    assert s.iterm_tab_index is None
    assert s.iterm_session_id == "abc-123"
    assert s.iterm_tty is None


async def test_build_sessions_with_iterm_applescript_location(isolated_log_dir):
    """AppleScript fallback path: tab_index + tty populated, tab_id is None."""
    log_dir, cwd = isolated_log_dir
    procs = [_proc(pid=6666, cwd=cwd)]
    state = LinkerState()
    state.log_dir = log_dir
    iterm_cache = FakeItermCache(
        locations={
            6666: ItermLocation(
                window_id=99,
                tab_id=None,
                tab_index=2,
                session_id="xyz-456",
                tab_title="zsh",
                tty="/dev/ttys007",
            )
        }
    )

    with patch("backend.detectors.linker.scan_claude_processes", return_value=procs):
        sessions = await build_sessions(
            DEFAULT_CONFIG, state, iterm_cache=iterm_cache, tmux_cache=FakeTmuxCache()
        )

    s = sessions[0]
    assert s.location_type == "iterm"
    assert s.iterm_window_id == 99
    assert s.iterm_tab_id is None
    assert s.iterm_tab_index == 2
    assert s.iterm_tty == "/dev/ttys007"


async def test_build_sessions_disambiguates_two_pids_same_cwd(tmp_path):
    """Two claudes in the same cwd, each with its own --resume id → each gets the right log."""
    cwd = "/tmp/dupcwd"
    folder = tmp_path / "-tmp-dupcwd"
    folder.mkdir()
    (folder / "alpha.jsonl").write_text(
        '{"type":"assistant","message":{"model":"claude-opus-4-7","content":[],"usage":{"input_tokens":100}}}\n'
    )
    (folder / "beta.jsonl").write_text(
        '{"type":"assistant","message":{"model":"claude-sonnet-4-6","content":[],"usage":{"input_tokens":200}}}\n'
    )

    procs = [_proc(pid=1, cwd=cwd, session_id="alpha"), _proc(pid=2, cwd=cwd, session_id="beta")]
    state = LinkerState()
    state.log_dir = tmp_path

    with patch("backend.detectors.linker.scan_claude_processes", return_value=procs):
        sessions = await build_sessions(
            DEFAULT_CONFIG, state, iterm_cache=FakeItermCache(), tmux_cache=FakeTmuxCache()
        )

    by_pid = {s.pid: s for s in sessions}
    assert by_pid[1].conversation_id == "alpha"
    assert by_pid[1].usage.input_tokens == 100
    assert by_pid[1].model == "claude-opus-4-7"
    assert by_pid[2].conversation_id == "beta"
    assert by_pid[2].usage.input_tokens == 200
    assert by_pid[2].model == "claude-sonnet-4-6"


async def test_env_session_id_disambiguates_two_pids_same_cwd_no_cmdline(tmp_path):
    """Real-world case: two claudes in the same cwd, neither started with --resume.

    The only ground-truth link is the CLAUDE_CODE_SESSION_ID env var. Without it the
    linker falls back to "freshest log in the folder" and mis-attributes both PIDs to
    the same (latest-mtime) session.
    """
    cwd = "/tmp/envcwd"
    folder = tmp_path / "-tmp-envcwd"
    folder.mkdir()
    (folder / "alpha.jsonl").write_text(
        '{"type":"assistant","message":{"model":"claude-opus-4-7","content":[],"usage":{"input_tokens":100}}}\n'
    )
    (folder / "beta.jsonl").write_text(
        '{"type":"assistant","message":{"model":"claude-sonnet-4-6","content":[],"usage":{"input_tokens":200}}}\n'
    )

    procs = [
        _proc(pid=1, cwd=cwd, env_session_id="alpha"),
        _proc(pid=2, cwd=cwd, env_session_id="beta"),
    ]
    state = LinkerState()
    state.log_dir = tmp_path

    with patch("backend.detectors.linker.scan_claude_processes", return_value=procs):
        sessions = await build_sessions(
            DEFAULT_CONFIG, state, iterm_cache=FakeItermCache(), tmux_cache=FakeTmuxCache()
        )

    by_pid = {s.pid: s for s in sessions}
    assert by_pid[1].conversation_id == "alpha"
    assert by_pid[1].usage.input_tokens == 100
    assert by_pid[1].model == "claude-opus-4-7"
    assert by_pid[2].conversation_id == "beta"
    assert by_pid[2].usage.input_tokens == 200
    assert by_pid[2].model == "claude-sonnet-4-6"
    # Distinct sessions → not duplicates
    assert by_pid[1].duplicate_count == 1
    assert by_pid[2].duplicate_count == 1


async def test_env_session_id_from_other_cwd_is_ignored(tmp_path):
    """macOS leaks the caller's CLAUDE_CODE_SESSION_ID onto processes whose env it can't read.

    An env id whose log lives in a *different* project folder is bogus: it must be ignored
    (not trusted as this process's session, and never counted as a duplicate). The process
    should fall back to the real log in its own folder.
    """
    cwd = "/tmp/realcwd"
    (tmp_path / "-tmp-realcwd").mkdir()
    (tmp_path / "-tmp-realcwd" / "real-sess.jsonl").write_text(
        '{"type":"assistant","message":{"model":"claude-opus-4-7","content":[],"usage":{"input_tokens":10}}}\n'
    )
    # The leaked id's log lives in a *different* folder.
    (tmp_path / "-tmp-othercwd").mkdir()
    (tmp_path / "-tmp-othercwd" / "leaked-id.jsonl").write_text(
        '{"type":"assistant","message":{"content":[]}}\n'
    )

    procs = [_proc(pid=1, cwd=cwd, env_session_id="leaked-id")]
    state = LinkerState()
    state.log_dir = tmp_path

    with patch("backend.detectors.linker.scan_claude_processes", return_value=procs):
        sessions = await build_sessions(
            DEFAULT_CONFIG, state, iterm_cache=FakeItermCache(), tmux_cache=FakeTmuxCache()
        )

    s = sessions[0]
    assert s.session_id is None  # leaked env id rejected — its log isn't in this cwd
    assert s.conversation_id == "real-sess"  # fell back to the real in-folder log
    assert s.usage.input_tokens == 10
    assert s.duplicate_count == 1


async def test_duplicate_session_detection_same_env_session_id(tmp_path):
    """Same Claude session attached from two CLIs → same CLAUDE_CODE_SESSION_ID → flagged."""
    cwd = "/tmp/dupenv"
    folder = tmp_path / "-tmp-dupenv"
    folder.mkdir()
    (folder / "shared.jsonl").write_text(
        '{"type":"assistant","message":{"model":"claude-opus-4-7","content":[],"usage":{"input_tokens":50}}}\n'
    )

    procs = [
        _proc(pid=10, cwd=cwd, env_session_id="shared"),
        _proc(pid=11, cwd=cwd, env_session_id="shared"),
        _proc(pid=12, cwd=cwd, env_session_id="lonely"),  # unique id, no log file
    ]
    state = LinkerState()
    state.log_dir = tmp_path

    with patch("backend.detectors.linker.scan_claude_processes", return_value=procs):
        sessions = await build_sessions(
            DEFAULT_CONFIG, state, iterm_cache=FakeItermCache(), tmux_cache=FakeTmuxCache()
        )

    by_pid = {s.pid: s for s in sessions}
    assert by_pid[10].session_id == "shared"
    assert by_pid[10].duplicate_count == 2
    assert by_pid[11].duplicate_count == 2
    assert set(by_pid[10].duplicate_pids) == {11}
    assert set(by_pid[11].duplicate_pids) == {10}
    assert by_pid[12].duplicate_count == 1
    assert by_pid[12].duplicate_pids == []


async def test_context_tokens_includes_output(tmp_path):
    """Output tokens of the latest turn count toward the context window (they become part of
    the conversation history re-sent next turn), so context_tokens must include them."""
    cwd = "/tmp/ctxcwd"
    folder = tmp_path / "-tmp-ctxcwd"
    folder.mkdir()
    (folder / "ctx.jsonl").write_text(
        '{"type":"assistant","message":{"model":"claude-opus-4-7","content":[],'
        '"usage":{"input_tokens":1000,"output_tokens":500,'
        '"cache_read_input_tokens":2000,"cache_creation_input_tokens":100}}}\n'
    )
    procs = [_proc(pid=1, cwd=cwd, env_session_id="ctx")]
    state = LinkerState()
    state.log_dir = tmp_path

    with patch("backend.detectors.linker.scan_claude_processes", return_value=procs):
        sessions = await build_sessions(
            DEFAULT_CONFIG, state, iterm_cache=FakeItermCache(), tmux_cache=FakeTmuxCache()
        )

    s = sessions[0]
    assert s.context_tokens == 1000 + 500 + 2000 + 100  # input + output + cache_read + cache_creation


async def test_build_sessions_no_log_gracefully(tmp_path):
    procs = [_proc(pid=42, cwd="/some/path/that/has/no/logs")]
    state = LinkerState()
    state.log_dir = tmp_path

    with patch("backend.detectors.linker.scan_claude_processes", return_value=procs):
        sessions = await build_sessions(
            DEFAULT_CONFIG, state, iterm_cache=FakeItermCache(), tmux_cache=FakeTmuxCache()
        )

    s = sessions[0]
    assert s.usage is None
    assert s.conversation_id is None
    assert s.location_type == "headless"
    assert s.tool_calls.total == 0


async def test_build_sessions_without_caches_is_headless(tmp_path):
    """When caches aren't wired (e.g. in some test contexts), sessions still build but headless."""
    procs = [_proc(pid=77, cwd="/somewhere")]
    state = LinkerState()
    state.log_dir = tmp_path

    with patch("backend.detectors.linker.scan_claude_processes", return_value=procs):
        sessions = await build_sessions(DEFAULT_CONFIG, state)

    assert len(sessions) == 1
    assert sessions[0].location_type == "headless"

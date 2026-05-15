"""Tests for ItermLocationCache pid→location matching (AppleScript-only mode).

The cache walks the claudewatch pid's *ancestor TTYs* (every process in an iTerm
tab shares the tab's TTY) and looks them up against the AppleScript-derived
session list. Tests below lock in that behavior.
"""
from __future__ import annotations

from unittest.mock import patch

from backend.detectors.iterm_applescript import ItermSessionTty
from backend.detectors.iterm_cache import ItermLocationCache


def _make_session(*, tty: str, window_id: int = 100, tab_index: int = 1) -> ItermSessionTty:
    return ItermSessionTty(
        window_id=window_id,
        tab_index=tab_index,
        tty=tty,
        unique_id=f"sess-{tty}",
        name=f"title-{tty}",
    )


def test_get_locations_matches_via_tty():
    cache = ItermLocationCache()
    cache._sessions = [_make_session(tty="/dev/ttys003", window_id=42, tab_index=2)]
    with patch(
        "backend.detectors.iterm_cache._ancestor_ttys", return_value=["/dev/ttys003"]
    ):
        out = cache.get_locations([12345])
    assert 12345 in out
    assert out[12345].tty == "/dev/ttys003"
    assert out[12345].window_id == 42
    assert out[12345].tab_index == 2
    assert out[12345].tab_id is None


def test_get_locations_walks_ancestor_ttys():
    """The claudewatch pid may be deep in the process tree; ancestor TTYs lead to the tab."""
    cache = ItermLocationCache()
    cache._sessions = [_make_session(tty="/dev/ttys007")]

    # The deepest process has no tty (e.g. an MCP child with stdio piping);
    # walking ancestors eventually reaches the shell with the real tty.
    def fake_ancestors(pid: int, max_depth: int = 12) -> list[str]:
        return {76707: ["/dev/ttys007"], 76772: ["/dev/ttys007"]}.get(pid, [])

    with patch("backend.detectors.iterm_cache._ancestor_ttys", side_effect=fake_ancestors):
        out = cache.get_locations([76707, 76772])

    assert 76707 in out and 76772 in out
    assert out[76707].tty == "/dev/ttys007"


def test_get_locations_empty_when_no_match():
    cache = ItermLocationCache()
    cache._sessions = [_make_session(tty="/dev/ttys999")]
    with patch("backend.detectors.iterm_cache._ancestor_ttys", return_value=["/dev/ttys003"]):
        out = cache.get_locations([42])
    assert out == {}


def test_get_locations_skips_sentinel_tty():
    cache = ItermLocationCache()
    cache._sessions = [_make_session(tty="?")]
    with patch("backend.detectors.iterm_cache._ancestor_ttys", return_value=["?"]):
        out = cache.get_locations([42])
    assert out == {}


def test_get_locations_empty_pids_returns_empty():
    cache = ItermLocationCache()
    cache._sessions = [_make_session(tty="/dev/ttys001")]
    assert cache.get_locations([]) == {}


def test_refresh_interval_floor():
    """The refresh interval is clamped to a sane minimum so iTerm isn't hammered."""
    cache = ItermLocationCache(refresh_interval=0.1)
    assert cache.refresh_interval >= 5.0

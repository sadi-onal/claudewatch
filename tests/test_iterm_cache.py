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


# --- Smart + safe enumeration: no background polling, cache-once, circuit breaker ---


def test_no_enumeration_when_nothing_pending():
    """Steady state (every claude session already located, or none running) → zero iTerm calls."""
    cache = ItermLocationCache()
    cache.get_locations([])
    assert cache.should_enumerate() is False


def test_circuit_breaker_blocks_enumeration_after_failure():
    """The root-cause bug: a hanging AppleScript was re-run every cycle and wedged iTerm.

    One enumeration failure (hang/timeout) must open the breaker and stop further polling.
    """
    cache = ItermLocationCache(failure_threshold=1, breaker_cooldown=600.0)
    cache._pending = {123}
    assert cache.should_enumerate() is True
    cache._record_failure()
    assert cache.should_enumerate() is False


def test_circuit_breaker_recovers_after_cooldown():
    now = {"t": 1000.0}
    cache = ItermLocationCache(failure_threshold=1, breaker_cooldown=600.0, clock=lambda: now["t"])
    cache._pending = {123}
    cache._record_failure()
    assert cache.should_enumerate() is False
    now["t"] += 601
    assert cache.should_enumerate() is True


def test_success_resets_failure_count():
    cache = ItermLocationCache(failure_threshold=2)
    cache._pending = {1}
    cache._record_failure()
    cache._record_success([])
    assert cache._consecutive_failures == 0
    assert cache.should_enumerate() is True  # still pending, breaker closed


def test_located_pid_is_cached_and_stops_polling():
    """Once a session's tab is found, it's cached forever (tty/window never change) and the
    cache stops wanting to enumerate — no background churn against iTerm."""
    cache = ItermLocationCache()
    with patch("backend.detectors.iterm_cache._ancestor_ttys", return_value=["/dev/ttys003"]):
        cache.get_locations([123])  # _sessions empty → can't match yet → pending
        assert cache.should_enumerate() is True
        cache._record_success([_make_session(tty="/dev/ttys003")])  # enumeration locates it
        out = cache.get_locations([123])
    assert 123 in out and out[123].tty == "/dev/ttys003"
    assert cache.should_enumerate() is False


def test_gives_up_on_unlocatable_pid():
    """A Terminal.app (non-iTerm) session never matches. After a few attempts the cache must
    give up so it doesn't enumerate iTerm forever chasing a pid it can't place."""
    cache = ItermLocationCache(give_up_after=2)
    with patch("backend.detectors.iterm_cache._ancestor_ttys", return_value=[]):
        cache.get_locations([999])
        cache._record_success([_make_session(tty="/dev/ttys003")])  # attempt 1, no match
        cache.get_locations([999])
        assert cache.should_enumerate() is True
        cache._record_success([_make_session(tty="/dev/ttys003")])  # attempt 2 → give up
        cache.get_locations([999])
    assert cache.should_enumerate() is False
    assert cache.get_locations([999]) == {}

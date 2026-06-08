"""Tests for the CLI daemon stop logic.

Bug: `claudewatch stop` sent SIGTERM, waited, then unconditionally reported "Stopped"
and deleted the pid file — even when the process was still alive (uvicorn's graceful
shutdown blocks on long-lived SSE connections). The daemon kept running (and kept
poking iTerm). Stop must escalate to SIGKILL and report the true outcome.
"""
from __future__ import annotations

import signal

from backend.cli import _escalating_stop


def test_escalating_stop_returns_true_when_sigterm_works():
    sent: list[int] = []
    checks = {"n": 0}

    def kill(pid, sig):
        sent.append(sig)

    def alive(pid):
        checks["n"] += 1
        return checks["n"] < 2  # alive on first poll, gone on the second

    assert _escalating_stop(123, kill, alive, sleeper=lambda: None) is True
    assert signal.SIGTERM in sent
    assert signal.SIGKILL not in sent  # graceful was enough


def test_escalating_stop_falls_back_to_sigkill():
    sent: list[int] = []

    def kill(pid, sig):
        sent.append(sig)

    def alive(pid):
        # Survives SIGTERM; only dies once SIGKILL is delivered.
        return signal.SIGKILL not in sent

    assert _escalating_stop(123, kill, alive, sleeper=lambda: None) is True
    assert sent.count(signal.SIGTERM) == 1
    assert signal.SIGKILL in sent


def test_escalating_stop_reports_failure_if_process_never_dies():
    sent: list[int] = []

    def kill(pid, sig):
        sent.append(sig)

    def alive(pid):
        return True  # never dies → must report failure, not a false success

    assert _escalating_stop(123, kill, alive, sleeper=lambda: None) is False

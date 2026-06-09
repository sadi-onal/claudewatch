"""API endpoint tests using FastAPI TestClient.

These tests don't spin up uvicorn; they instantiate the app and call routes
directly via httpx. Lifespan runs the scheduler loop, so we suppress that by
overriding state setup.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.config import DEFAULT_CONFIG, STATE_DB
from backend.detectors.filesystem_watch import FilesystemWatcher
from backend.models import ClaudeSession, TokenUsage, ToolCallStats
from backend.server import AppState
from backend.state import State


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Build the FastAPI app with a tmp_path state DB and no real scheduler."""
    monkeypatch.setattr("backend.config.STATE_DB", tmp_path / "state.db")
    monkeypatch.setattr("backend.server.STATE_DB", tmp_path / "state.db")

    # Disable the scheduler by patching the loop to no-op.
    async def _no_scheduler(s):
        return None

    monkeypatch.setattr("backend.server._scheduler_loop", _no_scheduler)

    from backend.server import create_app

    app = create_app()

    with TestClient(app) as client:
        yield client, app


@pytest.fixture
def populated_app(app, tmp_path):
    client, fastapi_app = app
    now = datetime.now(timezone.utc)
    sess = ClaudeSession(
        pid=12345,
        cwd="/Users/me/Projects/x",
        started_at=now,
        duration_seconds=120,
        cpu_percent=4.2,
        memory_mb=512.0,
        status="working",
        location_type="iterm",
        iterm_window_id=42,
        iterm_tab_index=1,
        iterm_tty="/dev/ttys001",
        last_activity_at=now,
        model="claude-opus-4-7",
        usage=TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=2000,
            cost_estimate_usd=0.063,
        ),
        tool_calls=ToolCallStats(total=3, breakdown={"Edit": 2, "Bash": 1}, last_used="Edit"),
        permission_mode="dangerously-skip",
        message_count=4,
    )
    fastapi_app.state.s.sessions = {sess.pid: sess}
    return client, fastapi_app, sess


def test_sessions_endpoint_lists_active(populated_app):
    client, _, sess = populated_app
    r = client.get("/api/sessions")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["pid"] == 12345
    assert data[0]["model"] == "claude-opus-4-7"
    assert data[0]["usage"]["total_tokens"] == 3500


def test_sessions_get_single_404_when_missing(populated_app):
    client, _, _ = populated_app
    assert client.get("/api/sessions/99999").status_code == 404


def test_sessions_get_single_returns_full_object(populated_app):
    client, _, _ = populated_app
    r = client.get("/api/sessions/12345")
    assert r.status_code == 200
    d = r.json()
    assert d["iterm_window_id"] == 42
    assert d["tool_calls"]["total"] == 3
    assert d["tool_calls"]["breakdown"] == {"Edit": 2, "Bash": 1}
    assert d["permission_mode"] == "dangerously-skip"


def test_health_endpoint_shape(populated_app):
    client, _, _ = populated_app
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    for key in ("iterm_api", "automation", "tmux_available", "log_dir_found", "issues"):
        assert key in d


def test_stats_aggregates_active_tokens_and_cost(populated_app):
    client, _, _ = populated_app
    r = client.get("/api/stats")
    assert r.status_code == 200
    d = r.json()
    assert d["active"] == 1
    assert d["active_tokens"] == 3500
    assert d["active_cost"] == pytest.approx(0.063)


def test_config_get_and_post_roundtrip(populated_app, tmp_path, monkeypatch):
    client, fastapi_app, _ = populated_app
    monkeypatch.setattr("backend.config.CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr("backend.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("backend.config.LOGS_DIR", tmp_path / "logs")

    r = client.post("/api/config", json={"port": 7799})
    assert r.status_code == 200
    assert r.json()["port"] == 7799

    r2 = client.get("/api/config")
    assert r2.status_code == 200
    assert r2.json()["port"] == 7799


def test_history_returns_empty_initially(populated_app):
    client, _, _ = populated_app
    r = client.get("/api/history")
    assert r.status_code == 200
    assert r.json() == []


def test_focus_rejects_headless(populated_app):
    client, fastapi_app, _ = populated_app
    sess = list(fastapi_app.state.s.sessions.values())[0]
    sess.location_type = "headless"
    sess.iterm_tty = None
    sess.iterm_window_id = None
    r = client.post(f"/api/sessions/{sess.pid}/focus")
    assert r.status_code == 400
    assert "headless" in r.json()["detail"].lower()


def test_focus_rejects_404_for_unknown_pid(populated_app):
    client, _, _ = populated_app
    assert client.post("/api/sessions/99999/focus").status_code == 404


def test_halt_404_for_unknown_pid(populated_app):
    client, _, _ = populated_app
    assert client.post("/api/sessions/99999/halt").status_code == 404


def test_read_only_mode_blocks_actions(populated_app):
    client, fastapi_app, _ = populated_app
    fastapi_app.state.s.config["read_only"] = True
    assert client.post("/api/sessions/12345/halt").status_code == 403
    assert client.post("/api/sessions/12345/focus").status_code == 403
    assert client.post("/api/sessions/new", json={"cwd": str(Path.home())}).status_code == 403


def test_new_session_rejects_bad_cwd(populated_app):
    client, _, _ = populated_app
    r = client.post("/api/sessions/new", json={"cwd": "/nope/not/a/dir/xyz"})
    assert r.status_code == 400


def test_new_session_rejects_unsafe_flag(populated_app):
    client, _, _ = populated_app
    r = client.post(
        "/api/sessions/new",
        json={"cwd": str(Path.home()), "flags": ["--evil;rm"]},
    )
    assert r.status_code == 400


def test_log_tail_privacy_redacts_text_by_default(populated_app, tmp_path):
    client, fastapi_app, sess = populated_app
    fastapi_app.state.s.config["show_log_text"] = False
    log = tmp_path / "fake.jsonl"
    log.write_text(
        '{"type":"assistant","timestamp":"2026-01-01T00:00:00Z",'
        '"message":{"model":"claude-opus-4-7","content":['
        '{"type":"text","text":"secret content"},'
        '{"type":"tool_use","name":"Bash","input":{"command":"echo s3cret"}}'
        ']}}\n'
    )
    sess.conversation_log_path = str(log)
    r = client.get(f"/api/sessions/{sess.pid}/log-tail")
    assert r.status_code == 200
    data = r.json()
    assert data["privacy_mode"] is True
    blocks = data["entries"][0]["message"]["content"]
    text_block = next(b for b in blocks if b["type"] == "text")
    # Redacted text block: keeps {type} only, no "text" key with content
    assert text_block.get("text") is None or text_block.get("text") == ""
    tool_block = next(b for b in blocks if b["type"] == "tool_use")
    assert tool_block.get("name") == "Bash"
    assert "input" not in tool_block


def test_log_tail_shows_text_when_show_log_text_true(populated_app, tmp_path):
    client, fastapi_app, sess = populated_app
    log = tmp_path / "show.jsonl"
    log.write_text(
        '{"type":"assistant","message":{"content":[{"type":"text","text":"hello"}]}}\n'
    )
    sess.conversation_log_path = str(log)
    fastapi_app.state.s.config["show_log_text"] = True
    r = client.get(f"/api/sessions/{sess.pid}/log-tail")
    data = r.json()
    assert data["privacy_mode"] is False
    assert data["entries"][0]["message"]["content"][0]["text"] == "hello"


def test_index_cache_busts_static_assets(app):
    """A freshly-served index.html must reference versioned asset URLs (and be no-store),
    so the browser can't pair it with a stale cached app.js/styles.css."""
    client, _ = app
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "/static/app.js?v=" in body
    assert "/static/styles.css?v=" in body
    cc = r.headers.get("cache-control", "")
    assert "no-store" in cc

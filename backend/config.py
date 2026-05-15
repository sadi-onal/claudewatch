from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w

CONFIG_DIR = Path.home() / ".claudewatch"
CONFIG_PATH = CONFIG_DIR / "config.toml"
STATE_DB = CONFIG_DIR / "state.db"
LOGS_DIR = CONFIG_DIR / "logs"
PID_FILE = CONFIG_DIR / "server.pid"

DEFAULT_CONFIG: dict[str, Any] = {
    "port": 7788,
    "read_only": False,
    "privacy_mode": True,
    "show_log_text": False,
    "file_change_retention_minutes": 10,
    "process_scan_interval_seconds": 2,
    "log_scan_interval_seconds": 3,
    "git_refresh_interval_seconds": 10,
    "iterm_refresh_interval_seconds": 30,
    "tmux_refresh_interval_seconds": 5,
    "ignore_patterns": [
        ".git/",
        "node_modules/",
        "__pycache__/",
        ".venv/",
        "venv/",
        ".DS_Store",
        "*.pyc",
        "*.log",
        "dist/",
        "build/",
        "target/",
        ".next/",
    ],
    "pricing": {
        "claude-opus-4-7": {
            "input": 15.00,
            "output": 75.00,
            "cache_read": 1.50,
            "cache_write": 18.75,
        },
        "claude-opus-4-6": {
            "input": 15.00,
            "output": 75.00,
            "cache_read": 1.50,
            "cache_write": 18.75,
        },
        "claude-sonnet-4-6": {
            "input": 3.00,
            "output": 15.00,
            "cache_read": 0.30,
            "cache_write": 3.75,
        },
        "claude-sonnet-4-5": {
            "input": 3.00,
            "output": 15.00,
            "cache_read": 0.30,
            "cache_write": 3.75,
        },
        "claude-haiku-4-5": {
            "input": 1.00,
            "output": 5.00,
            "cache_read": 0.10,
            "cache_write": 1.25,
        },
    },
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    ensure_config_dir()
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)
    return _deep_merge(DEFAULT_CONFIG, data)


def save_config(cfg: dict[str, Any]) -> None:
    ensure_config_dir()
    merged = _deep_merge(DEFAULT_CONFIG, cfg)
    with open(CONFIG_PATH, "wb") as f:
        tomli_w.dump(merged, f)


def update_config(updates: dict[str, Any]) -> dict[str, Any]:
    current = load_config()
    merged = _deep_merge(current, updates)
    save_config(merged)
    return merged

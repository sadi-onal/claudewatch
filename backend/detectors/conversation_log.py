from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backend.models import ToolCallStats, TokenUsage

log = logging.getLogger(__name__)

LOG_DIR_CANDIDATES = [
    Path.home() / ".claude" / "projects",
    Path.home() / ".config" / "claude" / "projects",
    Path.home() / "Library" / "Application Support" / "Claude" / "projects",
]


def find_log_dir() -> Path | None:
    for p in LOG_DIR_CANDIDATES:
        if p.is_dir():
            return p
    return None


def cwd_to_project_folder(cwd: str) -> str:
    """Convert /Users/x/Projects/y to -Users-x-Projects-y (Claude Code convention).

    Claude Code encodes both `/` and `.` as `-`, so usernames like `first.last`
    (common with macOS AD-joined accounts) and dotted directories like
    `.claude` map to `first-last` and `-claude` respectively.
    """
    cwd = cwd.rstrip("/")
    if not cwd:
        return ""
    return cwd.replace("/", "-").replace(".", "-")


def find_logs_for_cwd(cwd: str, log_dir: Path | None = None) -> list[Path]:
    base = log_dir or find_log_dir()
    if base is None:
        return []
    folder = base / cwd_to_project_folder(cwd)
    if not folder.is_dir():
        return []
    files = [f for f in folder.glob("*.jsonl") if f.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        return None


@dataclass
class ParsedLog:
    conversation_id: str
    log_path: Path
    model: str | None = None
    cli_version: str | None = None
    permission_mode: str | None = None
    message_count: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    thinking_enabled: bool = False
    tool_calls: ToolCallStats = field(default_factory=ToolCallStats)
    last_activity_at: datetime | None = None
    cwd: str | None = None
    git_branch: str | None = None
    # The user's most recent typed prompt (best-effort; tool results and injected
    # command/system messages are skipped). Used as a human-readable session label.
    last_user_message: str | None = None
    # Latest assistant turn (NOT cumulative) — for context % and "current-turn" display
    last_assistant_at: datetime | None = None
    last_assistant_usage: TokenUsage = field(default_factory=TokenUsage)
    last_stop_reason: str | None = None
    # True when last entry is assistant with stop_reason="tool_use" or null/None,
    # meaning the model is mid-stream (called a tool, awaiting result).
    is_in_flight: bool = False
    # Currently in-progress Task* task (if any)
    current_task_subject: str | None = None
    current_task_active_form: str | None = None
    current_task_id: str | None = None
    current_task_started_at: datetime | None = None


def parse_log(path: Path) -> ParsedLog:
    """Walk all JSONL entries and aggregate into ParsedLog. Robust to schema drift."""
    pl = ParsedLog(conversation_id=path.stem, log_path=path)
    breakdown: dict[str, int] = {}
    last_tool_used: str | None = None
    last_tool_used_at: datetime | None = None
    # Task tracking: index every TaskCreate by its 1-based position within THIS file,
    # then walk TaskUpdate calls in order to determine which task is currently in_progress.
    # Note: Task IDs are global across the whole Claude session and may reference tasks
    # created in prior log files. We can't resolve those cross-session.
    tasks_created_here: list[dict] = []  # [{"subject", "active_form", "ts", "local_id"}]
    current_in_progress: dict | None = None
    current_in_progress_started_at: datetime | None = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = _parse_ts(entry.get("timestamp"))
                if ts and (pl.last_activity_at is None or ts > pl.last_activity_at):
                    pl.last_activity_at = ts
                if not pl.cwd and entry.get("cwd"):
                    pl.cwd = entry.get("cwd")
                if not pl.cli_version and entry.get("version"):
                    pl.cli_version = entry.get("version")
                if not pl.git_branch and entry.get("gitBranch"):
                    pl.git_branch = entry.get("gitBranch")

                etype = entry.get("type")
                if etype == "permission-mode":
                    pm = entry.get("permissionMode")
                    if pm:
                        pl.permission_mode = pm
                elif etype in ("user", "assistant"):
                    pl.message_count += 1
                if etype == "user":
                    msg = entry.get("message") or {}
                    content = msg.get("content")
                    text = None
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, str):
                                text = block
                                break
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text")
                                break
                            # tool_result blocks have no user-typed text → skip
                    if text:
                        t = text.strip()
                        # Skip injected command output / system-reminder / caveat wrappers,
                        # keeping only what the user actually typed.
                        if t and not t.startswith("<") and not t.startswith("Caveat:"):
                            pl.last_user_message = t[:280]
                if etype == "assistant":
                    msg = entry.get("message") or {}
                    model = msg.get("model")
                    if model:
                        pl.model = model
                    usage = msg.get("usage") or {}
                    in_t = int(usage.get("input_tokens") or 0)
                    out_t = int(usage.get("output_tokens") or 0)
                    cr_t = int(usage.get("cache_read_input_tokens") or 0)
                    cc_t = int(usage.get("cache_creation_input_tokens") or 0)
                    pl.usage.input_tokens += in_t
                    pl.usage.output_tokens += out_t
                    pl.usage.cache_read_input_tokens += cr_t
                    pl.usage.cache_creation_input_tokens += cc_t
                    # Latest assistant snapshot (overwritten each iteration)
                    pl.last_assistant_at = ts
                    pl.last_assistant_usage = TokenUsage(
                        input_tokens=in_t,
                        output_tokens=out_t,
                        cache_read_input_tokens=cr_t,
                        cache_creation_input_tokens=cc_t,
                    )
                    pl.last_stop_reason = msg.get("stop_reason")
                    # In-flight: assistant turn that's still mid-stream (tool_use stop
                    # means a tool call was issued and the assistant is waiting for the result,
                    # or null/None means streaming isn't finished).
                    pl.is_in_flight = pl.last_stop_reason in (None, "tool_use")

                    content = msg.get("content") or []
                    if isinstance(content, list):
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            btype = block.get("type")
                            if btype == "thinking":
                                pl.thinking_enabled = True
                            elif btype == "tool_use":
                                name = block.get("name") or "unknown"
                                breakdown[name] = breakdown.get(name, 0) + 1
                                pl.tool_calls.total += 1
                                last_tool_used = name
                                last_tool_used_at = ts
                                inp = block.get("input") or {}
                                if name == "TaskCreate" and isinstance(inp, dict):
                                    tasks_created_here.append(
                                        {
                                            "subject": inp.get("subject"),
                                            "active_form": inp.get("activeForm"),
                                            "ts": ts,
                                        }
                                    )
                                elif name == "TaskUpdate" and isinstance(inp, dict):
                                    tid = str(inp.get("taskId") or "")
                                    status = inp.get("status")
                                    if status == "in_progress":
                                        # Look up subject: prefer task created in this file with matching index
                                        subject = None
                                        active_form = None
                                        try:
                                            idx = int(tid)
                                            # Heuristic: count TaskCreate calls so far; if this taskId
                                            # equals (count of TaskCreates so far), it's the most recent create.
                                            if tasks_created_here:
                                                # Best match: the last TaskCreate before this update
                                                t = tasks_created_here[-1]
                                                subject = t["subject"]
                                                active_form = t["active_form"]
                                        except (TypeError, ValueError):
                                            pass
                                        current_in_progress = {
                                            "id": tid,
                                            "subject": subject,
                                            "active_form": active_form,
                                        }
                                        current_in_progress_started_at = ts
                                    elif status in ("completed", "deleted") and current_in_progress and current_in_progress.get("id") == tid:
                                        current_in_progress = None
                                        current_in_progress_started_at = None
    except OSError as e:
        log.warning("Cannot read log %s: %s", path, e)
        return pl
    pl.tool_calls.breakdown = dict(sorted(breakdown.items(), key=lambda kv: -kv[1]))
    pl.tool_calls.last_used = last_tool_used
    pl.tool_calls.last_used_at = last_tool_used_at
    if current_in_progress:
        pl.current_task_id = current_in_progress.get("id")
        pl.current_task_subject = current_in_progress.get("subject")
        pl.current_task_active_form = current_in_progress.get("active_form")
        pl.current_task_started_at = current_in_progress_started_at
    return pl


def parse_logs_for_cwd(
    cwd: str, log_dir: Path | None = None, max_files: int = 3
) -> ParsedLog | None:
    """Return the freshest log for the cwd (highest mtime)."""
    logs = find_logs_for_cwd(cwd, log_dir)
    if not logs:
        return None
    return parse_log(logs[0])

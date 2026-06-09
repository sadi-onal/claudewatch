from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_estimate_usd: float | None = None

    @computed_field
    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


class ToolCallStats(BaseModel):
    total: int = 0
    breakdown: dict[str, int] = Field(default_factory=dict)
    last_used: str | None = None
    last_used_at: datetime | None = None


class FileChange(BaseModel):
    path: str
    kind: Literal["created", "modified", "deleted"]
    ts: datetime


class GitContext(BaseModel):
    branch: str | None = None
    is_dirty: bool = False
    modified_count: int = 0
    insertions: int = 0
    deletions: int = 0


SessionStatus = Literal["working", "waiting", "idle", "ended"]
LocationType = Literal["iterm", "tmux", "headless"]


class ClaudeSession(BaseModel):
    pid: int
    cwd: str
    started_at: datetime
    duration_seconds: int = 0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    status: SessionStatus = "idle"

    location_type: LocationType = "headless"
    iterm_window_id: int | None = None
    iterm_tab_id: int | None = None
    iterm_tab_index: int | None = None
    iterm_session_id: str | None = None
    iterm_tab_title: str | None = None
    iterm_tty: str | None = None
    tmux_session: str | None = None
    tmux_window: str | None = None
    tmux_pane: str | None = None

    last_activity_at: datetime
    last_output_preview: str | None = None

    model: str | None = None
    cli_version: str | None = None
    conversation_id: str | None = None
    conversation_log_path: str | None = None
    # Ground-truth session id from the process env (CLAUDE_CODE_SESSION_ID). Present even
    # when no conversation log has been written yet, so it's the reliable de-dup key.
    session_id: str | None = None
    # How many live processes are attached to this same session (1 = unique). >1 means the
    # same Claude session is open from multiple CLIs — usually accidental.
    duplicate_count: int = 1
    duplicate_pids: list[int] = Field(default_factory=list)
    # The user's most recent typed prompt — a human-readable label for the session.
    # Only populated when show_log_text is enabled (privacy).
    last_user_message: str | None = None
    message_count: int = 0
    usage: TokenUsage | None = None
    thinking_enabled: bool | None = None
    permission_mode: str | None = None
    extra_flags: list[str] = Field(default_factory=list)
    tool_calls: ToolCallStats = Field(default_factory=ToolCallStats)

    recent_file_changes: list[FileChange] = Field(default_factory=list)
    git: GitContext | None = None

    # Live activity (from latest assistant turn)
    is_in_flight: bool = False
    last_turn_tokens: int | None = None             # output tokens of the latest assistant message
    last_turn_at: datetime | None = None
    last_stop_reason: str | None = None
    # Context window utilization (based on latest assistant.usage)
    context_tokens: int | None = None
    context_max_tokens: int | None = None
    context_pct: float | None = None
    # Currently in-progress Task* task
    current_task_subject: str | None = None
    current_task_active_form: str | None = None
    current_task_id: str | None = None
    current_task_started_at: datetime | None = None
    current_task_elapsed_seconds: int | None = None


class NewSessionRequest(BaseModel):
    cwd: str
    window_type: Literal["new-window", "new-tab"] = "new-window"
    flags: list[str] = Field(default_factory=list)
    command: str = "claude"


class HealthReport(BaseModel):
    iterm_api: bool
    automation: bool
    tmux_available: bool
    log_dir_found: bool
    log_dir_path: str | None = None
    issues: list[str] = Field(default_factory=list)

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.live import Live
from rich.table import Table

from backend.config import CONFIG_PATH, LOGS_DIR, PID_FILE, ensure_config_dir, load_config

app = typer.Typer(add_completion=False, help="ClaudeWatch — local Claude Code session monitor.")
console = Console()


def _server_url() -> str:
    return f"http://127.0.0.1:{int(load_config().get('port', 7788))}"


def _is_running() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    try:
        os.kill(pid, 0)
        return pid
    except (ProcessLookupError, PermissionError):
        return None


def _http_get(path: str):
    import urllib.request

    url = _server_url() + path
    with urllib.request.urlopen(url, timeout=2) as resp:
        return json.loads(resp.read())


@app.command()
def start(daemon: bool = typer.Option(False, "--daemon", "-d", help="Detach to background")) -> None:
    """Start the ClaudeWatch server."""
    ensure_config_dir()
    cfg = load_config()
    port = int(cfg.get("port", 7788))
    existing = _is_running()
    if existing:
        console.print(f"[yellow]Server already running on PID {existing}[/yellow]")
        return
    if daemon:
        log_path = LOGS_DIR / "server.log"
        with open(log_path, "ab") as logf:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "backend.server:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--log-level",
                    "info",
                    # Force-close lingering SSE connections after 3s on shutdown so
                    # SIGTERM (claudewatch stop) actually terminates the server.
                    "--timeout-graceful-shutdown",
                    "3",
                ],
                stdout=logf,
                stderr=logf,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        PID_FILE.write_text(str(proc.pid))
        time.sleep(1.0)
        console.print(f"[green]Started ClaudeWatch (PID {proc.pid}) → {_server_url()}[/green]")
        return
    uvicorn.run(
        "backend.server:app",
        host="127.0.0.1",
        port=port,
        log_level="info",
    )


def _escalating_stop(
    pid: int,
    kill,
    alive,
    term_polls: int = 30,
    kill_polls: int = 15,
    sleeper=None,
) -> bool:
    """SIGTERM, then SIGKILL if the process survives. Return True iff it is gone.

    uvicorn's graceful shutdown blocks on long-lived SSE connections, so a plain
    SIGTERM can hang forever — we must escalate. `kill(pid, sig)` and `alive(pid)`
    are injected for testability; `sleeper` runs between polls.
    """
    if sleeper is None:
        sleeper = lambda: time.sleep(0.2)
    try:
        kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    for _ in range(term_polls):
        if not alive(pid):
            return True
        sleeper()
    try:
        kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    for _ in range(kill_polls):
        if not alive(pid):
            return True
        sleeper()
    return not alive(pid)


@app.command()
def stop() -> None:
    """Stop a daemonized ClaudeWatch server."""
    pid = _is_running()
    if not pid:
        console.print("[yellow]No running server found[/yellow]")
        return

    def _alive(p: int) -> bool:
        try:
            os.kill(p, 0)
            return True
        except ProcessLookupError:
            return False

    gone = _escalating_stop(pid, os.kill, _alive)
    PID_FILE.unlink(missing_ok=True)
    if gone:
        console.print(f"[green]Stopped PID {pid}[/green]")
    else:
        console.print(f"[red]Failed to stop PID {pid} — still running[/red]")


@app.command()
def status() -> None:
    """Show whether the server is running."""
    pid = _is_running()
    if pid:
        console.print(f"[green]Running[/green] · PID {pid} · {_server_url()}")
        try:
            stats = _http_get("/api/stats")
            console.print(f"  active sessions: [bold]{stats.get('active', 0)}[/bold]")
        except Exception:
            console.print("  (no response from API yet)")
    else:
        console.print("[yellow]Not running[/yellow]")


@app.command(name="open")
def open_browser() -> None:
    """Open the dashboard in your default browser."""
    webbrowser.open(_server_url())


@app.command()
def sessions(once: bool = typer.Option(False, "--once", help="Print one snapshot and exit")) -> None:
    """Show active sessions in a live terminal table."""

    def render() -> Table:
        try:
            data = _http_get("/api/sessions")
        except Exception as e:
            t = Table(title="ClaudeWatch (server unreachable)")
            t.add_column("error")
            t.add_row(str(e))
            return t
        t = Table(title=f"ClaudeWatch · {len(data)} sessions · {_server_url()}")
        t.add_column("PID", justify="right")
        t.add_column("Status")
        t.add_column("Loc")
        t.add_column("Model")
        t.add_column("cwd", overflow="fold")
        t.add_column("Tokens", justify="right")
        t.add_column("Cost", justify="right")
        t.add_column("Tools", justify="right")
        for s in sorted(data, key=lambda x: -(x.get("usage", {}) or {}).get("cost_estimate_usd", 0) or 0):
            usage = s.get("usage") or {}
            cost = usage.get("cost_estimate_usd")
            t.add_row(
                str(s.get("pid")),
                s.get("status", ""),
                s.get("location_type", ""),
                s.get("model") or "—",
                s.get("cwd") or "—",
                f"{usage.get('total_tokens', 0):,}",
                f"${cost:.2f}" if cost else "—",
                str((s.get("tool_calls") or {}).get("total", 0)),
            )
        return t

    if once:
        console.print(render())
        return
    with Live(render(), refresh_per_second=1, console=console) as live:
        try:
            while True:
                time.sleep(2)
                live.update(render())
        except KeyboardInterrupt:
            return


@app.command()
def info(pid: int) -> None:
    """Show full detail for a session."""
    try:
        d = _http_get(f"/api/sessions/{pid}")
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        return
    console.print_json(json.dumps(d))


@app.command()
def new(directory: str = typer.Argument(..., help="Working directory")) -> None:
    """Open a new Claude session in a new iTerm window."""
    import urllib.request

    body = json.dumps(
        {
            "cwd": str(Path(directory).expanduser().resolve()),
            "window_type": "new-window",
            "flags": ["--dangerously-skip-permissions"],
            "command": "claude",
        }
    ).encode()
    req = urllib.request.Request(
        _server_url() + "/api/sessions/new",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            console.print(resp.read().decode())
    except Exception as e:
        console.print(f"[red]{e}[/red]")


@app.command()
def logs(tail: int = typer.Option(100, "-n", help="Lines to tail")) -> None:
    """Tail the server log."""
    log_path = LOGS_DIR / "server.log"
    if not log_path.is_file():
        console.print("[yellow]No log file yet[/yellow]")
        return
    subprocess.run(["tail", "-n", str(tail), "-f", str(log_path)])


@app.command()
def config() -> None:
    """Open the config TOML in $EDITOR."""
    editor = os.environ.get("EDITOR", "nano")
    ensure_config_dir()
    if not CONFIG_PATH.exists():
        load_config()  # writes default
    subprocess.run([editor, str(CONFIG_PATH)])


@app.command()
def pricing() -> None:
    """Edit pricing (alias for `config`; the [pricing] table lives there)."""
    config()


@app.command()
def uninstall() -> None:
    """Remove ~/.claudewatch/ data (does NOT remove the package)."""
    import shutil

    pid = _is_running()
    if pid:
        console.print("[yellow]Server is running — stop it first with `claudewatch stop`[/yellow]")
        raise typer.Exit(1)
    if not typer.confirm("Delete ~/.claudewatch/ (config + history + logs)?"):
        return
    from backend.config import CONFIG_DIR

    if CONFIG_DIR.exists():
        shutil.rmtree(CONFIG_DIR)
    console.print("[green]Removed ~/.claudewatch/[/green]")


if __name__ == "__main__":
    app()

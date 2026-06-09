from __future__ import annotations

import subprocess
from pathlib import Path

from backend.models import GitContext


def get_git_context(cwd: str, timeout: float = 1.0) -> GitContext | None:
    p = Path(cwd)
    if not (p / ".git").exists():
        return None
    branch: str | None = None
    porcelain = ""
    try:
        r1 = subprocess.run(
            ["git", "-C", str(p), "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r1.returncode == 0:
            branch = r1.stdout.strip() or None
        r2 = subprocess.run(
            ["git", "-C", str(p), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r2.returncode == 0:
            porcelain = r2.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return GitContext(branch=branch, is_dirty=False, modified_count=0)
    lines = [ln for ln in porcelain.splitlines() if ln.strip()]

    insertions = deletions = 0
    if lines:  # only pay for the diff when the tree is actually dirty
        try:
            r3 = subprocess.run(
                ["git", "-C", str(p), "diff", "--numstat", "HEAD"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if r3.returncode == 0:
                for ln in r3.stdout.splitlines():
                    cols = ln.split("\t")
                    if len(cols) >= 2 and cols[0].isdigit() and cols[1].isdigit():
                        insertions += int(cols[0])
                        deletions += int(cols[1])
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return GitContext(
        branch=branch,
        is_dirty=bool(lines),
        modified_count=len(lines),
        insertions=insertions,
        deletions=deletions,
    )

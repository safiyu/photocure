"""PID file location shared by the server and the stop/kill commands."""
from __future__ import annotations

import os
from pathlib import Path


def pid_file_path(port: int) -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) / "photocure" if runtime else Path.home() / ".cache" / "photocure" / "run"
    return base / f"{port}.pid"

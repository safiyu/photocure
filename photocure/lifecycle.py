"""PID file location shared by the server and the stop/kill commands."""
from __future__ import annotations

import os
from pathlib import Path


def pid_file_path(port: int) -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) / "photocure" if runtime else Path.home() / ".cache" / "photocure" / "run"
    return base / f"{port}.pid"


def process_cmdline(pid: int) -> str | None:
    """Command line of ``pid`` (Linux /proc, otherwise ps), or None if unreadable."""
    import platform
    import subprocess
    try:
        if platform.system() == "Linux":
            return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ").strip()
        out = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


_SERVER_COMMANDS = {"serve", "start", "setup"}


def is_photocure_server(cmdline: str | None) -> bool:
    """True for ``photocure serve|start|setup`` (also ``python -m photocure ...``).

    A bare substring match would also hit editors or tails that merely mention the name.
    """
    if not cmdline:
        return False
    tokens = cmdline.split()
    names = {Path(t).name for t in tokens}
    runs_photocure = "photocure" in names or (
        "-m" in tokens and tokens.index("-m") + 1 < len(tokens)
        and tokens[tokens.index("-m") + 1] == "photocure")
    return runs_photocure and any(t in _SERVER_COMMANDS for t in tokens)


def running_servers() -> list[int]:
    """Ports with a live Photocure server according to the PID files."""
    import os
    base = pid_file_path(0).parent
    ports = []
    for f in base.glob("*.pid"):
        try:
            pid = int(f.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, OSError):
            continue
        if is_photocure_server(process_cmdline(pid)):
            ports.append(int(f.stem))
    return sorted(ports)

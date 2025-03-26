#!/usr/bin/env python3
"""Photocure first-launch bootstrap (Python standard library only).

Installed at /opt/photocure/bootstrap.py and started by /usr/bin/photocure when the
per-user environment is missing (first run) or older than the package (upgrade).

First run builds ~/.local/share/photocure/venv from the bundled wheel. Upgrade only
upgrades the wheel inside the existing venv, so PyTorch, CUDA libraries and cached model
weights are kept. While it works, a small page on the dashboard port shows progress;
/healthz answers 503 until the real server takes over, and the page reloads on 200.
"""
from __future__ import annotations

import http.server
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

APP_DIR = Path(os.environ.get("PCUR_APP_DIR", "/opt/photocure"))
SERVER_COMMANDS = ("serve", "start")
PASSTHROUGH = ("stop", "kill", "--version", "-V", "version")

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Photocure</title>
<style>body{font:16px system-ui,sans-serif;display:grid;place-items:center;min-height:100vh;margin:0;
background:#0c0e12;color:#f1f3f7}main{max-width:32rem;padding:2rem}h1{font-size:1.4rem;margin:0 0 .5rem}
p{color:#949eb2;margin:.25rem 0}.err{color:#ff8a80}</style></head><body><main>
<h1>Setting up Photocure</h1><p id="step">Starting...</p>
<p>The first launch installs Photocure for your user account. This page reloads when it is ready.</p>
<p id="err" class="err" role="alert"></p></main>
<script>
async function poll() {
  try {
    const r = await fetch('/healthz', {cache: 'no-store'});
    if (r.status === 200) { location.reload(); return; }
    const j = await r.json();
    document.getElementById('step').textContent = j.step || '';
    if (j.error) { document.getElementById('err').textContent = j.error; return; }
  } catch (e) { /* server is switching over */ }
  setTimeout(poll, 1000);
}
poll();
</script></body></html>
"""


# ---------------------------------------------------------------- helpers
def user_venv() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "photocure" / "venv"


def read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def newest_wheel(app_dir: Path = APP_DIR) -> Path | None:
    wheels = sorted((app_dir / "wheels").glob("photocure-*.whl"))
    return wheels[-1] if wheels else None


def option_value(args: list[str], names: tuple[str, ...]) -> str | None:
    for i, a in enumerate(args):
        if a in names and i + 1 < len(args):
            return args[i + 1]
        for n in names:
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def requested_port(args: list[str]) -> int:
    raw = option_value(args, ("--port", "-p")) or os.environ.get("PCUR_PORT") or "8080"
    try:
        return int(raw)
    except ValueError:
        return 8080


def with_port(args: list[str], port: int) -> list[str]:
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in ("--port", "-p"):
            skip = True
        elif not a.startswith("--port="):
            out.append(a)
    return out + ["--port", str(port)]


def find_port(host: str, start: int, tries: int = 50) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port found from {start}")


def exec_args(venv: Path, args: list[str], port: int | None) -> list[str]:
    """Command for the real CLI. Servers get the chosen port and never open a browser themselves."""
    cmd = [str(venv / "bin" / "photocure"), *args]
    if port is not None:
        cmd = [cmd[0], *with_port(args, port)]
        if "--no-open" not in cmd:
            cmd.append("--no-open")
        cmd = [c for c in cmd if c != "--open"]
    return cmd


# --------------------------------------------------------------- progress
class Progress:
    def __init__(self) -> None:
        self.step = "Starting..."
        self.error: str | None = None

    def set(self, step: str) -> None:
        self.step = step
        print(step, file=sys.stderr, flush=True)

    def fail(self, message: str) -> None:
        self.error = message
        print(message, file=sys.stderr, flush=True)


def make_handler(progress: Progress):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                body = {"status": "provisioning", "step": progress.step}
                if progress.error:
                    body["error"] = progress.error
                self._send(500 if progress.error else 503, json.dumps(body).encode(), "application/json")
            elif self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *a) -> None:
            pass

    return Handler


def serve_progress(host: str, port: int, progress: Progress):
    server = http.server.ThreadingHTTPServer((host, port), make_handler(progress))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


# -------------------------------------------------------------- provision
def provision(venv: Path, wheel: Path, version: str, progress: Progress,
              upgrade: bool = False, run=subprocess.run) -> None:
    """Build or upgrade the user environment. The version marker is written last."""
    marker = venv / ".photocure_version"
    py = str(venv / "bin" / "python")
    if upgrade:
        progress.set("Upgrading Photocure...")
    else:
        progress.set("Creating the Python environment...")
        if venv.exists():  # a previous attempt that never finished
            shutil.rmtree(venv)
        venv.parent.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "-m", "venv", str(venv)], check=True)
        progress.set("Installing Photocure (this can take a few minutes)...")
    cmd = [py, "-m", "pip", "install", "--no-cache-dir"] + (["--upgrade"] if upgrade else []) + [str(wheel)]
    run(cmd, check=True)
    marker.write_text(version + "\n")


# -------------------------------------------------------------- commands
def stop_without_venv(args: list[str]) -> int:
    """`photocure stop` before any environment exists: only the progress page can be running."""
    port = int(option_value(args, ("--port", "-p")) or os.environ.get("PCUR_PORT") or 8080)
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) / "photocure" if runtime else Path.home() / ".cache" / "photocure" / "run"
    pid_file = base / f"{port}.pid"
    if not pid_file.exists():
        print(f"No active Photocure server found on port {port}.")
        return 0
    try:
        pid = int(pid_file.read_text().strip())
        cmdline = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True,
                                 text=True, timeout=3).stdout
    except (ValueError, OSError, subprocess.SubprocessError):
        pid_file.unlink(missing_ok=True)
        print(f"No active Photocure server found on port {port}.")
        return 0
    if "bootstrap.py" not in cmdline:
        print(f"PID {pid} is not a Photocure process. Nothing was stopped.")
        return 1
    os.kill(pid, signal.SIGTERM)
    pid_file.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    upgrade = bool(args) and args[0] == "--upgrade"
    if upgrade:
        args = args[1:]
    venv = user_venv()
    cli = venv / "bin" / "photocure"
    marker = venv / ".photocure_version"
    system_version = read_text(APP_DIR / "VERSION")
    command = args[0] if args else ""

    if command in PASSTHROUGH:
        if cli.exists():  # never upgrade or install just to stop a server or print a version
            os.execv(str(cli), [str(cli), *args])
        if command in ("--version", "-V", "version"):
            print(f"photocure {system_version or 'unknown'}")
            return 0
        return stop_without_venv(args)

    complete = cli.exists() and marker.exists()
    if not upgrade and complete:
        os.execv(str(cli), [str(cli), *args])
    wheel = newest_wheel()
    if wheel is None:
        print(f"No Photocure wheel found in {APP_DIR / 'wheels'}. Reinstall the package.", file=sys.stderr)
        return 1

    progress = Progress()
    is_server = command in SERVER_COMMANDS
    port = None
    otc = None
    server = None
    if is_server:
        host = option_value(args, ("--host",)) or os.environ.get("PCUR_HOST") or "127.0.0.1"
        port = find_port(host, requested_port(args))
        server, _ = serve_progress(host, port, progress)
        otc = secrets.token_urlsafe(32)
        shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
        url = f"http://{shown}:{port}/#code={otc}"
        if "--no-open" in args:
            print(f"Open {url} (single use) or paste the code into the sign-in box.", file=sys.stderr)
        else:
            webbrowser.open(url)

    try:
        provision(venv, wheel, system_version, progress, upgrade=upgrade and venv.exists())
    except (subprocess.CalledProcessError, OSError) as e:
        progress.fail(f"Setup failed: {e}. Run the command again, or reinstall the package.")
        if server is not None:
            time.sleep(60)  # keep the explanation on screen
        return 1

    if server is not None:
        progress.set("Starting Photocure...")
        server.shutdown()
        server.server_close()
        os.environ["PCUR_OTC"] = otc
    os.execv(str(cli), exec_args(venv, args, port))
    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())

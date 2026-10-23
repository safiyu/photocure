"""One command setup: detect hardware, install the AI libraries, fetch the model, open the UI.

Everything here is split into small functions that take their side effects (running
commands, looking up programs) as arguments, so the logic is testable without
installing anything.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import site
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

TORCH_BASE = "https://download.pytorch.org/whl/"
TORCH_INDEX = {
    "cuda": TORCH_BASE + "cu124",
    "rocm": TORCH_BASE + "rocm6.2",
    "xpu": TORCH_BASE + "xpu",
    "cpu": TORCH_BASE + "cpu",
}
KIND_LABEL = {
    "cuda": "NVIDIA GPU (CUDA)", "rocm": "AMD GPU (ROCm)", "xpu": "Intel GPU (XPU)",
    "mps": "Apple GPU (Metal)", "cpu": "CPU only",
}
# Rough download sizes in MB, shown before anything big is fetched.
TORCH_MB = {"cuda": 2800, "rocm": 3500, "xpu": 2200, "mps": 90, "cpu": 200}
CLIP_WEIGHTS_MB = 350
MEDIAPIPE_MB = 200

# Reported by a fresh interpreter, so this process never imports torch itself.
_PROBE = r"""
import json
out = {"torch": False, "open_clip": None, "mediapipe": False}
try:
    import torch
    out.update(torch=True, version=torch.__version__,
               cuda_build=getattr(torch.version, "cuda", None) is not None,
               hip_build=getattr(torch.version, "hip", None) is not None,
               has_xpu=getattr(torch, "xpu", None) is not None,
               cuda_ok=bool(torch.cuda.is_available()),
               mps_ok=bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
except Exception as e:
    out["error"] = str(e)[:200]
try:
    import open_clip
    out["open_clip"] = open_clip.__version__
except Exception:
    pass
import importlib.util
out["mediapipe"] = importlib.util.find_spec("mediapipe") is not None
print(json.dumps(out))
"""

_VERIFY = r"""
import json, time
import numpy as np
from photocure.config import CuratorConfig
from photocure.inference.backends import describe_backend
from photocure.inference.semantic_engine import SemanticEngine
cfg = CuratorConfig()
t = time.time()
eng = SemanticEngine(cfg)
eng.classify_and_embed([np.full((224, 224, 3), 127, np.uint8)])
info = describe_backend(eng.bs)
print(json.dumps({"ok": eng.bs.semantic_backend == "clip", "device": info.get("selected"),
                  "backend": eng.bs.semantic_backend, "seconds": round(time.time() - t, 1),
                  "model": cfg.taxonomy.clip_model_name}))
"""

@dataclass
class Step:
    title: str
    cmd: list[str] = field(default_factory=list)
    optional: bool = False       # a failure is reported but does not stop setup
    size_mb: int = 0
    note: str = ""
    after: list[list[str]] = field(default_factory=list)  # run only if the main command worked


Runner = Callable[..., "subprocess.CompletedProcess"]


# ------------------------------------------------------------------ detection
def _render_vendors(dev_root: str, sys_root: str) -> set[str]:
    """PCI vendor ids of the GPUs whose render nodes are visible, e.g. {'0x8086'}."""
    out: set[str] = set()
    dri = os.path.join(dev_root, "dri")
    try:
        nodes = [n for n in os.listdir(dri) if n.startswith("renderD")]
    except OSError:
        return out
    for n in nodes:
        try:
            with open(os.path.join(sys_root, "class", "drm", n, "device", "vendor")) as f:
                out.add(f.read().strip().lower())
        except OSError:
            continue
    return out


def detect_kind(which: Callable[[str], str | None] = shutil.which,
                system: str | None = None, machine: str | None = None,
                dev_root: str = "/dev", sys_root: str = "/sys",
                env: dict | None = None) -> str:
    """Best accelerator guess: cuda, rocm, xpu, mps or cpu.

    Looks at installed tools and at device nodes, so it also works inside a
    container where only the devices are passed through. ``PCUR_ACCEL`` overrides.
    """
    env = os.environ if env is None else env
    forced = (env.get("PCUR_ACCEL") or "").strip().lower()
    if forced in KIND_LABEL:
        return forced
    system = system or platform.system()
    machine = machine or platform.machine()
    linux = system == "Linux"
    exists = lambda *p: os.path.exists(os.path.join(dev_root, *p))  # noqa: E731
    if which("nvidia-smi") or (linux and (exists("nvidiactl") or exists("nvidia0"))):
        return "cuda"
    if linux and (which("rocm-smi") or which("rocminfo") or exists("kfd")):
        return "rocm"
    vendors = _render_vendors(dev_root, sys_root) if linux else set()
    if which("sycl-ls") or which("xpu-smi") or "0x8086" in vendors:
        return "xpu"
    if system == "Darwin" and machine == "arm64":
        return "mps"
    return "cpu"


def probe_state(runner: Runner = subprocess.run, python: str | None = None) -> dict:
    """What the ML stack looks like right now, measured in a clean child process."""
    try:
        r = runner([python or sys.executable, "-c", _PROBE], capture_output=True,
                   text=True, timeout=180)
        lines = [ln for ln in (r.stdout or "").strip().splitlines() if ln.startswith("{")]
        return json.loads(lines[-1]) if lines else {"torch": False}
    except Exception as e:  # a broken interpreter must not crash setup
        return {"torch": False, "error": str(e)[:200]}


def torch_needs_reinstall(kind: str, st: dict) -> bool:
    """True when torch is present but built for the wrong hardware."""
    if not st.get("torch"):
        return False
    if kind == "cuda":
        return not st.get("cuda_build")
    if kind == "rocm":
        return not st.get("hip_build")
    if kind == "xpu":
        return not st.get("has_xpu")
    return False


# --------------------------------------------------------------------- plan
def user_install_wanted() -> bool:
    """Install into the user site when PYTHONUSERBASE is set (the Docker data volume)."""
    return bool(os.environ.get("PYTHONUSERBASE")) and sys.prefix == sys.base_prefix


def activate_user_site() -> None:
    """Make libraries installed during this run importable without restarting."""
    try:
        path = site.getusersitepackages()
        if os.path.isdir(path):
            site.addsitedir(path)
    except Exception:
        pass


def plan_steps(kind: str, st: dict, *, faces: bool = True, force: bool = False,
               python: str | None = None, system: str | None = None,
               user: bool | None = None) -> list[Step]:
    """The ordered list of things that still need doing. Empty means all set."""
    py = python or sys.executable
    system = system or platform.system()
    user = user_install_wanted() if user is None else user
    pip = [py, "-m", "pip", "install"] + (["--user"] if user else [])
    steps: list[Step] = []

    wrong = torch_needs_reinstall(kind, st)
    if not st.get("torch") or wrong or force:
        index_kind = kind if kind in TORCH_INDEX else None
        if kind == "cpu" and system == "Darwin":
            index_kind = None  # macOS wheels are already CPU and Metal builds
        if kind == "mps":
            index_kind = None
        cmd = pip + (["--upgrade", "--force-reinstall"] if (wrong or force) and st.get("torch") else []) \
            + ["torch", "torchvision"]
        override = (os.environ.get("PCUR_TORCH_INDEX") or "").strip()
        if override and kind != "mps":
            cmd += ["--index-url", override]  # for example an older CUDA build
        elif index_kind:
            cmd += ["--index-url", TORCH_INDEX[index_kind]]
        why = "wrong build for this hardware, replacing it" if wrong else ""
        steps.append(Step(f"Install PyTorch for {KIND_LABEL.get(kind, kind)}", cmd,
                          size_mb=TORCH_MB.get(kind, 200), note=why))
    if not st.get("open_clip"):
        steps.append(Step("Install OpenCLIP (the photo classifier)",
                          pip + ["open-clip-torch>=2.20"], size_mb=30))
    if faces and not st.get("mediapipe"):
        # MediaPipe pulls in the GUI build of OpenCV, which shadows the headless one this
        # project ships and needs desktop libraries in containers. Put the headless build back.
        steps.append(Step("Install blink detection (MediaPipe)",
                          pip + ["mediapipe>=0.10.14"], optional=True, size_mb=MEDIAPIPE_MB,
                          after=[[py, "-m", "pip", "uninstall", "-y", "opencv-contrib-python", "opencv-python"],
                                 pip + ["--force-reinstall", "--no-deps", "opencv-python-headless"]],
                          note="Optional. Without it faces are still found, but eyes are not "
                               "measured, so no blink badges appear."))
    return steps


# ----------------------------------------------------------------------- run
def run_step(step: Step, runner: Runner = subprocess.run) -> bool:
    if runner(step.cmd).returncode != 0:
        return False
    for extra in step.after:  # tidy-up commands; their failure does not undo the install
        runner(extra)
    return True


def verify_model(runner: Runner = subprocess.run, python: str | None = None) -> dict:
    """Load the real model in a clean process (downloads weights once if missing)."""
    env = {k: v for k, v in os.environ.items() if k != "PCUR_BACKEND"}
    try:
        r = runner([python or sys.executable, "-c", _VERIFY], capture_output=True,
                   text=True, env=env, timeout=1800)
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}
    lines = [ln for ln in (r.stdout or "").strip().splitlines() if ln.startswith("{")]
    if r.returncode == 0 and lines:
        return json.loads(lines[-1])
    tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
    return {"ok": False, "error": " | ".join(tail)[:500] or "model check failed"}


# ------------------------------------------------------------------- serving
def pick_port(host: str, port: int, tries: int = 20) -> int:
    """First free port at or above ``port`` (so a running server does not block setup)."""
    bind = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        family = socket.getaddrinfo(bind, None, type=socket.SOCK_STREAM)[0][0]
    except OSError:
        family = socket.AF_INET
    for p in range(port, port + tries):
        with socket.socket(family, socket.SOCK_STREAM) as s:
            if os.name == "nt":  # SO_REUSEADDR would let bind succeed on a busy port here
                s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((bind, p))
                return p
            except OSError:
                continue
    return port


def open_browser_when_ready(host: str, port: int, timeout: float = 180.0,
                            opener: Callable[[str], object] = webbrowser.open,
                            fetch: Callable[..., object] = urllib.request.urlopen,
                            code: str | None = None) -> threading.Thread:
    """Open the browser as soon as the server answers its health check."""
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{shown}:{port}/"

    def wait() -> None:
        end = time.time() + timeout
        while time.time() < end:
            try:
                fetch(url + "healthz", timeout=2)
                opener(url + (f"#code={code}" if code else ""))
                return
            except Exception:
                time.sleep(0.5)

    t = threading.Thread(target=wait, daemon=True)
    t.start()
    return t


def total_download_mb(steps: Sequence[Step], weights_cached: bool) -> int:
    return sum(s.size_mb for s in steps) + (0 if weights_cached else CLIP_WEIGHTS_MB)

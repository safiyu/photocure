"""Runtime capability detection & device selection for the inference layer.

Supported accelerators: NVIDIA (cuda), AMD (rocm, which torch exposes through
the cuda API), Intel GPUs incl. integrated Iris Xe and Arc (xpu), Apple (mps),
Windows DirectML (best effort) and cpu. ``auto`` tries them in that order and a
requested device that is missing falls down the same order, with the reason
recorded. When torch or the AI model is absent the semantic stage uses a
deterministic stub so the pipeline stays exercisable end to end.
"""
from __future__ import annotations

import logging
import os
import platform
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..config import (
    CuratorConfig,
    DeviceSettings,
    auto_decode_workers,
    auto_torch_threads,
)

log = logging.getLogger("photocure.backends")

_lock = threading.Lock()
_cached: BackendState | None = None
_stub_warned = False
_threads_applied = False

# Preference order for "auto" and for falling back from an unavailable device.
_DEVICE_ORDER = ("cuda", "rocm", "xpu", "mps", "directml", "cpu")
# Aliases that expand to a preference list instead of one device.
_ALIASES = {"integrated": ("xpu", "mps")}


def in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def pip_install_cmd(spec: str) -> str:
    """A command that is safe to show a user.

    Inside a venv this is that venv's own pip, never a bare ``pip`` that may belong to
    another Python. Outside one, installing into the system Python is not advised, so
    point to the guided setup instead.
    """
    if in_venv():
        return f"{Path(sys.prefix) / 'bin' / 'pip'} install {spec}"
    return "photocure setup --reinstall"


def ml_install_cmd() -> str:
    return pip_install_cmd('"photocure[ml]"')

_LABELS = {
    "cuda": "NVIDIA GPU (CUDA)",
    "rocm": "AMD GPU (ROCm)",
    "xpu": "Intel GPU (Iris Xe, Arc)",
    "mps": "Apple GPU (Metal)",
    "directml": "Windows GPU (DirectML)",
    "cpu": "CPU",
}
_TORCH_INDEX_URLS = {
    "cuda": "https://download.pytorch.org/whl/cu124",
    "rocm": "https://download.pytorch.org/whl/rocm6.2",
    "xpu": "https://download.pytorch.org/whl/xpu",
}


def _install_hint(acc: str) -> str:
    if acc in _TORCH_INDEX_URLS:
        return pip_install_cmd(f"torch --index-url {_TORCH_INDEX_URLS[acc]}")
    if acc == "mps":
        return "Needs an Apple Silicon Mac with macOS 12.3 or newer and a current PyTorch"
    if acc == "directml":
        return pip_install_cmd("torch-directml")
    return ""


# onnxruntime execution providers to prefer per accelerator (CPU always last).
_PROVIDER_PREFS = {
    "cuda": ("CUDAExecutionProvider",),
    "rocm": ("ROCMExecutionProvider", "MIGraphXExecutionProvider"),
    "xpu": ("OpenVINOExecutionProvider",),
    "mps": ("CoreMLExecutionProvider",),
    "directml": ("DmlExecutionProvider",),
}
# Accelerators where fp16 autocast is worthwhile.
_FP16_ACCELERATORS = ("cuda", "rocm", "xpu")


@dataclass
class BackendState:
    device: str                 # torch device string: cuda | xpu | mps | cpu | ...
    fp16: bool
    torch: object | None
    cuda_available: bool
    memory_fraction: float
    # Face backend: "yunet" | "stub"
    face_backend: str
    # Semantic backend: "clip" | "stub"
    semantic_backend: str
    mps_available: bool = False
    # Accelerator id: cuda | rocm | xpu | mps | directml | cpu (rocm uses device "cuda").
    accelerator: str = ""
    requested: str = "auto"
    fallback_reason: str = ""
    device_index: int = 0
    torch_threads: int = 0
    decode_workers: int = 0
    probes: dict = field(default_factory=dict)
    # Detected backends; engines restore these after a transient load failure downgraded them.
    face_wanted: str = ""
    semantic_wanted: str = ""

    def __post_init__(self) -> None:
        self.face_wanted = self.face_wanted or self.face_backend
        self.semantic_wanted = self.semantic_wanted or self.semantic_backend

    def retry_downgraded(self) -> None:
        """Undo a downgrade to stub caused by a failed model load, so it is tried again."""
        self.face_backend = self.face_wanted
        self.semantic_backend = self.semantic_wanted
        if not self.accelerator:
            self.accelerator = self.device if self.device in _DEVICE_ORDER else (
                "cpu" if self.device == "cpu" else self.device)


def _force_env() -> str:
    return os.environ.get("PCUR_BACKEND", "").strip().lower()


# ---------------------------------------------------------------- workers
def _cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def resolve_decode_workers(config: CuratorConfig | None = None) -> int:
    """Image decode threads: the configured count, or auto = min(8, cpu_count)."""
    ds = getattr(config or CuratorConfig(), "device", None)
    n = int(getattr(ds, "decode_workers", 0) or 0)
    return n if n > 0 else auto_decode_workers(_cpu_count())


def resolve_torch_threads(config: CuratorConfig | None = None,
                          accelerator: str = "cpu") -> int:
    """Torch CPU threads: configured, else all cores (max 8) on CPU, 2 with a GPU."""
    ds = getattr(config or CuratorConfig(), "device", None)
    n = int(getattr(ds, "torch_threads", 0) or 0)
    return n if n > 0 else auto_torch_threads(accelerator == "cpu", _cpu_count())


def configure_threads_env(config: CuratorConfig | None = None) -> bool:
    """Set OMP_NUM_THREADS before torch/onnxruntime load, if that is still possible.

    OpenMP reads the variable once at library load, so it is only set when
    neither library is imported yet and the user has not set it. Returns True
    when it was set. numpy and OpenCV are normally loaded already and keep
    their own thread settings.
    """
    if "OMP_NUM_THREADS" in os.environ:
        return False
    if "torch" in sys.modules or "onnxruntime" in sys.modules:
        return False
    os.environ["OMP_NUM_THREADS"] = str(resolve_torch_threads(config, "cpu"))
    return True


def apply_thread_settings(torch, n_threads: int) -> None:
    """Apply torch thread counts once per process (interop can only be set once)."""
    global _threads_applied
    if torch is None or _threads_applied:
        return
    _threads_applied = True
    for fn, value in (("set_num_threads", n_threads),
                      ("set_num_interop_threads", 2 if n_threads >= 2 else 1)):
        try:
            getattr(torch, fn)(value)
        except Exception as e:  # interop raises once parallel work has started
            log.debug("torch.%s(%s) not applied: %s", fn, value, e)


# ------------------------------------------------------------ accelerators
def _import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _hip_build(torch) -> bool:
    return bool(getattr(getattr(torch, "version", None), "hip", None))


def _mps_available(torch) -> bool:
    """True when this torch build can use Apple's Metal (MPS) backend."""
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(mps is not None and _safe(mps.is_available, False))


def _xpu_available(torch) -> bool:
    xpu = getattr(torch, "xpu", None)
    return bool(xpu is not None and _safe(xpu.is_available, False))


def _directml_module():
    try:
        import torch_directml
        return torch_directml
    except Exception:
        return None


def _enumerate(api, props) -> list[dict]:
    """[{index, name, memory_gb}] for every GPU the torch backend can see."""
    count_fn = getattr(api, "device_count", None)
    n = _safe(count_fn, 0) if callable(count_fn) else 0
    name_fn = getattr(api, "get_device_name", None)
    out = []
    for i in range(int(n or 0)):
        total = _safe(lambda i=i: props(i).total_memory)
        name = _safe(lambda i=i: name_fn(i), "") if callable(name_fn) else ""
        out.append({"index": i, "name": name or f"GPU {i}",
                    "memory_gb": round(total / 1e9, 1) if total else None})
    return out


def _probe_one(acc: str, torch, index: int = 0) -> dict:
    """Describe one accelerator: availability, a reason when missing, name, memory."""
    info = {"id": acc, "label": _LABELS[acc], "name": "", "kind": "discrete",
            "memory_gb": None, "available": False, "reason": "",
            "install_hint": _install_hint(acc)}
    if acc == "cpu":
        info.update(kind="cpu", available=True,
                    name=f"{platform.processor() or platform.machine() or 'CPU'} "
                         f"({_cpu_count()} threads)")
        return info
    if torch is None and acc != "directml":
        info.update(reason="PyTorch is not installed", install_hint=ml_install_cmd())
        return info

    if acc in ("cuda", "rocm"):
        hip = _hip_build(torch)
        cuda_api = bool(_safe(torch.cuda.is_available, False)) if hasattr(torch, "cuda") else False
        if acc == "cuda":
            if hip:
                info["reason"] = "this PyTorch build is for AMD GPUs (ROCm)"
            elif not hasattr(torch, "cuda") or getattr(getattr(torch, "version", None), "cuda", "x") is None:
                info["reason"] = "torch build has no CUDA support"
            elif not cuda_api:
                info["reason"] = "no NVIDIA GPU or driver detected"
        else:
            if not hip:
                info["reason"] = "torch build has no ROCm support"
            elif not cuda_api:
                info["reason"] = "no AMD GPU or ROCm driver detected"
        if not info["reason"]:
            info["available"] = True
            info["name"] = _safe(lambda: torch.cuda.get_device_name(index), "") or ""
            total = _safe(lambda: torch.cuda.get_device_properties(index).total_memory)
            info["memory_gb"] = round(total / 1e9, 1) if total else None
            info["devices"] = _enumerate(torch.cuda, getattr(torch.cuda, "get_device_properties", lambda i: None))
            info["count"] = len(info["devices"])
        return info

    if acc == "xpu":
        if getattr(torch, "xpu", None) is None:
            info["reason"] = "torch build has no Intel GPU (XPU) support"
        elif not _xpu_available(torch):
            info["reason"] = "no Intel GPU or driver detected"
        else:
            name = _safe(lambda: torch.xpu.get_device_name(index), "") or ""
            low = name.lower()
            discrete = any(w in low for w in ("arc", "max", "flex")) and "graphics" not in low
            total = _safe(lambda: torch.xpu.get_device_properties(index).total_memory)
            info.update(available=True, name=name,
                        kind="discrete" if discrete else "integrated",
                        memory_gb=round(total / 1e9, 1) if total else None)
            info["devices"] = _enumerate(torch.xpu, getattr(torch.xpu, "get_device_properties", lambda i: None))
            info["count"] = len(info["devices"])
        return info

    if acc == "mps":
        info["kind"] = "integrated"
        if getattr(getattr(torch, "backends", None), "mps", None) is None:
            info["reason"] = "torch build has no Metal (MPS) support"
        elif not _mps_available(torch):
            info["reason"] = "needs an Apple Silicon Mac running macOS 12.3 or newer"
        else:
            info.update(available=True, name="Apple GPU (unified memory)")
        return info

    # directml: best effort, only when the optional package imports.
    dml = _directml_module()
    if dml is None:
        info["reason"] = "package not installed"
    elif torch is None:
        info["reason"] = "PyTorch is not installed"
        info["install_hint"] = ml_install_cmd()
    else:
        count = _safe(dml.device_count, 0) or 0
        if count < 1:
            info["reason"] = "no DirectML capable GPU detected"
        else:
            info.update(available=True,
                        name=_safe(lambda: dml.device_name(index), "") or "DirectML GPU")
    return info


def list_accelerators(torch=None, device_index: int = 0) -> list[dict]:
    """Every accelerator id with availability and a reason when unavailable.

    Keys per entry: id, label, name, kind ('discrete'|'integrated'|'cpu'),
    memory_gb (float or None), available (bool), reason (str), install_hint (str);
    GPU entries that are available also carry count and devices [{index, name, memory_gb}].
    """
    torch = torch if torch is not None else _import_torch()
    probes = [_probe_one(acc, torch, device_index) for acc in _DEVICE_ORDER]
    if any(p["available"] and p["id"] != "cpu" for p in probes):
        # A GPU backend already works; another torch build could break it, so do not suggest one.
        for p in probes:
            if not p["available"]:
                p["install_hint"] = ""
    return probes


def _resolve_accelerator(requested: str, probes: dict) -> tuple[str, str]:
    """Pick the best available accelerator at or below the request, plus why."""
    want = (requested or "auto").strip().lower()
    if want in _ALIASES:
        preferred = [d for d in _ALIASES[want] if probes[d]["available"]]
        if preferred:
            return preferred[0], ""
        start = _DEVICE_ORDER.index(_ALIASES[want][-1])
        label = want
    elif want in _DEVICE_ORDER:
        start, label = _DEVICE_ORDER.index(want), want
    else:
        start, label = 0, "auto"
    for dev in _DEVICE_ORDER[start:]:
        if probes[dev]["available"]:
            if label == "auto" or dev == want:
                return dev, ""
            if want in _ALIASES:
                reasons = "; ".join(
                    f"{d}: {probes[d]['reason']}" for d in _ALIASES[want])
            else:
                reasons = probes[want]["reason"]
            return dev, f"{label} is not available ({reasons}); using {dev}."
    return "cpu", ""


def _device_string(acc: str, torch) -> str:
    if acc in ("cuda", "rocm"):
        return "cuda"
    if acc == "directml":
        dml = _directml_module()
        return str(_safe(dml.device, "cpu")) if dml is not None else "cpu"
    return acc


def _apply_device_index(acc: str, torch, index: int) -> None:
    """Select the GPU for multi GPU machines (best effort)."""
    if torch is None or index <= 0:
        return
    try:
        if acc in ("cuda", "rocm"):
            torch.cuda.set_device(index)
        elif acc == "xpu":
            torch.xpu.set_device(index)
    except Exception as e:
        log.warning("could not select device index %s for %s: %s", index, acc, e)


def select_onnx_providers(accelerator: str, available: list[str] | None = None) -> list[str]:
    """onnxruntime providers for an accelerator: preferred ones present, CPU last."""
    if available is None:
        try:
            import onnxruntime
            available = list(onnxruntime.get_available_providers())
        except Exception:
            available = []
    chosen = [p for p in _PROVIDER_PREFS.get(accelerator, ()) if p in available]
    return chosen + ["CPUExecutionProvider"]


# ---------------------------------------------------------------- detection
def detect_backend(config: CuratorConfig | None = None) -> BackendState:
    """Resolve the concrete backends once (cached for the process lifetime)."""
    global _cached
    if _cached is not None:
        return _cached
    with _lock:
        if _cached is not None:
            return _cached

        cfg = config or CuratorConfig()
        ds: DeviceSettings = cfg.device
        configure_threads_env(cfg)

        torch = _import_torch()
        index = int(getattr(ds, "device_index", 0) or 0)
        probes = {p["id"]: p for p in list_accelerators(torch, index)}
        requested = (ds.device or "auto").strip().lower()
        if _force_env() in ("cpu", "cuda"):
            requested = _force_env()  # falls back to cpu when cuda is missing
        accelerator, fallback_reason = _resolve_accelerator(requested, probes)
        device = _device_string(accelerator, torch)
        _apply_device_index(accelerator, torch, index)

        cuda_available = bool(torch is not None and hasattr(torch, "cuda")
                              and _safe(torch.cuda.is_available, False))
        mps_available = bool(torch is not None and _mps_available(torch))

        fp16 = bool(ds.use_fp16 and accelerator in _FP16_ACCELERATORS and torch is not None)
        if fp16 and accelerator == "cuda":
            # Tensor-core fp16 needs compute capability 7.0+ (Volta and newer).
            fp16 = bool(_safe(lambda: torch.cuda.get_device_capability()[0] >= 7, False))

        # Cap PyTorch's share of VRAM on NVIDIA and AMD (headroom for the desktop).
        if accelerator in ("cuda", "rocm") and torch is not None:
            _safe(lambda: torch.cuda.set_per_process_memory_fraction(ds.cuda_memory_fraction))

        threads = resolve_torch_threads(cfg, accelerator)
        apply_thread_settings(torch, threads)

        # Force override (PCUR_BACKEND=cpu / cuda / stub / clip).
        force = _force_env()

        # Face backend.
        if force in ("stub", "off"):
            face_backend = "stub"
        else:
            face_backend = _pick_face_backend(device)

        # Semantic backend.
        if force in ("stub", "off"):
            semantic_backend = "stub"
        elif force == "clip":
            semantic_backend = "clip" if _have_open_clip() else "stub"
        else:
            semantic_backend = _pick_semantic_backend(device)

        _cached = BackendState(
            device=device,
            fp16=fp16,
            torch=torch,
            cuda_available=cuda_available,
            memory_fraction=ds.cuda_memory_fraction,
            face_backend=face_backend,
            semantic_backend=semantic_backend,
            mps_available=mps_available,
            accelerator=accelerator,
            requested=requested,
            fallback_reason=fallback_reason,
            device_index=index,
            torch_threads=threads,
            decode_workers=resolve_decode_workers(cfg),
            probes=probes,
        )
        if fallback_reason:
            log.warning(fallback_reason)
        _warn_if_stub(semantic_backend, force)
        return _cached


def _warn_if_stub(semantic_backend: str, force: str) -> None:
    """Log once when the placeholder classifier runs without being asked for."""
    global _stub_warned
    if semantic_backend != "stub" or force in ("stub", "off") or _stub_warned:
        return
    _stub_warned = True
    log.warning(
        "No AI model is installed, so photos cannot be recognised and most will "
        "be marked uncertain. Run: photocure setup   (or: %s)", ml_install_cmd())


def describe_backend(state: BackendState | None = None) -> dict:
    """Plain summary of what will run, for the CLI and the dashboard.

    Keys: device, semantic_backend, face_backend, semantic_is_stub, forced_by_env,
    face_is_fallback, blink_available, aesthetic_available, aesthetic_model, hint,
    accelerators (see list_accelerators), selected (accelerator id), requested,
    fallback_reason.
    """
    from .aesthetic import get_head
    bs = state or detect_backend()
    stub = bs.semantic_backend == "stub"
    head = get_head()
    # The model can be installed yet switched off by PCUR_BACKEND, which needs a different fix.
    forced = stub and _force_env() in ("stub", "off") and _have_open_clip()
    if forced:
        hint = ("The AI model is installed but switched off by PCUR_BACKEND="
                f"{_force_env()} in your environment. Run: unset PCUR_BACKEND   "
                "then restart the app.")
    elif stub:
        hint = ("The AI model is not installed, so photos cannot be recognised yet. "
                f"Run: photocure setup   (or install it with: {ml_install_cmd()})")
    else:
        hint = ""
    return {
        "forced_by_env": forced,
        "device": bs.device,
        "semantic_backend": bs.semantic_backend,
        "face_backend": bs.face_backend,
        "semantic_is_stub": stub,
        "face_is_fallback": bs.face_backend != "yunet",
        "blink_available": bs.face_backend == "yunet" and _have_mediapipe(),
        "aesthetic_available": not head.stub,
        "aesthetic_model": "" if head.stub else head.name,
        "hint": hint,
        "accelerators": list_accelerators(bs.torch, bs.device_index),
        "selected": bs.accelerator,
        "requested": bs.requested,
        "fallback_reason": bs.fallback_reason,
    }


def _have_mediapipe() -> bool:
    """MediaPipe is an optional extra; without it eye state stays unknown."""
    import importlib.util
    try:
        return importlib.util.find_spec("mediapipe") is not None
    except (ImportError, ValueError):
        return False


def _have_open_clip() -> bool:
    try:
        import open_clip  # noqa: F401
        return True
    except Exception:
        return False


def _pick_face_backend(device: str) -> str:
    """YuNet when OpenCV can run it and the bundled model is intact; otherwise the stub."""
    try:
        import cv2
        from .. import models
        if hasattr(cv2, "FaceDetectorYN") and models.verify_model(models.YUNET):
            return "yunet"
    except Exception:
        pass
    return "stub"


def _pick_semantic_backend(device: str) -> str:
    """Prefer OpenCLIP; fall back to a deterministic stub."""
    if _have_open_clip():
        return "clip"
    return "stub"


def reset_backend_cache() -> None:
    """Clear the cached backend state (used by tests)."""
    global _cached, _stub_warned, _threads_applied
    with _lock:
        _cached = None
        _stub_warned = False
        _threads_applied = False


def is_cuda() -> bool:
    """True when inference actually runs on the torch cuda API (NVIDIA or ROCm)."""
    return detect_backend().device == "cuda"


def is_oom(exc: BaseException) -> bool:
    """True for GPU / onnxruntime out-of-memory failures."""
    try:
        import torch
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "failed to allocate" in msg


def empty_cuda_cache(bs: BackendState) -> None:
    """Release cached accelerator blocks (cuda, rocm, xpu or mps)."""
    if bs.torch is None:
        return
    try:
        if bs.device == "cuda":
            bs.torch.cuda.empty_cache()
            # Drops shared-memory handles that empty_cache alone keeps around.
            try:
                bs.torch.cuda.ipc_collect()
            except Exception:
                pass
        elif bs.device == "xpu":
            bs.torch.xpu.empty_cache()
        elif bs.device == "mps":
            bs.torch.mps.empty_cache()
    except Exception:
        pass

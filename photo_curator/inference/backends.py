"""Runtime capability detection & device selection for the inference layer.

The host has an RTX 4060 (8 GiB) shared with the desktop. When CUDA is present
we run on it with FP16 and a capped memory fraction; otherwise (CI, laptops) we
fall back to CPU with FP32 and deterministic stub backends for face/CLIP so the
pipeline is always exercisable end-to-end.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from ..config import CuratorConfig, DeviceSettings

_lock = threading.Lock()
_cached: BackendState | None = None


@dataclass
class BackendState:
    device: str                 # "cuda" | "cpu"
    fp16: bool
    torch: object | None
    cuda_available: bool
    memory_fraction: float
    # Face backend: "insightface" | "cv2" | "stub"
    face_backend: str
    # Semantic backend: "clip" | "stub"
    semantic_backend: str


def _force_env() -> str:
    return os.environ.get("PCUR_BACKEND", "").strip().lower()


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

        torch = None
        cuda_available = False
        try:
            import torch  # noqa: F401
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            torch = None
            cuda_available = False

        # Device resolution.
        if ds.device == "cuda":
            device = "cuda" if cuda_available else "cpu"
        elif ds.device == "cpu":
            device = "cpu"
        else:  # "auto"
            device = "cuda" if cuda_available else "cpu"

        fp16 = bool(ds.use_fp16 and device == "cuda" and torch is not None)
        if fp16:
            try:
                import torch as _t
                if not _t.cuda.is_bf16_supported() and not _t.cuda.is_half_supported():
                    fp16 = False
            except Exception:
                fp16 = False

        # Cap PyTorch's share of VRAM (keeps headroom for Xorg/Jellyfin).
        if device == "cuda" and torch is not None:
            try:
                torch.cuda.set_per_process_memory_fraction(ds.cuda_memory_fraction)
            except Exception:
                pass

        # Force override (PCUR_BACKEND=cpu / cuda / stub / insightface / clip).
        force = _force_env()

        # Face backend.
        if force in ("stub", "off"):
            face_backend = "stub"
        elif force == "insightface":
            face_backend = "insightface" if _have_insightface() else "cv2"
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
        )
        return _cached


def _have_insightface() -> bool:
    try:
        import insightface  # noqa: F401
        return True
    except Exception:
        return False


def _have_open_clip() -> bool:
    try:
        import open_clip  # noqa: F401
        return True
    except Exception:
        return False


def _pick_face_backend(device: str) -> str:
    """Prefer InsightFace (SCRFD); fall back to OpenCV Haar; then stub."""
    if _have_insightface():
        return "insightface"
    try:
        import cv2  # noqa: F401
        return "cv2"
    except Exception:
        return "stub"


def _pick_semantic_backend(device: str) -> str:
    """Prefer OpenCLIP; fall back to a deterministic stub."""
    if _have_open_clip():
        return "clip"
    return "stub"


def reset_backend_cache() -> None:
    """Clear the cached backend state (used by tests)."""
    global _cached
    with _lock:
        _cached = None


def is_cuda() -> bool:
    return detect_backend().cuda_available

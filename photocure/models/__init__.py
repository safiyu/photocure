"""Bundled model files, verified against MANIFEST.json before use."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger("photocure.models")

MODELS_DIR = Path(__file__).resolve().parent
YUNET = "face_detection_yunet_2023mar.onnx"
LANDMARKER = "face_landmarker.task"

_verified: dict[tuple[str, int, int], bool] = {}


def load_manifest(models_dir: Path = MODELS_DIR) -> list[dict]:
    try:
        return list(json.loads((models_dir / "MANIFEST.json").read_text(encoding="utf-8"))["models"])
    except (OSError, ValueError, KeyError):
        return []


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model(name: str, models_dir: Path = MODELS_DIR) -> bool:
    """True when the file exists and its SHA-256 matches the manifest."""
    entry = next((m for m in load_manifest(models_dir) if m.get("file") == name), None)
    path = models_dir / name
    if entry is None or not path.is_file():
        return False
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _verified:
        _verified[key] = _sha256(path) == entry.get("sha256")
        if not _verified[key]:
            log.warning("model %s does not match its manifest hash; it will not be used", name)
    return _verified[key]


def model_path(name: str, models_dir: Path = MODELS_DIR) -> Path | None:
    """Path of a verified model, or None."""
    return models_dir / name if verify_model(name, models_dir) else None


def active_models(models_dir: Path = MODELS_DIR) -> list[dict]:
    """Manifest entries whose file is present and intact (what the About card may list)."""
    return [m for m in load_manifest(models_dir) if verify_model(m["file"], models_dir)]

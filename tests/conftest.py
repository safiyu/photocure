"""Shared pytest fixtures for the photocure test suite.

All inference tests force the deterministic CPU backends (Haar face detector
and the stub CLIP engine) via the ``PCUR_BACKEND`` environment variable so the
suite runs without GPU, network, or model downloads.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the package importable when running from a source checkout.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Force deterministic CPU backends for the whole suite (stub face + stub clip),
# unless the operator explicitly wants the real engines.
os.environ.setdefault("PCUR_BACKEND", os.environ.get("PCUR_BACKEND", "stub"))
os.environ.setdefault("PCUR_DEVICE_DEVICE", "cpu")
# Starlette's TestClient sends "Host: testserver"; pin a token so no auth_token file is written.
os.environ.setdefault("PCUR_ALLOWED_HOSTS", "testserver")
os.environ.setdefault("PCUR_AUTH_TOKEN", "test-token-for-the-suite")


@pytest.fixture()
def tmp_db(tmp_path):
    """A fresh SQLite state store path inside the temp sandbox."""
    return tmp_path / "photocure.db"


@pytest.fixture()
def cfg(tmp_db):
    from photocure.config import CuratorConfig

    c = CuratorConfig()
    c.db_path = tmp_db
    c.device.device = "cpu"
    return c


@pytest.fixture()
def session(cfg, tmp_db):
    """An open, committed-at-exit SQLModel session over a fresh schema."""
    from photocure.db.session import get_session, init_db

    init_db(tmp_db, drop=True)
    with get_session(tmp_db) as s:
        yield s


@pytest.fixture()
def repo(session):
    from photocure.db.repository import PhotoRepository

    return PhotoRepository(session)


def make_image(path: Path, size=(480, 320), rng_seed: int = 0,
               blur: bool = False, dark: bool = False,
               white: bool = False) -> Path:
    """Create a deterministic synthetic JPEG at ``path``."""
    import cv2
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = size
    rng = np.random.default_rng(rng_seed)
    if dark:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = rng.integers(0, 6, 3)
    elif white:
        img = np.full((h, w, 3), 250, dtype=np.uint8)
        noise = rng.integers(0, 6, (h, w, 3), dtype=np.uint8)
        img = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255
                      ).astype(np.uint8)
    else:
        # Structured content: blocks + gradient so the Laplacian is meaningful.
        img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        step = max(8, min(h, w) // 16)
        for y in range(0, h, step):
            for x in range(0, w, step):
                v = int(255 * ((x * 7 + y * 13 + rng_seed * 37) % 97) / 97)
                img[y:y + step, x:x + step] = v
        # Stamp the seed into the first MCU so every seed yields distinct bytes.
        img[:16, :16] = (rng_seed & 255, (rng_seed >> 8) & 255, (rng_seed * 7) & 255)
        if blur:
            img = cv2.GaussianBlur(img, (21, 21), 11)
    ok = cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok, f"could not write {path}"
    return path

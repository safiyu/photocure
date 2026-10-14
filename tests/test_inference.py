"""Tests for the inference layer (face + semantic engines + pipeline).

The default suite runs with the deterministic stub backends (``PCUR_BACKEND=stub``)
so it is fast and network-free. The real OpenCLIP path is additionally verified
when the ViT-B-16 model weights are cached locally (skipped otherwise).
"""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from tests.conftest import make_image
from photo_curator.config import CuratorConfig
from photo_curator.db.models import ImageRecord
from photo_curator.db.repository import PhotoRepository, dequantize_embedding
from photo_curator.db.session import get_session, init_db
from photo_curator.inference.backends import detect_backend, reset_backend_cache
from photo_curator.inference.face_engine import (
    FaceEngine,
    eye_aspect_ratio,
)
from photo_curator.inference.pipeline import InferencePipeline
from photo_curator.inference.semantic_engine import SemanticEngine


def _cfg(tmp_db) -> CuratorConfig:
    c = CuratorConfig()
    c.db_path = tmp_db
    c.device.device = "cpu"
    return c


@pytest.fixture(autouse=True)
def _reset_backend():
    reset_backend_cache()
    yield
    reset_backend_cache()


# ---------------------------------------------------------------------------
# EAR geometry
# ---------------------------------------------------------------------------

def test_eye_aspect_ratio_open_eye():
    # Canonical open eye: [p1,p2,p3,p4,p5,p6] with p1/p4 the horizontal
    # corners and p2,p3 (top) / p5,p6 (bottom) the vertical pairs.
    pts = [
        (0.0, 0.5), (0.25, 0.35), (0.75, 0.35), (1.0, 0.5), (0.75, 0.65), (0.25, 0.65),
    ]
    ear = eye_aspect_ratio(pts)
    assert 0.1 < ear < 0.5


def test_eye_aspect_ratio_closed_eye_is_small():
    pts = [
        (0.0, 0.5), (0.25, 0.48), (0.75, 0.48), (1.0, 0.5), (0.75, 0.52), (0.25, 0.52),
    ]
    ear = eye_aspect_ratio(pts)
    assert ear < 0.08


def test_eye_aspect_ratio_degenerate_returns_zero():
    assert eye_aspect_ratio([(0, 0)] * 6) == 0.0
    assert eye_aspect_ratio([(0, 0), (1, 0)]) == 0.0


# ---------------------------------------------------------------------------
# Stub backends (deterministic, default for the suite)
# ---------------------------------------------------------------------------

def test_stub_face_backend_reports_no_faces(tmp_db):
    reset_backend_cache()
    bs = detect_backend(_cfg(tmp_db))
    assert bs.face_backend == "stub"
    fe = FaceEngine(_cfg(tmp_db), backend=bs)
    img = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    res = fe.detect_faces_batch([img, img])
    assert res == [[], []]
    fe.release()


def test_stub_semantic_is_deterministic_and_normalized(tmp_db):
    reset_backend_cache()
    bs = detect_backend(_cfg(tmp_db))
    assert bs.semantic_backend == "stub"
    se = SemanticEngine(_cfg(tmp_db), backend=bs)
    img = np.random.default_rng(1).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    r1 = se.classify_and_embed([img])[0]
    r2 = se.classify_and_embed([img])[0]
    # Deterministic.
    assert r1.embedding == r2.embedding
    assert r1.top1 == r2.top1
    # Embedding is normalized and 512-dim.
    e = np.asarray(r1.embedding)
    assert e.shape == (512,)
    assert abs(float(np.linalg.norm(e)) - 1.0) < 1e-3
    # Distribution sums to ~1 and is consistent with top-1/top-2.
    assert abs(sum(r1.distribution.values()) - 1.0) < 1e-3
    assert r1.top1_score >= r1.top2_score
    assert r1.margin == pytest.approx(r1.top1_score - r1.top2_score, abs=1e-6)
    se.release()


def test_stub_semantic_black_frame_is_low_confidence(tmp_db):
    reset_backend_cache()
    bs = detect_backend(_cfg(tmp_db))
    se = SemanticEngine(_cfg(tmp_db), backend=bs)
    black = np.zeros((200, 200, 3), dtype=np.uint8)
    r = se.classify_and_embed([black])[0]
    assert r.top1_score < 0.3  # pushed low -> will route to Uncertain
    se.release()


# ---------------------------------------------------------------------------
# Real OpenCLIP path (skipped when model weights are not cached)
# ---------------------------------------------------------------------------

def _clip_available() -> bool:
    try:
        import open_clip  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _clip_available(), reason="open_clip not installed")
def test_real_clip_embed_and_classify(tmp_db):
    """ViT-B-16 on CPU: 512-dim L2-normalized embeddings, stable per image."""
    import os
    import torch
    c = _cfg(tmp_db)
    c.device.device = "cpu"
    os.environ["PCUR_BACKEND"] = "clip"
    # Single-thread CPU inference is deterministic; on a multi-core box the
    # intra-op reduction order can vary the low-order bits and flip a
    # borderline top-1 between runs.
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        reset_backend_cache()
        bs = detect_backend(c)
        bs.face_backend = "stub"
        se = SemanticEngine(c, backend=bs)
        # A structured scene with an unambiguous top-1 (sky + ground + sun),
        # not noise: random noise lands on a decision boundary where tiny FP
        # differences change the argmax.
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:102, :] = (210, 150, 80)        # sky (BGR)
        img[102:, :] = (60, 150, 60)         # ground
        cv2.circle(img, (190, 48), 24, (120, 210, 255), -1)  # sun
        r = se.classify_and_embed([img])[0]
        assert r.embedding and len(r.embedding) == 512
        e = np.asarray(r.embedding)
        assert abs(float(np.linalg.norm(e)) - 1.0) < 1e-3
        assert abs(sum(r.distribution.values()) - 1.0) < 1e-2
        # Deterministic for identical input.
        r2 = se.classify_and_embed([img])[0]
        assert r2.top1 == r.top1
        assert abs(r2.top1_score - r.top1_score) < 1e-3
        se.release()
    finally:
        torch.set_num_threads(prev_threads)
        del os.environ["PCUR_BACKEND"]
        reset_backend_cache()


# ---------------------------------------------------------------------------
# Pipeline end-to-end (stub backends)
# ---------------------------------------------------------------------------

def test_pipeline_marks_images_inferred(tmp_db, tmp_path):
    reset_backend_cache()
    root = tmp_path / "lib"
    paths = [
        make_image(root / "IMG_20240301_100001.jpg", rng_seed=21),
        make_image(root / "IMG_20240301_100002.jpg", rng_seed=22),
    ]
    from photo_curator.scanner.file_scanner import FileScanner
    cfg = _cfg(tmp_db)
    init_db(tmp_db, drop=True)
    res = FileScanner(cfg).scan_directory(root, db_path=tmp_db,
                                          show_progress=False)
    assert res.new == 2

    os_environ_backup = None
    import os
    prev = os.environ.get("PCUR_BACKEND")
    os.environ["PCUR_BACKEND"] = "stub"
    try:
        reset_backend_cache()
        pipe = InferencePipeline(cfg)
        ires = pipe.run(db_path=tmp_db, show_progress=False)
    finally:
        if prev is None:
            os.environ.pop("PCUR_BACKEND", None)
        else:
            os.environ["PCUR_BACKEND"] = prev
        reset_backend_cache()

    assert ires.processed == 2
    assert ires.failures == 0
    assert ires.quality_computed == 2
    assert ires.semantic_computed == 2

    with get_session(tmp_db) as s:
        repo = PhotoRepository(s)
        for p in paths:
            rec = repo.get_by_path(str(p))
            assert rec.infer_status == "inferred"
            assert rec.quality is not None
            assert rec.semantic is not None
            assert len(dequantize_embedding(rec.semantic.embedding) or []) == 512


def test_pipeline_resumable_skips_inferred(tmp_db, tmp_path):
    import os
    from photo_curator.scanner.file_scanner import FileScanner
    cfg = _cfg(tmp_db)
    init_db(tmp_db, drop=True)
    root = tmp_path / "lib"
    make_image(root / "IMG_20240301_100001.jpg", rng_seed=31)
    FileScanner(cfg).scan_directory(root, db_path=tmp_db, show_progress=False)

    prev = os.environ.get("PCUR_BACKEND")
    os.environ["PCUR_BACKEND"] = "stub"
    try:
        reset_backend_cache()
        first = InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
        second = InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
    finally:
        if prev is None:
            os.environ.pop("PCUR_BACKEND", None)
        else:
            os.environ["PCUR_BACKEND"] = prev
        reset_backend_cache()

    assert first.processed == 1
    assert second.processed == 0
    assert second.skipped_already_inferred == 1

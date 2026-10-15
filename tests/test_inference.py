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
from photocure.config import CuratorConfig
from photocure.db.repository import PhotoRepository, dequantize_embedding
from photocure.db.session import get_session, init_db
from photocure.inference.backends import detect_backend, reset_backend_cache
from photocure.inference.face_engine import (
    FaceEngine,
    eye_aspect_ratio,
)
from photocure.inference.pipeline import InferencePipeline
from photocure.inference.semantic_engine import SemanticEngine


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
    prev_backend = os.environ.get("PCUR_BACKEND")
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
        # Restore the suite default (stub) instead of deleting it for later tests.
        if prev_backend is None:
            os.environ.pop("PCUR_BACKEND", None)
        else:
            os.environ["PCUR_BACKEND"] = prev_backend
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
    from photocure.scanner.file_scanner import FileScanner
    cfg = _cfg(tmp_db)
    init_db(tmp_db, drop=True)
    res = FileScanner(cfg).scan_directory(root, db_path=tmp_db,
                                          show_progress=False)
    assert res.new == 2

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
    from photocure.scanner.file_scanner import FileScanner
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


def test_pipeline_streams_across_chunks(tmp_db, tmp_path):
    """Memory-bounded streaming: with a small ``infer_chunk_size`` the whole
    library is processed in several chunks. No image may be dropped across a
    chunk boundary, and each surviving record must carry quality, a (stub)
    semantic row and a full 512-dim embedding."""
    import os
    from photocure.scanner.file_scanner import FileScanner
    cfg = _cfg(tmp_db)
    # Force the library to be split into many chunks (5 images, chunk of 2 ->
    # 3 chunks), exercising the per-chunk decode/persist/release loop.
    cfg.device.infer_chunk_size = 2
    init_db(tmp_db, drop=True)
    root = tmp_path / "lib"
    paths = [make_image(root / f"IMG_20240301_10000{i}.jpg", rng_seed=40 + i)
             for i in range(5)]
    FileScanner(cfg).scan_directory(root, db_path=tmp_db, show_progress=False)

    prev = os.environ.get("PCUR_BACKEND")
    os.environ["PCUR_BACKEND"] = "stub"
    try:
        reset_backend_cache()
        ires = InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
    finally:
        if prev is None:
            os.environ.pop("PCUR_BACKEND", None)
        else:
            os.environ["PCUR_BACKEND"] = prev
        reset_backend_cache()

    # Every image processed exactly once, none lost between chunks.
    assert ires.processed == 5
    assert ires.failures == 0
    assert ires.quality_computed == 5
    assert ires.semantic_computed == 5

    with get_session(tmp_db) as s:
        repo = PhotoRepository(s)
        for p in paths:
            rec = repo.get_by_path(str(p))
            assert rec.infer_status == "inferred"
            assert rec.quality is not None
            assert rec.semantic is not None
            assert len(dequantize_embedding(rec.semantic.embedding) or []) == 512


# ---------------------------------------------------------------------------
# Review fixes: face engine
# ---------------------------------------------------------------------------

import sys
import types
import logging

from photocure.inference.backends import BackendState, is_oom
from photocure.inference.face_engine import (
    HAAR_CONFIDENCE,
    _LEFT_EYE_106,
    _RIGHT_EYE_106,
    eye_openness_from_106,
)


def _bs(device="cpu", cuda_available=False, face="insightface", sem="stub"):
    return BackendState(device=device, fp16=False, torch=None,
                        cuda_available=cuda_available, memory_fraction=0.75,
                        face_backend=face, semantic_backend=sem)


def _lm106(open_eye: bool) -> np.ndarray:
    """106 landmarks where only the eye contours matter, open or closed."""
    pts = np.zeros((106, 2), dtype=float)
    half = 0.0 if not open_eye else 6.0
    for base_x, idx in ((100.0, _LEFT_EYE_106), (200.0, _RIGHT_EYE_106)):
        c1, t1, t2, c2, b2, b1 = idx
        pts[c1] = (base_x, 50)
        pts[c2] = (base_x + 40, 50)
        pts[t1] = (base_x + 13, 50 - half)
        pts[t2] = (base_x + 27, 50 - half)
        pts[b1] = (base_x + 13, 50 + half)
        pts[b2] = (base_x + 27, 50 + half)
    return pts


def test_eye_openness_from_106_open_vs_closed():
    open_ear = eye_openness_from_106(_lm106(True))
    closed_ear = eye_openness_from_106(_lm106(False))
    assert open_ear is not None and 0.2 < open_ear < 0.4
    # Fully closed lids collapse to zero and are reported as unknown.
    assert closed_ear is None


def test_eye_openness_unknown_for_bad_input():
    assert eye_openness_from_106(np.zeros((5, 2))) is None
    assert eye_openness_from_106(None) is None
    assert eye_openness_from_106(np.zeros((106, 2))) is None


def test_five_landmarks_never_invent_eye_openness(tmp_db):
    class _Face(dict):
        pass

    f = _Face(bbox=[10, 10, 90, 90], det_score=0.9,
              kps=np.array([[30, 40], [70, 40], [50, 60], [35, 75], [65, 75]]))
    fe = FaceEngine(_cfg(tmp_db), backend=_bs())
    fe._insightface = types.SimpleNamespace(get=lambda bgr: [f])
    out = fe._detect_insightface(np.zeros((100, 100, 3), np.uint8), 100, 100)
    assert out[0].eye_aspect_ratio is None
    assert out[0].embedding is None


def test_insightface_detect_reads_106_and_embedding(tmp_db):
    emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
    f = {"bbox": [10, 10, 90, 90], "det_score": 0.9,
         "kps": np.zeros((5, 2)), "landmark_2d_106": _lm106(True),
         "normed_embedding": emb}
    fe = FaceEngine(_cfg(tmp_db), backend=_bs())
    fe._insightface = types.SimpleNamespace(get=lambda bgr: [f])
    out = fe._detect_insightface(np.zeros((100, 100, 3), np.uint8), 100, 100)[0]
    assert out.eye_aspect_ratio is not None
    assert out.embedding is not None and len(out.embedding) == 512


def _fake_insightface(monkeypatch, calls, fail_modules=()):
    class FaceAnalysis:
        def __init__(self, name, allowed_modules, providers):
            calls.append((tuple(allowed_modules), tuple(providers)))
            if tuple(allowed_modules) in fail_modules:
                raise RuntimeError("model files missing")

        def prepare(self, ctx_id, det_size):
            calls.append(("prepare", ctx_id))

    app_mod = types.ModuleType("insightface.app")
    app_mod.FaceAnalysis = FaceAnalysis
    pkg = types.ModuleType("insightface")
    pkg.app = app_mod
    monkeypatch.setitem(sys.modules, "insightface", pkg)
    monkeypatch.setitem(sys.modules, "insightface.app", app_mod)


def test_load_insightface_imports_face_analysis(tmp_db, monkeypatch):
    calls: list = []
    _fake_insightface(monkeypatch, calls)
    fe = FaceEngine(_cfg(tmp_db), backend=_bs())
    app = fe._load_insightface()
    assert app is not None
    assert calls[0][0] == ("detection", "landmark_2d_106", "recognition")


def test_load_insightface_uses_requested_device_not_cuda_presence(tmp_db, monkeypatch):
    calls: list = []
    _fake_insightface(monkeypatch, calls)
    # CUDA exists but the run was resolved to cpu: no CUDA provider, ctx -1.
    fe = FaceEngine(_cfg(tmp_db), backend=_bs(device="cpu", cuda_available=True))
    fe._load_insightface()
    assert calls[0][1] == ("CPUExecutionProvider",)
    assert ("prepare", -1) in calls


def test_load_insightface_falls_back_to_detection_only(tmp_db, monkeypatch):
    calls: list = []
    _fake_insightface(monkeypatch, calls,
                      fail_modules={("detection", "landmark_2d_106", "recognition")})
    fe = FaceEngine(_cfg(tmp_db), backend=_bs())
    assert fe._load_insightface() is not None
    assert calls[-2][0] == ("detection",)


def test_insightface_failure_is_logged_as_warning_with_reason(tmp_db, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "insightface", None)
    monkeypatch.setitem(sys.modules, "insightface.app", None)
    fe = FaceEngine(_cfg(tmp_db), backend=_bs())
    with caplog.at_level(logging.WARNING, logger="photocure.face"):
        assert fe._load_insightface() is None
    assert any("InsightFace import failed" in r.getMessage()
               and r.levelno == logging.WARNING for r in caplog.records)


def test_haar_confidence_is_below_the_face_gate():
    cfg = CuratorConfig()
    assert HAAR_CONFIDENCE < cfg.taxonomy.face_confidence_min


def test_oom_is_not_swallowed_by_insightface_detect(tmp_db):
    def boom(bgr):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")

    fe = FaceEngine(_cfg(tmp_db), backend=_bs())
    fe._insightface = types.SimpleNamespace(get=boom)
    with pytest.raises(RuntimeError):
        fe._detect_insightface(np.zeros((10, 10, 3), np.uint8), 10, 10)


# ---------------------------------------------------------------------------
# Review fixes: backends
# ---------------------------------------------------------------------------

def _fake_torch(capability=(8, 9), cuda=True):
    rec = {"fraction": None}
    cuda_ns = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_capability=lambda: capability,
        set_per_process_memory_fraction=lambda f: rec.__setitem__("fraction", f),
        empty_cache=lambda: None,
    )
    mod = types.ModuleType("torch")
    mod.cuda = cuda_ns
    return mod, rec


@pytest.mark.parametrize("cap,expected", [((8, 9), True), ((6, 1), False)])
def test_fp16_follows_compute_capability(monkeypatch, cap, expected):
    torch_mod, _ = _fake_torch(capability=cap)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    reset_backend_cache()
    c = CuratorConfig()
    c.device.device = "cuda"
    assert detect_backend(c).fp16 is expected


def test_memory_fraction_applies_only_on_resolved_cuda(monkeypatch):
    torch_mod, rec = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    reset_backend_cache()
    c = CuratorConfig()
    c.device.device = "cpu"
    bs = detect_backend(c)
    assert bs.device == "cpu" and rec["fraction"] is None
    reset_backend_cache()
    c.device.device = "cuda"
    bs = detect_backend(c)
    assert bs.device == "cuda" and rec["fraction"] == c.device.cuda_memory_fraction


def test_is_oom_detection():
    assert is_oom(RuntimeError("CUDA out of memory"))
    assert is_oom(RuntimeError("onnxruntime: Failed to allocate memory"))
    assert not is_oom(ValueError("bad shape"))


# ---------------------------------------------------------------------------
# Review fixes: semantic engine
# ---------------------------------------------------------------------------

def test_clip_preprocess_receives_a_pil_image_in_rgb_order():
    from PIL import Image
    seen = {}

    def preprocess(img):
        seen["type"] = type(img)
        seen["px"] = img.getpixel((0, 0))
        return "tensor"

    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[:, :] = (10, 20, 30)  # B, G, R
    assert SemanticEngine._bgr_to_tensor(bgr, preprocess) == "tensor"
    assert seen["type"] is Image.Image
    assert seen["px"] == (30, 20, 10)


def test_ensemble_text_features_mean_pools_and_renormalizes():
    from photocure.inference.semantic_engine import ensemble_text_features
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)   # two prompts
    b = np.array([[0.0, 2.0]], dtype=np.float32)               # one prompt
    out = ensemble_text_features([a, b])
    assert out.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(out[0], [np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)
    np.testing.assert_allclose(out[1], [0.0, 1.0], atol=1e-6)


def test_every_prompt_contributes_to_class_features(tmp_db, tmp_path, monkeypatch):
    import json
    tax1 = tmp_path / "t1.json"
    tax2 = tmp_path / "t2.json"
    tax1.write_text(json.dumps([{"name": "food", "prompts": ["a", "b"]}]))
    tax2.write_text(json.dumps([{"name": "food", "prompts": ["a", "c"]}]))
    feats = []
    for t in (tax1, tax2):
        monkeypatch.setenv("PCUR_TAXONOMY_FILE", str(t))
        se = SemanticEngine(_cfg(tmp_db), backend=_bs(sem="stub"))
        feats.append(se.text_features_numpy())
    assert not np.allclose(feats[0], feats[1])


def test_rescore_from_embeddings_uses_stored_vectors_only(tmp_db, tmp_path, monkeypatch):
    import json
    from photocure.db.models import ImageRecord, SemanticScore
    from photocure.db.repository import quantize_embedding
    from photocure.inference.semantic_engine import rescore_from_embeddings
    init_db(tmp_db, drop=True)
    rng = np.random.default_rng(3)
    with get_session(tmp_db) as s:
        for i in range(3):
            img = ImageRecord(path=f"/p/{i}.jpg", filename=f"{i}.jpg",
                              infer_status="inferred")
            s.add(img)
            s.flush()
            v = rng.standard_normal(512).astype(np.float32)
            v /= np.linalg.norm(v)
            s.add(SemanticScore(image_id=img.id, top1_category="food",
                                embedding=quantize_embedding(v.tolist())))
    tax = tmp_path / "t.json"
    tax.write_text(json.dumps([
        {"name": "scenery", "prompts": ["mountains"]},
        {"name": "kids", "prompts": ["children playing"], "folder": "Family/Kids"},
    ]))
    monkeypatch.setenv("PCUR_TAXONOMY_FILE", str(tax))
    reset_backend_cache()
    with get_session(tmp_db) as s:
        n = rescore_from_embeddings(s, _cfg(tmp_db), backend=_bs(sem="stub"))
    assert n == 3
    with get_session(tmp_db) as s:
        from sqlmodel import select
        for row in s.exec(select(SemanticScore)).all():
            dist = json.loads(row.distribution)
            assert set(dist) == {"scenery", "kids"}
            assert row.top1_category in {"scenery", "kids"}
            assert abs(sum(dist.values()) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Review fixes: image loader
# ---------------------------------------------------------------------------

def test_16bit_png_is_not_blown_out(tmp_path):
    from PIL import Image
    from photocure.quality.image_loader import load_image
    grad = np.tile(np.linspace(0, 65535, 64, dtype=np.uint16), (64, 1))
    p = tmp_path / "deep.png"
    Image.fromarray(grad).save(p)
    img = load_image(p)
    gray = img.bgr[:, :, 0].astype(int)
    assert gray.min() < 10 and gray.max() > 245
    assert 100 < gray.mean() < 156            # mid-grey on average, not 255


def test_16bit_scaling_helpers():
    from photocure.quality.image_loader import _to_uint8
    a = np.array([[0, 30000, 65535]], dtype=np.int32)
    out = _to_uint8(a)
    assert out.max() == 255 and out.min() == 0 and 100 < out[0, 1] < 156
    f = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    assert list(_to_uint8(f)[0]) == [0, 128, 255]
    wide = np.array([[-1000.0, 0.0, 1000.0]], dtype=np.float32)
    assert list(_to_uint8(wide)[0]) == [0, 128, 255]


def test_oversize_frame_is_a_decode_failure(tmp_path, monkeypatch):
    from photocure.quality import image_loader
    p = tmp_path / "big.jpg"
    make_image(p, size=(200, 100))
    monkeypatch.setattr(image_loader, "MAX_PIXELS", 1000)
    with pytest.raises(image_loader.DecodeError):
        image_loader.load_image(p)


def test_jpeg_draft_decodes_at_reduced_scale_and_tracks_original(tmp_path):
    from photocure.quality.image_loader import load_image
    p = tmp_path / "wide.jpg"
    make_image(p, size=(1200, 1600))   # (h, w)
    img = load_image(p, max_side=400)
    assert max(img.width, img.height) <= 400
    assert (img.orig_width, img.orig_height) == (1600, 1200)
    assert img.scale == pytest.approx(400 / 1600, rel=0.02)


def test_truncated_jpeg_still_decodes(tmp_path):
    from photocure.quality.image_loader import load_image
    p = tmp_path / "cut.jpg"
    make_image(p, size=(640, 480))
    data = p.read_bytes()
    p.write_bytes(data[: int(len(data) * 0.8)])
    assert load_image(p).bgr.shape[2] == 3


def test_garbage_file_raises(tmp_path):
    from photocure.quality.image_loader import load_image
    p = tmp_path / "junk.jpg"
    p.write_bytes(b"not an image at all")
    with pytest.raises(Exception):
        load_image(p)


def test_heic_orientation_is_not_applied_twice(tmp_path):
    from photocure.quality import image_loader
    if image_loader.pillow_heif is None:
        pytest.skip("pillow_heif not installed")
    src = tmp_path / "a.jpg"
    make_image(src, size=(200, 300))   # (h, w)
    heic_named = tmp_path / "a.heic"          # decoded by content via Pillow
    heic_named.write_bytes(src.read_bytes())
    rotated = image_loader.load_image(src, orientation=6)
    heif = image_loader.load_image(heic_named, orientation=6)
    assert rotated.bgr.shape[:2] == (300, 200)  # jpeg rotated 90 degrees
    assert heif.bgr.shape[:2] == (200, 300)     # heif already oriented: untouched


# ---------------------------------------------------------------------------
# Review fixes: pipeline
# ---------------------------------------------------------------------------

from photocure.inference.pipeline import run_with_retry


def test_run_with_retry_halves_on_oom_until_it_fits():
    sizes = []
    ooms = []

    def fn(items):
        sizes.append(len(items))
        if len(items) > 2:
            raise RuntimeError("CUDA out of memory")
        return [i * 10 for i in items]

    out = run_with_retry(fn, list(range(8)), on_oom=lambda: ooms.append(1))
    assert out == [i * 10 for i in range(8)]
    assert sizes[0] == 8 and max(sizes[1:]) <= 4
    assert len(ooms) >= 1


def test_run_with_retry_isolates_a_bad_item():
    def fn(items):
        if 3 in items:
            raise ValueError("corrupt")
        return [i for i in items]

    out = run_with_retry(fn, [1, 2, 3, 4, 5])
    assert out == [1, 2, None, 4, 5]


def test_run_with_retry_empty():
    assert run_with_retry(lambda x: x, []) == []


def _scan(tmp_path, tmp_db, n, prefix="IMG_20240301_1000", size=(480, 320)):
    from photocure.scanner.file_scanner import FileScanner
    cfg = _cfg(tmp_db)
    init_db(tmp_db, drop=True)
    root = tmp_path / "lib"
    paths = [make_image(root / f"{prefix}{i:02d}.jpg", size=size, rng_seed=60 + i)
             for i in range(n)]
    FileScanner(cfg).scan_directory(root, db_path=tmp_db, show_progress=False)
    return cfg, paths


def _statuses(tmp_db):
    with get_session(tmp_db) as s:
        from sqlmodel import select
        from photocure.db.models import ImageRecord
        return {r.filename: r.infer_status for r in s.exec(select(ImageRecord)).all()}


def test_pipeline_commits_each_chunk_and_resumes_after_crash(tmp_db, tmp_path, monkeypatch):
    cfg, _ = _scan(tmp_path, tmp_db, 5)
    cfg.device.infer_chunk_size = 2
    calls = {"n": 0}
    real = SemanticEngine.classify_and_embed

    def crashy(self, images):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt  # not an Exception: simulates Ctrl-C
        return real(self, images)

    monkeypatch.setattr(SemanticEngine, "classify_and_embed", crashy)
    with pytest.raises(KeyboardInterrupt):
        InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
    st = _statuses(tmp_db)
    assert list(st.values()).count("inferred") == 4
    assert list(st.values()).count("faces_done") == 1
    assert "pending" not in st.values()

    monkeypatch.setattr(SemanticEngine, "classify_and_embed", real)
    res = InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
    assert res.processed == 1
    assert res.failures == 0
    assert set(_statuses(tmp_db).values()) == {"inferred"}


def test_faces_done_rows_skip_phase_a_on_resume(tmp_db, tmp_path, monkeypatch):
    cfg, _ = _scan(tmp_path, tmp_db, 2)
    with get_session(tmp_db) as s:
        from sqlmodel import select
        from photocure.db.models import ImageRecord
        for r in s.exec(select(ImageRecord)).all():
            r.infer_status = "faces_done"
    seen = {"a": 0}
    real = FaceEngine.detect_faces_batch

    def spy(self, images):
        seen["a"] += len(images)
        return real(self, images)

    monkeypatch.setattr(FaceEngine, "detect_faces_batch", spy)
    res = InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
    assert seen["a"] == 0
    assert res.processed == 2


def test_infer_chunk_size_is_not_capped_at_64(tmp_db, tmp_path, monkeypatch):
    cfg, _ = _scan(tmp_path, tmp_db, 70, size=(96, 64))
    cfg.device.infer_chunk_size = 512
    sizes = []
    real = SemanticEngine.classify_and_embed

    def spy(self, images):
        sizes.append(len(images))
        return real(self, images)

    monkeypatch.setattr(SemanticEngine, "classify_and_embed", spy)
    InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
    assert sizes == [70]


def test_phase_b_decodes_at_clip_max_side(tmp_db, tmp_path, monkeypatch):
    cfg, _ = _scan(tmp_path, tmp_db, 2, size=(640, 480))
    cfg.device.face_max_side = 320
    cfg.device.clip_max_side = 96
    face_sides, clip_sides = [], []
    real_f, real_s = FaceEngine.detect_faces_batch, SemanticEngine.classify_and_embed

    def f_spy(self, images):
        face_sides.extend(max(i.shape[:2]) for i in images)
        return real_f(self, images)

    def s_spy(self, images):
        clip_sides.extend(max(i.shape[:2]) for i in images)
        return real_s(self, images)

    monkeypatch.setattr(FaceEngine, "detect_faces_batch", f_spy)
    monkeypatch.setattr(SemanticEngine, "classify_and_embed", s_spy)
    InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
    assert face_sides and max(face_sides) == 320
    assert clip_sides and max(clip_sides) == 96


def test_face_batch_size_is_honored(tmp_db, tmp_path, monkeypatch):
    cfg, _ = _scan(tmp_path, tmp_db, 5, size=(96, 64))
    cfg.device.face_batch_size = 2
    sizes = []
    real = FaceEngine.detect_faces_batch

    def spy(self, images):
        sizes.append(len(images))
        return real(self, images)

    monkeypatch.setattr(FaceEngine, "detect_faces_batch", spy)
    InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
    assert sizes == [2, 2, 1]


def test_retry_failed_reprocesses_failed_rows(tmp_db, tmp_path):
    cfg, paths = _scan(tmp_path, tmp_db, 2)
    hidden = paths[0].with_suffix(".hidden")
    paths[0].rename(hidden)
    first = InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
    assert first.failures == 1 and first.processed == 1
    hidden.rename(paths[0])
    plain = InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
    assert plain.processed == 0                       # failed rows stay failed
    again = InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False,
                                       retry_failed=True)
    assert again.processed == 1 and again.failures == 0
    assert set(_statuses(tmp_db).values()) == {"inferred"}


def test_limit_and_only_paths_query_by_path(tmp_db, tmp_path):
    cfg, paths = _scan(tmp_path, tmp_db, 4)
    res = InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False, limit=1)
    assert res.processed == 1
    done = {n for n, st in _statuses(tmp_db).items() if st == "inferred"}
    target = next(p for p in paths if p.name not in done)
    res = InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False,
                                     only_paths=[str(target)])
    assert res.processed == 1
    assert list(_statuses(tmp_db).values()).count("inferred") == 2


def test_progress_is_stopped_when_the_run_aborts(tmp_db, tmp_path, monkeypatch):
    from photocure.inference import pipeline as pl
    cfg, _ = _scan(tmp_path, tmp_db, 2)
    events = []

    class FakeProgress:
        def __init__(self, *a, **k):
            pass

        def add_task(self, *a, **k):
            return 0

        def advance(self, *a, **k):
            pass

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    def boom(self, images):
        raise KeyboardInterrupt

    monkeypatch.setattr(pl, "Progress", FakeProgress)
    monkeypatch.setattr(SemanticEngine, "classify_and_embed", boom)
    with pytest.raises(KeyboardInterrupt):
        InferencePipeline(cfg).run(db_path=tmp_db, show_progress=True)
    assert events == ["start", "stop"]


def test_pipeline_stores_low_contrast_flag_and_aesthetic(tmp_db, tmp_path):
    from photocure.db.models import QualityMetric
    from photocure.scanner.file_scanner import FileScanner
    cfg = _cfg(tmp_db)
    init_db(tmp_db, drop=True)
    root = tmp_path / "lib"
    flat = root / "IMG_20240301_100000.jpg"
    flat.parent.mkdir(parents=True)
    cv2.imwrite(str(flat), np.full((200, 300, 3), 128, np.uint8))
    FileScanner(cfg).scan_directory(root, db_path=tmp_db, show_progress=False)
    InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
    with get_session(tmp_db) as s:
        rec = PhotoRepository(s).get_by_path(str(flat))
        assert rec.quality.is_low_contrast is True
        assert rec.semantic.aesthetic is not None
        assert 0.0 <= rec.semantic.aesthetic <= 1.0


def test_pipeline_persists_face_embedding_and_scale(tmp_db, tmp_path, monkeypatch):
    from photocure.db.models import FaceDetection
    from photocure.db.repository import dequantize_embedding
    from photocure.inference.face_engine import FaceInfo
    cfg, _ = _scan(tmp_path, tmp_db, 1, size=(640, 480))
    cfg.device.face_max_side = 320
    emb = (np.ones(512) / np.sqrt(512)).tolist()

    def fake(self, images):
        return [[FaceInfo(1, 2, 30, 40, 0.9, embedding=emb)] for _ in images]

    monkeypatch.setattr(FaceEngine, "detect_faces_batch", fake)
    InferencePipeline(cfg).run(db_path=tmp_db, show_progress=False)
    with get_session(tmp_db) as s:
        from sqlmodel import select
        f = s.exec(select(FaceDetection)).first()
        assert f.scale == pytest.approx(0.5, rel=0.02)
        assert len(dequantize_embedding(f.embedding_json)) == 512


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def test_onnx_export_reports_missing_torch(tmp_path, monkeypatch):
    from photocure.inference.onnx_export import export_models
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="torch"):
        export_models(tmp_path / "out", CuratorConfig())

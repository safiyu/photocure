"""Learning from corrections, stub re-analysis and per-device batch sizes."""
from __future__ import annotations

import json

import numpy as np
import pytest
from sqlmodel import select

from photocure.config import CuratorConfig
from photocure.db.models import ImageRecord, SemanticScore
from photocure.db.repository import quantize_embedding
from photocure.db.session import get_session, init_db
from photocure.inference import pipeline as pipeline_mod
from photocure.inference.backends import BackendState, reset_backend_cache
from photocure.inference.pipeline import InferencePipeline, mark_stub_rows_for_reanalysis
from photocure.inference.semantic_engine import (
    SemanticEngine,
    effective_clip_batch,
    learn_from_corrections,
    rescore_from_embeddings,
)

from .test_inference import _scan


def stub_bs(sem="stub") -> BackendState:
    return BackendState(device="cpu", fp16=False, torch=None, cuda_available=False,
                        memory_fraction=0.75, face_backend="stub", semantic_backend=sem)


@pytest.fixture(autouse=True)
def _reset():
    reset_backend_cache()
    yield
    reset_backend_cache()


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def add_image(session, name, emb, manual=None, locked=False):
    img = ImageRecord(path=f"/lib/{name}.jpg", filename=f"{name}.jpg", extension=".jpg",
                      infer_status="inferred", manual_bucket=manual, locked=locked)
    session.add(img)
    session.flush()
    session.add(SemanticScore(image_id=img.id, model_name="t", backend="clip",
                              embedding=quantize_embedding([float(x) for x in emb])))
    session.flush()
    return img


def sem_of(session, name):
    return session.exec(
        select(SemanticScore).join(ImageRecord, ImageRecord.id == SemanticScore.image_id)
        .where(ImageRecord.filename == f"{name}.jpg")).one()


@pytest.fixture()
def world(tmp_db):
    cfg = CuratorConfig()
    cfg.db_path = tmp_db
    init_db(tmp_db, drop=True)
    engine = SemanticEngine(cfg, backend=stub_bs())
    text = engine.text_features_numpy()
    names = engine.taxonomy.names
    rng = np.random.default_rng(0)
    t = {n: text[i] for i, n in enumerate(names)}
    q = rng.standard_normal(text.shape[1]).astype(np.float32)
    # Remove food/scenery components so the zero shot tie between them is exact.
    basis = np.stack([t["food"], t["scenery"]], axis=1)
    q = unit(q - basis @ np.linalg.lstsq(basis, q, rcond=None)[0])
    return cfg, tmp_db, t, q, rng


def test_learn_from_corrections_shifts_an_ambiguous_photo(world):
    cfg, db, t, q, rng = world
    ambiguous = unit(t["food"] + t["scenery"] + 1.5 * q)
    with get_session(db) as s:
        add_image(s, "ambig", ambiguous)
        add_image(s, "locked", ambiguous, locked=True)
        add_image(s, "people", t["people"])
        for i in range(3):
            noise = 0.05 * unit(rng.standard_normal(len(q)))
            add_image(s, f"fixed{i}", unit(q + 0.3 * t["scenery"] + noise), manual="scenery")
        add_image(s, "lonely_food", t["food"], manual="food")   # one example only
        rescore_from_embeddings(s, cfg, stub_bs())
        before = {n: (sem_of(s, n).top1_category, sem_of(s, n).top1_score)
                  for n in ("ambig", "locked", "people", "fixed0")}
        assert before["ambig"][0] in ("food", "scenery") and before["ambig"][1] < 0.7
        assert before["people"][0] == "people"

        out = learn_from_corrections(s, cfg, stub_bs())
        assert out["classes_used"] == 1 and out["examples"] == 3
        assert out["updated"] >= 2                       # ambig + people (+ lonely is manual)

        a = sem_of(s, "ambig")
        assert a.top1_category == "scenery" and a.top1_score > before["ambig"][1]
        assert a.top1_score > 0.9
        dist = json.loads(a.distribution)
        assert dist["scenery"] == pytest.approx(a.top1_score, abs=1e-6)
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-3)
        # locked and manually corrected photos are never rewritten
        assert (sem_of(s, "locked").top1_category, sem_of(s, "locked").top1_score) == before["locked"]
        assert sem_of(s, "fixed0").top1_score == before["fixed0"][1]
        # an unrelated photo keeps its zero shot answer
        assert sem_of(s, "people").top1_category == "people"


def test_learn_needs_enough_examples(world):
    cfg, db, t, q, rng = world
    with get_session(db) as s:
        add_image(s, "ambig", unit(t["food"] + t["scenery"] + 1.5 * q))
        add_image(s, "one", unit(q + 0.3 * t["scenery"]), manual="scenery")
        rescore_from_embeddings(s, cfg, stub_bs())
        before = sem_of(s, "ambig").top1_score
        out = learn_from_corrections(s, cfg, stub_bs())
        assert out == {"updated": 0, "classes_used": 0, "examples": 0}
        assert sem_of(s, "ambig").top1_score == before


def test_learn_ignores_manual_buckets_that_are_not_classes(world):
    cfg, db, t, q, rng = world
    with get_session(db) as s:
        add_image(s, "a", unit(t["food"] + 0.2 * q))
        for i in range(3):
            add_image(s, f"x{i}", unit(q), manual="not_a_class")
        rescore_from_embeddings(s, cfg, stub_bs())
        assert learn_from_corrections(s, cfg, stub_bs())["classes_used"] == 0


def test_learn_respects_configured_weight_and_minimum(world):
    cfg, db, t, q, rng = world
    cfg.taxonomy.feedback_min_examples = 1
    cfg.taxonomy.feedback_weight = 0.0
    with get_session(db) as s:
        add_image(s, "ambig", unit(t["food"] + t["scenery"] + 1.5 * q))
        add_image(s, "one", unit(q + 0.3 * t["scenery"]), manual="scenery")
        rescore_from_embeddings(s, cfg, stub_bs())
        before = sem_of(s, "ambig").top1_score
        out = learn_from_corrections(s, cfg, stub_bs())
        assert out["classes_used"] == 1
        assert sem_of(s, "ambig").top1_score == pytest.approx(before, abs=1e-5)  # weight 0


def test_learn_below_min_cosine_does_not_shift_unrelated_photo(world):
    cfg, db, t, q, rng = world
    cfg.taxonomy.feedback_min_examples = 2
    cfg.taxonomy.feedback_min_cosine = 0.60
    with get_session(db) as s:
        # Product photo with moderate baseline similarity (0.50) to people prototype
        # but below the 0.60 threshold: should NOT receive any boost.
        add_image(s, "jewelry", unit(t["product"]))
        for i in range(2):
            add_image(s, f"person{i}", unit(t["people"]), manual="people")
        rescore_from_embeddings(s, cfg, stub_bs())
        before = sem_of(s, "jewelry")
        assert before.top1_category == "product"
        learn_from_corrections(s, cfg, stub_bs())
        after = sem_of(s, "jewelry")
        assert after.top1_category == "product"
        assert after.top1_score == pytest.approx(before.top1_score, abs=1e-4)


# --------------------------------------------------------- stub re-analysis
def _rows(db):
    with get_session(db) as s:
        return {r.filename: (r.infer_status, sem.backend)
                for r, sem in s.exec(select(ImageRecord, SemanticScore)
                                     .join(SemanticScore, SemanticScore.image_id == ImageRecord.id))}


class FakeClip(SemanticEngine):
    """Pretends to be the real model without loading anything."""

    def _ensure_loaded(self):
        pass

    def classify_and_embed(self, images):
        return [self._stub_result(b) for b in images]


def test_pipeline_records_backend_per_row(tmp_db, tmp_path):
    cfg, _ = _scan(tmp_path, tmp_db, 3)
    InferencePipeline(cfg, backend=stub_bs()).run(db_path=tmp_db, show_progress=False)
    assert set(_rows(tmp_db).values()) == {("inferred", "stub")}


def test_label_reflects_fallback_when_the_model_cannot_load(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(SemanticEngine, "_load_clip", lambda self: False)
    cfg, _ = _scan(tmp_path, tmp_db, 2)
    bs = stub_bs("clip")
    InferencePipeline(cfg, backend=bs).run(db_path=tmp_db, show_progress=False)
    assert bs.semantic_backend == "stub"
    assert set(_rows(tmp_db).values()) == {("inferred", "stub")}


def test_stub_rows_are_reanalysed_when_a_real_backend_appears(tmp_db, tmp_path, monkeypatch):
    cfg, _ = _scan(tmp_path, tmp_db, 3)
    InferencePipeline(cfg, backend=stub_bs()).run(db_path=tmp_db, show_progress=False)
    monkeypatch.setattr(pipeline_mod, "SemanticEngine", FakeClip)
    real = stub_bs("clip")
    second = InferencePipeline(cfg, backend=real).run(db_path=tmp_db, show_progress=False)
    assert second.reanalyzed_stub == 3 and second.processed == 3
    assert set(_rows(tmp_db).values()) == {("inferred", "clip")}
    third = InferencePipeline(cfg, backend=real).run(db_path=tmp_db, show_progress=False)
    assert third.reanalyzed_stub == 0 and third.processed == 0   # real rows are never redone


def test_reanalysis_can_be_disabled(tmp_db, tmp_path, monkeypatch):
    cfg, _ = _scan(tmp_path, tmp_db, 2)
    InferencePipeline(cfg, backend=stub_bs()).run(db_path=tmp_db, show_progress=False)
    monkeypatch.setattr(pipeline_mod, "SemanticEngine", FakeClip)
    res = InferencePipeline(cfg, backend=stub_bs("clip")).run(
        db_path=tmp_db, show_progress=False, reanalyze_stub=False)
    assert res.reanalyzed_stub == 0 and res.processed == 0
    assert set(_rows(tmp_db).values()) == {("inferred", "stub")}


def test_stub_backend_never_triggers_reanalysis(tmp_db, tmp_path):
    cfg, _ = _scan(tmp_path, tmp_db, 2)
    InferencePipeline(cfg, backend=stub_bs()).run(db_path=tmp_db, show_progress=False)
    again = InferencePipeline(cfg, backend=stub_bs()).run(db_path=tmp_db, show_progress=False)
    assert again.reanalyzed_stub == 0 and again.processed == 0


def test_mark_stub_rows_targets_stub_and_unknown_only(tmp_db, tmp_path):
    cfg, _ = _scan(tmp_path, tmp_db, 3)
    InferencePipeline(cfg, backend=stub_bs()).run(db_path=tmp_db, show_progress=False)
    with get_session(tmp_db) as s:
        rows = s.exec(select(SemanticScore).order_by(SemanticScore.id)).all()
        rows[0].backend = "clip"
        rows[1].backend = ""
        rows[2].backend = "stub"
    with get_session(tmp_db) as s:
        assert mark_stub_rows_for_reanalysis(s) == 2
    states = sorted(v[0] for v in _rows(tmp_db).values())
    assert states == ["inferred", "pending", "pending"]


def test_effective_clip_batch():
    assert effective_clip_batch(64, "cuda") == 64
    assert effective_clip_batch(64, "mps") == 32
    assert effective_clip_batch(64, "xpu") == 32
    assert effective_clip_batch(64, "cpu") == 16
    assert effective_clip_batch(8, "cpu") == 8
    assert effective_clip_batch(0, "cuda") == 1


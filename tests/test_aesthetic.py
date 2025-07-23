"""Aesthetic head, backfill and highlights query."""
from __future__ import annotations

import numpy as np
import pytest

from photocure.db.models import ImageRecord, SemanticScore
from photocure.db.repository import quantize_embedding
from photocure.inference import aesthetic
from photocure.inference.aesthetic import (
    AestheticHead,
    highlights,
    score_embedding,
    sync_aesthetic,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("PCUR_AESTHETIC_WEIGHTS", raising=False)
    aesthetic.reset_head()
    yield
    aesthetic.reset_head()


def _vec(seed):
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_stub_head_is_deterministic_and_bounded():
    a, b = AestheticHead(), AestheticHead()
    assert a.stub and np.array_equal(a.w, b.w)
    for s in range(10):
        sc = a.score(_vec(s))
        assert 0.0 <= sc <= 1.0
        assert sc == b.score(_vec(s))


def test_score_is_scale_invariant_and_zero_vector_is_neutral():
    h = AestheticHead()
    v = _vec(1)
    assert h.score(v) == pytest.approx(h.score(v * 7.0))
    assert h.score(np.zeros(512)) == 0.5


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        AestheticHead().score(np.ones(10))


def test_real_weights_load_from_env(tmp_path, monkeypatch):
    w = np.zeros(512, dtype=np.float32)
    w[0] = 1.0
    path = tmp_path / "head.npz"
    np.savez(path, w=w, b=0.0)
    monkeypatch.setenv("PCUR_AESTHETIC_WEIGHTS", str(path))
    head = AestheticHead.load()
    assert not head.stub
    up, down = np.zeros(512), np.zeros(512)
    up[0], down[0] = 1.0, -1.0
    assert head.score(up) > 0.9 > 0.1 > head.score(down)
    aesthetic.reset_head()
    assert score_embedding(up) > 0.9


def test_unreadable_weights_fall_back_to_stub(tmp_path, monkeypatch):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"garbage")
    monkeypatch.setenv("PCUR_AESTHETIC_WEIGHTS", str(bad))
    assert AestheticHead.load().stub


def test_score_many_matches_score():
    h = AestheticHead()
    vs = np.stack([_vec(i) for i in range(5)])
    assert h.score_many(vs) == pytest.approx([h.score(v) for v in vs], abs=1e-5)


def _row(session, i, aest=None, emb=True, backend="clip", model="", **img_kw):
    img = ImageRecord(path=f"/lib/{i}.jpg", filename=f"{i}.jpg", **img_kw)
    session.add(img)
    session.flush()
    session.add(SemanticScore(image_id=img.id, aesthetic=aest, backend=backend,
                              aesthetic_model=model,
                              embedding=quantize_embedding(_vec(i).tolist()) if emb else ""))
    session.flush()
    return img


def _rows(session):
    from sqlmodel import select
    return {r.image_id: r for r in session.exec(select(SemanticScore)).all()}


def test_linear10_kind_maps_raw_rating_to_unit_range(tmp_path, monkeypatch):
    # A LAION style head rates 1..10; the stored kind says how to map it.
    w = np.zeros(512, dtype=np.float32)
    w[0] = 4.5
    path = tmp_path / "laion.npz"
    np.savez(path, w=w, b=np.float32(3.4), kind="linear10")
    monkeypatch.setenv("PCUR_AESTHETIC_WEIGHTS", str(path))
    head = AestheticHead.load()
    assert head.kind == "linear10" and not head.stub
    e = np.zeros(512)
    e[0] = 1.0
    assert head.score(e) == pytest.approx((4.5 + 3.4 - 1.0) / 9.0)
    assert head.score(-e) == 0.0
    assert head.score_many(np.stack([e, -e])) == pytest.approx(
        [(4.5 + 3.4 - 1.0) / 9.0, 0.0], abs=1e-6)


def test_bundled_laion_head_loads_without_env():
    head = AestheticHead.load()
    assert not head.stub
    assert head.kind == "linear10"
    assert head.name == "laion-sa-v1-vit-b-16"
    assert head.w.shape == (512,)
    assert AestheticHead().name == "stub"


def test_sync_scores_rows_from_other_or_unknown_heads(session):
    head = aesthetic.get_head()
    a = _row(session, 1, aest=0.42)                       # scored by an unknown head
    b = _row(session, 2)                                  # never scored
    c = _row(session, 3, emb=False)                       # nothing to score
    d = _row(session, 4, aest=0.9, model=head.name)       # already current
    e = _row(session, 5, aest=0.7, backend="stub")        # placeholder embedding
    assert sync_aesthetic(session) == 3
    rows = _rows(session)
    assert rows[a.id].aesthetic == pytest.approx(head.score(_vec(1)), abs=1e-3)  # float16 embedding
    assert rows[a.id].aesthetic_model == head.name
    assert rows[b.id].aesthetic_model == head.name
    assert rows[c.id].aesthetic is None
    assert rows[d.id].aesthetic == 0.9
    assert rows[e.id].aesthetic is None and rows[e.id].aesthetic_model == ""
    assert sync_aesthetic(session) == 0


def test_sync_with_stub_head_clears_placeholder_scores(session, monkeypatch):
    a = _row(session, 1, aest=0.42)
    b = _row(session, 2, aest=0.5, model="laion-sa-v1-vit-b-16")
    monkeypatch.setattr(aesthetic, "_head", AestheticHead())
    assert sync_aesthetic(session) == 2
    rows = _rows(session)
    assert rows[a.id].aesthetic is None and rows[a.id].aesthetic_model == ""
    assert rows[b.id].aesthetic is None and rows[b.id].aesthetic_model == ""
    assert sync_aesthetic(session) == 0


def test_describe_backend_reports_aesthetic_head(monkeypatch):
    from photocure.inference.backends import describe_backend, detect_backend, reset_backend_cache
    reset_backend_cache()
    d = describe_backend(detect_backend())
    assert d["aesthetic_model"] == "laion-sa-v1-vit-b-16"
    assert d["aesthetic_available"] is True
    monkeypatch.setattr(aesthetic, "_head", AestheticHead())
    d = describe_backend(detect_backend())
    assert d["aesthetic_model"] == "" and d["aesthetic_available"] is False
    reset_backend_cache()


def test_highlights_orders_filters_and_excludes_junk(session):
    best = _row(session, 1, aest=0.95)
    good = _row(session, 2, aest=0.80)
    low = _row(session, 3, aest=0.30)
    dup = _row(session, 4, aest=0.99, is_duplicate=True)
    alt = _row(session, 5, aest=0.97, is_alternate=True)
    _row(session, 6, aest=None)
    res = highlights(session, min_score=0.6, limit=10)
    assert [i for i, _ in res] == [best.id, good.id]
    assert [s for _, s in res] == [0.95, 0.80]
    assert highlights(session, min_score=0.0, limit=1) == [(best.id, 0.95)]
    ids = {i for i, _ in highlights(session, min_score=0.0, limit=50)}
    assert dup.id not in ids and alt.id not in ids and low.id in ids


def test_reanalysis_clears_scores_the_new_run_cannot_produce(session):
    from sqlmodel import select
    from photocure.db.repository import PhotoRepository
    img = _row(session, 1, aest=0.7, model="laion-sa-v1-vit-b-16")
    sem = session.exec(select(SemanticScore)).one()
    sem.preference = 0.9
    session.flush()
    PhotoRepository(session).upsert_semantic(SemanticScore(
        image_id=img.id, backend="stub", embedding=quantize_embedding(_vec(2).tolist()),
        aesthetic=None, aesthetic_model=""))
    row = session.exec(select(SemanticScore)).one()
    assert row.backend == "stub"
    assert row.aesthetic is None and row.aesthetic_model == ""
    assert row.preference is None          # the old taste score belonged to the old embedding


def test_sync_pages_through_a_large_table_in_small_chunks(session):
    head = aesthetic.get_head()
    stale = [_row(session, i, aest=0.1) for i in range(1, 8)]                 # unknown head, every row
    stub = _row(session, 20, aest=0.4, backend="stub")                        # must be cleared
    current = _row(session, 21, aest=0.9, model=head.name)                    # already current
    assert sync_aesthetic(session, batch=2) == 8
    rows = _rows(session)
    assert all(rows[r.id].aesthetic_model == head.name for r in stale)
    assert rows[stub.id].aesthetic is None
    assert rows[current.id].aesthetic == 0.9
    assert sync_aesthetic(session, batch=2) == 0

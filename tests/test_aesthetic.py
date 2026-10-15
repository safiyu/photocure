"""Aesthetic head, backfill and highlights query."""
from __future__ import annotations

import numpy as np
import pytest

from photocure.db.models import ImageRecord, SemanticScore
from photocure.db.repository import quantize_embedding
from photocure.inference import aesthetic
from photocure.inference.aesthetic import (
    AestheticHead,
    backfill_aesthetic,
    highlights,
    score_embedding,
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


def _row(session, i, aest=None, emb=True, **img_kw):
    img = ImageRecord(path=f"/lib/{i}.jpg", filename=f"{i}.jpg", **img_kw)
    session.add(img)
    session.flush()
    session.add(SemanticScore(image_id=img.id, aesthetic=aest,
                              embedding=quantize_embedding(_vec(i).tolist()) if emb else ""))
    session.flush()
    return img


def test_backfill_fills_only_missing(session):
    a = _row(session, 1)
    b = _row(session, 2, aest=0.42)
    c = _row(session, 3, emb=False)
    assert backfill_aesthetic(session) == 1
    from sqlmodel import select
    rows = {r.image_id: r.aesthetic for r in session.exec(select(SemanticScore)).all()}
    assert rows[a.id] is not None
    assert rows[b.id] == 0.42
    assert rows[c.id] is None


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

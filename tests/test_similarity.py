"""Embedding similarity search over stored CLIP embeddings."""
from __future__ import annotations

import numpy as np
import pytest

from photocure.clustering import similarity
from photocure.clustering.similarity import find_similar
from photocure.db.models import ImageRecord, SemanticScore
from photocure.db.repository import dequantize_embedding, quantize_embedding


def _populate(session, n=40, dim=512, seed=0):
    rng = np.random.default_rng(seed)
    vecs = {}
    for i in range(n):
        img = ImageRecord(path=f"/lib/{i}.jpg", filename=f"{i}.jpg")
        session.add(img)
        session.flush()
        v = rng.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        emb = quantize_embedding(v.tolist())
        session.add(SemanticScore(image_id=img.id, embedding=emb))
        vecs[img.id] = np.asarray(dequantize_embedding(emb), dtype=np.float32)
    session.flush()
    return vecs


def _brute(vecs, qid, k):
    q = vecs[qid] / np.linalg.norm(vecs[qid])
    scored = [(i, float(np.dot(v / np.linalg.norm(v), q))) for i, v in vecs.items() if i != qid]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[:k]


def test_matches_bruteforce_and_excludes_self(session):
    vecs = _populate(session)
    qid = sorted(vecs)[3]
    got = find_similar(session, qid, k=7)
    exp = _brute(vecs, qid, 7)
    assert [i for i, _ in got] == [i for i, _ in exp]
    assert all(i != qid for i, _ in got)
    assert [s for _, s in got] == pytest.approx([s for _, s in exp], abs=1e-5)
    assert [s for _, s in got] == sorted((s for _, s in got), reverse=True)


def test_near_copy_ranks_first(session):
    vecs = _populate(session, n=20)
    qid = sorted(vecs)[0]
    twin = ImageRecord(path="/lib/twin.jpg", filename="twin.jpg")
    session.add(twin)
    session.flush()
    v = vecs[qid] + 0.01 * np.random.default_rng(1).standard_normal(512).astype(np.float32)
    session.add(SemanticScore(image_id=twin.id, embedding=quantize_embedding((v / np.linalg.norm(v)).tolist())))
    session.flush()
    top = find_similar(session, qid, k=1)[0]
    assert top[0] == twin.id and top[1] > 0.95


def test_k_larger_than_library_and_zero(session):
    vecs = _populate(session, n=5)
    qid = sorted(vecs)[0]
    assert len(find_similar(session, qid, k=50)) == 4
    assert find_similar(session, qid, k=0) == []


def test_chunked_scan_gives_same_answer(session, monkeypatch):
    vecs = _populate(session, n=37)
    qid = sorted(vecs)[5]
    full = find_similar(session, qid, k=10)
    monkeypatch.setattr(similarity, "_CHUNK", 4)
    chunked = find_similar(session, qid, k=10)
    assert [i for i, _ in chunked] == [i for i, _ in full]
    assert [s for _, s in chunked] == pytest.approx([s for _, s in full], abs=1e-6)


def test_missing_embedding_raises(session):
    img = ImageRecord(path="/lib/x.jpg", filename="x.jpg")
    session.add(img)
    session.flush()
    with pytest.raises(ValueError):
        find_similar(session, img.id)
    with pytest.raises(ValueError):
        find_similar(session, 99999)


def test_rows_with_bad_embeddings_are_skipped(session):
    vecs = _populate(session, n=6)
    bad = ImageRecord(path="/lib/bad.jpg", filename="bad.jpg")
    session.add(bad)
    session.flush()
    session.add(SemanticScore(image_id=bad.id, embedding=""))
    session.flush()
    res = find_similar(session, sorted(vecs)[0], k=10)
    assert bad.id not in {i for i, _ in res}

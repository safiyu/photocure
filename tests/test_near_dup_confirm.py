"""Perceptual-hash near-duplicate links must be confirmed by CLIP embeddings."""
from __future__ import annotations

import numpy as np

from photocure.db import models as M
from photocure.db.repository import PhotoRepository, quantize_embedding
from photocure.db.session import get_session

from .world import build_world


def _set(world, link_emb_a, link_emb_b):
    """Link image 2 as a near duplicate of image 1 with the given embeddings."""
    a, b = world.ids[0], world.ids[1]
    with get_session(world.db) as s:
        for iid, v in ((a, link_emb_a), (b, link_emb_b)):
            if v is None:
                continue
            v = np.asarray(v, dtype=np.float32)
            s.add(M.SemanticScore(image_id=iid, top1_category="screenshot", top1_score=0.9,
                                  top2_category="document", top2_score=0.05, score_margin=0.85,
                                  distribution="{}", embedding=quantize_embedding(v / np.linalg.norm(v)),
                                  backend="clip"))
        PhotoRepository(s).get_by_id(b).near_dup_of = a


def _run(world):
    with get_session(world.db) as s:
        recs = PhotoRepository(s).all_images()
        n = world.planner._confirm_near_duplicates(recs)
        link = {r.id: r.near_dup_of for r in recs}
    return n, link


def test_unrelated_embeddings_clear_the_link(tmp_path):
    w = build_world(tmp_path, n=2)
    rng = np.random.default_rng(1)
    _set(w, rng.standard_normal(512), rng.standard_normal(512))
    n, link = _run(w)
    assert n == 1 and link[w.ids[1]] is None


def test_similar_embeddings_keep_the_link(tmp_path):
    w = build_world(tmp_path, n=2)
    base = np.random.default_rng(2).standard_normal(512)
    _set(w, base, base + np.random.default_rng(3).standard_normal(512) * 0.02)
    n, link = _run(w)
    assert n == 0 and link[w.ids[1]] == w.ids[0]


def test_missing_embeddings_leave_the_link_alone(tmp_path):
    w = build_world(tmp_path, n=2)
    _set(w, None, None)
    n, link = _run(w)
    assert n == 0 and link[w.ids[1]] == w.ids[0]

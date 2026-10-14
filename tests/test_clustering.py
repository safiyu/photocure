"""Tests for burst clustering (temporal + cosine window) and best-shot scoring.

Uses the milestone-3 acceptance scenario: a 5-shot burst taken within 4
seconds (1 sharp/open-eye, 2 blurry, 2 closed-eye) plus a far-away singleton.
"""
from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime, timedelta

from photo_curator.config import CuratorConfig
from photo_curator.clustering.burst_detector import (
    BurstDetector,
    BurstGroupResult,
    BurstInput,
    cosine_similarity,
)
from photo_curator.clustering.best_shot_selector import (
    BestShotSelector,
    ShotCandidate,
)


# ---------------------------------------------------------------------------
# Cosine helper
# ---------------------------------------------------------------------------

def test_cosine_similarity_basic():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    assert cosine_similarity(a, c) == pytest.approx(0.0, abs=1e-6)
    assert cosine_similarity(np.zeros(3), b) == 0.0


# ---------------------------------------------------------------------------
# Burst detector
# ---------------------------------------------------------------------------

def _unit(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _near(v: np.ndarray, rng: np.random.Generator, scale: float = 0.03) -> np.ndarray:
    """A unit vector within cosine >= 0.99 of ``v``.

    ``scale`` is the tilt ``eps`` toward a random unit direction: the result is
    the renormalisation of ``(1 - eps) * v + eps * u``. Because ``v`` and ``u``
    are (near-)orthogonal in high dimensions, the cosine with ``v`` is
    ``~1 - O(eps**2)`` — robust to embedding dimension, unlike per-element
    Gaussian noise (whose std must be ~1/sqrt(dim) to stay close).
    """
    u = rng.normal(size=v.shape).astype(np.float32)
    u /= (np.linalg.norm(u) + 1e-9)
    return _unit((1.0 - scale) * v + scale * u)


def _far(v: np.ndarray) -> np.ndarray:
    return _unit(-v)


def test_burst_grouping_within_window(cfg):
    t0 = datetime(2024, 5, 12, 15, 30, 0)
    rng = np.random.default_rng(0)
    base = _unit(rng.normal(size=512))
    items = []
    for i in range(5):
        items.append(BurstInput(
            image_id=100 + i,
            taken_at=t0 + timedelta(seconds=i),
            embedding=_near(base, rng).tolist(),
            sharpness_norm=0.8,
        ))
    # A distant photo (same scene embedding but 10 min later) must NOT join.
    items.append(BurstInput(
        image_id=999,
        taken_at=t0 + timedelta(minutes=10),
        embedding=_near(base, rng).tolist(),
    ))
    # A temporally close but visually different photo must NOT join either.
    items.append(BurstInput(
        image_id=998,
        taken_at=t0 + timedelta(seconds=1),
        embedding=_far(base).tolist(),
    ))

    det = BurstDetector(cfg)
    groups = det.cluster(items)
    assert len(groups) == 1
    assert sorted(groups[0]) == [100, 101, 102, 103, 104]


def test_burst_window_boundary(cfg):
    t0 = datetime(2024, 5, 12, 15, 30, 0)
    rng = np.random.default_rng(1)
    base = _unit(rng.normal(size=512))
    # 30 s apart -> inside window (<= 30). 31 s apart -> outside.
    inside = [BurstInput(1, t0, _near(base, rng).tolist()),
              BurstInput(2, t0 + timedelta(seconds=30), _near(base, rng).tolist())]
    outside = [BurstInput(3, t0, _near(base, rng).tolist()),
               BurstInput(4, t0 + timedelta(seconds=31), _near(base, rng).tolist())]
    assert len(BurstDetector(cfg).cluster(inside)) == 1
    assert len(BurstDetector(cfg).cluster(outside)) == 0


def test_transitive_burst_chaining(cfg):
    """A-B similar, B-C similar (each within 30 s) -> one group of 3 even
    though A and C are 40 s apart (union-find transitivity)."""
    t0 = datetime(2024, 5, 12, 15, 30, 0)
    rng = np.random.default_rng(2)
    base = _unit(rng.normal(size=512))
    a = _near(base, rng)
    b = _near(base, rng)
    c = _near(base, rng)
    items = [
        BurstInput(1, t0, a.tolist()),
        BurstInput(2, t0 + timedelta(seconds=10), b.tolist()),
        BurstInput(3, t0 + timedelta(seconds=40), c.tolist()),
    ]
    groups = BurstDetector(cfg).cluster(items)
    assert len(groups) == 1
    assert sorted(groups[0]) == [1, 2, 3]


def test_singletons_excluded(cfg):
    t0 = datetime(2024, 5, 12, 15, 30, 0)
    rng = np.random.default_rng(3)
    base = _unit(rng.normal(size=512))
    items = [BurstInput(i, t0 + timedelta(minutes=i), _near(base, rng).tolist())
             for i in range(5)]
    assert BurstDetector(cfg).cluster(items) == []


def test_missing_embedding_or_time_excluded(cfg):
    t0 = datetime(2024, 5, 12, 15, 30, 0)
    rng = np.random.default_rng(4)
    base = _unit(rng.normal(size=512))
    items = [
        BurstInput(1, t0, _near(base, rng).tolist()),
        BurstInput(2, t0 + timedelta(seconds=1), None),          # no embedding
        BurstInput(3, None, _near(base, rng).tolist()),          # no timestamp
        BurstInput(4, t0 + timedelta(seconds=2), _near(base, rng).tolist()),
    ]
    groups = BurstDetector(cfg).cluster(items)
    # Only 1 and 4 can form a burst.
    assert len(groups) == 1
    assert sorted(groups[0]) == [1, 4]


# ---------------------------------------------------------------------------
# Best-shot selector — milestone-3 acceptance scenario
# ---------------------------------------------------------------------------

def test_best_shot_prefers_sharp_open_eye(cfg):
    sel = BestShotSelector(cfg)
    sharp = ShotCandidate(image_id=1, sharpness_norm=0.95, face_sharpness_norm=0.9,
                          eye_open=0.28, width=4000, height=3000, has_face=True)
    blurry = ShotCandidate(image_id=2, sharpness_norm=0.10, face_sharpness_norm=0.1,
                           eye_open=0.27, width=4000, height=3000, has_face=True)
    closed1 = ShotCandidate(image_id=3, sharpness_norm=0.80, face_sharpness_norm=0.7,
                            eye_open=0.10, width=4000, height=3000, has_face=True)
    closed2 = ShotCandidate(image_id=4, sharpness_norm=0.78, face_sharpness_norm=0.65,
                            eye_open=0.12, width=4000, height=3000, has_face=True)
    lowres = ShotCandidate(image_id=5, sharpness_norm=0.95, face_sharpness_norm=0.9,
                           eye_open=0.28, width=640, height=480, has_face=True)
    best = sel.pick_best_shot([closed1, blurry, sharp, closed2, lowres])
    assert best is not None
    assert best.image_id == 1
    # Sanity: the composite ordering matches the tie-break chain.
    ranked = sel.rank([sharp, blurry, closed1, closed2, lowres])
    assert ranked[0].image_id == 1
    assert ranked[-1].image_id in (2, 3, 4)


def test_best_shot_empty_list_returns_none(cfg):
    assert BestShotSelector(cfg).pick_best_shot([]) is None


def test_face_components_ignored_without_face(cfg):
    sel = BestShotSelector(cfg)
    with_face = ShotCandidate(image_id=1, sharpness_norm=0.5,
                              face_sharpness_norm=0.9, eye_open=0.3,
                              width=1000, height=800, has_face=True)
    no_face = ShotCandidate(image_id=2, sharpness_norm=0.5,
                            face_sharpness_norm=0.9, eye_open=0.3,
                            width=1000, height=800, has_face=False)
    s_wf, _ = sel.score(with_face)
    s_nf, _ = sel.score(no_face)
    assert s_wf > s_nf


def test_eye_openness_mapping(cfg):
    from photo_curator.clustering.best_shot_selector import _eye_open_score
    t = cfg.burst
    assert _eye_open_score(0.10, t) == 0.0
    assert _eye_open_score(t.ear_open, t) == 1.0
    assert _eye_open_score(0.5, t) == 1.0
    mid = _eye_open_score((t.ear_closed + t.ear_open) / 2, t)
    assert 0.4 < mid < 0.6

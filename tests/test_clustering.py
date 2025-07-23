"""Tests for burst clustering (temporal + cosine window) and best-shot scoring.

Uses the milestone-3 acceptance scenario: a 5-shot burst taken within 4
seconds (1 sharp/open-eye, 2 blurry, 2 closed-eye) plus a far-away singleton.
"""
from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime, timedelta

from photocure.clustering.burst_detector import (
    BurstDetector,
    BurstInput,
    cosine_similarity,
)
from photocure.clustering.best_shot_selector import (
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
    ``~1 - O(eps**2)`` - robust to embedding dimension, unlike per-element
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
# Best-shot selector - milestone-3 acceptance scenario
# ---------------------------------------------------------------------------

def test_best_shot_prefers_sharp_open_eye(cfg):
    sel = BestShotSelector(cfg)
    sharp = ShotCandidate(image_id=1, sharpness_norm=0.95, face_sharpness_norm=0.9,
                          eye_open=0.95, width=4000, height=3000, has_face=True)
    blurry = ShotCandidate(image_id=2, sharpness_norm=0.10, face_sharpness_norm=0.1,
                           eye_open=0.93, width=4000, height=3000, has_face=True)
    closed1 = ShotCandidate(image_id=3, sharpness_norm=0.80, face_sharpness_norm=0.7,
                            eye_open=0.10, width=4000, height=3000, has_face=True)
    closed2 = ShotCandidate(image_id=4, sharpness_norm=0.78, face_sharpness_norm=0.65,
                            eye_open=0.12, width=4000, height=3000, has_face=True)
    lowres = ShotCandidate(image_id=5, sharpness_norm=0.95, face_sharpness_norm=0.9,
                           eye_open=0.95, width=640, height=480, has_face=True)
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
    from photocure.clustering.best_shot_selector import _eye_open_score
    # Openness is 1 - eye_closed_score and is only clipped to [0, 1].
    assert _eye_open_score(0.10) == pytest.approx(0.10)
    assert _eye_open_score(1.5) == 1.0
    assert _eye_open_score(-0.2) == 0.0
    assert _eye_open_score(None) is None


# ---------------------------------------------------------------------------
# Review fixes: burst detector
# ---------------------------------------------------------------------------

def _scene(n, spacing_s, source="exif:dtoriginal", start_id=1, seed=11):
    t0 = datetime(2024, 5, 12, 15, 0, 0)
    rng = np.random.default_rng(seed)
    base = _unit(rng.normal(size=512))
    return [BurstInput(start_id + i, t0 + timedelta(seconds=i * spacing_s),
                       _near(base, rng).tolist(), taken_at_source=source)
            for i in range(n)]


def test_mtime_and_unknown_sources_never_form_bursts(cfg):
    for src in ("mtime", "unknown"):
        assert BurstDetector(cfg).cluster(_scene(6, 1, source=src)) == []


def test_filename_and_exif_sources_can_form_bursts(cfg):
    for src in ("filename", "exif:dtoriginal", "exif:create", ""):
        assert len(BurstDetector(cfg).cluster(_scene(4, 1, source=src))) == 1


def test_mixed_sources_only_cluster_trustworthy_frames(cfg):
    good = _scene(3, 1, start_id=1)
    bad = _scene(3, 1, source="mtime", start_id=10)
    groups = BurstDetector(cfg).cluster(good + bad)
    assert [sorted(g) for g in groups] == [[1, 2, 3]]


def test_burst_span_is_capped(cfg):
    # 25 frames 25 s apart chain through the 30 s window for 600 s overall.
    items = _scene(25, 25)
    groups = BurstDetector(cfg).cluster(items)
    assert len(groups) > 1
    times = {i.image_id: i.taken_at for i in items}
    for g in groups:
        span = (max(times[i] for i in g) - min(times[i] for i in g)).total_seconds()
        assert span <= 120
    assert sum(len(g) for g in groups) >= 20


def test_burst_span_cap_is_configurable(cfg):
    cfg.burst.__dict__["max_burst_duration_s"] = 10_000
    assert len(BurstDetector(cfg).cluster(_scene(25, 25))) == 1


def test_vectorized_clustering_matches_bruteforce(cfg):
    rng = np.random.default_rng(5)
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    centers = [_unit(rng.normal(size=64)) for _ in range(4)]
    items = []
    for i in range(80):
        c = centers[int(rng.integers(0, 4))]
        items.append(BurstInput(i, t0 + timedelta(seconds=float(rng.integers(0, 90))),
                                _near(c, rng, 0.2).tolist(),
                                taken_at_source="exif:dtoriginal"))
    cfg.burst.__dict__["max_burst_duration_s"] = 1e9
    got = {frozenset(g) for g in BurstDetector(cfg).cluster(items)}

    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    thr, win = cfg.burst.similarity_threshold, cfg.burst.window_seconds
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            dt = abs((items[a].taken_at - items[b].taken_at).total_seconds())
            sim = float(np.dot(items[a].embedding, items[b].embedding))
            if dt <= win and sim >= thr:
                parent[find(a)] = find(b)
    groups = {}
    for i in range(len(items)):
        groups.setdefault(find(i), set()).add(items[i].image_id)
    expected = {frozenset(g) for g in groups.values() if len(g) >= 2}
    assert got == expected


def test_cluster_records_skips_mtime_dated_rows(cfg):
    from photocure.db.models import ImageRecord, SemanticScore
    from photocure.db.repository import quantize_embedding
    rng = np.random.default_rng(2)
    base = _unit(rng.normal(size=512))
    recs = []
    for i, src in enumerate(["mtime", "mtime", "exif:dtoriginal", "exif:dtoriginal"]):
        r = ImageRecord(id=i + 1, path=f"/p/{i}.jpg", filename=f"{i}.jpg",
                        taken_at=datetime(2024, 5, 12, 10, 0, i), taken_at_source=src,
                        infer_status="inferred")
        r.semantic = SemanticScore(embedding=quantize_embedding(_near(base, rng).tolist()))
        recs.append(r)
    groups = BurstDetector(cfg).cluster_records(recs)
    assert [sorted(g.image_ids) for g in groups] == [[3, 4]]


# ---------------------------------------------------------------------------
# Review fixes: best-shot scoring
# ---------------------------------------------------------------------------

def test_unknown_eye_openness_is_excluded_and_weights_renormalized(cfg):
    sel = BestShotSelector(cfg)
    unknown = ShotCandidate(image_id=1, sharpness_norm=0.8, face_sharpness_norm=0.8,
                            eye_open=None, width=4000, height=3000, has_face=True)
    score, br = sel.score(unknown)
    assert br["eye_known"] is False
    w = cfg.burst
    expected = ((w.weight_blur * 0.8 + w.weight_face_sharp * 0.8
                 + w.weight_resolution * br["resolution"])
                / (w.weight_blur + w.weight_face_sharp + w.weight_resolution))
    assert score == pytest.approx(expected)


def test_unknown_eye_openness_is_not_penalized_against_open_eyes(cfg):
    sel = BestShotSelector(cfg)
    kw = dict(sharpness_norm=0.9, face_sharpness_norm=0.9, width=4000, height=3000,
              has_face=True)
    unknown = sel.score(ShotCandidate(image_id=1, eye_open=None, **kw))[0]
    half_open = sel.score(ShotCandidate(image_id=2, eye_open=0.20, **kw))[0]
    # Unknown sits with the other terms (~0.9), above a half-open score of 0.5.
    assert unknown > half_open


def test_closed_eyes_still_lose_when_known(cfg):
    sel = BestShotSelector(cfg)
    kw = dict(sharpness_norm=0.9, face_sharpness_norm=0.9, width=4000, height=3000,
              has_face=True)
    open_ = ShotCandidate(image_id=1, eye_open=0.30, **kw)
    closed = ShotCandidate(image_id=2, eye_open=0.05, **kw)
    assert sel.pick_best_shot([closed, open_]).image_id == 1


def test_aesthetic_is_a_fifth_term_and_optional(cfg):
    sel = BestShotSelector(cfg)
    kw = dict(sharpness_norm=0.6, width=4000, height=3000)
    pretty = ShotCandidate(image_id=1, aesthetic=0.95, **kw)
    plain = ShotCandidate(image_id=2, aesthetic=0.10, **kw)
    assert sel.pick_best_shot([plain, pretty]).image_id == 1
    none = sel.score(ShotCandidate(image_id=3, **kw))[1]
    assert none["aesthetic"] == 0.0
    assert sel.score(pretty)[1]["aesthetic"] == pytest.approx(0.95)


def test_aesthetic_does_not_override_sharpness_by_much(cfg):
    sel = BestShotSelector(cfg)
    kw = dict(width=4000, height=3000)
    sharp = ShotCandidate(image_id=1, sharpness_norm=0.95, aesthetic=0.3, **kw)
    soft = ShotCandidate(image_id=2, sharpness_norm=0.30, aesthetic=0.9, **kw)
    assert sel.pick_best_shot([soft, sharp]).image_id == 1


def test_candidate_uses_taxonomy_face_gate(cfg):
    from photocure.clustering.best_shot_selector import _candidate_from_record
    from photocure.db.models import FaceDetection, ImageRecord
    rec = ImageRecord(id=1, path="/p", filename="p", width=100, height=100)
    rec.faces = [FaceDetection(bbox_x1=0, bbox_y1=0, bbox_x2=9, bbox_y2=9,
                               confidence=0.40, face_sharpness=200,
                               eye_aspect_ratio=0.3)]
    assert _candidate_from_record(rec, cfg.taxonomy.face_confidence_min).has_face is False
    rec.faces[0].confidence = 0.9
    assert _candidate_from_record(rec, cfg.taxonomy.face_confidence_min).has_face is True


def test_candidate_derives_openness_and_blink_count_from_the_blink_fields(cfg):
    from photocure.clustering.best_shot_selector import _candidate_from_record
    from photocure.db.models import FaceDetection, ImageRecord
    rec = ImageRecord(id=1, path="/p", filename="p", width=100, height=100, blink_count=2)
    rec.faces = [FaceDetection(bbox_x1=0, bbox_y1=0, bbox_x2=9, bbox_y2=9, confidence=0.9,
                               face_sharpness=200, eye_closed_score=0.8)]
    c = _candidate_from_record(rec)
    assert c.eye_open == pytest.approx(0.2) and c.blink_count == 2


def test_candidate_carries_unknown_eye_and_aesthetic(cfg):
    from photocure.clustering.best_shot_selector import _candidate_from_record
    from photocure.db.models import FaceDetection, ImageRecord, SemanticScore
    rec = ImageRecord(id=1, path="/p", filename="p", width=100, height=100)
    rec.faces = [FaceDetection(bbox_x1=0, bbox_y1=0, bbox_x2=9, bbox_y2=9,
                               confidence=0.9, face_sharpness=200,
                               eye_aspect_ratio=None)]
    rec.semantic = SemanticScore(aesthetic=0.7)
    c = _candidate_from_record(rec)
    assert c.eye_open is None and c.aesthetic == 0.7


def test_eye_open_score_unknown_is_none(cfg):
    from photocure.clustering.best_shot_selector import _eye_open_score
    assert _eye_open_score(None, cfg.burst) is None

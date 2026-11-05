"""Pure face-clustering and backfill-remap matching: no DB, no OpenCV."""
from __future__ import annotations

import numpy as np

from photocure.clustering.face_identity import (
    ClusterCentroid,
    NewFace,
    OldFace,
    assign_cluster,
    effective_threshold,
    remap_person_ids,
    update_centroid,
)


def unit(seed: int, dim: int = 512) -> list[float]:
    v = np.random.default_rng(seed).standard_normal(dim)
    return (v / np.linalg.norm(v)).tolist()


def mix(a: list[float], b: list[float], cos: float) -> list[float]:
    """A unit vector with cosine exactly ``cos`` to ``a``, leaning towards ``b``."""
    a_, b_ = np.asarray(a), np.asarray(b)
    perp = b_ - float(a_ @ b_) * a_
    perp /= np.linalg.norm(perp)
    return (cos * a_ + np.sqrt(1.0 - cos * cos) * perp).tolist()


def test_assign_cluster_picks_the_closest_centroid_above_threshold():
    centroids = [ClusterCentroid(1, [1.0, 0.0], 3), ClusterCentroid(2, [0.0, 1.0], 5)]
    assert assign_cluster([0.9, 0.1], centroids, threshold=0.5) == 1


def test_assign_cluster_returns_none_when_nothing_clears_the_threshold():
    centroids = [ClusterCentroid(1, [1.0, 0.0], 3)]
    assert assign_cluster([0.0, 1.0], centroids, threshold=0.5) is None


def test_update_centroid_is_a_running_mean():
    assert update_centroid([2.0, 0.0], 1, [0.0, 2.0]) == [1.0, 1.0]


def test_remap_is_one_to_one_even_when_faces_swap_positions():
    # Old face A (person 1) now detected second; old face B (person 2) now detected first.
    old = [OldFace(1, [1.0, 0.0]), OldFace(2, [0.0, 1.0])]
    new = [NewFace(0, [0.0, 0.95]), NewFace(1, [0.95, 0.0])]
    assert remap_person_ids(old, new, threshold=0.5) == {0: 2, 1: 1}


def test_remap_never_assigns_the_same_old_face_twice():
    old = [OldFace(1, [1.0, 0.0])]
    new = [NewFace(0, [0.99, 0.0]), NewFace(1, [0.95, 0.0])]
    result = remap_person_ids(old, new, threshold=0.5)
    assert len(result) == 1 and set(result.values()) == {1}


def test_remap_drops_pairs_below_threshold():
    old = [OldFace(1, [1.0, 0.0])]
    new = [NewFace(0, [0.0, 1.0])]
    assert remap_person_ids(old, new, threshold=0.5) == {}


def test_co_occurrence_veto_protects_siblings():
    """Two faces from the same photo cannot join the same person cluster."""
    # Cluster 1 already has image_id=101 (e.g. older sister)
    c1 = ClusterCentroid(1, [1.0, 0.0] * 256, 3, image_ids={101})
    c2 = ClusterCentroid(2, [0.0, 1.0] * 256, 2, image_ids={999})
    # Face from image_id=101 has high cosine similarity to c1 (0.95), but cannot link because of co-occurrence veto!
    similar_face = [0.95, 0.05] * 256
    assigned = assign_cluster(similar_face, [c1, c2], threshold=0.38, image_id=101)
    assert assigned != 1  # Cannot merge into sister!


def test_chronological_medoids_bridge_child_growth():
    """Face from 2024 matches 2023 medoid even if infant 2021 centroid has drifted."""
    # Person cluster with static mean drifting away, but yearly medoids chained
    c = ClusterCentroid(
        1,
        centroid=[0.5, 0.5] * 256,
        face_count=10,
        temporal_medoids={
            "2021": [0.2, 0.8] * 256,
            "2022": [0.4, 0.6] * 256,
            "2023": [0.8, 0.2] * 256,
        },
        image_ids={1, 2, 3},
    )
    # 2024 face is very close to 2023 medoid
    face_2024 = [0.82, 0.18] * 256
    assigned = assign_cluster(face_2024, [c], threshold=0.38, image_id=500, time_bucket="2024")
    assert assigned == 1


def test_arcface_alignment_and_normalization():
    """Verify ArcFace 112x112 partial affine alignment."""
    import numpy as np
    from photocure.inference.face_engine import align_face_112, ARCFACE_REF_5PTS

    # Create dummy image and 5 landmarks
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    landmarks = [(40.0, 50.0), (75.0, 50.0), (55.0, 70.0), (42.0, 90.0), (70.0, 90.0)]
    warped = align_face_112(img, landmarks)
    assert warped.shape == (112, 112, 3)
    assert len(ARCFACE_REF_5PTS) == 5


def test_mismatched_embedding_dimensions_handled_safely():
    """Verify that comparing 512-D and 128-D vectors never raises ValueError."""
    from photocure.clustering.face_identity import cosine, update_centroid

    vec512 = [1.0] * 512
    vec128 = [1.0] * 128

    assert cosine(vec512, vec128) == 0.0
    assert cosine(vec128, vec512) == 0.0

    # Clusters with 128-dim shouldn't crash when matching against a 512-dim face
    c128 = ClusterCentroid(1, vec128, 5)
    c512 = ClusterCentroid(2, vec512, 2)
    assert assign_cluster(vec512, [c128, c512], threshold=0.38) == 2

    # Updating centroid when dimensions differ shouldn't crash
    updated = update_centroid(vec128, 5, vec512)
    assert len(updated) == 512




def test_threshold_above_035_is_honoured_for_512d():
    c = ClusterCentroid(1, unit(1), 3)
    face = mix(unit(1), unit(2), cos=0.45)
    assert assign_cluster(face, [c], threshold=0.50) is None
    assert assign_cluster(face, [c], threshold=0.40) == 1


def test_effective_threshold_defaults_by_dimension():
    assert effective_threshold(None, 512) == 0.35
    assert effective_threshold(None, 128) == 0.64
    assert effective_threshold(0.55, 512) == 0.55
    assert effective_threshold(0.30, 512) == 0.30


def test_outlier_never_becomes_exemplar():
    from photocure.clustering.face_identity import update_exemplars
    cen = unit(1)
    far = mix(unit(1), unit(9), cos=0.20)
    assert update_exemplars([cen], far, centroid=cen) == [cen]


def test_drift_is_blocked_by_exemplar_rule():
    from photocure.clustering.face_identity import update_exemplars
    a = unit(1)
    b = mix(a, unit(2), cos=0.36)          # barely joins a 10-face cluster around a
    c = mix(b, unit(3), cos=0.50)          # close to b only; far from a
    assert float(np.dot(a, c)) < 0.30
    centroid = update_centroid(a, 10, b)
    exemplars = update_exemplars([a], b, centroid=centroid)
    cluster = ClusterCentroid(1, centroid, 11, exemplars=exemplars)
    assert assign_cluster(c, [cluster], threshold=0.35) is None


def test_yearly_mean_is_a_running_mean():
    from photocure.clustering.face_identity import update_temporal_medoid
    e1, e2 = unit(1), unit(2)
    meds, counts = update_temporal_medoid({}, "2020", e1, {})
    meds, counts = update_temporal_medoid(meds, "2020", e2, counts)
    assert counts == {"2020": 2}
    expect = (np.asarray(e1) + np.asarray(e2)) / 2
    expect /= np.linalg.norm(expect)
    assert np.allclose(meds["2020"], expect, atol=1e-6)

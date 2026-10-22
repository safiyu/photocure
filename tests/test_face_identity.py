"""Pure face-clustering and backfill-remap matching: no DB, no OpenCV."""
from __future__ import annotations

from photocure.clustering.face_identity import (
    ClusterCentroid,
    NewFace,
    OldFace,
    assign_cluster,
    remap_person_ids,
    update_centroid,
)


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

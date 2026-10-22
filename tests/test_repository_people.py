"""PersonCluster centroid storage and the repository CRUD around it."""
from __future__ import annotations

import pytest

from photocure.db.repository import dequantize_embedding


def test_create_and_update_person_cluster_round_trips_the_centroid(repo):
    p = repo.create_person_cluster([1.0, 0.0])
    assert p.id is not None and p.face_count == 1
    repo.update_person_cluster(p.id, [0.5, 0.5], 2)
    clusters = repo.all_person_clusters()
    assert clusters[0].face_count == 2 and dequantize_embedding(clusters[0].centroid_json) == [0.5, 0.5]


def test_find_person_by_name_is_case_and_whitespace_insensitive(repo):
    repo.rename_person(repo.create_person_cluster([1.0]).id, "Safi")
    assert repo.find_person_by_name("  safi ") is not None


def test_find_person_by_name_returns_none_when_unknown(repo):
    assert repo.find_person_by_name("nobody") is None


def test_rename_person_raises_for_unknown_id(repo):
    with pytest.raises(ValueError):
        repo.rename_person(999, "x")

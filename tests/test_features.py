"""Feature settings: highlight weights, keyword export and watch folder, stored beside the database."""
from __future__ import annotations

import json

import pytest

from photocure import features


def test_defaults_are_returned_when_nothing_is_stored(tmp_path):
    f = features.load_features(tmp_path / "a.db")
    assert f == features.DEFAULTS
    assert f["write_keywords"] is False and f["watch_enabled"] is False and f["watch_path"] == ""


def test_save_merges_and_round_trips(tmp_path):
    db = tmp_path / "a.db"
    out = features.save_features(db, {"highlight_rating_weight": 0.2})
    assert out["highlight_rating_weight"] == 0.2
    out = features.save_features(db, {"write_keywords": True, "watch_path": str(tmp_path)})
    assert out["highlight_rating_weight"] == 0.2 and out["write_keywords"] is True
    assert features.load_features(db) == out
    assert features.features_path(db).name == "a.features.json"


@pytest.mark.parametrize("bad", [
    {"nope": 1},
    {"highlight_rating_weight": 0.9},
    {"highlight_favorite_weight": -0.1},
    {"highlight_preference_weight": 1.5},
    {"highlight_rating_weight": "high"},
    {"highlight_rating_weight": True},
    {"write_keywords": "yes"},
    {"watch_enabled": 1},
    {"watch_path": 5},
    {"watch_path": "x" * 5000},
])
def test_invalid_values_are_rejected_and_nothing_is_stored(tmp_path, bad):
    db = tmp_path / "a.db"
    with pytest.raises(ValueError):
        features.save_features(db, bad)
    assert not features.features_path(db).exists()


def test_corrupt_or_out_of_range_stored_values_fall_back_to_defaults(tmp_path):
    db = tmp_path / "a.db"
    features.features_path(db).write_text("not json")
    assert features.load_features(db) == features.DEFAULTS
    features.features_path(db).write_text(json.dumps(
        {"highlight_rating_weight": 99, "write_keywords": True, "mystery": 1}))
    f = features.load_features(db)
    assert f["highlight_rating_weight"] == features.DEFAULTS["highlight_rating_weight"]
    assert f["write_keywords"] is True and "mystery" not in f


def test_save_none_resets_a_key(tmp_path):
    db = tmp_path / "a.db"
    features.save_features(db, {"highlight_rating_weight": 0.3})
    out = features.save_features(db, {"highlight_rating_weight": None})
    assert out["highlight_rating_weight"] == features.DEFAULTS["highlight_rating_weight"]

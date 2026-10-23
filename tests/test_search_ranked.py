"""Search that needs every part of the query and stops at the strong matches."""
from __future__ import annotations

import numpy as np
import pytest

from photocure import search
from photocure.db.models import ImageRecord, SemanticScore
from photocure.db.repository import quantize_embedding


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _axis(i: int) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[i] = 1.0
    return v


class _Encoder:
    def __init__(self, table):
        self.table = table

    def encode(self, text):
        return _unit(self.table[text])


def _add(session, i, emb):
    img = ImageRecord(path=f"/lib/{i}.jpg", filename=f"{i}.jpg")
    session.add(img)
    session.flush()
    session.add(SemanticScore(image_id=img.id, backend="clip",
                              embedding=quantize_embedding(_unit(emb).tolist())))
    session.flush()
    return img


@pytest.mark.parametrize("text, parts", [
    ("playing with dog", ["playing", "dog"]),
    ("a dog in the snow", ["dog", "snow"]),
    ("man on a bike holding a coffee", ["man", "bike", "coffee"]),
    ("photos of dogs", []),
    ("sunset", []),
    ("quiet moment", []),
])
def test_queries_split_into_the_things_they_ask_for(text, parts):
    assert search.split_concepts(text) == parts


@pytest.fixture()
def library(session):
    rng = np.random.default_rng(7)
    noise = [_add(session, 100 + i, rng.standard_normal(512)) for i in range(60)]
    both = _add(session, 1, _axis(0) + _axis(1))
    playing_only = _add(session, 2, _axis(1) + 0.05 * rng.standard_normal(512))
    dog_only = _add(session, 3, _axis(0) + 0.05 * rng.standard_normal(512))
    enc = _Encoder({"playing with dog": _axis(0) + _axis(1), "playing": _axis(1), "dog": _axis(0)})
    return both, playing_only, dog_only, noise, enc


def test_the_photo_with_every_part_ranks_first(session, library):
    both, playing_only, dog_only, _noise, enc = library
    res = search.search_ranked(session, enc, "playing with dog", 20, strict=False)
    ids = [i for i, _ in res.hits]
    assert ids[0] == both.id
    assert set(ids[1:3]) == {playing_only.id, dog_only.id}
    assert res.parts == ["playing", "dog"]


def test_strict_stops_at_the_strong_matches_and_reports_the_rest(session, library):
    *_, enc = library
    strict = search.search_ranked(session, enc, "playing with dog", 60)
    loose = search.search_ranked(session, enc, "playing with dog", 60, strict=False)
    assert len(strict.hits) == max(strict.strong, search.MIN_SHOWN)
    assert len(strict.hits) < len(loose.hits) == 60
    assert strict.more == len(loose.hits) - len(strict.hits) > 0
    assert loose.more == 0
    assert [i for i, _ in strict.hits] == [i for i, _ in loose.hits][:len(strict.hits)]


def test_a_single_idea_query_has_no_parts_and_keeps_the_best_first(session, library):
    both, _po, dog_only, _noise, enc = library
    enc.table["dog"] = _axis(0)
    res = search.search_ranked(session, enc, "dog", 5, strict=False)
    assert res.parts == []
    assert {i for i, _ in res.hits[:2]} == {both.id, dog_only.id}


def test_an_empty_library_returns_nothing(session):
    enc = _Encoder({"playing with dog": _axis(0) + _axis(1), "playing": _axis(1), "dog": _axis(0)})
    res = search.search_ranked(session, enc, "playing with dog")
    assert res.hits == [] and res.strong == 0 and res.more == 0


def test_extra_filters_narrow_the_search(session, library):
    both, *_rest, enc = library
    res = search.search_ranked(session, enc, "playing with dog", 20, strict=False,
                               extra_filters=(ImageRecord.id != both.id,))
    assert both.id not in [i for i, _ in res.hits]

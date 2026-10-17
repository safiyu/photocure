"""Natural-language search: ranking over stored embeddings, the text encoder, API and CLI."""
from __future__ import annotations

import types

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import select
from typer.testing import CliRunner

import cv2

from photocure import search
from photocure.config import CuratorConfig
from photocure.db import models as M
from photocure.db.models import ImageRecord, SemanticScore
from photocure.db.repository import quantize_embedding
from photocure.db.session import get_session
from photocure.search import SearchUnavailable, TextEncoder, search_embeddings
from photocure.ui.app import create_app

from .conftest import make_image
from .test_c3_faces import PORTRAIT


def _axis(i: int) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[i] = 1.0
    return v


def _add(session, i, emb, backend="clip", **img_kw):
    img = ImageRecord(path=f"/lib/{i}.jpg", filename=f"{i}.jpg", **img_kw)
    session.add(img)
    session.flush()
    session.add(SemanticScore(image_id=img.id, backend=backend,
                              embedding=quantize_embedding(np.asarray(emb).tolist())))
    session.flush()
    return img


def test_results_are_ranked_by_cosine_to_the_query(session):
    a = _add(session, 1, _axis(0))
    b = _add(session, 2, _axis(1))
    c = _add(session, 3, _axis(2))
    q = _axis(1) + 0.5 * _axis(0)
    res = search_embeddings(session, q, k=3)
    assert [i for i, _ in res] == [b.id, a.id, c.id]
    expected = float(np.dot(_axis(1), q / np.linalg.norm(q)))
    assert res[0][1] == pytest.approx(expected, abs=1e-3)


def test_placeholder_missing_duplicate_and_alternate_rows_are_skipped(session):
    good = _add(session, 1, _axis(0))
    _add(session, 2, _axis(0), backend="stub")
    _add(session, 3, _axis(0), scan_status="missing")
    dup = _add(session, 4, _axis(0), is_duplicate=True)
    alt = _add(session, 5, _axis(0), is_alternate=True)
    assert [i for i, _ in search_embeddings(session, _axis(0), k=10)] == [good.id]
    everything = {i for i, _ in search_embeddings(session, _axis(0), k=10, include_duplicates=True)}
    assert everything == {good.id, dup.id, alt.id}


def test_k_and_min_score_limit_the_results(session):
    for i in range(6):
        _add(session, i, _axis(i))
    q = _axis(0) + 0.3 * _axis(1)
    assert len(search_embeddings(session, q, k=2)) == 2
    only_close = search_embeddings(session, q, k=10, min_score=0.5)
    assert [s >= 0.5 for _, s in only_close] == [True] * len(only_close) and len(only_close) == 1
    assert search_embeddings(session, q, k=0) == []


def test_zero_query_vector_is_rejected(session):
    with pytest.raises(ValueError):
        search_embeddings(session, np.zeros(512), k=3)


class _FakeEngine:
    def __init__(self, available=True):
        self.available = available
        self.calls: list[list[str]] = []
        self.released = 0

    def text_search_available(self):
        return self.available

    def encode_queries(self, texts):
        self.calls.append(list(texts))
        rng = np.random.default_rng(abs(hash(texts[0])) % (2 ** 32))
        return rng.standard_normal((len(texts), 512)).astype(np.float32)

    def release(self):
        self.released += 1


def test_encoder_returns_a_unit_vector_and_caches_repeats():
    eng = _FakeEngine()
    enc = TextEncoder(lambda: eng, idle_s=60)
    v1 = enc.encode("  dog   on the beach ")
    v2 = enc.encode("dog on the beach")
    assert v1.shape == (512,) and abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5
    assert np.array_equal(v1, v2)
    assert len(eng.calls) == 1
    assert eng.calls[0] == ["dog on the beach", "a photo of dog on the beach"]
    enc.close()


def test_encoder_rejects_empty_queries_and_missing_model():
    enc = TextEncoder(lambda: _FakeEngine(), idle_s=60)
    with pytest.raises(ValueError):
        enc.encode("   ")
    enc.close()
    missing = TextEncoder(lambda: _FakeEngine(available=False), idle_s=60)
    with pytest.raises(SearchUnavailable):
        missing.encode("cat")
    missing.close()


def test_encoder_frees_the_model_and_reloads_on_demand():
    made: list[_FakeEngine] = []

    def factory():
        made.append(_FakeEngine())
        return made[-1]

    enc = TextEncoder(factory, idle_s=60)
    enc.encode("one")
    assert enc.loaded
    enc.release()
    assert not enc.loaded and made[0].released == 1
    enc.encode("two")
    assert len(made) == 2
    enc.close()


def test_default_encoder_on_the_placeholder_backend_is_unavailable(cfg):
    from photocure.inference.backends import reset_backend_cache
    reset_backend_cache()
    enc = search.default_encoder(cfg)
    with pytest.raises(SearchUnavailable):
        enc.encode("a cat")
    enc.close()
    reset_backend_cache()


# ---------------------------------------------------------------- API and CLI
@pytest.fixture()
def env(tmp_path):
    app = create_app(str(tmp_path / "s.db"))
    app.state.cfg.cache_dir = tmp_path / "cache"
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    lib = tmp_path / "photos"
    for i in range(3):
        make_image(lib / f"p{i}.jpg", rng_seed=10 + i)
    assert client.post("/api/scan", json={"path": str(lib)}).status_code == 200
    assert client.post("/api/infer", json={"device": "cpu"}).status_code == 200
    with get_session(app.state.db) as s:
        rows = list(s.exec(select(M.SemanticScore).order_by(M.SemanticScore.image_id)).all())
        for n, row in enumerate(rows):
            row.backend = "clip"
            row.embedding = quantize_embedding(_axis(n).tolist())
        ids = [r.image_id for r in rows]
    return types.SimpleNamespace(app=app, client=client, ids=ids, tmp=tmp_path)


class _VecEncoder:
    def __init__(self, vec):
        self.vec = vec

    def encode(self, text):
        return self.vec / np.linalg.norm(self.vec)

    def close(self):
        pass


def test_api_search_returns_ranked_items_with_scores(env):
    env.app.state.text_encoder = _VecEncoder(_axis(2) + 0.2 * _axis(1))
    r = env.client.get("/api/search", params={"q": "anything", "k": 3})
    assert r.status_code == 200
    items = r.json()["items"]
    assert [it["id"] for it in items][:2] == [env.ids[2], env.ids[1]]
    assert items[0]["score"] > items[1]["score"] > items[2]["score"]
    assert r.json()["query"] == "anything"


def test_api_search_validates_input(env):
    assert env.client.get("/api/search", params={"q": "  "}).status_code == 400
    assert env.client.get("/api/search").status_code == 422
    assert env.client.get("/api/search", params={"q": "x", "k": 0}).status_code == 422


def test_api_search_explains_a_missing_model(env):
    class _Missing:
        def encode(self, text):
            raise SearchUnavailable("The AI model is not installed.")

        def close(self):
            pass

    env.app.state.text_encoder = _Missing()
    r = env.client.get("/api/search", params={"q": "cat"})
    assert r.status_code == 503 and "AI model" in r.json()["detail"]


def test_api_search_requires_a_session(env):
    anon = TestClient(env.app)
    assert anon.get("/api/search", params={"q": "cat"}).status_code == 401


def test_cli_search_prints_the_best_matches(env, monkeypatch):
    from photocure import cli
    monkeypatch.setattr(search, "default_encoder", lambda cfg=None: _VecEncoder(_axis(1)))
    res = CliRunner().invoke(cli.app, ["search", "a cat", "--db", str(env.app.state.db), "-n", "2"],
                             env={"COLUMNS": "400"})
    assert res.exit_code == 0, res.output
    with get_session(env.app.state.db) as s:
        best = s.get(ImageRecord, env.ids[1]).filename  # the photo whose embedding the query matches
    assert best in res.output


def _clip_ready() -> bool:
    try:
        import open_clip  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _clip_ready(), reason="open_clip not installed")
def test_real_clip_text_prefers_the_matching_scene(monkeypatch):
    from photocure.inference.backends import detect_backend, reset_backend_cache
    from photocure.inference.semantic_engine import SemanticEngine

    monkeypatch.setenv("PCUR_BACKEND", "clip")
    reset_backend_cache()
    try:
        c = CuratorConfig()
        c.device.device = "cpu"
        engine = SemanticEngine(c, backend=detect_backend(c))
        enc = TextEncoder(lambda: engine, idle_s=60)
        sky = np.concatenate([np.full((100, 256, 3), (210, 150, 80), np.uint8),
                              np.full((156, 256, 3), (60, 150, 60), np.uint8)])
        black = np.zeros((256, 256, 3), np.uint8)
        portrait = cv2.imread(str(PORTRAIT))
        e_sky, e_black, e_face = (np.asarray(r.embedding) for r in
                                  engine.classify_and_embed([sky, black, portrait]))
        land = enc.encode("a landscape with blue sky and green grass")
        assert float(land @ e_sky) > float(land @ e_face)
        assert float(land @ e_sky) > float(land @ e_black)
        dark = enc.encode("a black screen")
        assert float(dark @ e_black) > float(dark @ e_sky)
        assert float(dark @ e_black) > float(dark @ e_face)
        enc.close()
    finally:
        reset_backend_cache()


def test_the_encoder_is_created_once_when_first_searches_arrive_together(env, monkeypatch):
    import threading
    import time
    made = []

    def factory(cfg=None):
        made.append(1)
        time.sleep(0.2)                      # widen the window in which two requests could both build one
        return _VecEncoder(_axis(1))

    monkeypatch.setattr(search, "default_encoder", factory)
    env.app.state.text_encoder = None
    statuses = []

    def go():
        c = TestClient(env.app)
        c.headers["X-PCUR-Token"] = env.app.state.token
        statuses.append(c.get("/api/search", params={"q": "x"}).status_code)

    threads = [threading.Thread(target=go) for _ in range(6)]
    [t.start() for t in threads]
    [t.join(timeout=20) for t in threads]
    assert statuses == [200] * 6 and len(made) == 1

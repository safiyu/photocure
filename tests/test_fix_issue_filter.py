import types

import pytest
from fastapi.testclient import TestClient

from photocure.ui.app import create_app

from .conftest import make_image


@pytest.fixture()
def env(tmp_path):
    app = create_app(str(tmp_path / "ui.db"))
    app.state.cfg.cache_dir = tmp_path / "cache"
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    lib = tmp_path / "photos"
    make_image(lib / "a.jpg", rng_seed=1)
    make_image(lib / "b_blur.jpg", rng_seed=2, blur=True)
    make_image(lib / "c_dark.jpg", rng_seed=3, dark=True)
    assert client.post("/api/scan", json={"path": str(lib)}).status_code == 200
    assert client.post("/api/infer", json={"device": "cpu"}).status_code == 200
    return types.SimpleNamespace(client=client)


def items(env, **params):
    r = env.client.get("/api/images", params=params)
    assert r.status_code == 200, r.text
    return r.json()["items"]


def names(env, **params):
    return sorted(i["filename"] for i in items(env, **params))


def flagged(env, key):
    return sorted(i["filename"] for i in items(env) if i[key])


def test_issue_filter_matches_each_photos_own_flags(env):
    assert names(env, issue="blurry") == flagged(env, "is_blurry") != []
    assert names(env, issue="dark") == flagged(env, "is_too_dark") != []
    assert "b_blur.jpg" in names(env, issue="blurry")
    assert "c_dark.jpg" in names(env, issue="dark")
    assert "a.jpg" not in names(env, issue="blurry,dark")


def test_issue_filter_combines_with_or_and_other_filters(env):
    both = sorted(set(flagged(env, "is_blurry")) | set(flagged(env, "is_too_dark")))
    assert names(env, issue="blurry,dark") == both
    assert names(env, issue="blurry,dark", q="blur") == [n for n in both if "blur" in n]


def test_unknown_issue_is_rejected(env):
    assert env.client.get("/api/images", params={"issue": "bogus"}).status_code == 422
    assert env.client.get("/api/images", params={"issue": "blurry,bogus"}).status_code == 422


def test_facets_count_the_issues(env):
    issues = env.client.get("/api/facets").json()["issues"]
    assert issues["blurry"] == len(flagged(env, "is_blurry"))
    assert issues["dark"] == len(flagged(env, "is_too_dark"))
    assert set(issues) == {"blurry", "dark", "overexposed", "blink", "duplicate", "any"}
    assert issues["any"] >= max(issues["blurry"], issues["dark"])

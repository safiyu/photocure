"""Folder picker and new overview/preview endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from photocure.ui.app import create_app

from .conftest import make_image


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("PCUR_BROWSE_ROOTS", raising=False)
    app = create_app(str(tmp_path / "ui.db"))
    app.state.cfg.cache_dir = tmp_path / "cache"
    c = TestClient(app)
    c.headers["X-PCUR-Token"] = app.state.token
    c.tmp = tmp_path
    c.app_ = app
    return c


def test_browse_requires_token(client):
    r = client.get("/api/browse", headers={"X-PCUR-Token": "wrong"})
    assert r.status_code in (401, 403)


def test_browse_lists_folders_and_counts_photos(client):
    lib = client.tmp / "lib"
    (lib / "trips").mkdir(parents=True)
    (lib / ".hidden").mkdir()
    make_image(lib / "a.jpg", rng_seed=1)
    (lib / "notes.txt").write_text("x")
    d = client.get("/api/browse", params={"path": str(lib)}).json()
    assert [e["name"] for e in d["entries"]] == ["trips"]
    assert d["photos"] == 1
    assert d["parent"] == str(lib.resolve().parent)
    assert d["writable"] is True
    assert client.get("/api/browse", params={"path": str(lib), "hidden": True}
                      ).json()["entries"][0]["name"] == ".hidden"


def test_browse_rejects_bad_paths(client):
    assert client.get("/api/browse", params={"path": "relative/x"}).status_code == 400
    assert client.get("/api/browse", params={"path": str(client.tmp / "nope")}
                      ).status_code == 404


def test_browse_default_has_roots(client):
    d = client.get("/api/browse").json()
    assert d["roots"] and d["roots"][0]["name"] == "Home"


def test_browse_confined_to_roots(tmp_path, monkeypatch):
    allowed = tmp_path / "photos"
    (allowed / "sub").mkdir(parents=True)
    other = tmp_path / "secret"
    other.mkdir()
    monkeypatch.setenv("PCUR_BROWSE_ROOTS", str(allowed))
    app = create_app(str(tmp_path / "ui.db"))
    c = TestClient(app)
    c.headers["X-PCUR-Token"] = app.state.token
    d = c.get("/api/browse").json()
    assert d["path"] == str(allowed.resolve()) and d["parent"] is None
    assert [e["name"] for e in d["entries"]] == ["sub"]
    assert c.get("/api/browse", params={"path": str(other)}).status_code == 403
    esc = str(allowed / ".." / "secret")
    assert c.get("/api/browse", params={"path": esc}).status_code == 403
    assert c.post("/api/browse/mkdir", json={"parent": str(other), "name": "x"}
                  ).status_code == 403


def test_mkdir(client):
    base = client.tmp / "out"
    base.mkdir()
    r = client.post("/api/browse/mkdir", json={"parent": str(base), "name": "Organized"})
    assert r.status_code == 200 and (base / "Organized").is_dir()
    assert client.post("/api/browse/mkdir", json={"parent": str(base), "name": "Organized"}
                       ).status_code == 409
    for bad in ["..", "a/b", "", "x:y"]:
        assert client.post("/api/browse/mkdir", json={"parent": str(base), "name": bad}
                           ).status_code == 400
    assert client.post("/api/browse/mkdir", json={"parent": str(base), "name": "z"},
                       headers={"X-PCUR-Token": "bad"}).status_code in (401, 403)


def test_overview_and_preview(client):
    lib = client.tmp / "lib"
    make_image(lib / "a.jpg", rng_seed=1)
    make_image(lib / "b.jpg", rng_seed=2)
    assert client.post("/api/scan", json={"path": str(lib)}).status_code == 200
    assert client.post("/api/infer", json={"device": "cpu"}).status_code == 200
    ov = client.get("/api/overview").json()
    assert ov["total"] == 2 and sum(ov["by_category"].values()) == 2
    r = client.get("/api/preview/1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/api/preview/9999").status_code == 404

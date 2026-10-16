"""C1 acceptance: route protection, OTC exchange, Host allowlist, CSRF Origin, lifecycle."""
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from photocure.auth import build_allowed_hosts, origin_allowed
from photocure.db.session import init_db, snapshot_database
from photocure.engine.locks import LEDGER_WRITING
from photocure.ui.app import create_app


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_OTC", "otc-for-test")
    return create_app(str(tmp_path / "t.db"))


def test_api_and_media_routes_need_auth(app):
    c = TestClient(app)
    assert c.get("/api/images").status_code == 401
    assert c.get("/api/thumbnail/1").status_code == 401
    assert c.get("/healthz").status_code == 200
    assert c.get("/").status_code == 200


def test_otc_exchange_sets_strict_httponly_cookie_once(app):
    c = TestClient(app)
    r = c.post("/auth/exchange", json={"code": "otc-for-test"})
    assert r.status_code == 204
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert c.get("/api/images").status_code == 200  # cookie now authenticates
    assert TestClient(app).post("/auth/exchange", json={"code": "otc-for-test"}).status_code == 401


def test_query_string_code_is_not_accepted(app):
    c = TestClient(app, follow_redirects=False)
    r = c.get("/?code=otc-for-test")
    assert "set-cookie" not in r.headers
    assert c.get("/api/images").status_code == 401


def test_unlisted_host_is_rejected(app):
    r = TestClient(app).get("/healthz", headers={"Host": "evil.com"})
    assert r.status_code == 400


def test_cross_origin_mutation_is_rejected(app):
    h = {"X-PCUR-Token": app.state.token}
    r = TestClient(app).post("/api/settings", json={"device": "cpu"},
                             headers={**h, "Origin": "http://evil.com"})
    assert r.status_code == 403
    r = TestClient(app).post("/api/settings", json={"device": "cpu"},
                             headers={**h, "Origin": "http://127.0.0.1.evil.com"})
    assert r.status_code == 403


def test_origin_matching_is_by_authority_not_prefix():
    hosts = build_allowed_hosts("127.0.0.1", 8080, "photos.lan")
    assert origin_allowed("http://127.0.0.1:8080", hosts)
    assert origin_allowed("http://photos.lan:8080/x", hosts)
    assert not origin_allowed("http://127.0.0.1.evil.com", hosts)
    assert not origin_allowed("null", hosts)


def test_restart_refused_while_ledger_writes_even_if_forced(app):
    c = TestClient(app, headers={"X-PCUR-Token": app.state.token})
    assert LEDGER_WRITING.acquire(blocking=False)
    try:
        r = c.post("/api/restart?force=true")
        assert r.status_code == 409 and r.json()["ok"] is False
    finally:
        LEDGER_WRITING.release()


def test_snapshot_is_consistent_during_concurrent_writes(tmp_path):
    db = tmp_path / "s.db"
    init_db(db)
    stop = threading.Event()

    def writer():
        con = sqlite3.connect(db, timeout=30)
        n = 0
        while not stop.is_set():
            con.execute("CREATE TABLE IF NOT EXISTS junk(x)")
            con.execute("INSERT INTO junk VALUES (?)", (n,))
            con.commit()
            n += 1
        con.close()

    t = threading.Thread(target=writer)
    t.start()
    try:
        bak = snapshot_database(db, tmp_path / "bk")
    finally:
        stop.set()
        t.join()
    con = sqlite3.connect(bak)
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_snapshot_rotation_keeps_newest(tmp_path):
    db = tmp_path / "r.db"
    init_db(db)
    bk = tmp_path / "bk"
    bk.mkdir()
    for i in range(12):
        (bk / f"r_2020010{i % 10}T0000{i:02d}Z.bak.db").write_bytes(b"x")
    snapshot_database(db, bk, keep=5)
    assert len(list(bk.glob("r_*.bak.db"))) == 5

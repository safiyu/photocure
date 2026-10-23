import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from photocure import setup_cmd
from photocure.auth import build_allowed_hosts, origin_allowed
from photocure.ui.app import create_app, validate_output_dir


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_OTC", "otc-x")
    return create_app(str(tmp_path / "t.db"))


def test_origin_requires_matching_port():
    hosts = build_allowed_hosts("127.0.0.1", 8080)
    assert origin_allowed("http://localhost:8080", hosts)
    assert not origin_allowed("http://localhost:3000", hosts)
    assert origin_allowed("http://localhost:3000", hosts, host="localhost:3000")
    assert not origin_allowed("http://localhost", hosts)


def test_other_localhost_port_rejected_and_missing_origin(app):
    h = {"X-PCUR-Token": app.state.token}
    c = TestClient(app)
    r = c.post("/api/settings", json={"device": "cpu"},
               headers={**h, "Origin": "http://localhost:3000"})
    assert r.status_code == 403
    c2 = TestClient(app, cookies={"pcur_session": app.state.token})
    r = c2.post("/api/settings", json={"device": "cpu"}, headers={"Sec-Fetch-Site": "same-site"})
    assert r.status_code == 403


def test_exchange_rate_limit_and_cookie_flags(app):
    c = TestClient(app)
    for _ in range(10):
        assert c.post("/auth/exchange", json={"code": "bad"}).status_code == 401
    assert c.post("/auth/exchange", json={"code": "bad"}).status_code == 429


def test_cookie_max_age_and_secure_on_https(app):
    r = TestClient(app, base_url="https://testserver").post("/auth/exchange", json={"code": "otc-x"})
    ck = r.headers["set-cookie"].lower()
    assert "max-age=" in ck and "secure" in ck


def test_sort_and_threshold_validated(app):
    c = TestClient(app, headers={"X-PCUR-Token": app.state.token})
    assert c.get("/api/people?sort=bogus").status_code == 422
    assert c.post("/api/people/recluster?threshold=5").status_code == 422


def test_output_dir_rejects_hidden_home_and_confinement(tmp_path, monkeypatch):
    dbp = tmp_path / "t.db"
    create_app(str(dbp))
    with pytest.raises(Exception):
        validate_output_dir(str(Path.home() / ".ssh" / "x"), dbp, False)
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setenv("PCUR_BROWSE_ROOTS", str(root))
    with pytest.raises(Exception):
        validate_output_dir(str(tmp_path / "other" / "o"), dbp, False)
    assert validate_output_dir(str(root / "out"), dbp, False) == (root / "out").resolve()


def test_scan_outside_browse_roots_rejected(app, tmp_path, monkeypatch):
    root = tmp_path / "allowed"
    root.mkdir()
    (tmp_path / "other").mkdir()
    monkeypatch.setenv("PCUR_BROWSE_ROOTS", str(root))
    c = TestClient(app, headers={"X-PCUR-Token": app.state.token})
    assert c.post("/api/scan", json={"path": str(tmp_path / "other")}).status_code == 403
    assert c.post("/api/watch", json={"enabled": True, "path": str(tmp_path / "other")}).status_code == 403


def test_pick_port_ipv6_and_busy():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert setup_cmd.pick_port("127.0.0.1", busy) != busy
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s6:
            s6.bind(("::1", 0))
            s6.listen()
            busy6 = s6.getsockname()[1]
            assert setup_cmd.pick_port("::1", busy6) != busy6
    except OSError:
        pytest.skip("no IPv6")

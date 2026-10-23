import hashlib
import time

import pytest
from fastapi.testclient import TestClient

from photocure.auth import AuthManager, build_allowed_hosts
from photocure.ui.app import _restart_args, create_app


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PCUR_OTC", "otc-for-test")
    return create_app(str(tmp_path / "t.db"))


def test_non_ascii_code_is_rejected_not_500(app):
    c = TestClient(app)
    r = c.post("/auth/exchange", json={"code": "café’"})
    assert r.status_code == 401
    r = c.get("/api/images", headers={"X-PCUR-Token": "café".encode("latin-1")})
    assert r.status_code == 401


def test_auth_manager_handles_unicode_and_still_matches(tmp_path):
    mgr = AuthManager(tmp_path / "cache", token="x" * 40)
    assert mgr.exchange("é") is None
    assert not mgr.validate("’" * 40)
    assert mgr.validate("x" * 40)
    assert mgr.exchange("x" * 40) == "x" * 40


def test_wildcard_bind_allows_machine_names(monkeypatch):
    monkeypatch.setattr("socket.gethostname", lambda: "box.lan")
    monkeypatch.setattr("socket.gethostbyname_ex", lambda h: (h, [], ["192.168.1.20", "127.0.1.1"]))
    hosts = build_allowed_hosts("0.0.0.0", 8080)
    assert {"box.lan:8080", "192.168.1.20:8080", "192.168.1.20"} <= hosts
    assert "127.0.1.1" not in hosts
    assert "evil.com" not in hosts


def test_specific_bind_does_not_add_machine_names(monkeypatch):
    monkeypatch.setattr("socket.gethostname", lambda: "box.lan")
    assert "box.lan" not in build_allowed_hosts("127.0.0.1", 8080)


def test_restart_args_force_no_open():
    assert _restart_args(["py", "photocure", "serve", "--open"]) == ["photocure", "serve", "--no-open"]
    assert _restart_args(["py", "photocure", "serve"]) == ["photocure", "serve", "--no-open"]
    assert _restart_args(["py", "photocure", "serve", "--no-open"]).count("--no-open") == 1
    assert _restart_args(["py", "-m", "uvicorn", "x:app"]) == ["-m", "uvicorn", "x:app"]


def test_face_route_unicode_digit_is_404(app):
    c = TestClient(app, headers={"X-PCUR-Token": app.state.token})
    assert c.get("/api/faces/%C2%B2.jpg").status_code == 404
    assert c.get("/api/faces/abc.jpg").status_code == 404


def test_reset_removes_webp_thumbnail_pyramid(app, tmp_path):
    c = TestClient(app, headers={"X-PCUR-Token": app.state.token})
    db_fp = hashlib.sha1(str((tmp_path / "t.db").resolve()).encode()).hexdigest()[:8]
    thumbs = app.state.cfg.cache_dir / "thumbs"
    mine = [thumbs / t / f"{db_fp}_1_abc.webp" for t in ("micro", "medium", "preview")]
    other = thumbs / "micro" / "deadbeef_1_abc.webp"
    for p in [*mine, other]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    assert c.post("/api/reset", headers={"Origin": "http://testserver"}).status_code == 200
    assert not any(p.exists() for p in mine)
    assert other.exists()


def _wait(c, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] not in ("running", "stopping"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_learn_job_is_done_even_if_stop_pressed(app, monkeypatch):
    from photocure.services import Planner

    def slow_learn(self):
        time.sleep(0.3)
        return {"updated": 2, "moved_out_of_uncertain": 1}

    monkeypatch.setattr(Planner, "learn_from_corrections", slow_learn, raising=False)
    c = TestClient(app, headers={"X-PCUR-Token": app.state.token, "Origin": "http://testserver"})
    job_id = c.post("/api/learn").json()["job_id"]
    c.post("/api/stop")
    j = _wait(c, job_id)
    assert j["status"] == "done"
    assert j["result"]["updated"] == 2

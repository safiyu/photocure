"""Plans write into the one library root; only move and copy exist."""
from __future__ import annotations

import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from photocure.cli import cli
from photocure.engine.action_engine import ActionEngine
from photocure.rules.organizer import PlannedAction
from photocure.rules.taxonomy import CategoryType
from photocure.ui.app import create_app

from .conftest import make_image
from .test_ui_api import env, wait_job  # noqa: F401  (shared fixture)


@pytest.fixture()
def env_no_root(tmp_path):
    app = create_app(str(tmp_path / "ui.db"))
    app.state.cfg.cache_dir = tmp_path / "cache"
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    lib = tmp_path / "photos"
    make_image(lib / "a.jpg", rng_seed=1)
    assert client.post("/api/scan", json={"path": str(lib)}).status_code == 200
    assert client.post("/api/infer", json={"device": "cpu"}).status_code == 200
    return types.SimpleNamespace(client=client, lib=lib, tmp=tmp_path, db=tmp_path / "ui.db")


@pytest.mark.parametrize("path", ["/api/plan/rebuild", "/api/plan/diff", "/api/plan/apply"])
def test_plan_endpoints_409_without_root(env_no_root, path):
    r = env_no_root.client.post(path, json={"operation": "move"})
    assert r.status_code == 409
    assert r.json()["detail"] == "library folder not set"


def test_readiness_reports_missing_root(env_no_root):
    j = env_no_root.client.get("/api/plan/readiness").json()
    assert j["library_root_set"] is False
    assert "library folder not set" in j["issues"]
    assert j["ready"] is False


def test_plan_apply_moves_into_library_root(env):
    r = env.client.post("/api/plan/apply", json={"operation": "move", "trash": False})
    assert r.status_code == 202
    j = wait_job(env.client, r.json()["job_id"])
    assert j["result"]["executed"] > 0
    moved = list(env.out.rglob("*.jpg"))
    assert moved and all(Path(p).is_relative_to(env.out) for p in moved)


def test_diff_reports_the_library_root(env):
    d = env.client.post("/api/plan/diff", json={"operation": "copy"}).json()
    assert d["organize_root"] == str(env.out.resolve())
    assert d["mode"] == "category_first"


def test_hardlink_and_symlink_rejected_by_api(env):
    for op in ("hardlink", "symlink"):
        assert env.client.post("/api/plan/diff", json={"operation": op}).status_code == 422


def test_settings_endpoint_validates_the_library_folder(tmp_path):
    app = create_app(str(tmp_path / "ui.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    for bad in ("relative", "/etc/photos", "/"):
        assert client.post("/api/settings", json={"library_root": bad}).status_code == 400


def _cli_env(tmp_path):
    runner = CliRunner()
    db = tmp_path / "c.db"
    lib = tmp_path / "photos"
    make_image(lib / "a.jpg", rng_seed=1)
    assert runner.invoke(cli, ["scan", str(lib), "--db", str(db)]).exit_code == 0
    assert runner.invoke(cli, ["infer", "--db", str(db), "--device", "cpu", "-q"]).exit_code == 0
    return runner, db


def test_cli_plan_uses_setting(tmp_path):
    from photocure.services import save_settings
    runner, db = _cli_env(tmp_path)
    save_settings(db, {"library_root": str(tmp_path / "Library")})
    r = runner.invoke(cli, ["plan", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert "Plan ID" in r.output


def test_cli_plan_without_root_exits_2(tmp_path):
    runner, db = _cli_env(tmp_path)
    r = runner.invoke(cli, ["plan", "--db", str(db)])
    assert r.exit_code == 2
    assert "library folder" in r.output.lower()


def test_cli_rejects_symlink_operation(tmp_path):
    runner, db = _cli_env(tmp_path)
    r = runner.invoke(cli, ["plan", "--db", str(db), "--library-root", str(tmp_path / "L"),
                            "--operation", "symlink"])
    assert r.exit_code == 2


def test_engine_rejects_hardlink(cfg, tmp_path):
    from photocure.db.session import init_db
    init_db(cfg.db_path)
    src = make_image(tmp_path / "s" / "a.jpg", rng_seed=3)
    act = PlannedAction(image_id=0, source=src, target=tmp_path / "t" / "a.jpg",
                        category=CategoryType.PEOPLE, operation="hardlink")
    rep = ActionEngine(cfg, db_path=cfg.db_path).execute([act], organize_root=tmp_path / "t")
    assert rep.failures == 1
    assert "unsupported operation" in rep.errors[0]
    assert not (tmp_path / "t" / "a.jpg").exists()

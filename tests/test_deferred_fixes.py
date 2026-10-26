"""The review items that were first left for later: engine races, stage leaks, reset, sync, docs and small UI fixes."""
from __future__ import annotations

import os
import re
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select
from typer.testing import CliRunner

from photocure import trash_actions
from photocure import library_actions, library_migration
from photocure.cli import app as cli_app
from photocure.db import models as M
from photocure.db.repository import PhotoRepository
from photocure.db.session import get_session
from photocure.engine.action_engine import ActionEngine
from photocure.rules.decision_matrix import Decision
from photocure.rules.organizer import FolderOrganizer
from photocure.rules.taxonomy import CategoryType, parse_taxonomy
from photocure.scanner.file_scanner import FileScanner
from photocure.ui.app import create_app

from .conftest import make_image
from .test_ui_api import env, wait_job  # noqa: F401  (shared fixture)

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "photocure" / "ui" / "static"
JS = (STATIC / "app.js").read_text()
HTML = (STATIC / "index.html").read_text()
CSS = (STATIC / "style.css").read_text()
runner = CliRunner()


@pytest.fixture()
def lib(env):
    r = env.client.post("/api/plan/apply", json={"operation": "move", "trash": False})
    assert wait_job(env.client, r.json()["job_id"])["status"] == "done"
    env.ids = sorted(i["id"] for i in env.client.get("/api/images?stage=library&limit=50").json()["items"])
    return env


def _row(env, iid):
    with get_session(env.db) as s:
        r = s.get(M.ImageRecord, iid)
        return {c: getattr(r, c) for c in ("path", "stage", "scan_status", "route_status")}


# ------------------------------------------------------------------ reset
def test_reset_also_clears_a_library_folder_chosen_in_settings(env):
    r = env.client.post("/api/reset")
    assert r.status_code == 200 and r.json()["library_root_cleared"] is True
    s = env.client.get("/api/settings").json()
    assert s["library_root"] is None and s["library_root_source"] == "none"


def test_reset_keeps_a_library_folder_that_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_LIBRARY_ROOT", str(tmp_path / "EnvLib"))
    app = create_app(str(tmp_path / "ui.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    r = client.post("/api/reset")
    assert r.status_code == 200 and r.json()["library_root_cleared"] is False
    assert client.get("/api/settings").json()["library_root"] == str(tmp_path / "EnvLib")


def test_the_ui_asks_for_a_library_folder_again_after_a_reset():
    handler = JS.split('await post("/api/reset")', 1)[1].split("\n    } catch", 1)[0]
    assert "refreshSettings(" in handler and "maybeFirstRun()" in handler


# ------------------------------------------------------------------ sync
def test_sync_brings_back_a_library_photo_whose_file_reappears(lib):
    iid = lib.ids[0]
    path = Path(_row(lib, iid)["path"])
    keep = path.read_bytes()
    path.unlink()
    assert iid in lib.client.post("/api/library/sync").json()["missing_ids"]
    assert _row(lib, iid)["scan_status"] == "missing"
    path.write_bytes(keep)
    j = lib.client.post("/api/library/sync").json()
    assert j["revived"] == 1
    row = _row(lib, iid)
    assert (row["stage"], row["scan_status"], row["route_status"]) == ("library", "scanned", "executed")


def test_sync_reads_the_disk_before_it_opens_a_write_transaction():
    body = (ROOT / "photocure" / "ui" / "app.py").read_text().split("def _sync_library", 1)[1].split(
        '    @app.post("/api/plan/apply"', 1)[0]
    assert body.index("os.walk") < body.index("compute_sha256(cand)") < body.index("# Write phase")
    assert body.count("get_session(dbp)") == 2  # one read, then one short write


# ------------------------------------------------------------------ stats and listings do not write
def test_stats_are_read_only_and_people_count_leaves_out_trash_people(session, repo):
    imgs = [M.ImageRecord(path=f"/p/{i}.jpg", filename=f"{i}.jpg", infer_status="inferred") for i in range(3)]
    session.add_all(imgs)
    session.flush()
    bg = M.BurstGroup(name="Burst_x", image_count=3, best_shot_id=imgs[0].id)
    session.add(bg)
    session.flush()
    for r in imgs:
        r.burst_id = bg.id
    imgs[1].scan_status = imgs[2].scan_status = "missing"  # two frames are gone, leaving a lone frame
    session.add_all([M.PersonCluster(name="A", face_count=1, stage="library"),
                     M.PersonCluster(name="B", face_count=1, stage="trash")])
    session.flush()
    st = repo.stats()
    assert st["burst_groups"] == 0 and st["people"] == 1
    assert imgs[1].burst_id == bg.id and imgs[0].burst_id == bg.id  # nothing was rewritten


def test_the_burst_list_and_overview_do_not_rewrite_rows(env):
    ids = sorted(i["id"] for i in env.client.get("/api/images?limit=50").json()["items"])
    with get_session(env.db) as s:
        bg = M.BurstGroup(name="Burst_y", image_count=2, best_shot_id=ids[0])
        s.add(bg)
        s.flush()
        for iid in ids[:2]:
            s.get(M.ImageRecord, iid).burst_id = bg.id
        s.get(M.ImageRecord, ids[1]).scan_status = "missing"
        s.commit()
    assert env.client.get("/api/bursts").json()["items"] == []
    env.client.get("/api/overview")
    env.client.get("/api/stats")
    with get_session(env.db) as s:
        assert s.get(M.ImageRecord, ids[1]).burst_id is not None


# ------------------------------------------------------------------ engine
def test_a_move_never_overwrites_a_file_that_appeared_after_the_check(tmp_path):
    src = make_image(tmp_path / "s" / "a.jpg", rng_seed=1)
    target = tmp_path / "t" / "a.jpg"
    target.parent.mkdir()
    target.write_bytes(b"someone else's file")
    with pytest.raises(FileExistsError):
        ActionEngine._move(src, target, "")
    assert target.read_bytes() == b"someone else's file" and src.exists()


def test_restore_does_not_adopt_an_identical_file_that_already_sits_at_the_target(lib):
    iid = lib.ids[0]
    before = Path(_row(lib, iid)["path"])
    library_actions.trash_image(lib.db, lib.app.state.cfg, iid)
    trashed = Path(_row(lib, iid)["path"])
    before.parent.mkdir(parents=True, exist_ok=True)
    before.write_bytes(trashed.read_bytes())  # a backup copy with the same content
    out = trash_actions.restore(lib.db, lib.app.state.cfg, [iid])
    assert out["restored"] == 1
    row = _row(lib, iid)
    assert Path(row["path"]) != before and Path(row["path"]).exists()
    assert before.exists() and not trashed.exists()  # the backup stays, the trashed file moved


def test_migration_rewrites_paths_when_the_old_root_is_given_through_a_symlink(env, tmp_path):
    from photocure.services import save_settings
    real = tmp_path / "real_lib"
    real.mkdir()
    link = tmp_path / "link_lib"
    link.symlink_to(real)
    save_settings(env.db, {"library_root": None}, force=True)
    env.app.state.cfg.library_root = link  # a root from the environment is used exactly as given
    r = env.client.post("/api/plan/apply", json={"operation": "copy", "trash": False})
    wait_job(env.client, r.json()["job_id"])
    with get_session(env.db) as s:
        assert all(str(link) in x.path for x in s.exec(select(M.ImageRecord).where(M.ImageRecord.stage == "library")).all())
    out = library_migration.migrate_library(env.db, env.app.state.cfg, tmp_path / "moved")
    assert out["ok"]
    with get_session(env.db) as s:
        rows = s.exec(select(M.ImageRecord).where(M.ImageRecord.stage == "library")).all()
        assert rows and all(Path(r.path).is_relative_to((tmp_path / "moved").resolve()) for r in rows)


# ------------------------------------------------------------------ stage-scoped listings
def test_facets_and_highlights_can_be_limited_to_a_stage(lib):
    with get_session(lib.db) as s:
        s.get(M.ImageRecord, lib.ids[0]).stage = "new"
        s.commit()
    assert lib.client.get("/api/facets?stage=library").json()["total"] == 3
    assert lib.client.get("/api/facets?stage=new").json()["total"] == 1
    assert lib.client.get("/api/facets").json()["total"] == 4
    assert lib.client.get("/api/highlights/events?stage=library").status_code == 200
    assert 'stage=${encodeURIComponent(state.stage)}' in JS.split("async function loadFacets", 1)[1].split("\n  }\n", 1)[0]


# ------------------------------------------------------------------ rules and taxonomy
def test_the_library_layout_is_flat_even_for_a_class_that_asks_for_date_folders(cfg, tmp_path):
    org = FolderOrganizer(cfg, organize_root=tmp_path / "L")
    rec = M.ImageRecord(path="/in/a.jpg", filename="a.jpg", taken_at=datetime(2024, 5, 12, 9, tzinfo=timezone.utc))
    dec = Decision(CategoryType.CUSTOM, "x", needs_date=True, folder="Trips")
    assert org.resolve_target(rec, dec) == tmp_path / "L" / "Trips"
    tax = parse_taxonomy([{"name": "trips", "prompts": ["a trip"], "folder": "Trips", "date_grouped": True}])
    assert tax.category_of("trips") is not None  # old files that still say date_grouped keep loading


@pytest.mark.parametrize("bad", ["a,b", "a/b", "a\\b", "x" * 80, "line\nbreak"])
def test_class_names_that_would_break_sidecars_or_paths_are_refused(bad):
    data = [{"name": bad, "prompts": ["a photo"], "folder": "Custom"}]
    with pytest.raises(ValueError, match="name"):
        parse_taxonomy(data)


def test_ordinary_class_names_still_work():
    tax = parse_taxonomy([{"name": "boat trips", "prompts": ["a boat"], "folder": "Boats"}])
    assert tax.category_of("boat trips") is not None


def test_watch_events_inside_the_library_are_ignored(cfg, tmp_db, tmp_path):
    root = (tmp_path / "Photos").resolve()
    lib_dir = root / "Library"
    make_image(root / "x.jpg", rng_seed=1)
    inside = make_image(lib_dir / "People" / "y.jpg", rng_seed=2)
    cfg.library_root = lib_dir
    res = FileScanner(cfg).scan_paths(root, [root / "x.jpg", inside], db_path=tmp_db)
    assert res.new == 1


# ------------------------------------------------------------------ people
def test_a_manually_assigned_face_without_an_embedding_still_counts_down_on_reanalysis(cfg, session):
    import types
    from photocure.inference.pipeline import InferencePipeline
    repo = PhotoRepository(session)
    img = M.ImageRecord(path="/p/x.jpg", filename="x.jpg", infer_status="inferred")
    session.add(img)
    session.flush()
    person = repo.create_person_cluster([1.0, 0.0])
    session.add(M.FaceDetection(image_id=img.id, person_id=person.id, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1,
                                confidence=0.9, x1=0.1, y1=0.1, x2=0.2, y2=0.2))  # no embedding
    session.flush()
    InferencePipeline(cfg)._assign_people(repo, {}, img.id, [])
    gone = session.get(M.PersonCluster, person.id)
    assert gone is None or gone.face_count == 0  # an empty unnamed person is dropped


def test_naming_a_trash_face_makes_a_trash_person_and_never_joins_a_library_one(session, tmp_db):
    from photocure.people_actions import reassign_face
    lib_img = M.ImageRecord(path="/p/a.jpg", filename="a.jpg", stage="library")
    trash_img = M.ImageRecord(path="/p/b.jpg", filename="b.jpg", stage="trash", scan_status="missing")
    session.add_all([lib_img, trash_img])
    session.flush()
    session.add(M.PersonCluster(name="Sam", face_count=1, stage="library"))
    face = M.FaceDetection(image_id=trash_img.id, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1, confidence=0.9,
                           x1=0.1, y1=0.1, x2=0.2, y2=0.2)
    session.add(face)
    session.commit()
    out = reassign_face(tmp_db, face.id, new_name="Sam")
    with get_session(tmp_db) as s:
        person = s.get(M.PersonCluster, out["person_id"])
        assert person.stage == "trash"
        assert s.query(M.PersonCluster).filter_by(name="Sam").count() == 2


# ------------------------------------------------------------------ API behaviour
def test_undo_with_nothing_to_undo_is_an_error_not_a_quiet_success(env):
    assert env.client.post("/api/undo", json={}).status_code == 404


def test_learn_without_a_library_folder_is_refused_before_it_changes_anything(tmp_path):
    app = create_app(str(tmp_path / "ui.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    assert client.post("/api/learn").status_code == 409


def test_the_about_endpoint_needs_the_token(env, tmp_path):
    anon = TestClient(env.app)
    assert anon.get("/api/about").status_code in (401, 403)
    assert env.client.get("/api/about").status_code == 200


# ------------------------------------------------------------------ CLI and packaging
def test_apply_warns_about_every_option_that_a_saved_plan_ignores(tmp_path):
    db = tmp_path / "c.db"
    lib = tmp_path / "photos"
    make_image(lib / "a.jpg", rng_seed=1)
    assert runner.invoke(cli_app, ["scan", str(lib), "--db", str(db)]).exit_code == 0
    assert runner.invoke(cli_app, ["infer", "--db", str(db), "--device", "cpu", "-q"]).exit_code == 0
    out = runner.invoke(cli_app, ["plan", "--db", str(db), "-o", str(tmp_path / "L")]).output
    pid = re.search(r"PLAN_\w+", out).group(0)
    r = runner.invoke(cli_app, ["apply", "--db", str(db), "--plan-id", pid, "-o", str(tmp_path / "L"),
                                "--window-seconds", "5", "--similarity-threshold", "0.9", "-y"])
    text = r.output.replace("\n", " ")
    for flag in ("--library-root", "--window-seconds", "--similarity-threshold"):
        assert flag in text, flag


def test_the_binary_answers_dash_dash_version():
    r = runner.invoke(cli_app, ["--version"])
    assert r.exit_code == 0 and "photocure" in r.output


def test_requirements_txt_lists_every_core_dependency():
    core = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    names = lambda lines: {re.split(r"[<>=\[ ]", x.strip())[0].lower().replace("_", "-")  # noqa: E731
                           for x in lines if x.strip() and not x.strip().startswith("#")}
    assert names(core) <= names((ROOT / "requirements.txt").read_text().splitlines())


def test_the_stop_alias_test_checks_behaviour_not_a_constant():
    src = (ROOT / "tests" / "test_stop_c1.py").read_text()
    assert "assert os.name" not in src


# ------------------------------------------------------------------ UI details
def test_the_stage_switch_is_a_button_group_with_pressed_states():
    sw = re.search(r'<div class="top-mode-switch" id="stage-switch".*?</div>', HTML, re.S).group(0)
    assert 'role="group"' in sw and 'role="tablist"' not in sw and 'role="tab"' not in sw
    assert "aria-pressed" in sw and "aria-pressed" in JS.split("function setStage", 1)[1].split("\n    }\n", 1)[0]


def test_banners_never_sit_above_dialogs_and_the_hidden_version_badge_cannot_take_focus():
    banners = int(re.search(r"\.banners\s*\{[^}]*z-index:\s*(\d+)", CSS).group(1))
    modal = int(re.search(r"\n\.modal\s*\{[^}]*z-index:\s*(\d+)", CSS).group(1))
    assert modal > banners
    badge = re.search(r'<button[^>]*id="header-version-badge"[^>]*>', HTML).group(0)
    assert 'tabindex="-1"' in badge and 'aria-hidden="true"' in badge


def test_dead_code_from_removed_features_is_gone():
    for dead in ("async function showFaces", "triageSel", "scopeParam"):
        assert dead not in JS, dead


def test_the_folder_picker_says_what_the_folder_is_for():
    assert "opts.info" in JS and JS.count("info:") >= 2  # the clean export and move dialogs set their own text


def test_stale_answers_cannot_overwrite_the_stats_or_the_people_list():
    assert "Seq" in JS.split("async function loadStats", 1)[1].split("\n  }\n", 1)[0]
    assert "Seq" in JS.split("async function renderPeople", 1)[1].split("\n  }\n", 1)[0]

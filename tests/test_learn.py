"""learn command, Planner.learn_from_corrections and backend_info, repo helpers, migration."""
from __future__ import annotations

import sqlite3

import pytest
from typer.testing import CliRunner

from photocure.cli import app
from photocure.db.models import ImageRecord, SemanticScore
from photocure.db.repository import PhotoRepository
from photocure.db.session import SCHEMA_VERSION, close_engine, get_session, init_db
from photocure.inference import semantic_engine as se

from .world import build_world, set_bucket

runner = CliRunner()


def flat(out: str) -> str:
    for ch in "│┃┏┓┗┛┡┩╇━─╭╮╰╯┳┻╋┠┨":
        out = out.replace(ch, " ")
    return " ".join(out.split())


def make_uncertain(world):
    with get_session(world.db) as s:
        for r in PhotoRepository(s).all_images():
            r.category = "uncertain"


def confident_food(image_id, backend="clip"):
    return SemanticScore(image_id=image_id, top1_category="food", top1_score=0.9,
                         top2_category="scenery", top2_score=0.05, score_margin=0.85,
                         backend=backend)


@pytest.fixture()
def world(tmp_path):
    w = build_world(tmp_path, n=4)
    w.cfg.organize_root = w.out
    make_uncertain(w)
    return w


@pytest.fixture()
def fake_learn(monkeypatch, world):
    """Pretend the learner made two photos confident."""
    def fn(session, cfg=None, backend=None):
        repo = PhotoRepository(session)
        for i in world.ids[:2]:
            repo.upsert_semantic(confident_food(i))
        return {"updated": 2, "classes_used": 1, "examples": 2}

    monkeypatch.setattr(se, "learn_from_corrections", fn, raising=False)
    return fn


# ----------------------------------------------------------------- planner
def test_learn_from_corrections_moves_photos_out_of_uncertain(world, fake_learn):
    out = world.planner.learn_from_corrections()
    assert out["updated"] == 2 and out["classes_used"] == 1 and out["examples"] == 2
    assert out["uncertain_before"] == 4
    assert out["uncertain_after"] == 2
    assert out["moved_out_of_uncertain"] == 2
    assert out["plan_id"]
    with get_session(world.db) as s:
        cats = {r.id: r.category for r in PhotoRepository(s).all_images()}
    assert [cats[i] for i in world.ids[:2]] == ["food", "food"]
    assert [cats[i] for i in world.ids[2:]] == ["uncertain", "uncertain"]


def test_learn_without_rebuild_leaves_routing_alone(world, fake_learn):
    out = world.planner.learn_from_corrections(rebuild=False)
    assert out["plan_id"] is None
    assert out["moved_out_of_uncertain"] == 0 and out["uncertain_after"] == 4


def test_learn_never_overrides_manual_choices(world, fake_learn):
    set_bucket(world, world.ids[0], "document")
    world.planner.learn_from_corrections()
    with get_session(world.db) as s:
        repo = PhotoRepository(s)
        assert repo.count_category("document") == 1
        assert repo.get_by_id(world.ids[0]).manual_bucket == "document"


def test_learn_reports_when_unavailable(world, monkeypatch):
    monkeypatch.delattr(se, "learn_from_corrections", raising=False)
    with pytest.raises(RuntimeError, match="not available"):
        world.planner.learn_from_corrections()


def test_backend_info_counts_placeholder_rows(world):
    with get_session(world.db) as s:
        repo = PhotoRepository(s)
        repo.upsert_semantic(SemanticScore(image_id=world.ids[0], backend=""))
        repo.upsert_semantic(SemanticScore(image_id=world.ids[1], backend="stub"))
        repo.upsert_semantic(SemanticScore(image_id=world.ids[2], backend="clip"))
    info = world.planner.backend_info()
    assert info["stub_rows"] == 2
    for key in ("device", "semantic_backend", "face_backend", "semantic_is_stub",
                "face_is_fallback", "hint"):
        assert key in info


def test_backend_info_survives_a_broken_backend(world, monkeypatch):
    import photocure.inference.backends as bk

    def boom(cfg=None):
        raise RuntimeError("torch exploded")

    monkeypatch.setattr(bk, "detect_backend", boom)
    info = world.planner.backend_info()
    assert info["semantic_is_stub"] is True and "torch exploded" in info["hint"]


# -------------------------------------------------------------------- repo
def test_repo_helpers(world):
    with get_session(world.db) as s:
        repo = PhotoRepository(s)
        repo.upsert_semantic(SemanticScore(image_id=world.ids[0], backend="stub",
                                           embedding="[1,2,3]"))
        repo.upsert_semantic(SemanticScore(image_id=world.ids[1], backend="clip",
                                           embedding=""))
        repo.upsert_semantic(SemanticScore(image_id=world.ids[2], backend="clip",
                                           embedding="[1,2,3]"))
    set_bucket(world, world.ids[0], "food")     # corrected, has embedding
    set_bucket(world, world.ids[1], "food")     # corrected, no embedding
    with get_session(world.db) as s:
        repo = PhotoRepository(s)
        assert repo.count_stub_rows() == 1
        assert repo.count_labeled_examples() == 1
        assert repo.count_category("food") == 2         # manual override wins
        assert repo.count_category("uncertain") == 2    # ids[2], ids[3]
        assert repo.count_category("nothing") == 0


def test_backend_column_defaults_and_updates(world):
    assert SemanticScore(image_id=1).backend == ""
    with get_session(world.db) as s:
        repo = PhotoRepository(s)
        repo.upsert_semantic(SemanticScore(image_id=world.ids[0], backend="stub"))
        repo.upsert_semantic(SemanticScore(image_id=world.ids[0], backend="clip"))
    with get_session(world.db) as s:
        row = PhotoRepository(s).s.exec(
            __import__("sqlmodel").select(SemanticScore)).all()
        assert [r.backend for r in row] == ["clip"]


# --------------------------------------------------------------- migration
def test_schema_version_is_at_least_three():
    assert SCHEMA_VERSION >= 3


def test_migration_adds_semantic_backend_column(tmp_path):
    db = tmp_path / "old.db"
    init_db(db, drop=True)
    with get_session(db) as s:
        img = ImageRecord(path="/x/a.jpg", filename="a.jpg", extension=".jpg",
                          file_size=1, relative_path="a.jpg")
        s.add(img)
        s.flush()
        s.add(SemanticScore(image_id=img.id, top1_category="food", backend="clip"))
    close_engine(db)
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE semantic_scores DROP COLUMN backend")
    con.execute("PRAGMA user_version=2")
    con.commit()
    assert "backend" not in {r[1] for r in con.execute("PRAGMA table_info(semantic_scores)")}
    con.close()

    init_db(db)
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(semantic_scores)")}
    ver = con.execute("PRAGMA user_version").fetchone()[0]
    stored = con.execute("SELECT backend FROM semantic_scores").fetchall()
    con.close()
    assert "backend" in cols and ver == SCHEMA_VERSION
    assert stored == [("",)]          # old rows read as unknown
    with get_session(db) as s:        # ORM works against the migrated table
        assert PhotoRepository(s).count_stub_rows() == 1


# --------------------------------------------------------------------- cli
def test_learn_command_reports_a_summary(world, fake_learn):
    r = runner.invoke(app, ["learn", "--db", str(world.db)])
    out = flat(r.stdout)
    assert r.exit_code == 0, r.stdout
    assert "Learned from your corrections" in out
    assert "Corrected photos used 2" in out
    assert "Uncertain before 4" in out and "Uncertain after 2" in out
    assert "Moved out of Uncertain 2" in out


def test_learn_command_without_corrections_explains_what_to_do(world, monkeypatch):
    monkeypatch.setattr(se, "learn_from_corrections",
                        lambda session, cfg=None, backend=None: {"examples": 0},
                        raising=False)
    r = runner.invoke(app, ["learn", "--db", str(world.db)])
    assert r.exit_code == 0
    assert "No corrections found yet" in flat(r.stdout)


def test_learn_command_no_rebuild_flag(world, fake_learn):
    r = runner.invoke(app, ["learn", "--db", str(world.db), "--no-rebuild"])
    assert r.exit_code == 0
    assert "Uncertain after 4" in flat(r.stdout)


def test_learn_command_fails_clearly_when_unavailable(world, monkeypatch):
    monkeypatch.delattr(se, "learn_from_corrections", raising=False)
    r = runner.invoke(app, ["learn", "--db", str(world.db)])
    assert r.exit_code == 1
    assert "not available" in flat(r.stdout)

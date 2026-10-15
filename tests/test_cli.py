"""End-to-end CLI tests using Typer's CliRunner."""
from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner

from photocure.cli import app
from .conftest import make_image

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "photocure" in result.stdout


def test_cli_stats_empty(tmp_path):
    db_file = tmp_path / "test.db"
    result = runner.invoke(app, ["stats", "--db", str(db_file)])
    assert result.exit_code == 0
    assert "Total Images" in result.stdout


def test_cli_full_workflow(tmp_path):
    src_dir = tmp_path / "photos"
    src_dir.mkdir()
    out_dir = tmp_path / "organized"
    db_file = tmp_path / "curator.db"

    img1 = src_dir / "2024-05-12_photo1.jpg"
    img2 = src_dir / "2024-05-12_photo2.jpg"
    make_image(img1, rng_seed=1)
    make_image(img2, rng_seed=2)

    # 1. scan
    r_scan = runner.invoke(app, ["scan", str(src_dir), "--db", str(db_file)])
    assert r_scan.exit_code == 0
    assert "Files seen" in r_scan.stdout

    # 2. infer
    r_infer = runner.invoke(app, ["infer", "--db", str(db_file), "--device", "cpu"])
    assert r_infer.exit_code == 0
    assert "Processed" in r_infer.stdout

    # 3. cluster
    r_cluster = runner.invoke(app, ["cluster", "--db", str(db_file)])
    assert r_cluster.exit_code == 0

    # 4. plan
    report_file = tmp_path / "plan.html"
    r_plan = runner.invoke(app, [
        "plan", "--db", str(db_file), "-o", str(out_dir),
        "--operation", "copy", "--report", str(report_file)
    ])
    assert r_plan.exit_code == 0
    assert report_file.exists()

    # 5. apply
    r_apply = runner.invoke(app, [
        "apply", "--db", str(db_file), "-o", str(out_dir),
        "--operation", "copy", "-y"
    ])
    assert r_apply.exit_code == 0
    assert "Transaction" in r_apply.stdout

    # 6. undo
    r_undo = runner.invoke(app, ["undo", "--db", str(db_file)])
    assert r_undo.exit_code == 0
    assert "Rollback" in r_undo.stdout


# ---------------------------------------------------------------------------
# persisted plans, safety flags and the new commands
# ---------------------------------------------------------------------------
def _world_cli(tmp_path, n=3):
    from .world import build_world
    return build_world(tmp_path, n=n)


def _plan_id_from(output: str) -> str:
    import re
    return re.search(r"Plan ID: (\S+)", output).group(1)


def test_cli_apply_plan_id_applies_reviewed_plan_and_refuses_drift(tmp_path):
    from .world import set_bucket
    w = _world_cli(tmp_path)
    args = ["--db", str(w.db), "-o", str(w.out), "--operation", "copy"]
    pid = _plan_id_from(runner.invoke(app, ["plan", *args]).stdout)
    set_bucket(w, w.ids[0], "food")
    r = runner.invoke(app, ["apply", "--db", str(w.db), "--plan-id", pid, "-y"])
    assert r.exit_code == 1 and "stale" in r.stdout
    assert not w.out.exists()
    r = runner.invoke(app, ["apply", "--db", str(w.db), "--plan-id", pid, "-y", "--force"])
    assert r.exit_code == 0 and "Transaction" in r.stdout
    r = runner.invoke(app, ["apply", "--db", str(w.db), "--plan-id", pid, "-y"])
    assert r.exit_code == 1 and "already applied" in r.stdout


def test_cli_apply_unknown_plan_id(tmp_path):
    w = _world_cli(tmp_path, n=1)
    r = runner.invoke(app, ["apply", "--db", str(w.db), "--plan-id", "nope", "-y"])
    assert r.exit_code == 1 and "Unknown plan id" in r.stdout


def test_cli_apply_in_place_keeps_originals(tmp_path):
    w = _world_cli(tmp_path)
    r = runner.invoke(app, ["apply", "--db", str(w.db), "-o", str(w.out),
                            "--mode", "in_place", "-y"])
    assert r.exit_code == 0
    assert all(p.exists() for p in w.paths)


def test_cli_undo_no_verify_is_passed_through(tmp_path, monkeypatch):
    from photocure.services import Planner
    w = _world_cli(tmp_path, n=1)
    seen = {}
    real = Planner.undo

    def spy(self, transaction_id=None, verify=True):
        seen["verify"] = verify
        return real(self, transaction_id, verify=verify)

    monkeypatch.setattr(Planner, "undo", spy)
    runner.invoke(app, ["apply", "--db", str(w.db), "-o", str(w.out),
                        "--operation", "copy", "-y"])
    runner.invoke(app, ["undo", "--db", str(w.db), "--no-verify"])
    assert seen["verify"] is False
    runner.invoke(app, ["undo", "--db", str(w.db)])
    assert seen["verify"] is True


def test_cli_cluster_thresholds_default_to_env_config(tmp_path, monkeypatch):
    from photocure.services import Planner
    w = _world_cli(tmp_path, n=1)
    monkeypatch.setenv("PCUR_BURST_SIMILARITY_THRESHOLD", "0.93")
    seen = {}
    monkeypatch.setattr(Planner, "cluster",
                        lambda self, win=None, sim=None: seen.update(win=win, sim=sim) or 0)
    r = runner.invoke(app, ["cluster", "--db", str(w.db)])
    assert r.exit_code == 0 and seen == {"win": None, "sim": None}
    assert "sim>=0.93" in r.stdout


def test_cli_trash_verify_repair_and_purge(tmp_path):
    from .world import set_bucket
    w = _world_cli(tmp_path, n=2)
    set_bucket(w, w.ids[0], "screenshot")
    r = runner.invoke(app, ["apply", "--db", str(w.db), "-o", str(w.out),
                            "--trash", "-y"])
    assert r.exit_code == 0 and (w.out / ".trash").is_dir()
    assert runner.invoke(app, ["verify-txn", "--db", str(w.db)]).exit_code == 0
    assert "Repaired 0" in runner.invoke(app, ["repair", "--db", str(w.db)]).stdout
    kept = runner.invoke(app, ["trash-purge", "--db", str(w.db), "-o", str(w.out), "-y"])
    assert "Purged 0" in kept.stdout and (w.out / ".trash").is_dir()
    gone = runner.invoke(app, ["trash-purge", "--db", str(w.db), "-o", str(w.out),
                               "--older-than-days", "-1", "-y"])
    assert "Purged 1" in gone.stdout and not (w.out / ".trash").exists()


def test_cli_apply_only_category(tmp_path):
    from .world import set_bucket
    w = _world_cli(tmp_path, n=2)
    set_bucket(w, w.ids[0], "food")
    r = runner.invoke(app, ["apply", "--db", str(w.db), "-o", str(w.out),
                            "--only-category", "food", "-y"])
    assert r.exit_code == 0
    assert not sorted(w.paths)[0].exists() and sorted(w.paths)[1].exists()


def test_cli_missing_optional_module_fails_gracefully(tmp_path, monkeypatch):
    import sys
    import typer
    import pytest
    from photocure.cli import _lazy
    with pytest.raises(typer.Exit):
        _lazy(".clustering.does_not_exist", "nothing")
    monkeypatch.setitem(sys.modules, "photocure.clustering.events", None)
    w = _world_cli(tmp_path, n=1)
    r = runner.invoke(app, ["events", "--db", str(w.db)])
    assert r.exit_code == 1 and "not available" in r.stdout


def test_cli_infer_retry_failed_resets_rows(tmp_path):
    from photocure.db.session import get_session
    from photocure.db.repository import PhotoRepository
    w = _world_cli(tmp_path, n=1)
    with get_session(w.db) as s:
        PhotoRepository(s).get_by_id(w.ids[0]).infer_status = "failed"
    r = runner.invoke(app, ["infer", "--db", str(w.db), "--device", "cpu",
                            "--retry-failed"])
    assert r.exit_code == 0 and "Reset 1 failed" in r.stdout


def test_cli_transactions_lists_ledger(tmp_path):
    w = _world_cli(tmp_path, n=1)
    runner.invoke(app, ["apply", "--db", str(w.db), "-o", str(w.out),
                        "--operation", "copy", "-y"])
    r = runner.invoke(app, ["transactions", "--db", str(w.db)])
    assert r.exit_code == 0 and "committed=1" in r.stdout


def test_cli_serve_and_start_options(tmp_path, monkeypatch):
    import uvicorn
    from photocure import setup_cmd as sc
    opened = []
    served = []
    monkeypatch.setattr(sc, "open_browser_when_ready", lambda host, port: opened.append((host, port)))
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port, **kw: served.append((host, port)))

    db = tmp_path / "test.db"
    r = runner.invoke(app, ["serve", "--db", str(db), "--port", "8099", "--open"])
    assert r.exit_code == 0
    assert len(opened) == 1 and opened[0] == ("127.0.0.1", 8099)
    assert len(served) == 1 and served[0] == ("127.0.0.1", 8099)

    r2 = runner.invoke(app, ["serve", "--db", str(db), "--port", "8099", "--no-open"])
    assert r2.exit_code == 0
    assert len(opened) == 1
    assert len(served) == 2

    r3 = runner.invoke(app, ["start", "--db", str(db), "--port", "8099", "--no-open"])
    assert r3.exit_code == 0
    assert len(served) == 3

"""End-to-end CLI tests using Typer's CliRunner."""
from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner

from photo_curator.cli import app
from .conftest import make_image

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "photo-curator" in result.stdout


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

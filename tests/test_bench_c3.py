"""C3: photocure bench blinks / pipeline."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from photocure import bench
from photocure import cli as cli_mod
from photocure.config import CuratorConfig

from .conftest import make_image

runner = CliRunner()
PORTRAIT = Path(__file__).parent / "fixtures" / "face_portrait.jpg"


def test_sweep_counts_and_metrics():
    scored = [("closed", .9), ("closed", .8), ("closed", .4), ("open", .1), ("open", .6), ("open", .2)]
    rows = {r.threshold: r for r in bench.sweep(scored, (0.5, 0.75))}
    r = rows[0.5]
    assert (r.tp, r.fp, r.fn, r.tn) == (2, 1, 1, 2)
    assert r.precision == pytest.approx(2 / 3) and r.recall == pytest.approx(2 / 3)
    assert rows[0.75].precision == 1.0 and rows[0.75].recall == pytest.approx(2 / 3)


def test_unknown_scores_are_neither_hits_nor_misses():
    rows = bench.sweep([("closed", None), ("open", None), ("closed", .9)], (0.5,))
    assert (rows[0].tp, rows[0].fn, rows[0].fp, rows[0].tn) == (1, 0, 0, 0)


def test_metrics_are_none_not_zero_when_undefined():
    row = bench.sweep([("open", .1)], (0.5,))[0]
    assert row.precision is None and row.recall is None and row.f1 is None
    assert bench.best_threshold([row]) is None


def test_best_threshold_picks_the_highest_f1():
    scored = [("closed", .9), ("closed", .85), ("open", .5), ("open", .2), ("open", .1)]
    assert bench.best_threshold(bench.sweep(scored)).threshold >= 0.55


def test_read_labels_resolves_relative_paths_and_skips_bad_rows(tmp_path):
    (tmp_path / "labels.csv").write_text("path,label\na.jpg,open\nb.jpg,CLOSED\nc.jpg,maybe\n,open\n")
    got = bench.read_labels(tmp_path / "labels.csv")
    assert got == [(tmp_path / "a.jpg", "open"), (tmp_path / "b.jpg", "closed")]


def test_bench_blinks_skips_gracefully_without_labels(tmp_path):
    r = runner.invoke(cli_mod.app, ["bench", "blinks", "--labels", str(tmp_path / "missing.csv")])
    assert r.exit_code == 0 and "Nothing to calibrate" in " ".join(r.output.split())


def test_bench_blinks_rejects_a_labels_file_with_no_usable_rows(tmp_path):
    f = tmp_path / "l.csv"
    f.write_text("path,label\nx.jpg,maybe\n")
    assert runner.invoke(cli_mod.app, ["bench", "blinks", "--labels", str(f)]).exit_code == 1


def test_bench_blinks_uses_the_real_models_when_available(tmp_path, monkeypatch):
    # Real detector. With MediaPipe the sweep prints; without it nothing is measured.
    monkeypatch.delenv("PCUR_BACKEND", raising=False)
    from photocure.inference.backends import reset_backend_cache
    reset_backend_cache()
    (tmp_path / "p.jpg").write_bytes(PORTRAIT.read_bytes())
    (tmp_path / "l.csv").write_text("path,label\np.jpg,open\n")
    r = runner.invoke(cli_mod.app, ["bench", "blinks", "--labels", str(tmp_path / "l.csv")])
    reset_backend_cache()
    try:
        import mediapipe  # noqa: F401
        assert r.exit_code == 0 and "Closed-eye threshold sweep" in r.output
    except ImportError:
        assert r.exit_code == 1 and "No eye scores" in r.output


def test_bench_pipeline_times_each_stage_and_flags_placeholders(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_BACKEND", "stub")
    for i in range(3):
        make_image(tmp_path / f"IMG_{i}.jpg", size=(240, 320), rng_seed=i)
    cfg = CuratorConfig()
    cfg.device.device = "cpu"
    res = bench.bench_pipeline(tmp_path, cfg, limit=3)
    assert res.photos == 3
    names = {row[0] for row in res.summary()}
    assert {"decode", "CLIP", "quality metrics", "SQLite commit"} <= names
    assert "face detection" not in names  # stub face backend: skipped, not faked
    assert any("placeholder" in n for n in res.notes) and any("No face detector" in n for n in res.notes)
    assert all(mean >= 0 and p95 >= med for _, _, mean, med, p95 in res.summary())


def test_bench_pipeline_cli_and_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_BACKEND", "stub")
    for i in range(4):
        make_image(tmp_path / f"IMG_{i}.jpg", size=(240, 320), rng_seed=i)
    r = runner.invoke(cli_mod.app, ["bench", "pipeline", str(tmp_path), "--limit", "2", "--device", "cpu"])
    assert r.exit_code == 0 and "2 photos" in r.output and "decode" in r.output
    assert runner.invoke(cli_mod.app, ["bench", "pipeline", str(tmp_path), "--device", "warp"]).exit_code == 2


def test_bench_pipeline_with_the_real_face_detector(tmp_path, monkeypatch):
    monkeypatch.delenv("PCUR_BACKEND", raising=False)
    from photocure.inference.backends import reset_backend_cache
    reset_backend_cache()
    shutil.copy(PORTRAIT, tmp_path / "IMG_a.jpg")
    cfg = CuratorConfig()
    cfg.device.device = "cpu"
    res = bench.bench_pipeline(tmp_path, cfg, limit=1)
    reset_backend_cache()
    names = {row[0] for row in res.summary()}
    assert res.face_backend == "yunet" and {"face detection", "crop + sharpness"} <= names

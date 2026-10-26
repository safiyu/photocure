"""Fixes from the final review of the CLI and the MCP server."""
from __future__ import annotations

import os
import types

import pytest
from typer.testing import CliRunner

from photocure.cli import app
from photocure.db import models as M
from photocure.db.session import get_session

from .test_doctor import scanned  # noqa: F401  (shared fixture)
from .test_mcp_server import q  # noqa: F401  (shared fixture)

runner = CliRunner()


def test_serve_keeps_the_one_time_code_the_installer_already_opened_the_browser_with(tmp_path, monkeypatch):
    import uvicorn
    for var in ("PCUR_HOST", "PCUR_PORT", "PCUR_AUTH_TOKEN"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    monkeypatch.setenv("PCUR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PCUR_OTC", "bootstrap-code-0123456789abcdef")
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)
    r = runner.invoke(app, ["serve", "--db", str(tmp_path / "t.db"), "--port", "8097", "--no-open"])
    assert r.exit_code == 0
    assert os.environ["PCUR_OTC"] == "bootstrap-code-0123456789abcdef"
    assert "bootstrap-code-0123456789abcdef" in r.output.replace("\n", "")


def _capture(monkeypatch):
    from photocure.inference import pipeline as pl
    seen: dict = {}

    def fake_run(self, db_path=None, only_paths=None, show_progress=True, reanalyze_stub=True,
                 stage="new", reanalyze=False, limit=0, **kw):
        seen.update(only_paths=only_paths, stage=stage, reanalyze=reanalyze, limit=limit)
        return types.SimpleNamespace(processed=0, faces_total=0, quality_computed=0, semantic_computed=0,
                                     failures=0, duration_s=0.0)

    monkeypatch.setattr(pl.InferencePipeline, "run", fake_run)
    return seen


def test_infer_limit_reaches_the_pipeline_instead_of_picking_pending_photos(scanned, monkeypatch):  # noqa: F811
    seen = _capture(monkeypatch)
    r = runner.invoke(app, ["infer", "--db", str(scanned), "--stage", "trash", "--limit", "1"])
    assert r.exit_code == 0
    assert seen["limit"] == 1 and seen["only_paths"] is None and seen["stage"] == "trash"


def test_a_bad_stage_is_refused_before_anything_is_changed(scanned, monkeypatch):  # noqa: F811
    _capture(monkeypatch)
    with get_session(scanned) as s:
        row = s.query(M.ImageRecord).first()
        row.infer_status = "failed"
        s.commit()
        rid = row.id
    r = runner.invoke(app, ["infer", "--db", str(scanned), "--stage", "library", "--retry-failed"])
    assert r.exit_code == 2
    with get_session(scanned) as s:
        assert s.get(M.ImageRecord, rid).infer_status == "failed"


def test_the_mcp_overview_leaves_out_missing_and_trashed_photos(q):  # noqa: F811
    with get_session(q.w.db) as s:
        r = s.get(M.ImageRecord, q.w.ids[0])
        r.stage, r.scan_status, r.route_status = "trash", "missing", "trashed"
        s.commit()
    assert q.q.overview()["photos"] == 2


def test_mcp_bursts_count_only_frames_that_are_still_in_the_library(q):  # noqa: F811
    with get_session(q.w.db) as s:
        bg = M.BurstGroup(name="Burst_t", image_count=3, best_shot_id=q.w.ids[1])
        s.add(bg)
        s.flush()
        for iid in q.w.ids:
            s.get(M.ImageRecord, iid).burst_id = bg.id
        gone = s.get(M.ImageRecord, q.w.ids[0])
        gone.stage, gone.scan_status, gone.route_status = "trash", "missing", "trashed"
        s.commit()
    items = q.q.list_bursts()["items"]
    assert items and items[0]["photos"] == 2

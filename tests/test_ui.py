"""Tests for the Web UI backend endpoints."""
from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from photocure.ui.app import create_app
from .conftest import make_image


def test_ui_scan_and_ingest(tmp_path):
    db_file = tmp_path / "ui_test.db"
    app = create_app(str(db_file))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token

    # Initial stats
    r_stats = client.get("/api/stats")
    assert r_stats.status_code == 200
    assert r_stats.json()["total_images"] == 0

    # Scan invalid directory
    r_bad = client.post("/api/scan", json={"path": "/nonexistent/path/12345"})
    assert r_bad.status_code == 400

    # Create dummy images
    src_dir = tmp_path / "photos"
    src_dir.mkdir()
    img1 = src_dir / "pic1.jpg"
    img2 = src_dir / "pic2.jpg"
    make_image(img1, rng_seed=1)
    make_image(img2, rng_seed=2)

    # Scan valid directory
    r_scan = client.post("/api/scan", json={"path": str(src_dir)})
    assert r_scan.status_code == 200
    data = r_scan.json()
    assert data["ok"] is True
    assert data["scanned"] == 2
    assert data["new"] == 2

    # Infer
    r_infer = client.post("/api/infer", json={"device": "cpu"})
    assert r_infer.status_code == 200
    assert r_infer.json()["ok"] is True
    assert r_infer.json()["processed"] == 2

    # Cluster
    r_cluster = client.post("/api/cluster", json={})
    assert r_cluster.status_code == 200
    assert r_cluster.json()["ok"] is True

    # Check updated stats
    r_stats2 = client.get("/api/stats")
    assert r_stats2.json()["total_images"] == 2

    # Check progress endpoint
    r_prog = client.get("/api/progress")
    assert r_prog.status_code == 200
    prog_data = r_prog.json()
    assert "active" in prog_data
    assert "percent" in prog_data
    assert "detail" in prog_data
    assert prog_data["active"] is False


def test_ui_mutations_require_token(tmp_path):
    app = create_app(str(tmp_path / "tok.db"))
    client = TestClient(app)
    assert client.post("/api/cluster", json={}).status_code == 403
    r = client.post("/api/cluster", json={}, headers={"X-PCUR-Token": "wrong"})
    assert r.status_code == 403
    r = client.post("/api/cluster", json={}, headers={"X-PCUR-Token": app.state.token})
    assert r.status_code == 200

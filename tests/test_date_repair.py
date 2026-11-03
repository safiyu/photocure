"""Tests for EXIF date audit, timezone shift, and sidecar sync."""
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from photocure.db.models import ImageRecord
from photocure.db.session import get_session, init_db
from photocure.scanner.xmp import find_sidecar, update_xmp_sidecar, parse_xmp_date, read_xmp
from photocure.ui.app import create_app
from tests.conftest import make_image


def test_update_xmp_sidecar_with_date(tmp_path):
    img = make_image(tmp_path / "IMG_20230814_100000.jpg")
    target_dt = datetime(2023, 8, 14, 10, 0, 0)

    ok, path = update_xmp_sidecar(img, date=target_dt)
    assert ok
    sc = find_sidecar(img)
    assert sc is not None
    info = read_xmp(img)
    assert info.taken_at == target_dt


def test_date_audit_and_fix_dates_endpoint(tmp_path, monkeypatch):
    db_path = tmp_path / "dates.db"
    init_db(db_path, drop=True)

    img1 = make_image(tmp_path / "IMG-20230814-WA0002.jpg")
    img2 = make_image(tmp_path / "camera_exif.jpg")

    dt1 = datetime(2023, 8, 14, 0, 0, 0, tzinfo=timezone.utc)
    dt2 = datetime(2024, 5, 1, 14, 30, 0, tzinfo=timezone.utc)

    with get_session(db_path) as s:
        r1 = ImageRecord(path=str(img1), filename=img1.name, taken_at=dt1,
                         taken_at_source="filename", file_size=100, quick_hash="q1")
        r2 = ImageRecord(path=str(img2), filename=img2.name, taken_at=dt2,
                         taken_at_source="exif", file_size=200, quick_hash="q2")
        s.add_all([r1, r2])
        s.commit()
        s.refresh(r1)
        r1_id = r1.id

    app = create_app(str(db_path))
    c = TestClient(app)
    c.headers["X-PCUR-Token"] = app.state.token

    # 1. Audit endpoint
    r_audit = c.get("/api/images/date-audit")
    assert r_audit.status_code == 200
    data = r_audit.json()
    assert data["counts"]["filename"] == 1
    assert data["counts"]["exif"] == 1
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["id"] == r1_id

    # 2. Fix dates endpoint (default: fixes filename/mtime photos and creates XMP sidecar)
    r_fix = c.post("/api/images/fix-dates", json={"write_sidecar": True})
    assert r_fix.status_code == 200
    res = r_fix.json()
    assert res["ok"] is True
    assert res["repaired_count"] == 1

    # Verify database was updated to "repaired"
    with get_session(db_path) as s:
        rec = s.get(ImageRecord, r1_id)
        assert rec.taken_at_source == "repaired"

    # Verify sidecar was created and has the date
    info = read_xmp(img1)
    assert info.taken_at == dt1.replace(tzinfo=None)

    # 3. Timezone shift
    r_shift = c.post("/api/images/fix-dates", json={"ids": [r1_id], "shift_hours": 3.0, "write_sidecar": True})
    assert r_shift.status_code == 200
    with get_session(db_path) as s:
        rec = s.get(ImageRecord, r1_id)
        assert rec.taken_at == dt1 + timedelta(hours=3)

"""Tests for Google Takeout JSON sidecar reading and multi-camera clock offset."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from photocure.db.models import ImageRecord
from photocure.db.session import get_session, init_db
from photocure.scanner.exif_reader import extract_exif, find_takeout_sidecar, read_takeout_sidecar
from photocure.ui.app import create_app
from tests.conftest import make_image


def test_google_takeout_sidecar_detection(tmp_path):
    img = make_image(tmp_path / "photo_travel.jpg")
    json_path = tmp_path / "photo_travel.jpg.json"
    json_path.write_text(json.dumps({
        "title": "photo_travel.jpg",
        "description": "Family trip to Kyoto",
        "photoTakenTime": {
            "timestamp": "1620000000",
            "formatted": "May 2, 2021, 10:40:00 PM UTC"
        },
        "geoData": {
            "latitude": 35.0116,
            "longitude": 135.7681,
            "altitude": 50.0
        },
        "favorited": True
    }))

    sidecar = find_takeout_sidecar(img)
    assert sidecar == json_path

    data = read_takeout_sidecar(img)
    assert data is not None
    assert data["gps"] == (35.0116, 135.7681)
    assert data["is_favorite"] is True
    assert data["description"] == "Kyoto" or "Kyoto" in data["description"]
    assert data["taken_at"] is not None

    # Exif reader integration
    info = extract_exif(img)
    assert info.taken_at_source == "google_takeout"
    assert info.gps_lat == 35.0116
    assert info.is_favorite is True


def test_multi_camera_clock_offset_endpoint(tmp_path):
    db_path = tmp_path / "camera_offset.db"
    init_db(db_path, drop=True)

    img_sony = make_image(tmp_path / "DSC0001.jpg")
    img_canon = make_image(tmp_path / "IMG0001.jpg")

    t_sony = datetime(2023, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    t_canon = datetime(2023, 5, 1, 12, 0, 0, tzinfo=timezone.utc)

    with get_session(db_path) as s:
        r1 = ImageRecord(path=str(img_sony), filename=img_sony.name, taken_at=t_sony,
                         camera_model="Sony ILCE-7M3", file_size=100, quick_hash="q1")
        r2 = ImageRecord(path=str(img_canon), filename=img_canon.name, taken_at=t_canon,
                         camera_model="Canon EOS R6", file_size=200, quick_hash="q2")
        s.add_all([r1, r2])
        s.commit()
        s.refresh(r1)
        s.refresh(r2)
        r1_id = r1.id
        r2_id = r2.id

    app = create_app(str(db_path))
    c = TestClient(app)
    c.headers["X-PCUR-Token"] = app.state.token

    # 1. Audit endpoint lists cameras
    r_audit = c.get("/api/images/date-audit")
    assert r_audit.status_code == 200
    data = r_audit.json()
    cams = {cam["name"]: cam["count"] for cam in data["cameras"]}
    assert cams.get("Sony ILCE-7M3") == 1
    assert cams.get("Canon EOS R6") == 1

    # 2. Offset only Sony by +15 minutes (900 seconds)
    r_fix = c.post("/api/images/fix-dates", json={
        "camera_model": "Sony ILCE-7M3",
        "shift_minutes": 15,
        "write_sidecar": False
    })
    assert r_fix.status_code == 200
    assert r_fix.json()["repaired_count"] == 1

    with get_session(db_path) as s:
        updated_sony = s.get(ImageRecord, r1_id)
        unchanged_canon = s.get(ImageRecord, r2_id)
        assert updated_sony.taken_at == t_sony + timedelta(minutes=15)
        assert unchanged_canon.taken_at == t_canon

    # 3. Test ingest_takeout batch action
    img_takeout = make_image(tmp_path / "takeout_test.jpg")
    (tmp_path / "takeout_test.jpg.json").write_text(json.dumps({
        "photoTakenTime": {"timestamp": "1650000000"},
        "geoData": {"latitude": 40.7128, "longitude": -74.0060},
        "favorited": True
    }))
    with get_session(db_path) as s:
        r_t = ImageRecord(path=str(img_takeout), filename=img_takeout.name,
                          taken_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                          taken_at_source="mtime", file_size=300, quick_hash="qt")
        s.add(r_t)
        s.commit()
        s.refresh(r_t)
        rt_id = r_t.id

    r_ingest = c.post("/api/images/fix-dates", json={"ids": [rt_id], "ingest_takeout": True, "write_sidecar": False})
    assert r_ingest.status_code == 200
    assert r_ingest.json()["repaired_count"] == 1

    with get_session(db_path) as s:
        rec = s.get(ImageRecord, rt_id)
        assert rec.taken_at_source == "google_takeout"
        assert rec.gps_lat == 40.7128
        assert rec.is_favorite is True

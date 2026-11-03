"""Tests for Library Insights endpoint and calculation."""
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from photocure.db.models import BurstGroup, ImageRecord
from photocure.db.session import get_session, init_db
from photocure.insights import get_library_insights
from photocure.ui.app import create_app
from tests.conftest import make_image


def test_library_insights_calculation_and_endpoint(tmp_path):
    db_path = tmp_path / "insights.db"
    init_db(db_path, drop=True)

    img1 = make_image(tmp_path / "dup.jpg")
    img2 = make_image(tmp_path / "burst_winner.jpg")
    img3 = make_image(tmp_path / "burst_alt.jpg")
    img4 = make_image(tmp_path / "junk.jpg")
    img5 = make_image(tmp_path / "keeper.jpg")

    with get_session(db_path) as s:
        r1 = ImageRecord(path=str(img1), filename=img1.name, file_size=2000,
                         is_duplicate=True, stage="new", quick_hash="q1")
        r2 = ImageRecord(path=str(img2), filename=img2.name, file_size=5000,
                         is_best_shot=True, stage="new", quick_hash="q2")
        r3 = ImageRecord(path=str(img3), filename=img3.name, file_size=4000,
                         is_alternate=True, stage="new", quick_hash="q3")
        r4 = ImageRecord(path=str(img4), filename=img4.name, file_size=1000,
                         category="accidental", stage="new", quick_hash="q4")
        r5 = ImageRecord(path=str(img5), filename=img5.name, file_size=8000,
                         category="people", camera_model="Sony A7", stage="new", quick_hash="q5")
        s.add_all([r1, r2, r3, r4, r5])
        s.commit()
        s.refresh(r2)
        s.refresh(r3)

        bg = BurstGroup(name="burst_1", best_shot_id=r2.id, image_count=2)
        s.add(bg)
        s.commit()
        s.refresh(bg)

        r2.burst_id = bg.id
        r3.burst_id = bg.id
        s.commit()

        # Direct function check
        insights = get_library_insights(s, stage="all")
        assert insights["total_photos"] == 5
        assert insights["total_bytes"] == 20000
        assert insights["duplicates"]["count"] == 1
        assert insights["duplicates"]["bytes"] == 2000
        assert insights["burst_alternates"]["count"] == 1
        assert insights["burst_alternates"]["bytes"] == 4000
        assert insights["junk"]["count"] == 1
        assert insights["junk"]["bytes"] == 1000
        # Reclaimable = 2000 + 4000 + 1000 = 7000
        assert insights["reclaimable_bytes"] == 7000
        assert insights["reclaimable_percent"] == 35.0

    # Test via API
    app = create_app(str(db_path))
    c = TestClient(app)
    c.headers["X-PCUR-Token"] = app.state.token

    r = c.get("/api/insights?stage=all")
    assert r.status_code == 200
    data = r.json()
    assert data["total_photos"] == 5
    assert data["reclaimable_bytes"] == 7000
    assert len(data["by_camera"]) == 1
    assert data["by_camera"][0]["name"] == "Sony A7"

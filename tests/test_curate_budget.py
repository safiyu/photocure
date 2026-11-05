"""Tests for Target Keeper Budget ("Cull to N Photos")."""
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from photocure.db.models import ImageRecord, SemanticScore
from photocure.db.session import get_session, init_db
from photocure.ui.app import create_app
from tests.conftest import make_image


def test_budget_curate_preview_and_stage_delete(tmp_path):
    db_path = tmp_path / "budget.db"
    init_db(db_path, drop=True)

    img_paths = [make_image(tmp_path / f"img_{i:03d}.jpg") for i in range(10)]

    with get_session(db_path) as s:
        # 10 photos: img_000 is favorite, img_001 is locked, img_002 is 5-star
        records = []
        for i, p in enumerate(img_paths):
            r = ImageRecord(
                path=str(p),
                filename=p.name,
                file_size=1000 * (i + 1),
                quick_hash=f"qh_{i}",
                taken_at=datetime(2023, 6, 1, 10, i, 0, tzinfo=timezone.utc),
                stage="new",
                is_favorite=(i == 0),
                locked=(i == 1),
                rating=5 if i == 2 else 0,
                is_best_shot=(i == 3),
                is_alternate=(i == 4),
            )
            s.add(r)
            records.append(r)
        s.commit()
        for r in records:
            s.refresh(r)
        rec_ids = [r.id for r in records]

        # Semantic scores: aesthetic ranging from 0.1 to 0.9
        for i, r in enumerate(records):
            sem = SemanticScore(image_id=r.id, aesthetic=0.1 * i, backend="stub")
            s.add(sem)
        s.commit()

    app = create_app(str(db_path))
    c = TestClient(app)
    c.headers["X-PCUR-Token"] = app.state.token

    # 1. Preview budget: target 4 photos out of 10
    r_preview = c.post("/api/curate/budget", json={"target_count": 4, "action": "preview"})
    assert r_preview.status_code == 200
    p_data = r_preview.json()
    assert p_data["ok"] is True
    assert p_data["total"] == 10
    assert p_data["budget"] == 4
    assert p_data["kept_count"] == 4
    assert p_data["culled_count"] == 6
    # Guaranteed keepers (fav=img0, locked=img1, 5-star=img2) must be in kept_ids
    assert rec_ids[0] in p_data["kept_ids"]
    assert rec_ids[1] in p_data["kept_ids"]
    assert rec_ids[2] in p_data["kept_ids"]

    # 2. Apply action: stage_delete
    r_stage = c.post("/api/curate/budget", json={"target_count": 4, "action": "stage_delete"})
    assert r_stage.status_code == 200
    s_data = r_stage.json()
    assert s_data["action_applied"] == "stage_delete"

    # Verify database: culled photos have delete_approved = True
    with get_session(db_path) as s:
        for rid in rec_ids:
            fresh = s.get(ImageRecord, rid)
            if fresh.id in s_data["kept_ids"]:
                assert fresh.delete_approved is False
            else:
                assert fresh.delete_approved is True


def test_budget_curate_excludes_rejects_junk_utility(tmp_path):
    db_path = tmp_path / "budget_filter.db"
    init_db(db_path, drop=True)

    img_good = make_image(tmp_path / "good.jpg")
    img_rejected = make_image(tmp_path / "rejected.jpg")
    img_junk = make_image(tmp_path / "junk.jpg")
    img_util = make_image(tmp_path / "util.jpg")

    with get_session(db_path) as s:
        r_good = ImageRecord(path=str(img_good), filename=img_good.name, file_size=500, quick_hash="qg", stage="new")
        r_rej = ImageRecord(path=str(img_rejected), filename=img_rejected.name, file_size=500, rating=-1, quick_hash="qr", stage="new")
        r_junk = ImageRecord(path=str(img_junk), filename=img_junk.name, file_size=500, category="accidental", quick_hash="qj", stage="new")
        r_util = ImageRecord(path=str(img_util), filename=img_util.name, file_size=500, category="document", quick_hash="qu", stage="new")
        s.add_all([r_good, r_rej, r_junk, r_util])
        s.commit()
        s.refresh(r_good)
        good_id = r_good.id

    app = create_app(str(db_path))
    c = TestClient(app)
    c.headers["X-PCUR-Token"] = app.state.token

    res = c.post("/api/curate/budget", json={"target_count": 5, "action": "preview"})
    assert res.status_code == 200
    data = res.json()
    # Only good photo should be considered
    assert data["total"] == 1
    assert data["kept_ids"] == [good_id]



def test_budget_uses_scale_labels(session, monkeypatch):
    from photocure.curate_budget import budget_curate
    from photocure.db.models import ImageRecord, SemanticScore
    for i in range(1, 7):
        session.add(ImageRecord(id=i, path=f"/b/{i}.jpg", filename=f"{i}.jpg", stage="new"))
    session.flush()
    for i in range(1, 7):
        session.add(SemanticScore(image_id=i, aesthetic=0.4 + i / 20, backend="clip"))
    session.commit()
    monkeypatch.setattr("photocure.curate_budget.scale_labels",
                        lambda s, ids, enc: {i: ("wide" if i % 2 else "closeup") for i in ids})
    monkeypatch.setattr("photocure.search.default_encoder", lambda cfg=None: object())

    res = budget_curate(session, target_count=4, stage="new")

    assert set(res["scale_coverage"]) == {"wide", "closeup"}

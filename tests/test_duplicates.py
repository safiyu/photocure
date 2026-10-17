"""Duplicate review: exact and near groups, keeper suggestion, resolving and the API."""
from __future__ import annotations

import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from photocure import duplicates as dup
from photocure.db.models import ImageRecord, QualityMetric, SemanticScore
from photocure.db.repository import quantize_embedding
from photocure.db.session import get_session
from photocure.rules.decision_matrix import DecisionMatrix
from photocure.rules.taxonomy import CategoryType
from photocure.ui.app import create_app


def _img(session, i, *, sha="", size=1000, w=100, h=100, **kw):
    img = ImageRecord(path=f"/lib/{i}.jpg", filename=f"{i}.jpg", sha256=sha, file_size=size,
                      width=w, height=h, **kw)
    session.add(img)
    session.flush()
    return img


def _emb(session, img, vec, quality=None, aesthetic=None, backend="clip"):
    session.add(SemanticScore(image_id=img.id, backend=backend, aesthetic=aesthetic,
                              embedding=quantize_embedding(np.asarray(vec).tolist())))
    if quality is not None:
        session.add(QualityMetric(image_id=img.id, quality_score=quality))
    session.flush()


def _axis(i):
    v = np.zeros(512, dtype=np.float32)
    v[i] = 1.0
    return v


def _ids(group):
    return [m["id"] for m in group["members"]]


# ------------------------------------------------------------- exact groups
def test_identical_files_form_a_group_with_reclaimable_space(session):
    a = _img(session, 1, sha="s1", size=500)
    b = _img(session, 2, sha="s1", size=500)
    c = _img(session, 3, sha="s1", size=500)
    _img(session, 4, sha="other")
    _img(session, 5, sha="s1", scan_status="missing")
    _img(session, 6, sha="")
    groups = dup.find_groups(session)
    assert len(groups) == 1 and groups[0]["kind"] == "exact"
    assert set(_ids(groups[0])) == {a.id, b.id, c.id}
    assert groups[0]["reclaimable_bytes"] == 1000
    assert groups[0]["keeper_id"] == a.id and groups[0]["members"][0]["suggested"] is True


@pytest.mark.parametrize(("better", "worse"), [
    (dict(is_favorite=True), dict()),
    (dict(rating=5), dict(rating=2)),
    (dict(w=400, h=300), dict(w=100, h=100)),
    (dict(size=9000), dict(size=100)),
])
def test_the_better_copy_is_suggested_as_keeper(session, better, worse):
    first = _img(session, 1, sha="s", **worse)
    second = _img(session, 2, sha="s", **better)
    g = dup.find_groups(session)[0]
    assert g["keeper_id"] == second.id and _ids(g)[0] == second.id
    assert first.id in _ids(g)


def test_a_rejected_copy_is_never_the_keeper_and_ties_go_to_the_oldest(session):
    old = _img(session, 1, sha="s")
    _img(session, 2, sha="s", rating=-1, is_favorite=True, w=999, h=999)
    new = _img(session, 3, sha="s")
    g = dup.find_groups(session)[0]
    assert g["keeper_id"] == old.id
    assert dup.suggest_keeper([m for m in g["members"] if m["id"] in (old.id, new.id)]) == old.id


# -------------------------------------------------------------- near groups
def test_near_duplicates_are_grouped_through_their_links(session):
    keep = _img(session, 1, w=400, h=300)
    a = _img(session, 2, near_dup_of=keep.id)
    b = _img(session, 3, near_dup_of=a.id)                  # chain: b -> a -> keep
    solo = _img(session, 4)
    gone = _img(session, 5, near_dup_of=solo.id, scan_status="missing")
    groups = dup.find_groups(session, kind="near")
    assert len(groups) == 1 and set(_ids(groups[0])) == {keep.id, a.id, b.id}
    assert groups[0]["keeper_id"] == keep.id and groups[0]["kind"] == "near"
    assert gone.id not in _ids(groups[0])


def test_burst_members_exact_duplicates_and_user_placed_photos_stay_out_of_near_groups(session):
    keep = _img(session, 1)
    _img(session, 2, near_dup_of=keep.id, burst_id=7)
    _img(session, 3, near_dup_of=keep.id, is_duplicate=True)
    _img(session, 4, near_dup_of=keep.id, manual_bucket="scenery")
    assert dup.find_groups(session, kind="near") == []


def test_links_the_embeddings_do_not_support_are_dropped(session):
    keep = _img(session, 1)
    close = _img(session, 2, near_dup_of=keep.id)
    far = _img(session, 3, near_dup_of=keep.id)
    unknown = _img(session, 4, near_dup_of=keep.id)
    _emb(session, keep, _axis(0))
    _emb(session, close, _axis(0) + 0.05 * _axis(1))
    _emb(session, far, _axis(5))
    g = dup.find_groups(session, kind="near", near_min_cosine=0.93)[0]
    assert set(_ids(g)) == {keep.id, close.id, unknown.id}
    sim = {m["id"]: m["similarity"] for m in g["members"]}
    assert sim[close.id] > 0.99 and sim[unknown.id] is None and sim[keep.id] == 1.0


def test_groups_can_be_filtered_by_kind(session):
    _img(session, 1, sha="s")
    _img(session, 2, sha="s")
    n = _img(session, 3)
    _img(session, 4, near_dup_of=n.id)
    assert [g["kind"] for g in dup.find_groups(session, kind="exact")] == ["exact"]
    assert [g["kind"] for g in dup.find_groups(session, kind="near")] == ["near"]
    assert len(dup.find_groups(session)) == 2
    with pytest.raises(ValueError):
        dup.find_groups(session, kind="weird")


# ---------------------------------------------------------------- resolving
def test_resolving_exact_copies_keeps_one_and_routes_the_rest_as_duplicates(session):
    a = _img(session, 1, sha="s", size=300)
    b = _img(session, 2, sha="s", size=300, is_duplicate=True, duplicate_of=1)   # scanner's choice
    c = _img(session, 3, sha="s", size=300, is_duplicate=True, duplicate_of=1)
    # The user prefers the copy the scanner called a duplicate.
    out = dup.resolve_group(session, keep_id=b.id, discard_ids=[a.id, c.id], kind="exact")
    assert out["kept"] == b.id and set(out["discarded"]) == {a.id, c.id}
    assert out["reclaimable_bytes"] == 600
    assert (b.is_duplicate, b.duplicate_of) == (False, None)
    for gone in (a, c):
        assert gone.is_duplicate and gone.duplicate_of == b.id and gone.manual_bucket == "duplicate"
    m = DecisionMatrix()
    assert m.evaluate_record(a).category == CategoryType.DUPLICATE
    assert m.evaluate_record(b).category != CategoryType.DUPLICATE


def test_a_resolved_group_leaves_the_queue_but_can_be_listed(session):
    a = _img(session, 1, sha="s")
    b = _img(session, 2, sha="s")
    dup.resolve_group(session, keep_id=a.id, discard_ids=[b.id], kind="exact")
    assert dup.find_groups(session) == []
    again = dup.find_groups(session, include_reviewed=True)
    assert len(again) == 1 and again[0]["reviewed"] is True


def test_resolving_near_duplicates_marks_them_and_clears_the_keepers_link(session):
    keep = _img(session, 1)
    a = _img(session, 2, near_dup_of=keep.id, manual_bucket=None)
    other = _img(session, 3, near_dup_of=a.id)                # pointed at a discard
    out = dup.resolve_group(session, keep_id=keep.id, discard_ids=[a.id, other.id], kind="near")
    assert out["discarded"] == [a.id, other.id]
    assert a.manual_bucket == "near_duplicate" and a.near_dup_of == keep.id
    assert other.manual_bucket == "near_duplicate" and other.near_dup_of == keep.id
    assert keep.near_dup_of is None and keep.manual_bucket is None
    assert DecisionMatrix().evaluate_record(a).category == CategoryType.NEAR_DUPLICATE


def test_keeping_a_photo_clears_an_old_duplicate_bucket(session):
    keep = _img(session, 1, manual_bucket="duplicate", near_dup_of=None)
    gone = _img(session, 2, near_dup_of=keep.id)
    dup.resolve_group(session, keep_id=keep.id, discard_ids=[gone.id], kind="near")
    assert keep.manual_bucket is None


def test_dismissing_near_duplicates_unlinks_them(session):
    keep = _img(session, 1)
    a = _img(session, 2, near_dup_of=keep.id)
    b = _img(session, 3, near_dup_of=keep.id)
    assert dup.dismiss_near(session, [keep.id, a.id, b.id]) == 2
    assert a.near_dup_of is None and b.near_dup_of is None
    assert dup.find_groups(session, kind="near") == []


@pytest.mark.parametrize("args", [
    dict(keep_id=1, discard_ids=[1], kind="exact"),
    dict(keep_id=1, discard_ids=[], kind="exact"),
    dict(keep_id=1, discard_ids=[2], kind="bogus"),
])
def test_resolve_rejects_bad_requests(session, args):
    _img(session, 1)
    _img(session, 2)
    with pytest.raises(ValueError):
        dup.resolve_group(session, **args)


def test_resolve_rejects_unknown_photos(session):
    a = _img(session, 1)
    with pytest.raises(LookupError):
        dup.resolve_group(session, keep_id=a.id, discard_ids=[999], kind="exact")


def test_summary_counts_open_groups_and_space(session):
    _img(session, 1, sha="s", size=400)
    _img(session, 2, sha="s", size=400)
    _img(session, 3, sha="s", size=400)
    k = _img(session, 4, size=50, w=300, h=300)
    _img(session, 5, size=70, near_dup_of=k.id)
    s = dup.summary(session)
    assert s == {"groups": 2, "exact_groups": 1, "near_groups": 1, "images": 5,
                 "reclaimable_bytes": 800 + 70}


def test_resolve_all_applies_the_suggested_keepers(session):
    a = _img(session, 1, sha="s", size=10, is_favorite=True)
    b = _img(session, 2, sha="s", size=10)
    k = _img(session, 3)
    n = _img(session, 4, near_dup_of=k.id, size=20)
    out = dup.resolve_all(session)
    assert out == {"groups": 2, "discarded": 2, "reclaimable_bytes": 30}
    assert b.manual_bucket == "duplicate" and n.manual_bucket == "near_duplicate"
    assert a.manual_bucket is None and k.manual_bucket is None
    assert dup.find_groups(session) == []


# ----------------------------------------------------------------------- API
@pytest.fixture()
def env(tmp_path):
    app = create_app(str(tmp_path / "d.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    with get_session(app.state.db) as s:
        a = _img(s, 1, sha="s", size=300, w=200, h=200)
        b = _img(s, 2, sha="s", size=300, w=100, h=100)
        k = _img(s, 3, size=40, w=300, h=300)
        n = _img(s, 4, size=60, near_dup_of=k.id)
        ids = types.SimpleNamespace(a=a.id, b=b.id, k=k.id, n=n.id)
    return types.SimpleNamespace(app=app, client=client, ids=ids)


def test_api_lists_groups_with_a_summary_and_items(env):
    r = env.client.get("/api/duplicates")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["groups"] == 2 and body["summary"]["reclaimable_bytes"] == 360
    exact = next(g for g in body["groups"] if g["kind"] == "exact")
    assert exact["keeper_id"] == env.ids.a
    assert exact["members"][0]["item"]["id"] == env.ids.a and "filename" in exact["members"][0]["item"]
    only = env.client.get("/api/duplicates", params={"kind": "near"}).json()
    assert [g["kind"] for g in only["groups"]] == ["near"]
    paged = env.client.get("/api/duplicates", params={"limit": 1}).json()
    assert len(paged["groups"]) == 1 and paged["summary"]["groups"] == 2


def test_api_validates_the_listing_parameters(env):
    assert env.client.get("/api/duplicates", params={"kind": "x"}).status_code == 422
    assert env.client.get("/api/duplicates", params={"limit": 0}).status_code == 422


def test_api_resolve_updates_the_photos_and_returns_the_new_summary(env):
    r = env.client.post("/api/duplicates/resolve", json={
        "keep_id": env.ids.b, "discard_ids": [env.ids.a], "kind": "exact"})
    assert r.status_code == 200 and r.json()["result"]["kept"] == env.ids.b
    assert r.json()["summary"]["groups"] == 1
    with get_session(env.app.state.db) as s:
        assert s.get(ImageRecord, env.ids.a).manual_bucket == "duplicate"


def test_api_resolve_reports_bad_requests(env):
    bad = env.client.post("/api/duplicates/resolve", json={
        "keep_id": env.ids.a, "discard_ids": [env.ids.a], "kind": "exact"})
    assert bad.status_code == 400
    missing = env.client.post("/api/duplicates/resolve", json={
        "keep_id": env.ids.a, "discard_ids": [9999], "kind": "exact"})
    assert missing.status_code == 404


def test_api_dismiss_and_resolve_all(env):
    r = env.client.post("/api/duplicates/dismiss", json={"ids": [env.ids.k, env.ids.n]})
    assert r.status_code == 200 and r.json()["summary"]["near_groups"] == 0
    r = env.client.post("/api/duplicates/resolve-all", json={})
    assert r.status_code == 200 and r.json()["result"]["groups"] == 1
    assert env.client.get("/api/duplicates").json()["summary"]["groups"] == 0


def test_api_duplicates_need_a_session(env):
    anon = TestClient(env.app)
    assert anon.get("/api/duplicates").status_code == 401
    assert anon.post("/api/duplicates/resolve-all", json={}).status_code == 401
    assert anon.post("/api/duplicates/dismiss", json={"ids": [1]}).status_code == 401
    assert anon.post("/api/duplicates/resolve", json={}).status_code == 401


def test_summarize_agrees_with_summary(session):
    _img(session, 1, sha="s", size=400)
    _img(session, 2, sha="s", size=400)
    k = _img(session, 3, size=50, w=300, h=300)
    _img(session, 4, size=70, near_dup_of=k.id)
    assert dup.summarize(dup.find_groups(session)) == dup.summary(session)
    assert dup.summarize([]) == {"groups": 0, "exact_groups": 0, "near_groups": 0, "images": 0,
                                 "reclaimable_bytes": 0}


def test_the_listing_finds_groups_only_once_per_request(env, monkeypatch):
    calls = []
    real = dup.find_groups
    monkeypatch.setattr(dup, "find_groups", lambda *a, **k: calls.append(1) or real(*a, **k))
    r = env.client.get("/api/duplicates", params={"kind": "exact"})
    assert r.status_code == 200 and len(calls) == 1
    assert r.json()["total"] == 1 and r.json()["summary"]["groups"] == 2   # the summary still spans every kind

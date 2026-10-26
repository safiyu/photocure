"""Fixes from the final review of the front end, and the API stage filters they rely on."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from photocure.db import models as M
from photocure.db.session import get_session

from .test_ui_api import env  # noqa: F401  (shared fixture)

STATIC = Path(__file__).resolve().parent.parent / "photocure" / "ui" / "static"
JS = (STATIC / "app.js").read_text()
HTML = (STATIC / "index.html").read_text()


def fn(name: str, kind: str = "function") -> str:
    """Source of one top-level function in app.js (up to the next two-space-indented closing brace)."""
    m = re.search(rf"\n  (?:async )?{kind} {name}\b.*?\n  \}}\n", JS, re.S)
    assert m, name
    return m.group(0)


# ---------------------------------------------------------------- lightbox
def test_the_lightbox_follows_a_photo_that_leaves_the_grid_under_it():
    body = fn("itemUpdated")
    assert "wasCurrent" in body
    assert body.index("wasCurrent") < body.index("g.rerender")  # decided before the grid drops the photo
    assert "closeLightbox" in body or "lbRender" in body.split("g.rerender", 1)[1]


def test_a_finished_trash_job_closes_the_viewer_and_the_confirm_names_the_photo():
    assert "closeLightbox" in fn("runTrashJob")
    assert "filename" in fn("purgePhotos")


# ---------------------------------------------------------------- people and stages
def test_people_deletes_and_recluster_carry_the_stage():
    for needle in ('api(`/api/people/${p.id}?stage=', "/api/people/bulk-delete?stage=",
                   "api(`/api/people/${activePpoPersonId}?stage="):
        assert needle in JS, needle
    assert "/api/people/recluster?stage=" in JS


def test_tabs_hidden_for_the_stage_are_not_reachable_by_links_or_back_button():
    assert "function tabAllowed" in JS
    assert "tabAllowed(" in fn("showTab")
    assert "data-stages" in fn("tabAllowed") or "dataset.stages" in fn("tabAllowed")


# ---------------------------------------------------------------- focus and modals
def test_the_sign_in_box_can_take_focus_while_another_dialog_traps_it():
    assert "#auth-modal" in JS.split("const onDocFocusIn", 1)[1].split("const trapRecord", 1)[0]


def test_every_dialog_is_in_the_modal_list_and_the_move_dialog_returns_focus():
    modals = re.search(r"const MODALS = \[(.*?)\];", JS, re.S).group(1)
    for sel in ("#firstrun", "#movelib", "#cleanexport"):
        assert sel in modals, sel
    assert "activeElement" in fn("openMoveLibrary")


# ---------------------------------------------------------------- results that arrive late
@pytest.mark.parametrize("name", ["showSimilar", "showBurst", "loadDuplicates"])
def test_overlapping_requests_cannot_overwrite_a_newer_answer(name):
    assert "Seq" in fn(name, "function") or "seq" in fn(name, "function")


# ---------------------------------------------------------------- trash buttons
def test_trash_buttons_are_disabled_while_busy_or_when_there_is_nothing_to_act_on():
    for bid in ("bulk-restore", "bulk-return", "bulk-purge"):
        assert "data-busy" in re.search(r'<button[^>]*id="%s"[^>]*>' % bid, HTML).group(0), bid
    counts = fn("renderCounts")
    assert "btn-trash-empty" in counts and "disabled" in counts


def test_trash_toasts_explain_a_file_outside_the_trash_and_failed_moves():
    assert "outside_trash" in JS and "failed" in fn("skippedText")


def test_keep_reports_only_what_it_kept_and_a_duplicate_stub_never_replaces_cached_data():
    assert "kept" in fn("bulkLock")
    assert "m.item" in fn("dupMemberNode").split("state.byId[m.id]", 1)[0] or "if (m.item)" in fn("dupMemberNode")


def test_the_burst_view_does_not_reuse_ids_that_the_library_grid_owns():
    burst = JS.split("async function showBurst", 1)[1].split("\n  }\n", 1)[0]
    assert 'id="grid"' not in burst and 'id="title"' not in burst and 'id="count"' not in burst


def test_icons_used_by_the_page_exist_in_the_sprite():
    sprite = set(re.findall(r'<symbol id="(i-[a-z0-9-]+)"', HTML))
    used = set(re.findall(r'href="#(i-[a-z0-9-]+)"', HTML))
    used |= {"i-" + n for n in re.findall(r'\bicon\("([a-z0-9-]+)"', JS)}
    assert not (used - sprite), sorted(used - sprite)


# ---------------------------------------------------------------- API: stage filters
def _burst(env, members):
    with get_session(env.db) as s:
        bg = M.BurstGroup(name="Burst_s", image_count=len(members), best_shot_id=members[0])
        s.add(bg)
        s.flush()
        for n, iid in enumerate(members):
            r = s.get(M.ImageRecord, iid)
            r.burst_id, r.is_best_shot, r.is_alternate = bg.id, n == 0, n != 0
        s.commit()
        return bg.id


def test_burst_detail_lists_only_the_requested_stage(env):
    ids = sorted(i["id"] for i in env.client.get("/api/images?limit=50").json()["items"])
    with get_session(env.db) as s:
        for iid in ids[2:]:
            r = s.get(M.ImageRecord, iid)
            r.stage, r.route_status = "library", "executed"
        s.commit()
    bid = _burst(env, ids)
    members = lambda q="": {m["id"] for m in env.client.get(f"/api/burst/{bid}{q}").json()["members"]}  # noqa: E731
    assert members("?stage=new") == set(ids[:2])
    assert members("?stage=library") == set(ids[2:])
    assert members() == set(ids)


def test_similar_photos_stay_inside_the_requested_stage(env):
    ids = sorted(i["id"] for i in env.client.get("/api/images?limit=50").json()["items"])
    with get_session(env.db) as s:
        for iid in ids[1:3]:
            r = s.get(M.ImageRecord, iid)
            r.stage, r.route_status = "library", "executed"
        s.commit()
    got = {i["id"] for i in env.client.get(f"/api/images/{ids[0]}/similar?stage=library").json()["items"]}
    assert got <= set(ids[1:3])
    got_new = {i["id"] for i in env.client.get(f"/api/images/{ids[0]}/similar?stage=new").json()["items"]}
    assert got_new.isdisjoint(set(ids[1:3]))

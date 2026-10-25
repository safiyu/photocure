"""Expanded taxonomy, group aware gating and plain reasons for uncertain photos."""
from __future__ import annotations

import json

import pytest

from photocure.config import CuratorConfig
from photocure.db.models import QualityMetric, SemanticScore
from photocure.rules.decision_matrix import (
    DecisionMatrix,
    assess_semantics,
    explain_uncertain,
)
from photocure.rules.organizer import FolderOrganizer, OrganizeMode
from photocure.rules.taxonomy import (
    CategoryType,
    DEFAULT_TAXONOMY,
    parse_taxonomy,
)

from .test_rules import _good_quality, _rec

NEW = ("illustration", "slide", "vehicle", "architecture", "plant", "product")
FOLDERS = {
    "illustration": "Non-Human/Art_and_Diagrams",
    "slide": "Utility_Reference/Notes_and_Whiteboards",
    "vehicle": "Non-Human/Vehicles",
    "architecture": "Non-Human/Architecture",
    "plant": "Non-Human/Plants",
    "product": "Non-Human/Products",
}


def sem(dist: dict, top1=None, top2=None) -> SemanticScore:
    """A semantic row built from a probability distribution (not normalized on purpose)."""
    ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    t1 = top1 or ranked[0][0]
    t2 = top2 or ranked[1][0]
    return SemanticScore(model_name="t", top1_category=t1, top1_score=dist[t1],
                         top2_category=t2, top2_score=dist[t2],
                         score_margin=round(dist[t1] - dist[t2], 4),
                         distribution=json.dumps(dist))


def evaluate(rec):
    return DecisionMatrix(CuratorConfig()).evaluate_record(rec)


# --------------------------------------------------------------- taxonomy
def test_new_categories_folders_and_flags():
    for name in NEW:
        cat = CategoryType(name)
        assert cat.folder == FOLDERS[name]
        assert not cat.is_junk
    assert not CategoryType.VEHICLE.is_date_grouped
    assert not CategoryType.ARCHITECTURE.is_date_grouped
    assert not CategoryType.PLANT.is_date_grouped
    assert not CategoryType.ILLUSTRATION.is_date_grouped
    assert not CategoryType.SLIDE.is_date_grouped
    assert not CategoryType.PRODUCT.is_date_grouped


def test_every_builtin_class_has_a_rich_prompt_ensemble_and_group():
    names = {e.key for e in DEFAULT_TAXONOMY.entries}
    assert set(NEW) <= names
    for e in DEFAULT_TAXONOMY.entries:
        assert 4 <= len(e.prompts) <= 6, e.key
        assert all(p.strip() for p in e.prompts)
        assert len(set(e.prompts)) == len(e.prompts)
        assert e.group, e.key


def test_groups():
    t = DEFAULT_TAXONOMY
    assert set(t.group_members("screen")) == {"screenshot", "document", "graphic",
                                              "illustration", "slide", "utility"}
    assert set(t.group_members("nature")) == {"scenery", "plant", "animal"}
    assert set(t.group_members("objects")) == {"food", "vehicle", "product", "architecture"}
    assert set(t.group_members("people")) == {"people", "event"}
    assert t.group_of("slide") == "screen"
    assert t.group_of("unknown_custom") == "unknown_custom"


def test_scenery_prompts_no_longer_cover_architecture():
    scenery = " ".join(DEFAULT_TAXONOMY.entry("scenery").prompts).lower()
    for word in ("architect", "cityscape", "city", "building", "skyline"):
        assert word not in scenery
    arch = " ".join(DEFAULT_TAXONOMY.entry("architecture").prompts).lower()
    assert "building" in arch and "city" in arch


def test_screenshot_prompts_cover_common_screens():
    text = " ".join(DEFAULT_TAXONOMY.entry("screenshot").prompts).lower()
    for phrase in ("dark mode", "code editor", "terminal", "chat", "dashboard",
                   "web page", "settings", "mobile phone"):
        assert phrase in text, phrase


def test_yaml_group_and_default_group():
    tax = parse_taxonomy([
        {"name": "food", "prompts": ["a photo of food"], "group": "meals"},
        {"name": "whiteboards", "prompts": ["a whiteboard"], "folder": "Work/Boards"},
        {"name": "invoices", "prompts": ["an invoice"], "folder": "Work/Invoices",
         "group": "paper"},
    ])
    assert tax.group_of("food") == "meals"
    assert tax.group_of("whiteboards") == "whiteboards"   # own name by default
    assert tax.group_of("invoices") == "paper"


def test_new_categories_route_to_folders(tmp_path):
    org = FolderOrganizer(CuratorConfig(), organize_root=tmp_path / "Org",
                          mode=OrganizeMode.CATEGORY_FIRST)
    for name in NEW:
        d = evaluate(_rec(quality=_good_quality(),
                          semantic=sem({name: 0.8, "scenery": 0.1, "food": 0.05})))
        assert d.category == CategoryType(name)
        target = org.full_target(_rec(quality=_good_quality(), id=1), d)
        assert FOLDERS[name] in target.as_posix()


# ---------------------------------------------------------------- gating
def test_screenshot_vs_document_split_is_a_confident_screenshot():
    d = evaluate(_rec(quality=_good_quality(), semantic=sem(
        {"screenshot": 0.30, "document": 0.28, "graphic": 0.10, "scenery": 0.05,
         "food": 0.02})))
    assert d.category == CategoryType.SCREENSHOT
    assert d.diagnostics["group"] == "screen"
    assert d.diagnostics["group_score"] == pytest.approx(0.68)


def test_old_rule_would_have_called_that_uncertain():
    s = sem({"screenshot": 0.30, "document": 0.28, "graphic": 0.10})
    assert s.score_margin < CuratorConfig().taxonomy.min_margin
    assert s.top1_score < 0.40  # also below the old top1 threshold


def test_close_call_across_groups_is_uncertain():
    d = evaluate(_rec(quality=_good_quality(), semantic=sem(
        {"food": 0.42, "scenery": 0.40, "screenshot": 0.05})))
    assert d.category == CategoryType.UNCERTAIN
    assert d.diagnostics["kind"] == "close_call"


def test_top_class_below_floor_is_uncertain_even_if_group_is_strong():
    # five look-alikes each below the 25% floor
    dist = {"screenshot": 0.22, "document": 0.21, "graphic": 0.20,
            "illustration": 0.19, "slide": 0.18}
    d = evaluate(_rec(quality=_good_quality(), semantic=sem(dist)))
    assert d.category == CategoryType.UNCERTAIN
    assert d.diagnostics["kind"] == "low_confidence"


def test_same_group_runner_up_uses_best_outside_class_for_margin():
    a = assess_semantics(
        sem({"scenery": 0.45, "plant": 0.40, "food": 0.05}),
        DEFAULT_TAXONOMY, CuratorConfig().taxonomy)
    assert a["confident"] and a["group"] == "nature"
    assert a["margin"] == pytest.approx(0.40)       # 0.45 - 0.05
    b = assess_semantics(
        sem({"scenery": 0.40, "plant": 0.10, "food": 0.38}),
        DEFAULT_TAXONOMY, CuratorConfig().taxonomy)
    assert not b["confident"] and b["kind"] == "close_call"


def test_missing_distribution_falls_back_to_old_rule():
    s = SemanticScore(model_name="t", top1_category="food", top1_score=0.85,
                      top2_category="scenery", top2_score=0.05, score_margin=0.80,
                      distribution="{}")
    assert evaluate(_rec(quality=_good_quality(), semantic=s)).category == CategoryType.FOOD
    weak = SemanticScore(model_name="t", top1_category="food", top1_score=0.30,
                         top2_category="scenery", top2_score=0.25, score_margin=0.05,
                         distribution="{}")
    assert evaluate(_rec(quality=_good_quality(), semantic=weak)).category == CategoryType.UNCERTAIN


# ----------------------------------------------- quality gate vs screens
def _dark_flat() -> QualityMetric:
    return QualityMetric(laplacian_variance=2, sharpness_norm=0.02, mean_luminance=12,
                         rms_contrast=3, black_clip_pct=0.9, white_clip_pct=0.0,
                         quality_score=0.1, is_too_dark=True, is_blurry=True)


def test_dark_flat_screenshot_is_not_an_accident():
    d = evaluate(_rec(quality=_dark_flat(), semantic=sem(
        {"screenshot": 0.62, "document": 0.10, "illustration": 0.08, "scenery": 0.02})))
    assert d.category == CategoryType.SCREENSHOT


def test_dark_photo_is_still_accidental():
    d = evaluate(_rec(quality=_dark_flat(), semantic=sem(
        {"scenery": 0.7, "food": 0.1, "screenshot": 0.02})))
    assert d.category == CategoryType.ACCIDENTAL
    assert d.reason.startswith("quality:")


def test_unsure_screen_content_that_is_dark_is_still_accidental():
    d = evaluate(_rec(quality=_dark_flat(), semantic=sem(
        {"food": 0.30, "screenshot": 0.28, "scenery": 0.27})))
    assert d.category == CategoryType.ACCIDENTAL


def test_dark_frame_without_semantics_is_still_accidental():
    assert evaluate(_rec(quality=_dark_flat())).category == CategoryType.ACCIDENTAL


# ------------------------------------------------------- explain_uncertain
def test_explain_low_confidence_text():
    e = explain_uncertain(sem({"food": 0.31, "scenery": 0.20, "people": 0.20,
                               "screenshot": 0.29}))
    assert e["kind"] == "low_confidence"
    assert e["text"] == "Best guess is Food at 31%, below the 40% needed."
    assert e["top1"] == "food" and e["top2"] == "screenshot"


def test_explain_close_call_text():
    e = explain_uncertain(sem({"food": 0.42, "scenery": 0.40, "screenshot": 0.05}))
    assert e["kind"] == "close_call"
    assert e["text"] == "Close call between Food (42%) and Scenery (40%)."
    assert e["top1_score"] == pytest.approx(0.42) and e["top2_score"] == pytest.approx(0.40)


def test_explain_close_call_names_outside_rival_when_runner_up_is_a_look_alike():
    e = explain_uncertain(sem({"scenery": 0.40, "plant": 0.10, "food": 0.38}))
    assert e["text"] == "Close call between Scenery (40%) and Food (38%)."


def test_explain_ok_and_no_data():
    ok = explain_uncertain(sem({"food": 0.80, "scenery": 0.05}))
    assert ok["kind"] == "ok"
    nd = explain_uncertain(None)
    assert nd["kind"] == "no_data" and "Analyze" in nd["text"]
    empty = explain_uncertain(SemanticScore())
    assert empty["kind"] == "no_data"


def test_explain_keys_and_diagnostics_match():
    s = sem({"food": 0.42, "scenery": 0.40, "screenshot": 0.05})
    e = explain_uncertain(s)
    assert set(e) == {"kind", "top1", "top1_score", "top2", "top2_score", "text"}
    d = evaluate(_rec(quality=_good_quality(), semantic=s))
    assert d.category == CategoryType.UNCERTAIN
    for k, v in e.items():
        assert d.diagnostics[k] == v


def test_no_semantic_data_decision_carries_no_data_diagnostics():
    d = evaluate(_rec(quality=_good_quality()))
    assert d.category == CategoryType.UNCERTAIN
    assert d.diagnostics["kind"] == "no_data"

"""Tests for the decision matrix, taxonomy and folder organizer."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from photo_curator.config import CuratorConfig
from photo_curator.db.models import FaceDetection, ImageRecord, QualityMetric, SemanticScore
from photo_curator.rules.decision_matrix import Decision, DecisionMatrix
from photo_curator.rules.organizer import FolderOrganizer, OrganizeMode, _safe_name
from photo_curator.rules.taxonomy import (
    CLIP_CATEGORIES,
    CategoryType,
    TAXONOMY_PROMPTS,
    flat_prompts,
)

T = datetime(2024, 5, 12, 15, 30, 45)


def _rec(**kw) -> ImageRecord:
    base = dict(path="/lib/IMG_x.jpg", filename="IMG_x.jpg",
                taken_at=T, taken_at_source="exif:dtoriginal")
    base.update(kw)
    return ImageRecord(**base)


def _good_quality() -> QualityMetric:
    return QualityMetric(laplacian_variance=400, sharpness_norm=0.9,
                         mean_luminance=128, rms_contrast=50,
                         black_clip_pct=0.01, white_clip_pct=0.01,
                         quality_score=0.9)


def _semantic(top1: str, top1_score: float, top2: str = "scenery",
              top2_score: float = 0.1) -> SemanticScore:
    return SemanticScore(model_name="test", top1_category=top1,
                         top1_score=top1_score, top2_category=top2,
                         top2_score=top2_score,
                         score_margin=round(top1_score - top2_score, 4),
                         distribution={top1: top1_score, top2: top2_score})


def _face(conf: float = 0.9) -> FaceDetection:
    return FaceDetection(bbox_x1=10, bbox_y1=10, bbox_x2=110, bbox_y2=110,
                         confidence=conf, landmarks="[]",
                         eye_aspect_ratio=0.25, face_sharpness=300)


def _eval(rec) -> Decision:
    return DecisionMatrix(CuratorConfig()).evaluate_record(rec)


# ---------------------------------------------------------------------------
# Decision matrix (spec 4.B)
# ---------------------------------------------------------------------------

def test_exact_duplicate_routes_to_duplicates():
    rec = _rec(is_duplicate=True, duplicate_of=1)
    d = _eval(rec)
    assert d.category == CategoryType.DUPLICATE
    assert d.reason == "exact sha256 match"


def test_burst_alternate_flag():
    rec = _rec(is_alternate=True)
    d = _eval(rec)
    assert d.is_alternate is True
    assert d.category is not None  # organizer overrides to the burst folder


def test_dark_frame_routes_to_accidental():
    rec = _rec(quality=_good_quality())
    rec.quality.is_too_dark = True
    rec.quality.quality_score = 0.05
    d = _eval(rec)
    assert d.category == CategoryType.ACCIDENTAL
    assert "too_dark" in d.reason


def test_blurry_frame_routes_to_accidental():
    q = _good_quality()
    q.is_blurry = True
    rec = _rec(quality=q)
    assert _eval(rec).category == CategoryType.ACCIDENTAL


def test_overexposed_frame_routes_to_accidental():
    q = _good_quality()
    q.is_overexposed = True
    rec = _rec(quality=q)
    d = _eval(rec)
    assert d.category == CategoryType.ACCIDENTAL
    assert d.subcategory == "overexposed"


def test_high_confidence_face_routes_to_people_with_date():
    rec = _rec(quality=_good_quality(), faces=[_face(0.92)])
    d = _eval(rec)
    assert d.category == CategoryType.PEOPLE
    assert d.needs_date is True
    assert d.confidence == pytest.approx(0.92)


def test_low_confidence_face_does_not_route_to_people():
    rec = _rec(quality=_good_quality(), faces=[_face(0.50)],
               semantic=_semantic("food", 0.8, top2_score=0.1))
    d = _eval(rec)
    assert d.category == CategoryType.FOOD


def test_clip_routes_categories(cfg):
    for cat in ("screenshot", "document", "graphic", "food", "animal", "scenery"):
        rec = _rec(quality=_good_quality(),
                   semantic=_semantic(cat, 0.85, top2_score=0.05))
        d = _eval(rec)
        assert d.category == CategoryType(cat), cat
        assert d.reason.startswith("clip top1")
    # date grouping only for the non-human world classes + people
    assert _eval(_rec(quality=_good_quality(),
                      semantic=_semantic("food", 0.85, top2_score=0.05))).needs_date
    assert not _eval(_rec(quality=_good_quality(),
                          semantic=_semantic("screenshot", 0.85, top2_score=0.05))).needs_date


def test_low_confidence_routes_to_uncertain():
    rec = _rec(quality=_good_quality(), semantic=_semantic("food", 0.30))
    d = _eval(rec)
    assert d.category == CategoryType.UNCERTAIN
    assert "low confidence" in d.reason


def test_low_margin_routes_to_uncertain():
    rec = _rec(quality=_good_quality(),
               semantic=_semantic("food", 0.55, top2_score=0.50))
    d = _eval(rec)
    assert d.category == CategoryType.UNCERTAIN
    assert "margin" in d.reason


def test_no_semantic_data_routes_to_uncertain():
    rec = _rec(quality=_good_quality())
    d = _eval(rec)
    assert d.category == CategoryType.UNCERTAIN
    assert d.reason == "no semantic data"


def test_face_beats_clip_precedence():
    """A confident face wins over a CLIP 'scenery' top-1."""
    rec = _rec(quality=_good_quality(), faces=[_face(0.95)],
               semantic=_semantic("scenery", 0.9, top2_score=0.05))
    assert _eval(rec).category == CategoryType.PEOPLE


def test_quality_beats_face_precedence():
    """A pitch-black photo with a 'detected face' is still junk."""
    q = _good_quality()
    q.is_too_dark = True
    rec = _rec(quality=q, faces=[_face(0.95)])
    assert _eval(rec).category == CategoryType.ACCIDENTAL


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def test_taxonomy_prompts_cover_all_clip_categories():
    for cat in CLIP_CATEGORIES:
        assert cat in TAXONOMY_PROMPTS
        assert len(TAXONOMY_PROMPTS[cat]) >= 1
        assert all(p.strip() for p in TAXONOMY_PROMPTS[cat])
    assert len(flat_prompts()) == len(CLIP_CATEGORIES)


def test_category_folders_match_spec():
    assert CategoryType.SCREENSHOT.folder == "Junk_Review/Screenshots"
    assert CategoryType.DOCUMENT.folder == "Junk_Review/Documents"
    assert CategoryType.FOOD.folder == "Non-Human/Food"
    assert CategoryType.ANIMAL.folder == "Non-Human/Animals"
    assert CategoryType.SCENERY.folder == "Non-Human/Scenery"
    assert CategoryType.PEOPLE.folder == "People"
    assert CategoryType.UNCERTAIN.folder == "Uncertain_Review"
    assert CategoryType.DUPLICATE.folder == "Junk_Review/Duplicates"
    assert CategoryType.UNKNOWN.folder == "Unknown_Date"


# ---------------------------------------------------------------------------
# Organizer
# ---------------------------------------------------------------------------

def test_organizer_people_date_hierachy(cfg):
    org = FolderOrganizer(cfg, organize_root=Path("/org"))
    from photo_curator.rules.decision_matrix import Decision
    rec = _rec(path="/lib/a.jpg", filename="a.jpg")
    dec = Decision(CategoryType.PEOPLE, "face", needs_date=True)
    target = org.full_target(rec, dec)
    assert target == Path("/org/People/2024/05/12/a.jpg")


def test_organizer_food_date_hierarchy(cfg):
    org = FolderOrganizer(cfg, organize_root=Path("/org"))
    from photo_curator.rules.decision_matrix import Decision
    rec = _rec(path="/lib/b.jpg", filename="b.jpg")
    dec = Decision(CategoryType.FOOD, "clip", needs_date=True)
    assert org.full_target(rec, dec) == Path("/org/Non-Human/Food/2024/05/12/b.jpg")


def test_organizer_screenshot_no_date(cfg):
    org = FolderOrganizer(cfg, organize_root=Path("/org"))
    from photo_curator.rules.decision_matrix import Decision
    rec = _rec(path="/lib/c.jpg", filename="c.jpg")
    dec = Decision(CategoryType.SCREENSHOT, "clip")
    assert org.full_target(rec, dec) == Path("/org/Junk_Review/Screenshots/c.jpg")


def test_organizer_missing_date_uses_unknown(cfg):
    org = FolderOrganizer(cfg, organize_root=Path("/org"))
    from photo_curator.rules.decision_matrix import Decision
    rec = _rec(path="/lib/d.jpg", filename="d.jpg",
               taken_at=None, taken_at_source="unknown")
    dec = Decision(CategoryType.SCENERY, "clip", needs_date=True)
    target = org.full_target(rec, dec)
    assert "Unknown_Date" in str(target)


def test_organizer_epoch_date_never_used(cfg):
    org = FolderOrganizer(cfg, organize_root=Path("/org"))
    from photo_curator.rules.decision_matrix import Decision
    rec = _rec(path="/lib/e.jpg", filename="e.jpg",
               taken_at=datetime(1970, 1, 1), taken_at_source="mtime")
    dec = Decision(CategoryType.PEOPLE, "face", needs_date=True)
    target = org.full_target(rec, dec)
    assert "1970" not in str(target)
    assert "Unknown_Date" in str(target)


def test_organizer_alternate_goes_to_burst_folder(cfg):
    from photo_curator.db.models import BurstGroup
    org = FolderOrganizer(cfg, organize_root=Path("/org"))
    bg = BurstGroup(name="Burst_20240512_0001")
    bg.id = 7
    org.set_bursts([bg])
    from photo_curator.rules.decision_matrix import Decision
    rec = _rec(path="/lib/f.jpg", filename="f.jpg", id=42, burst_id=7)
    dec = Decision(CategoryType.UNCERTAIN, "burst alternate", is_alternate=True)
    target = org.full_target(rec, dec)
    assert target == Path("/org/Bursts/Burst_20240512_0001/alternates/f.jpg")


def test_organizer_in_place_mode_uses_hardlink(cfg):
    from photo_curator.rules.decision_matrix import Decision
    org = FolderOrganizer(cfg, organize_root=Path("/org"),
                          mode=OrganizeMode.IN_PLACE)
    rec = _rec(path="/lib/g.jpg", filename="g.jpg")
    dec = Decision(CategoryType.SCREENSHOT, "clip")
    plan = org.build_destination_plan([rec], [dec], operation="move")
    assert len(plan) == 1
    assert plan[0].operation == "hardlink"


def test_safe_name_strips_unsafe_chars():
    assert _safe_name("a b/c:*.jpg") == "a_b_c_.jpg"
    assert _safe_name("...") == "unnamed"
    assert _safe_name("IMG_1234.JPG") == "IMG_1234.JPG"

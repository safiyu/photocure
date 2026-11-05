from photocure.rules.taxonomy import CategoryType
from tests.test_rules import _eval, _face, _good_quality, _rec, _semantic


def test_confident_screenshot_with_avatar_face_is_not_people():
    rec = _rec(quality=_good_quality(), faces=[_face(0.95)],
               semantic=_semantic("screenshot", 0.85, top2_score=0.05))
    assert _eval(rec).category == CategoryType.SCREENSHOT


def test_face_still_wins_over_non_screen_clip_category():
    rec = _rec(quality=_good_quality(), faces=[_face(0.95)],
               semantic=_semantic("scenery", 0.85, top2="food", top2_score=0.05))
    assert _eval(rec).category == CategoryType.PEOPLE


def test_face_still_wins_when_screenshot_is_not_confident():
    rec = _rec(quality=_good_quality(), faces=[_face(0.95)],
               semantic=_semantic("screenshot", 0.30, top2="scenery", top2_score=0.29))
    assert _eval(rec).category == CategoryType.PEOPLE


def test_camera_screenshot_prefers_document_over_scenery():
    import json
    from photocure.db.models import SemanticScore
    sem = SemanticScore(model_name="test", top1_category="screenshot", top1_score=0.5,
                        top2_category="document", top2_score=0.3, score_margin=0.2,
                        distribution=json.dumps({"screenshot": 0.5, "document": 0.3, "scenery": 0.1}))
    rec = _rec(quality=_good_quality(), semantic=sem, camera_model="Canon EOS R6")
    d = _eval(rec)
    assert d.category == CategoryType.DOCUMENT


def test_camera_screenshot_with_nothing_plausible_names_the_runner_up():
    import json
    from photocure.db.models import SemanticScore
    sem = SemanticScore(model_name="test", top1_category="screenshot", top1_score=0.6,
                        top2_category="food", top2_score=0.2, score_margin=0.4,
                        distribution=json.dumps({"screenshot": 0.6, "food": 0.2, "scenery": 0.05}))
    rec = _rec(quality=_good_quality(), semantic=sem, camera_model="Canon EOS R6")
    d = _eval(rec)
    assert d.category == CategoryType.UNCERTAIN and d.subcategory == "food"

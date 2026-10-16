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

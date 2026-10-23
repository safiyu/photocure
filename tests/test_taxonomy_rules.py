"""Rule fixes found in the taxonomy review: faces vs quality, presenters vs slides, new classes."""
from __future__ import annotations

import collections

import pytest

from photocure.rules.decision_matrix import FACE_OVERRIDDEN_BY
from photocure.rules.taxonomy import DEFAULT_TAXONOMY, TAXONOMY_PROMPTS, CategoryType

from .test_rules import _eval, _face, _good_quality, _rec, _semantic


# ----------------------------------------------------------- faces vs quality
def test_a_blurry_photo_of_a_person_stays_in_people():
    q = _good_quality()
    q.is_blurry = True
    d = _eval(_rec(quality=q, faces=[_face(0.92)]))
    assert d.category == CategoryType.PEOPLE


def test_an_overexposed_photo_of_a_person_stays_in_people():
    q = _good_quality()
    q.is_overexposed = True
    assert _eval(_rec(quality=q, faces=[_face(0.92)])).category == CategoryType.PEOPLE


def test_a_blurry_photo_without_a_face_is_still_junk():
    q = _good_quality()
    q.is_blurry = True
    d = _eval(_rec(quality=q))
    assert d.category == CategoryType.ACCIDENTAL and d.subcategory == "blurry"


def test_a_weak_face_does_not_protect_a_blurry_photo():
    q = _good_quality()
    q.is_blurry = True
    assert _eval(_rec(quality=q, faces=[_face(0.40)])).category == CategoryType.ACCIDENTAL


def test_dark_plus_blurry_with_a_face_reports_only_the_dark_verdict():
    q = _good_quality()
    q.is_too_dark = True
    q.is_blurry = True
    d = _eval(_rec(quality=q, faces=[_face(0.95)]))
    assert d.category == CategoryType.ACCIDENTAL and d.subcategory == "too_dark"


# ------------------------------------------------------ presenters vs slides
def test_a_presenter_in_front_of_a_slide_is_a_people_photo():
    rec = _rec(quality=_good_quality(), faces=[_face(0.9)],
               semantic=_semantic("slide", 0.85, top2="document", top2_score=0.05))
    assert _eval(rec).category == CategoryType.PEOPLE


@pytest.mark.parametrize("cls", sorted(FACE_OVERRIDDEN_BY))
def test_avatars_inside_flat_screen_content_do_not_make_it_a_people_photo(cls):
    rec = _rec(quality=_good_quality(), faces=[_face(0.9)],
               semantic=_semantic(cls, 0.85, top2="slide", top2_score=0.05))
    assert _eval(rec).category == CategoryType(cls)


def test_a_flat_dark_screenshot_with_a_face_is_not_junked():
    q = _good_quality()
    q.is_too_dark = True
    rec = _rec(quality=q, faces=[_face(0.9)],
               semantic=_semantic("screenshot", 0.9, top2="document", top2_score=0.02))
    assert _eval(rec).category == CategoryType.SCREENSHOT


# ----------------------------------------------------------------- prompts
def test_the_slide_prompts_describe_flat_content_not_rooms_with_people():
    text = " ".join(TAXONOMY_PROMPTS[CategoryType.SLIDE]).lower()
    assert "meeting room" not in text
    assert "close-up photo of a projector screen" in text


def test_posters_are_reference_documents_not_junk_graphics():
    assert any("poster" in p for p in TAXONOMY_PROMPTS[CategoryType.DOCUMENT])
    assert not any("poster" in p or "flyer" in p for p in TAXONOMY_PROMPTS[CategoryType.GRAPHIC])


def test_no_prompt_asks_for_an_absence():
    for prompts in TAXONOMY_PROMPTS.values():
        for p in prompts:
            assert " no people" not in p and not p.startswith("no "), p


def test_prompts_are_unique_across_classes():
    seen = collections.Counter(p for e in DEFAULT_TAXONOMY.entries for p in e.prompts)
    assert [p for p, n in seen.items() if n > 1] == []


def test_people_cover_presenters_children_and_parties():
    text = " ".join(TAXONOMY_PROMPTS[CategoryType.PEOPLE]).lower()
    for word in ("talk or presentation", "party", "child"):
        assert word in text


def test_every_class_keeps_a_balanced_prompt_ensemble():
    for e in DEFAULT_TAXONOMY.entries:
        assert 4 <= len(e.prompts) <= 6, e.key


# ------------------------------------------------------------------- events
def test_event_is_a_built_in_date_grouped_class():
    assert CategoryType.EVENT in DEFAULT_TAXONOMY.categories
    assert CategoryType.EVENT.folder == "Events"
    assert CategoryType.EVENT.is_date_grouped and not CategoryType.EVENT.is_junk
    assert DEFAULT_TAXONOMY.group_of("event") == DEFAULT_TAXONOMY.group_of("people")


def test_a_confident_event_photo_routes_to_events_with_a_date():
    rec = _rec(quality=_good_quality(),
               semantic=_semantic("event", 0.8, top2="scenery", top2_score=0.05))
    d = _eval(rec)
    assert d.category == CategoryType.EVENT and d.needs_date is True


def test_event_and_people_are_judged_as_one_group():
    """A near-tie inside the group still routes to the top class; a tie with another group is reviewed."""
    same_group = _rec(quality=_good_quality(),
                      semantic=_semantic("event", 0.45, top2="people", top2_score=0.40))
    assert _eval(same_group).category == CategoryType.EVENT
    other_group = _rec(quality=_good_quality(),
                       semantic=_semantic("event", 0.45, top2="scenery", top2_score=0.40))
    assert _eval(other_group).category == CategoryType.UNCERTAIN


def test_new_prompt_groups_cover_scenery_and_animals():
    assert any("waterfall" in p for p in TAXONOMY_PROMPTS[CategoryType.SCENERY])
    assert any("insect" in p and "fish" in p for p in TAXONOMY_PROMPTS[CategoryType.ANIMAL])
    assert any("horse" in p for p in TAXONOMY_PROMPTS[CategoryType.ANIMAL])

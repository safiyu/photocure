"""Utility / Reference: screenshots, receipts, notes, labels, signs and packaging leave Junk."""
from photocure.rules.taxonomy import DEFAULT_TAXONOMY, UTILITY_CATEGORIES, CategoryType


def test_utility_category_has_its_own_folder_and_is_not_junk():
    cat = CategoryType("utility")
    assert cat.folder == "Utility_Reference/Labels_and_Signs"
    assert not cat.is_junk and not cat.is_date_grouped


def test_the_utility_family_is_exactly_four_categories():
    assert set(UTILITY_CATEGORIES) == {CategoryType.UTILITY, CategoryType.SCREENSHOT,
                                       CategoryType.DOCUMENT, CategoryType.SLIDE}
    assert all(c.is_utility for c in UTILITY_CATEGORIES)
    assert not any(c.is_utility for c in CategoryType if c not in UTILITY_CATEGORIES)


def test_screenshots_documents_and_notes_move_under_utility_reference():
    assert CategoryType.SCREENSHOT.folder == "Utility_Reference/Screenshots"
    assert CategoryType.DOCUMENT.folder == "Utility_Reference/Documents"
    assert CategoryType.SLIDE.folder == "Utility_Reference/Notes_and_Whiteboards"
    assert not CategoryType.SCREENSHOT.is_junk and not CategoryType.DOCUMENT.is_junk


def test_real_junk_is_still_junk():
    for cat in (CategoryType.GRAPHIC, CategoryType.ACCIDENTAL, CategoryType.DUPLICATE,
                CategoryType.NEAR_DUPLICATE, CategoryType.UNREADABLE, CategoryType.JUNK_BLINKS):
        assert cat.is_junk, cat


def test_utility_prompts_cover_labels_parking_and_medicine():
    entry = next(e for e in DEFAULT_TAXONOMY.entries if e.category is CategoryType.UTILITY)
    text = " ".join(entry.prompts).lower()
    for needle in ("serial", "router", "parking", "medicine"):
        assert needle in text, needle
    assert entry.group_key == "screen"

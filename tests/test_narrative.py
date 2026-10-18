# ruff: noqa: E501
"""Event stories: zero-shot shot scale and a selection that reads like a story, not 50 medium shots."""
import numpy as np

from photocure.narrative import (
    SCALE_PROMPTS,
    SCALES,
    balance_selection,
    classify_scales,
    coverage,
)


def basis(i, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


TEXT = np.stack([basis(i) for i in range(len(SCALES))])


def test_every_scale_has_prompts_and_the_five_scales_are_fixed():
    assert SCALES == ("wide", "closeup", "action", "ambient", "medium")
    assert set(SCALE_PROMPTS) == set(SCALES)
    assert all(len(p) >= 3 and all(x.strip() for x in p) for p in SCALE_PROMPTS.values())


def test_each_image_gets_the_scale_of_its_nearest_prompt_set():
    emb = np.stack([basis(0) + 0.1 * basis(6), basis(2), basis(3) + 0.2 * basis(1), basis(1)])
    assert classify_scales(emb, TEXT) == ["wide", "action", "ambient", "closeup"]


def test_zero_vectors_and_ambiguous_images_fall_back_to_medium():
    emb = np.stack([np.zeros(8, dtype=np.float32), basis(0) + basis(1)])
    assert classify_scales(emb, TEXT, min_margin=0.1) == ["medium", "medium"]
    assert classify_scales(np.zeros((0, 8), dtype=np.float32), TEXT) == []


def pool(**counts):
    items, i = [], 0
    for scale, (n, top) in counts.items():
        for k in range(n):
            i += 1
            items.append((i, top - k * 0.001, scale))
    return items


def test_a_story_has_every_available_scale_even_when_medium_shots_score_highest():
    items = pool(medium=(50, 0.9), wide=(3, 0.5), closeup=(2, 0.5), action=(2, 0.5), ambient=(1, 0.5))
    chosen = balance_selection(items, 6)
    assert len(chosen) == 6
    assert set(coverage(chosen)) == set(SCALES)


def test_extra_slots_follow_the_mix_and_never_exceed_what_exists():
    items = pool(medium=(20, 0.9), wide=(1, 0.5), closeup=(10, 0.4), action=(10, 0.3), ambient=(10, 0.2))
    cov = coverage(balance_selection(items, 12))
    assert cov["wide"] == 1 and sum(cov.values()) == 12
    assert cov["medium"] <= 5


def test_fewer_slots_than_scales_keeps_the_heaviest_scales():
    items = pool(medium=(5, 0.5), wide=(5, 0.5), closeup=(5, 0.5), action=(5, 0.5), ambient=(5, 0.5))
    assert set(coverage(balance_selection(items, 3))) == {"medium", "wide", "closeup"}


def test_asking_for_more_than_the_pool_returns_everything_best_first():
    items = pool(medium=(2, 0.5), wide=(1, 0.9))
    chosen = balance_selection(items, 10)
    assert [c[1] for c in chosen] == sorted((c[1] for c in chosen), reverse=True) and len(chosen) == 3


def test_a_pool_of_one_scale_still_returns_the_best_n():
    items = pool(medium=(10, 0.9))
    chosen = balance_selection(items, 4)
    assert [c[0] for c in chosen] == [1, 2, 3, 4]


def test_empty_pool_and_zero_slots_are_fine():
    assert balance_selection([], 5) == [] and balance_selection(pool(wide=(3, 0.5)), 0) == []


def test_selection_is_deterministic_and_ties_prefer_the_lower_id():
    items = [(2, 0.5, "wide"), (1, 0.5, "wide"), (3, 0.5, "medium")]
    assert balance_selection(items, 2) == balance_selection(list(reversed(items)), 2)
    assert [c[0] for c in balance_selection([(2, 0.5, "wide"), (1, 0.5, "wide")], 1)] == [1]


def test_coverage_counts_only_scales_that_appear():
    assert coverage([(1, 0.5, "wide"), (2, 0.4, "wide"), (3, 0.3, "action")]) == {"wide": 2, "action": 1}

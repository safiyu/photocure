# Group Eyes-Open Scorer, Utility Vault and Event Stories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Add three features: a group burst scorer that finds the frame where everyone looks good, a Utility / Reference tab that isolates receipts, notes, labels, signs, medicine and screenshots, and event stories that balance shot scale.

**Architecture:** Three independent deliverables, each a pure module with unit tests, then an API change, then a UI change, each ending in its own commit. All three compute from data already stored (faces and eye states, category scores, OpenCLIP embeddings), so there is no schema change.

**Tech Stack:** Python (FastAPI, SQLModel, numpy), vanilla JS/CSS, pytest plus the CDP browser scenarios.

**Spec:** the user's request in conversation (three features under "Group and Relational Face Intelligence"); decisions below are mine.

## Global Constraints

- Branch `develop`, commit per task, no push, no Co-Authored-By trailer, no em dashes, one-line comments only.
- Do not break the baseline tokens, existing UI tests, or browser scenarios. No schema change (`SCHEMA_VERSION` stays 8).
- Flag, never cull: nothing here moves or deletes a file.
- Without the real CLIP model (placeholder backend) features that need text embeddings report unavailable instead of guessing.

## Review Focus

1. A burst with a single face, no faces, or one person missing from some frames must not crash or invent a "group".
2. Two different people sharing a person cluster id must not be merged into one subject when they appear in the same frame.
3. Placeholder (stub) embeddings must never produce scale tags.
4. An event with fewer photos than slots, or a scale with no photos, must still return a sensible list.
5. Photos reclassified into the new Utility category must not still be offered in Junk.

---

## Feature A: Group scorer

### Task A1: `photocure/clustering/group_shots.py`
**Files:** create `photocure/clustering/group_shots.py`, `tests/test_group_shots.py`
**Interfaces:** `FaceObs(box, eye_status, eye_closed, sharpness, person_id)`; `score_group(frames: Sequence[Sequence[FaceObs]], *, perfect_at=0.8, missing_q=0.3, tiebreak=None) -> GroupMatrix` with `status` in `not_group|perfect|composite|best_effort`, `subject_count`, `frames` (each `FrameScore(index, min_quality, mean_quality, subjects)`), `best`, `composite`.
- [ ] Tests: everyone open picks the sharpest-eyes frame as `perfect`; one blinker per frame picks the frame with the best minimum; complementary blinks give `composite` with the partner that fixes the weak subject; unfixable gives `best_effort`; single face or one frame gives `not_group`; a person missing from a frame is penalized; two people sharing a `person_id` in one frame stay separate subjects.
- [ ] Run RED, implement (track faces with `track_faces`, merge tracks by `person_id` only when frame sets are disjoint, core subjects are present in at least half the frames), run GREEN, commit.

### Task A2: API
**Files:** modify `photocure/ui/app.py` (`burst_detail`), test in `tests/test_group_api.py`
**Interfaces:** `/api/burst/{id}` gains `group: {status, subject_count, best_id, partner_id, frames: [{id, min_quality, subjects: [{state, q}]}]}`.
- [ ] Test through the existing API test fixtures with seeded faces, RED, implement, GREEN, commit.

### Task A3: UI
**Files:** modify `photocure/ui/static/app.js` (`showBurst`), `style.css`, test `tests/test_ui_group.py`
- [ ] Static tests for banner texts, badges, subject dots, styles. RED, implement, GREEN, commit.

## Feature B: Utility vault

### Task B1: taxonomy
**Files:** modify `photocure/rules/taxonomy.py`, update `tests/test_rules.py`, create `tests/test_utility_taxonomy.py`
**Interfaces:** `CategoryType.UTILITY = "utility"`, `CategoryType.is_utility` (utility, screenshot, document, slide), `UTILITY_CATEGORIES`; screenshot and document are no longer junk; folders move under `Utility_Reference/`; a UTILITY taxonomy entry with prompts for device labels and serial numbers, parking pillar and location signs, medicine bottles and packaging.
- [ ] Tests, RED, implement, GREEN, commit.

### Task B2: API
**Files:** modify `photocure/ui/app.py` (`group=utility` filter, `UTILITY_VALUES`, config flag, `POST /api/rescore`), test `tests/test_utility_api.py`
- [ ] Tests, RED, implement, GREEN, commit.

### Task B3: UI
**Files:** `index.html`, `app.js`, `style.css`, test `tests/test_ui_utility.py`, browser scenario
- [ ] Nav item and tab with kind chips (All, Screenshots, Receipts and documents, Notes and whiteboards, Labels, signs and packaging), grid, count badge, category color and label, junk filter options trimmed, "Re-check my photos" button. Tests, RED, implement, GREEN, commit.

## Feature C: Event stories

### Task C1: `photocure/narrative.py`
**Files:** create `photocure/narrative.py`, `tests/test_narrative.py`
**Interfaces:** `SCALES = ("wide","closeup","action","ambient","medium")`, `SCALE_PROMPTS: dict[str, tuple[str, ...]]`, `classify_scales(embeddings: np.ndarray, text: np.ndarray, min_margin=0.0) -> list[str]`, `balance_selection(items: Sequence[tuple[int, float, str]], n: int, mix=DEFAULT_MIX) -> list[tuple[int, float, str]]` (chronological order is the caller's job), `coverage(items) -> dict[str, int]`.
- [ ] Tests: every available scale appears when n allows; leftover slots go to the best remaining by score; n larger than the pool returns the pool; missing scales are skipped; deterministic ties. RED, implement, GREEN, commit.

### Task C2: service and API
**Files:** modify `photocure/narrative.py` (`scale_labels`), `photocure/highlights.py` (`best_of_events(..., balanced)`), `photocure/ui/app.py`, test `tests/test_story_api.py`
**Interfaces:** `scale_labels(session, image_ids, encoder) -> dict[int, str]` skipping stub rows; `/api/highlights/events?balanced=true&per_event=N` adds `scale` per item, `coverage` per event, and `balanced` plus `note` at top level.
- [ ] Tests with a fake encoder, RED, implement, GREEN, commit.

### Task C3: UI
**Files:** `index.html` (new `#hl-view` option `story`), `app.js`, `style.css`, test `tests/test_ui_story.py`
- [ ] Event stories view: more photos per event, scale badges, coverage line, unavailable note. Tests, RED, implement, GREEN, commit.

## Finish
- [ ] Full suite plus browser scenarios, lint the new files, one whole-branch review, report.

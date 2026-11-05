# Full Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every in-scope finding of the 2026-11-05 full review: file-safety bugs, API/DB defects, face-identity algorithm flaws, and vision/quality/editor issues, and make the docs describe the models that actually ship.

**Architecture:** No new subsystems. Each task edits the module that owns the defect and adds a focused pytest next to the existing tests for that module. Two small schema additions (`ActionLedger.companion_of`, `PersonCluster.yearly_json`) go through the existing `migrate()` which adds missing columns automatically. Tasks are grouped A (file safety), B (API/DB), C (faces), D (vision/docs); groups are independent of each other, tasks inside a group are ordered.

**Tech Stack:** Python 3.11, FastAPI, SQLModel/SQLite, numpy, OpenCV, open_clip, pytest (`.venv/bin/python -m pytest`).

**Spec:** `docs/superpowers/specs/2026-11-05-full-review-findings.md`

## Global Constraints

- Run tests with `.venv/bin/python -m pytest <path> -q`. The full suite must stay green: `.venv/bin/python -m pytest -q`.
- Comments: one line max, only when the why is non-obvious. No docstring paragraphs.
- Commit after every task; message style `fix(area): ...`; no Co-Authored-By line; never push.
- Do not touch `.claude/worktrees/` or `build/lib/`.
- Never change the ledger protocol order (PENDING row committed before mutation, COMMIT after).
- Threshold constants that users may tune live in `photocure/config.py`, not inline.

## Review Focus

Inputs the spec implies but which no single finding names; each has a pinned test in the owning task:

1. Two photos with the same stem and companions routed to one folder (Task A1, A2): both companions survive with distinct names; the report counts the failure if one cannot be placed.
2. Undo after the source volume is gone (Task A3): a copy-apply undo refuses and reports, never deletes.
3. `/api/reset` with `{"stage": "library"}` (Task B1): rejected with 422, nothing cleared.
4. A library with 3000 person clusters (Task C4): `list_people` completes in under 2 s on a laptop with the vectorized matcher.
5. Two faces of different people that appear together in one photo and score 0.6 (Task C5): never suggested as a merge.

---

## Group A: file safety

### Task A1: Implement companion collision handling in the engine

**Files:**
- Modify: `photocure/engine/action_engine.py:36-56` (`find_companion_files`), `:294-322` (companion loop), `:416-446` (`_prepare_target`)
- Modify: `photocure/db/models.py:254-284` (`ActionLedger`)
- Test: `tests/test_live_photos.py`

**Interfaces:**
- Produces: `ActionEngine._resolve_collision(self, target: Path) -> Path` (sequence rename `name (n).ext`, same loop as `_prepare_target` lines 442-446, no policy branch: companions never skip).
- Produces: `ActionLedger.companion_of: int | None = Field(default=None, index=True)`; companion rows get `image_id=None, companion_of=<parent entry id>`.
- Produces: `ExecutionReport.errors` gets `f"{comp}: {ce}"` and `rep.failures` is NOT incremented for companions (the photo moved); add `ExecutionReport.companion_failures: int = 0`.

- [ ] **Step 1: Write failing tests in `tests/test_live_photos.py`**

```python
def test_companion_collision_gets_sequence_name(tmp_path, engine_env):
    # A/IMG_001.HEIC+A/IMG_001.MOV and B/IMG_001.JPG+B/IMG_001.MOV both routed to out/
    ...
    rep = engine.execute(actions, organize_root=out)
    movs = sorted(p.name for p in out.rglob("*.mov"))
    assert movs == ["IMG_001 (1).mov", "IMG_001.mov"]
    assert rep.companion_failures == 0
    assert not list((tmp_path / "B").glob("*.MOV"))

def test_companion_ledger_row_has_no_image_id(tmp_path, engine_env):
    ...
    rows = session.exec(select(ActionLedger).where(ActionLedger.transaction_id == rep.transaction_id)).all()
    comp = [r for r in rows if r.source_path.endswith(".MOV")]
    assert comp and comp[0].image_id is None and comp[0].companion_of == parent.id

def test_find_companion_files_dedups_case_variants(tmp_path):
    # create IMG.jpg and IMG.MOV only; both ".mov" and ".MOV" candidates resolve to one inode on APFS
    assert len(find_companion_files(tmp_path / "IMG.jpg")) == 1
```

Use the existing fixtures in `tests/test_live_photos.py` for the engine and DB setup.

- [ ] **Step 2: Run tests, expect FAIL** (`AttributeError: _resolve_collision`, missing attribute `companion_of`).

- [ ] **Step 3: Implement**
  - `find_companion_files`: dedup by `(st_dev, st_ino)` from `os.stat`, not by path.
  - Add `_resolve_collision` as a method next to `_prepare_target`.
  - Companion loop: `repo.ledger_add(txn, None, ...)`, then `comp_entry.companion_of = entry_id`; on exception append to `rep.errors`, increment `rep.companion_failures`, keep the `log.warning`.
  - `ActionLedger.companion_of` column; `migrate()` picks it up.

- [ ] **Step 4: Run `tests/test_live_photos.py tests/test_action_engine.py tests/test_ledger_recovery.py -q`, expect PASS.**

- [ ] **Step 5: Commit** `fix(engine): place colliding companions with a sequence name and ledger them without the photo's image_id`

### Task A2: Remove the unledgered `_move_sidecar`

**Files:**
- Modify: `photocure/library_actions.py:24-40,68,97`, `photocure/trash_actions.py:321-323`, `photocure/burst_actions.py:241`
- Test: `tests/test_library_actions.py`, `tests/test_trash_actions.py`

**Interfaces:**
- Consumes: Task A1 engine companion handling.

- [ ] **Step 1: Write failing test in `tests/test_library_actions.py`**

```python
def test_move_to_category_never_overwrites_existing_companion(env):
    # out/Photos/IMG_001.mov already exists (another photo's video); moving A/IMG_001.jpg there
    before = (out / "Photos" / "IMG_001.mov").read_bytes()
    move_to_category(dbp, cfg, image_id, "people")
    assert (out / "Photos" / "IMG_001.mov").read_bytes() == before
    assert (out / "Photos" / "IMG_001 (1).mov").exists()
```

- [ ] **Step 2: Run, expect FAIL** (file overwritten).
- [ ] **Step 3: Delete `_move_sidecar` and all four call sites (grep `_move_sidecar` returns nothing).**
- [ ] **Step 4: Run `tests/test_library_actions.py tests/test_trash_actions.py tests/test_finalize_burst.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(library): drop unledgered sidecar move; the engine already moves companions safely`

### Task A3: Rollback must never delete the only copy

**Files:**
- Modify: `photocure/engine/rollback.py:210-232` (`_reverse_one`)
- Test: `tests/test_ledger_recovery.py`

- [ ] **Step 1: Write failing test**

```python
def test_undo_copy_refuses_when_source_is_gone(env):
    # apply with operation="copy", delete the source, roll back
    rep = RollbackEngine(cfg, db_path=dbp).rollback(txn)
    assert rep.failures == 1 and target.exists()
    assert "source gone" in rep.errors[0]
```

- [ ] **Step 2: Run, expect FAIL** (target deleted, failures 0).
- [ ] **Step 3: In `_reverse_one`, before the `op in ("copy", "hardlink", "symlink")` removal branch at line 229, add: `if op == "copy" and not os.path.lexists(source): return False, "source gone; the copy is the only remaining file (refusing to remove)"`.**
- [ ] **Step 4: Run `tests/test_ledger_recovery.py tests/test_undo_stages.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(rollback): refuse to delete a copied file when its source no longer exists`

### Task A4: `finalize_deletes` marks selected ids and survives partial failure

**Files:**
- Modify: `photocure/trash_actions.py:253-261,313-349`
- Test: `tests/test_trash_actions.py`

- [ ] **Step 1: Write failing tests**

```python
def test_finalize_unmarked_ids_lands_in_trash_folder(env):
    res = finalize_deletes(dbp, cfg, ids=[rec_id])  # rec not delete_approved
    rec = session.get(ImageRecord, rec_id)
    assert rec.stage == "trash" and cfg.trash_dir_name in Path(rec.path).parts

def test_finalize_partial_failure_still_finalizes_successes(env, monkeypatch):
    # make the second file unreadable so its move fails
    with pytest.raises(OSError):
        finalize_deletes(dbp, cfg, ids=[ok_id, bad_id])
    assert session.get(ImageRecord, ok_id).stage == "trash"
    assert session.get(ImageRecord, bad_id).stage != "trash"
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement**
  - When `ids` is given, set `delete_approved = True` on every selected record (commit) before building actions, replacing the `has_marked` branch.
  - After `engine.execute`, run the post-processing block for `a.image_id in rep.resolved_targets` only, then `raise OSError(...)` if `rep.failures`.

- [ ] **Step 4: Run `tests/test_trash_actions.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(trash): finalize selected ids through the trash path and keep successes on partial failure`

### Task A5: Crash recovery ignores companion rows

**Files:**
- Modify: `photocure/engine/rollback.py:120-160` (`_reconcile_pending`), `:60-99` (restore loop that calls `_restore_row`)
- Test: `tests/test_ledger_recovery.py`

**Interfaces:**
- Consumes: `ActionLedger.companion_of` from Task A1.

- [ ] **Step 1: Write failing test**

```python
def test_reconcile_pending_companion_does_not_repoint_photo_row(env):
    # leave a PENDING companion row (image_id None, companion_of=parent.id) whose .mov is on disk
    RollbackEngine(cfg, db_path=dbp).reconcile(txn)
    assert session.get(ImageRecord, image_id).path.endswith(".jpg")
```

- [ ] **Step 2: Run, expect FAIL** (passes trivially only if image_id is None already; assert the row update path is skipped when `entry.companion_of is not None` by also asserting the ledger row became `committed`).
- [ ] **Step 3: In both `_reconcile_pending` and the `_restore_row` call site, guard `if img is not None and entry.companion_of is None`.**
- [ ] **Step 4: Run `tests/test_ledger_recovery.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(rollback): companion ledger rows never update the photo row`

### Task A6: `db restore` safety copy via the sqlite backup API; sidecar flush skips moved rows

**Files:**
- Modify: `photocure/cli.py:1366-1371`, `photocure/sidecar_queue.py:72-86`
- Test: `tests/test_cli.py`, `tests/test_sidecar_sync.py`

- [ ] **Step 1: Write failing tests**

```python
def test_db_restore_keeps_wal_rows_in_pre_restore_copy(tmp_path, runner):
    # write a row, leave WAL unflushed (do not checkpoint), run restore, open .pre-restore and count rows
    assert count == 1

def test_sidecar_flush_skips_rows_whose_file_is_missing(env):
    rec.path = str(tmp_path / "gone.jpg")  # not on disk
    results = queue.flush()
    assert results[rec.id].status == "skipped"
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement**
  - `cli.py`: replace `shutil.copy2(db, keep)` with `sqlite3.connect(db).backup(sqlite3.connect(keep))` (closing both).
  - `sidecar_queue.flush`: `if not Path(rec.path).exists(): results[image_id] = SyncResult("skipped", "file moved"); continue` (use the existing `SyncResult` status vocabulary; check `sidecar_sync.py` for the skipped constant name and reuse it).

- [ ] **Step 4: Run both test files, expect PASS.**
- [ ] **Step 5: Commit** `fix(cli,sidecar): back up WAL rows before restore; never write a sidecar beside a moved file`

## Group B: API / DB

### Task B1: Validate `/api/reset` and the dict-body endpoints

**Files:**
- Modify: `photocure/ui/app.py:3003-3016` (`reset_library`), `:1378-1395` (`curate_budget`), `:1496-1513` (`post_fix_dates`), `:1915-1948` (`editor_bokeh`)
- Test: `tests/test_ui_api.py`

**Interfaces:**
- Produces Pydantic models in `app.py` next to the other request models: `ResetReq(stage: Literal["new","all"] = "all")`, `CurateBudgetReq(target_count: int | None = Field(None, ge=1), target_percent: float | None = Field(None, gt=0, le=100), event_id: int | None = None, stage: StageFilter = "new", action: Literal["preview","stage_delete","star_keepers"] = "preview")`, `FixDatesReq(ids: list[int] | None = Field(None, max_length=50000), shift_hours: float = 0, shift_minutes: float = 0, shift_seconds: float = 0, camera_model: str | None = None, ingest_takeout: bool = False, write_sidecar: bool = True)`.
- `editor_bokeh`: wrap the three `float()` calls in a helper `_form_float(value, default) -> float` that raises `HTTPException(400)` on `ValueError`, clamp `focus_x/focus_y` to [0,1] and `aperture` to [0,1].

- [ ] **Step 1: Write failing tests**

```python
def test_reset_rejects_unknown_stage_in_body(client):
    r = client.post("/api/reset", json={"stage": "library"})
    assert r.status_code == 422

def test_fix_dates_rejects_non_numeric_shift(client):
    assert client.post("/api/fix-dates", json={"shift_hours": "abc"}).status_code == 422

def test_curate_budget_rejects_bad_action(client):
    assert client.post("/api/curate/budget", json={"action": "delete_all"}).status_code == 422

def test_bokeh_rejects_non_numeric_focus(client):
    r = client.post("/api/editor/bokeh", json={"image": tiny_png_b64, "focus_x": "x"})
    assert r.status_code == 400
```

- [ ] **Step 2: Run, expect FAIL** (500 or 200).
- [ ] **Step 3: Implement the models and switch the four handlers to them; `reset_library` keeps the `stage` query param as a fallback only when no body is sent.**
- [ ] **Step 4: Run `tests/test_ui_api.py tests/test_c4_api.py tests/test_curate_budget.py tests/test_date_repair.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(api): validate reset, fix-dates, curate budget and bokeh request bodies`

### Task B2: Duplicates never point at a vanished original

**Files:**
- Modify: `photocure/scanner/file_scanner.py:679-700` (`_find_db_original`)
- Test: `tests/test_scanner.py`

- [ ] **Step 1: Write failing test**

```python
def test_renamed_file_is_not_a_duplicate_of_its_old_row(tmp_path, scanner_env):
    scan(root)                      # IMG_1.jpg scanned
    (root / "IMG_1.jpg").rename(root / "trip.jpg")
    scan(root)                      # no prune
    rec = by_filename("trip.jpg")
    assert rec.is_duplicate is False
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: In `_find_db_original`, skip any candidate whose `os.path.exists(cand.path)` is False (both the SHA hit and the quick-hash backfill loop). Apply the same check in `_find_in_library` line 674 before returning `cand`.**
- [ ] **Step 4: Run `tests/test_scanner.py tests/test_duplicates.py tests/test_watcher.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(scanner): a duplicate target must still exist on disk`

### Task B3: Short write transaction for burst clustering; `/api/bursts` without N+1

**Files:**
- Modify: `photocure/ui/app.py:2806-2882` (`do_cluster`), `:2513-2545` (`bursts`)
- Test: `tests/test_ui_api.py`

- [ ] **Step 1: Write failing tests**

```python
def test_bursts_endpoint_issues_bounded_queries(client, seeded_bursts, query_counter):
    with query_counter() as n:
        client.get("/api/bursts")
    assert n.count <= 6

def test_cluster_runs_detector_before_touching_rows(client, monkeypatch):
    # monkeypatch BurstDetector.cluster_records to assert no BurstGroup was deleted yet
```

If no `query_counter` fixture exists, add one in `tests/conftest.py` using `sqlalchemy.event.listen(engine, "before_cursor_execute", ...)`.

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement**
  - `do_cluster`: compute `results = BurstDetector(c).cluster_records(active)` and the best-shot picks into a local list first; only then null `burst_id`/flags, delete old groups, insert new ones, commit.
  - `bursts`: `names = dict(s.exec(select(ImageRecord.id, ImageRecord.filename).where(col(ImageRecord.id).in_(best_ids))).all())`.

- [ ] **Step 4: Run `tests/test_ui_api.py tests/test_clustering.py tests/test_group_api.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `perf(api): cluster bursts outside the write transaction; batch best-shot filenames`

### Task B4: Session id cookie with logout

**Files:**
- Modify: `photocure/auth.py` (`AuthManager`), `photocure/ui/app.py:700-715` (guard), `:885-900` (login response)
- Test: `tests/test_auth_c1.py`

**Interfaces:**
- Produces: `AuthManager.new_session() -> str` (random `secrets.token_urlsafe(32)`, stored in `self._sessions: set[str]`), `AuthManager.validate_session(sid: str | None) -> bool`, `AuthManager.revoke_session(sid: str) -> None`. `validate(token)` keeps validating the master token for the `x-pcur-token` header.
- Produces: `POST /auth/logout` clears the cookie and revokes the session.

- [ ] **Step 1: Write failing tests**

```python
def test_cookie_is_not_the_master_token(client, auth):
    r = client.post("/auth/login", json={"code": auth.token})
    assert r.cookies[COOKIE_NAME] != auth.token

def test_logout_revokes_cookie(client):
    client.post("/auth/logout")
    assert client.get("/api/stats").status_code == 401
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement; the guard accepts either a valid header token or a valid session cookie.**
- [ ] **Step 4: Run `tests/test_auth_c1.py tests/test_ui_api.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(auth): session cookie is a revocable id, not the persistent token`

### Task B5: XMP guard for non-UTF-8 sidecars, indexes, small server fixes

**Files:**
- Modify: `photocure/scanner/xmp.py:92,158`, `photocure/db/models.py:76,84,102,110`, `photocure/ui/app.py:731` (`exclusive`), `:738-740` (`jlog`)
- Test: `tests/test_xmp.py`, `tests/test_stage_schema.py`, `tests/test_ui_api.py`

- [ ] **Step 1: Write failing tests**

```python
def test_parse_xmp_rejects_utf16_doctype():
    data = "<!DOCTYPE x [<!ENTITY a 'b'>]><x/>".encode("utf-16")
    assert parse_xmp(data) == XmpInfo()

def test_hot_columns_are_indexed(engine):
    names = {i["name"] for i in inspect(engine).get_indexes("images")}
    assert {"ix_images_delete_approved", "ix_images_duplicate_of", "ix_images_near_dup_of", "ix_images_origin_path"} <= names

def test_busy_message_names_the_task(client, busy_progress):
    r = client.post("/api/scan", json={...})
    assert "Clustering" in r.json()["detail"]

def test_job_log_is_capped():
    job = {"log": []}
    for i in range(20000): jlog(job, str(i))
    assert len(job["log"]) <= 5000 and job["log"][-1]["msg"] == "19999"
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement**
  - `xmp.py`: add `_has_declarations(data: bytes) -> bool` that returns True when the bytes start with a UTF-16/32 BOM or contain `b"\x00"`, or contain the ASCII markers; use it in both places.
  - Models: `index=True` on the four columns.
  - `exclusive`: `label = cur.get("task") or "background task"`.
  - `jlog`: when `len(job["log"]) >= 5000`, drop the middle so 2000 head + 2999 tail remain and insert one `{"msg": "... log truncated ..."}` entry.

- [ ] **Step 4: Run the four test files, expect PASS.**
- [ ] **Step 5: Commit** `fix(server): reject non-UTF-8 XMP declarations, index hot columns, cap job logs, name the busy task`

### Task B6: Extend the mutating-route auth test list

**Files:**
- Modify: `tests/test_ui_api.py:70-88` (`MUTATING`)

- [ ] **Step 1: Add every mutating path listed in spec B.11 to `MUTATING` with a minimal valid body; add `test_reset_body_requires_auth`.**
- [ ] **Step 2: Run `tests/test_ui_api.py -q`, expect PASS (if any route is unguarded, fix it in `app.py` before committing).**
- [ ] **Step 3: Commit** `test(api): cover every mutating route in the auth test`

## Group C: face identity

### Task C1: Honour the threshold knob and tame the boosts

**Files:**
- Modify: `photocure/clustering/face_identity.py:37-98`, `photocure/people_actions.py:558,745-806,954`, `photocure/ui/app.py:2064`, `photocure/cli.py:537`, `photocure/config.py:205-210`
- Test: `tests/test_face_identity.py`, `tests/test_people_actions.py`

**Interfaces:**
- Produces: `face_identity.effective_threshold(threshold: float | None, dim: int, cfg_default_512: float = 0.35, cfg_default_128: float = 0.64) -> float` returns `threshold` when not None, else the per-dimension default. Every former `min(threshold, 0.35)` site calls it.
- Produces: `FaceSettings.cluster_join_threshold_sface: float = 0.64` in `config.py`; `cluster_join_threshold` (0.35) stays the 512-D default.
- `recluster_people_endpoint(threshold: float | None = Query(None, ge=0.0, le=1.0))`; CLI `--threshold` default `None`.
- Boosts in `recluster_people`: same burst +0.02, within 120 s +0.01, the 180-day boost removed.

- [ ] **Step 1: Write failing tests**

```python
def test_threshold_above_035_is_honoured_for_512d():
    c = ClusterCentroid(1, unit(seed=1), 3)
    face = mix(unit(seed=1), unit(seed=2), cos=0.45)   # cosine 0.45 to c
    assert assign_cluster(face, [c], threshold=0.50) is None
    assert assign_cluster(face, [c], threshold=0.40) == 1

def test_effective_threshold_defaults_by_dimension():
    assert effective_threshold(None, 512) == 0.35
    assert effective_threshold(None, 128) == 0.64
    assert effective_threshold(0.55, 512) == 0.55

def test_recluster_temporal_boost_capped(env):
    # two faces 0.33 apart, same burst: not merged at threshold 0.35 (boost 0.02 is not enough)
```

Add helpers `unit(seed) -> list[float]` (random 512-D unit vector) and `mix(a, b, cos) -> list[float]` to `tests/test_face_identity.py`; later tasks reuse them.

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement; also change `remap_person_ids` line 183 to use `effective_threshold(threshold, dim)` with the remap default `cfg.face.cluster_remap_threshold` passed by callers.**
- [ ] **Step 4: Run `tests/test_face_identity.py tests/test_people_actions.py tests/test_people_pipeline.py tests/test_c3_faces.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(faces): the identity threshold is the threshold; boosts are a small tie-break`

### Task C2: Exemplars stay near the centroid

**Files:**
- Modify: `photocure/clustering/face_identity.py:132-157` (`update_exemplars`), `photocure/people_actions.py:821-825,857-861`
- Test: `tests/test_face_identity.py`

**Interfaces:**
- `update_exemplars(current_exemplars, new_embedding, centroid: list[float], max_exemplars: int = 5, min_centroid_cosine: float = 0.45) -> list[list[float]]`: add only when `cosine(new, centroid) >= min_centroid_cosine`; when full, replace the exemplar farthest from the centroid if the new one is closer.
- `recluster_people` uses the same rule via `update_exemplars` (replace the inline block).

- [ ] **Step 1: Write failing test**

```python
def test_outlier_never_becomes_exemplar():
    cen = unit(1); far = mix(unit(1), unit(9), cos=0.20)
    assert update_exemplars([cen], far, centroid=cen) == [cen]

def test_drift_is_blocked_by_exemplar_rule():
    # chain a -> b -> c with cos(a,b)=0.5, cos(b,c)=0.5, cos(a,c)=0.1; after assigning b, c must not join
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `tests/test_face_identity.py tests/test_people_actions.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(faces): exemplars must be close to the centroid so identities cannot chain`

### Task C3: Persist yearly means; fix centroid double counting

**Files:**
- Modify: `photocure/db/models.py:288-297` (`PersonCluster`), `photocure/clustering/face_identity.py:101-120`, `photocure/inference/pipeline.py:429-438,760-783`, `photocure/people_actions.py:812-834,907-916`, `photocure/db/repository.py` (`update_person_cluster`, `create_person_cluster`)
- Test: `tests/test_face_identity.py`, `tests/test_people_pipeline.py`

**Interfaces:**
- Produces: `PersonCluster.yearly_json: str | None = Field(default=None)` storing `{"2019": {"n": 12, "v": <quantized embedding>}, ...}`.
- `update_temporal_medoid(current, time_bucket, new_embedding, counts: dict[str, int]) -> tuple[dict[str, list[float]], dict[str, int]]`: true running mean (weight 1/n), renormalized.
- `repository.update_person_cluster(pid, centroid, count, yearly: dict | None = None)` writes `yearly_json`; the pipeline loads it into `ClusterCentroid.temporal_medoids`.
- `recluster_people` line 815: `n = len(best_c["face_ids"])` only (pipeline faces are the whole population being re-clustered; existing cluster faces are not re-added).

- [ ] **Step 1: Write failing tests**

```python
def test_yearly_mean_is_a_running_mean():
    meds, counts = update_temporal_medoid({}, "2020", e1, {})
    meds, counts = update_temporal_medoid(meds, "2020", e2, counts)
    assert counts["2020"] == 2 and np.allclose(meds["2020"], unitize((np.array(e1)+np.array(e2))/2))

def test_yearly_vectors_survive_a_restart(env):
    # run the pipeline on faces from 2015 and 2020, reopen the DB, assert the loaded ClusterCentroid has both keys
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `tests/test_face_identity.py tests/test_people_pipeline.py tests/test_repository_people.py tests/test_stage_schema.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(faces): persist per-year running means; stop double counting faces in the recluster centroid`

### Task C4: Vectorize merge suggestions and recluster hot loops

**Files:**
- Modify: `photocure/people_actions.py:55-106` (`_find_merge_matches`), `:109-205` (`_arrange_by_similarity`), `:208-292` (`list_people`), `:789-802` (recluster boosts)
- Test: `tests/test_people_actions.py`

**Interfaces:**
- Produces: `_unit_centroids(clusters) -> list[tuple[M.PersonCluster, np.ndarray]]` shared by both functions.
- `_find_merge_matches(clusters, s=None, stage="all", cluster_images: dict[int, set[int]] | None = None) -> dict[int, dict]`: builds `S = V @ V.T` once per dimension group; named/unnamed masks via numpy; one `_best_face_id_for(target_id, s, stage=stage)` per distinct target.
- `_arrange_by_similarity(clusters) -> list[M.PersonCluster]` returns ordering only; no match info, no DB.
- `list_people` computes `has_any_face` once and passes `stage` into `_find_merge_matches`.
- `recluster_people`: precompute `cluster_bursts: set` per cluster and a sorted `np.ndarray` of timestamps per cluster, updated on assignment; nearest timestamp via `np.searchsorted`.

- [ ] **Step 1: Write failing tests**

```python
def test_find_merge_matches_is_vectorized(monkeypatch):
    calls = []; monkeypatch.setattr(np, "dot", lambda *a, **k: calls.append(1) or _real_dot(*a, **k))
    _find_merge_matches(make_clusters(300))
    assert len(calls) <= 5

def test_list_people_3000_clusters_under_2s(env_with_clusters(3000)):
    t = time.perf_counter(); list_people(dbp); assert time.perf_counter() - t < 2.0

def test_arrange_by_similarity_returns_same_order_as_before(env):  # pin current ordering on a small fixture
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `tests/test_people_actions.py tests/test_ui_people.py tests/test_fix_people_bar.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `perf(people): one matmul for merge suggestions; recluster boosts without per-face scans`

### Task C5: Merge suggestions respect co-occurrence and pick the best target

**Files:**
- Modify: `photocure/people_actions.py` (`_find_merge_matches` from Task C4, `list_people`)
- Test: `tests/test_people_actions.py`

**Interfaces:**
- Consumes: `_find_merge_matches(..., cluster_images=...)` from Task C4; `list_people` loads `cluster_images` with one `select(FaceDetection.person_id, FaceDetection.image_id)` query.
- Rules: skip pairs whose image sets intersect; a named target is chosen only when its score is the best overall or within 0.05 of the best; targets with `face_count <= 0` are never offered; response includes `"shared_photos": 0` (kept for the UI to show the reason later).

- [ ] **Step 1: Write failing tests**

```python
def test_cooccurring_clusters_are_never_suggested(env):
    # A and B both have a face in image 7, cosine 0.6 -> no pill
def test_named_target_loses_to_much_stronger_unnamed(env):
    # cand: 0.36 to named Alice, 0.92 to unnamed big cluster -> target is the unnamed cluster
def test_named_target_within_margin_wins(env):
    # 0.60 to Alice, 0.63 to unnamed -> Alice
def test_empty_named_cluster_is_not_a_target(env)
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `tests/test_people_actions.py tests/test_ui_people.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(people): merge pills skip co-occurring people and point at the strongest match`

### Task C6: Deterministic reassignment sweep in `recluster_people`

**Files:**
- Modify: `photocure/people_actions.py:734-740,882-887`
- Test: `tests/test_people_actions.py`

**Interfaces:**
- Sort key for both passes: `(-(face_sharpness or 0.0), f.id)`.
- After pass 2, one sweep: for each face, best final centroid (cosine, veto respected, no boosts, no exemplars) at `effective_threshold`; move the face when the best cluster differs and the score beats its current cluster by >= 0.02; recompute centroids once after the sweep.

- [ ] **Step 1: Write failing test**

```python
def test_recluster_is_order_invariant(env):
    # 3 identities x 20 faces, within cos ~0.55, between ~0.1; shuffle insertion order twice
    assert partition(run(order_a)) == partition(run(order_b))

def test_recluster_purity_on_synthetic_identities(env):
    # purity >= 0.98 and completeness >= 0.95 on the same fixture, plus 5 co-occurring pairs never merged
```

- [ ] **Step 2: Run, expect FAIL** (or flaky across orders).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `tests/test_people_actions.py -q`, expect PASS three times in a row.**
- [ ] **Step 5: Commit** `fix(people): deterministic order and a reassignment sweep after greedy clustering`

### Task C7: Face engine geometry gate, alignment and input name

**Files:**
- Modify: `photocure/inference/face_engine.py:55-90` (`is_valid_face_geometry`), `:100-106` (`align_face_112`), `:250-262` (`_detect_one`), `:355-372` (embedding gate)
- Modify: `photocure/scanner/phash.py` (delete `dhash_from_gray`, `compute_dhash`)
- Test: `tests/test_c3_faces.py`, `tests/test_phash.py`

**Interfaces:**
- Split: `is_plausible_box(face, w, h) -> bool` (box size, aspect 0.5-2.2, eye distance >= 12% of box, mouth >= 8%) used in `_detect_one`; `is_frontal_enough(face, w, h) -> bool` (nose projection, eye/mouth ratio >= 0.40) used only before embedding.
- `_detect_one` keeps unclipped landmarks in `FaceInfo.landmarks_raw` for alignment; `landmarks` stays clipped for storage.
- `align_face_112` uses `cv2.estimateAffinePartial2D(dst, ARCFACE_REF_5PTS, method=cv2.LMEDS)`.
- AuraFace input name from `self._aura_sess.get_inputs()[0].name`.

- [ ] **Step 1: Write failing tests**

```python
def test_profile_face_is_detected_but_not_embedded(synthetic_profile_face):
    faces = engine.detect(img); assert len(faces) == 1
    engine.embed(img, faces); assert faces[0].embedding is None

def test_edge_face_alignment_uses_unclipped_landmarks(): ...
def test_dhash_helpers_removed():
    import photocure.scanner.phash as m; assert not hasattr(m, "compute_dhash")
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `tests/test_c3_faces.py tests/test_phash.py tests/test_group_shots.py tests/test_composite.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(faces): keep profile faces for blink/group logic; align with unclipped landmarks`

## Group D: vision, quality, editor, docs

### Task D1: Docs and manifest describe the shipped models

**Files:**
- Modify: `README.md:32-37`, `photocure/models/MANIFEST.json:91-96`, `NOTICE` item 21, `photocure/inference/semantic_engine.py:168-185,204`, `photocure/models/__init__.py:56-68`, `photocure/editor_actions.py:726-730`
- Test: `tests/test_ui_about.py`

- [ ] **Step 1: Write failing tests**

```python
def test_readme_does_not_claim_unshipped_models():
    text = Path("README.md").read_text()
    for word in ("SigLIP", "TOPIQ", "LIQE", "multilingual"):
        assert word not in text

def test_manifest_has_no_siglip_entry(): ...
def test_model_path_has_no_developer_paths():
    assert "/home/safiyu" not in Path("photocure/models/__init__.py").read_text()
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement**
  - README line 33 becomes: `- **CLIP Vision-Language AI:** Zero-shot categorization and natural language photo search (English) with OpenAI CLIP ViT-B/16, with prompt ensembling and learned corrections.`
  - README line 37 becomes: `- **Joint Technical & Aesthetic Quality:** Laplacian sharpness, re-blur and motion checks combined with a LAION aesthetic predictor so blurred shots never crowd Highlights.`
  - README line 32: keep AuraFace; replace "chronological medoid growth chaining" with "per-year identity averaging".
  - Remove the SigLIP entry from MANIFEST `downloaded` and NOTICE item 21 (renumber the following items).
  - Delete the `naflex` parameter and branch in `_bgr_to_tensor`; call sites pass only `(b, self._preprocess)`.
  - Remove `Path("/home/safiyu/...")` from `model_path`; remove the generic `depth_anything.onnx` candidate.

- [ ] **Step 4: Run `tests/test_ui_about.py tests/test_inference.py tests/test_editor_actions.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `docs(models): describe the CLIP and quality stack that actually ships; drop dead naflex path`

### Task D2: Budget curation scale balancing actually runs

**Files:**
- Modify: `photocure/curate_budget.py:100-109`
- Test: `tests/test_curate_budget.py`

- [ ] **Step 1: Write failing test**

```python
def test_budget_uses_scale_labels(session, monkeypatch):
    monkeypatch.setattr("photocure.curate_budget.scale_labels", lambda s, ids, enc: {i: ("wide" if i % 2 else "close") for i in ids})
    res = budget_curate(session, target_count=4, stage="new")
    assert set(res["scale_coverage"]) == {"wide", "close"}
```

- [ ] **Step 2: Run, expect FAIL** (coverage is `{"medium": 4}`).
- [ ] **Step 3: Replace the import with `from .search import default_encoder` and `encoder = default_encoder(cfg)`; narrow the `except` to `(ImportError, RuntimeError, OSError)`.**
- [ ] **Step 4: Run `tests/test_curate_budget.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(curate): scale balancing was silently disabled by a bad import`

### Task D3: Highlights use technical quality and avoid near-duplicates

**Files:**
- Modify: `photocure/highlights.py:76-89,103-104,167-175`
- Test: `tests/test_highlights.py`

**Interfaces:**
- `highlight_expr(w)` multiplies the aesthetic term by `max(0.2, min(1.0, 0.35 + 0.65 * coalesce(QualityMetric.quality_score, 1.0)))` (same formula as `highlight_score`); `_joined` outer-joins `QualityMetric`.
- `top_of_year(session, w, year, n=50, max_cosine: float = 0.90)`: fetch `3n` ranked candidates with embeddings, greedy keep while cosine to every kept < `max_cosine`, stop at `n`.

- [ ] **Step 1: Write failing tests**

```python
def test_blurry_frame_ranks_below_sharp_one_with_equal_aesthetic(session): ...
def test_top_of_year_skips_near_identical_retakes(session):
    # 5 photos with identical embeddings and one different: top_of_year(n=2) returns one of the clones plus the different one
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `tests/test_highlights.py tests/test_story_api.py tests/test_keywords.py tests/test_insights.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(highlights): temper rankings by technical quality and skip near-duplicate retakes`

### Task D4: Calibrate learned corrections and the camera-model guard

**Files:**
- Modify: `photocure/inference/semantic_engine.py:429-433`, `photocure/config.py:227-231`, `photocure/rules/decision_matrix.py:258-272`
- Test: `tests/test_feedback.py`, `tests/test_fix_screenshot_face.py`

**Interfaces:**
- Config defaults: `feedback_min_examples = 3`, `feedback_min_cosine = 0.70`, new `feedback_max_bonus: float = 3.0` (logit units).
- Bonus: `logits[ci] += min(tcfg.feedback_max_bonus, weight * 100.0 * (sim - min_cosine))`.
- Camera guard: first candidate is the best class in `SCREEN_GROUP` other than `screenshot` (document/slide) if `>= min_class_floor`; else best outside the group; else `UNCERTAIN` with `subcategory=<name of the best non-screenshot class>`.

- [ ] **Step 1: Write failing tests**

```python
def test_feedback_bonus_is_capped(): ...  # two corrections, sim 0.95 -> bonus exactly 3.0
def test_camera_screenshot_prefers_document_over_scenery(): ...  # dist: screenshot .5, document .3, scenery .1
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `tests/test_feedback.py tests/test_learn.py tests/test_fix_screenshot_face.py tests/test_rules.py tests/test_taxonomy_rules.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(semantic): cap correction bonus; camera guard prefers document/slide over scenery`

### Task D5: Blur constants in config

**Files:**
- Modify: `photocure/quality/blur_detector.py:126-133`, `photocure/config.py` (`QualityThresholds`)
- Test: `tests/test_quality.py`

- [ ] **Step 1: Write failing test** `test_motion_rule_thresholds_come_from_config` (set `motion_anisotropy_min=0.99` and assert a streaky synthetic frame is not flagged).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Add `motion_anisotropy_min: float = 0.70`, `motion_retained_min: float = 0.48` to `QualityThresholds`; use them.**
- [ ] **Step 4: Run `tests/test_quality.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `refactor(quality): motion-blur thresholds are configurable`

### Task D6: MI-GAN crops around the mask; depth keeps aspect

**Files:**
- Modify: `photocure/editor_actions.py:685-707` (`inpaint_image`), depth preprocessing near `:744-770`
- Test: `tests/test_editor_actions.py`

**Interfaces:**
- `_inpaint_window(mask_bin: np.ndarray, h: int, w: int, pad: int = 32) -> tuple[int, int, int, int]`: square bbox around the dilated mask (`cv2.dilate`, 9x9), at least 512 px, clamped to the image.
- MI-GAN runs on the window resized to 512; output is resized back to the window size and pasted only where the dilated mask is set, feathered with a 15x15 blur.
- `_get_migan_session()` asserts two inputs with 3 and 1 channels, else returns None with a warning.
- Depth: resize so the short side is 518 and both sides are multiples of 14; resize the depth map back to the original size.

- [ ] **Step 1: Write failing tests**

```python
def test_inpaint_only_changes_pixels_near_mask(fake_migan_session): ...  # pixels > 60 px from the mask are byte-identical
def test_depth_input_keeps_aspect(fake_depth_session): ...  # 3000x2000 image -> input 777x518 rounded to 770x518
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `tests/test_editor_actions.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `fix(editor): inpaint a window around the mask at full detail; depth input keeps aspect ratio`

### Task D7: Cached embedding matrix for search and similarity

**Files:**
- Create: `photocure/inference/embedding_cache.py`
- Modify: `photocure/search.py:85-111` (`_scan`), `photocure/clustering/similarity.py` (`top_k`), `photocure/inference/semantic_engine.py` (`_write_result` callers commit), `photocure/db/repository.py` (`clear_library`)
- Test: `tests/test_search.py`

**Interfaces:**
- `embedding_cache.matrix(session) -> tuple[np.ndarray, np.ndarray]` returns `(ids int64 (N,), vecs float16 (N, D))` for all `SemanticScore` rows with non-stub embeddings; keyed per engine URL; validated against `(count(*), max(id), max(updated_at))` on `SemanticScore` (add `updated_at: datetime = Field(default_factory=utcnow)` to `SemanticScore` if missing and set it in `_write_result`).
- `embedding_cache.invalidate() -> None` called by `clear_library`.
- `_scan` and `top_k` apply their SQL filters by fetching the eligible id set and masking the cached matrix.

- [ ] **Step 1: Write failing tests**

```python
def test_second_search_does_not_decode_json(session, monkeypatch): ...  # count dequantize_embedding calls == 0 on the second call
def test_cache_refreshes_after_reanalysis(session): ...
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `tests/test_search.py tests/test_search_ranked.py tests/test_similarity.py tests/test_duplicates.py -q`, expect PASS.**
- [ ] **Step 5: Commit** `perf(search): keep a float16 embedding matrix in memory instead of decoding JSON per query`

## Group E: housekeeping and final verification

### Task E1: Remove the stale build artifact and run everything

- [ ] **Step 1: `rm -rf build/` (gitignored setuptools output; `git status` must stay clean afterwards).**
- [ ] **Step 2: Run `.venv/bin/python -m pytest -q`; expected: all pass.**
- [ ] **Step 3: Run `ruff check photocure tests` if ruff is in the venv; fix anything new.**
- [ ] **Step 4: Commit any leftover test fixes** `chore: full-suite green after review fixes`

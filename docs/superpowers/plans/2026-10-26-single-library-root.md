# Single Library Root Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One library root set in Settings, an explicit three-stage model (New, Library, Trash) across data, API and UI, AI reruns only in Trash, and a Trash stage with restore, return-to-origin and permanent delete.

**Architecture:** Add `stage` and `origin_path` columns and make `stage` the single source of truth for visibility; every endpoint that filtered on `route_status == "executed"` filters on `stage` instead. All filesystem changes (apply, manual category move, manual delete, restore, return) go through the existing `ActionEngine.execute` so the ledger, collisions and undo keep working. New modules `photocure/library.py` (root resolution and validation), `photocure/library_actions.py` (ledger-backed single-photo moves), `photocure/trash_actions.py` (Trash stage operations) and `photocure/library_migration.py` keep `ui/app.py` from growing further.

**Tech Stack:** Python 3.11, FastAPI, SQLModel/SQLite, Typer CLI, vanilla JS + CSS in `photocure/ui/static/`, pytest with the stub backends from `tests/conftest.py`.

**Spec:** `docs/superpowers/specs/2026-10-26-single-library-root-design.md`

## Global Constraints

- Stage values are exactly `"new"`, `"library"`, `"trash"`; person cluster stages are `"library"`, `"trash"`.
- Only `move` and `copy` are accepted operations anywhere (API, CLI, engine). `hardlink`, `symlink` and `in_place` are rejected.
- The trash folder is `<library_root>/.trash/<txn>/`. No shared `manual/` folder, no retention timer.
- A row with `stage = "library"` or `"trash"` always has `path` inside the library root.
- The AI never runs on `stage = "library"` rows, including via the CLI.
- No overwrite, ever: collisions use the configured `collision_policy`; return-to-origin skips an occupied origin.
- Comments one line; commit messages short, no co-author line; nothing is pushed.
- `PCUR_LIBRARY_ROOT` (env) and the settings file both feed `library_root`; settings file wins.

## Review Focus

1. Move apply where the library root is on another volume: cross-device fallback copies then unlinks; the row must still end `stage = "library"` with `path` under the root. Pinned in Task 4.
2. Restore when `<root>/<category>/<filename>` already exists: the configured collision policy applies and the existing file is untouched. Pinned in Task 9.
3. Trash reanalyse when a trashed file was removed from disk by hand: the row is marked `scan_status = "missing"` and the run continues. Pinned in Task 7.
4. Manual category change in Library when the file is missing on disk: 409 and no database change. Pinned in Task 8.
5. Any Trash operation while another job holds `_BUSY`: 409, nothing touched. Pinned in Task 9.

---

### Task 1: Schema: stage, origin_path, person stage

**Files:**
- Modify: `photocure/db/models.py:28-125` (ImageRecord), `:270-277` (PersonCluster)
- Modify: `photocure/db/session.py:71` (SCHEMA_VERSION), `:112-137` (migrate)
- Modify: `photocure/db/repository.py:508-537` (stats)
- Test: `tests/test_stage_schema.py`

**Interfaces:**
- Produces: `ImageRecord.stage: str` (default `"new"`, indexed), `ImageRecord.origin_path: str | None`, `PersonCluster.stage: str` (default `"library"`, indexed), module constants `STAGE_NEW = "new"`, `STAGE_LIBRARY = "library"`, `STAGE_TRASH = "trash"`, `STAGES = (STAGE_NEW, STAGE_LIBRARY, STAGE_TRASH)` in `photocure/db/models.py`.
- Produces: `PhotoRepository.stats()` gains `"stages": {"new": int, "library": int, "trash": int}` counting rows with `scan_status != "missing"`.
- Produces: `migrate()` backfills `images.stage` from `route_status` (`executed -> library`, `trashed -> trash`, else `new`) and `images.origin_path` from `path` for `library` rows that have no `origin_path`, only when the column was just added.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stage_schema.py
def test_new_image_defaults_to_stage_new(session): ...  # ImageRecord(path=..., filename=...).stage == "new"
def test_person_cluster_defaults_to_library(session): ...
def test_migrate_backfills_stage_from_route_status(tmp_path):
    # create a v10 db by hand with sqlite3: images rows with route_status executed / trashed / unrouted,
    # call init_db, assert stage == library / trash / new and origin_path == path for the library row
def test_stats_counts_per_stage(repo): ...  # 2 new, 1 library, 1 trash -> stats()["stages"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_stage_schema.py -v`
Expected: FAIL, `stage` attribute missing.

- [ ] **Step 3: Add the columns and constants, bump `SCHEMA_VERSION` to 11, extend `migrate()` with the backfill (run the two UPDATE statements only when `stage` was in the columns added this call)**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_stage_schema.py tests/test_c2_engine.py::test_older_database_gains_the_new_columns tests/test_c4_api.py::test_an_older_database_gains_geo_columns_and_every_declared_index -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add photocure/db tests/test_stage_schema.py
git commit -m "feat(db): stage and origin_path columns with backfill"
```

---

### Task 2: Library root service and settings

**Files:**
- Create: `photocure/library.py`
- Modify: `photocure/services.py:81-83` (SETTINGS_KEYS), `:92-125` (_validate_settings), `:145-160` (save_settings), `:162-178` (apply_settings), `:180-200` (effective_settings)
- Modify: `photocure/config.py:256-263` (drop `trash_retention_days`, keep `trash_dir_name`, `library_root`)
- Modify: `photocure/ui/app.py:295-350` (move validation out), `:888-898` (/api/config)
- Modify: `Dockerfile:41` (`PCUR_DEFAULT_OUTPUT` -> `PCUR_LIBRARY_ROOT=/output/Organized`)
- Modify: `.gitignore` (add `.superpowers/`)
- Test: `tests/test_library_root.py`

**Interfaces:**
- Produces, in `photocure/library.py`:
  - `class LibraryRootMissing(ValueError)`
  - `def library_root(dbp: Path, cfg: CuratorConfig | None = None) -> Path | None` (settings file, else `cfg.library_root` from env, else None)
  - `def library_root_source(dbp: Path, cfg: CuratorConfig | None = None) -> Literal["settings", "environment", "none"]`
  - `def require_library_root(dbp: Path, cfg: CuratorConfig | None = None) -> Path` raises `LibraryRootMissing("library folder not set")`
  - `def trash_dir(root: Path) -> Path` returns `root / ".trash"`
  - `def validate_library_root(raw: str, dbp: Path, *, browse_roots: list[Path] | None = None) -> Path` raises `ValueError` with the same messages `validate_output_dir` uses today, minus the in-library check; `browse_roots` enforces confinement when given
  - `def category_folder(cfg: CuratorConfig, category: str) -> str` (the taxonomy mapping `reassign` uses today: `folder_override_of` else `CategoryType.folder` else the value)
  - `def library_target(cfg: CuratorConfig, root: Path, rec: ImageRecord, category: str) -> Path` returns `root / category_folder(...) / rec.filename`
- Produces: `SETTINGS_KEYS` gains `"default_operation"` (`"move" | "copy"`, default `"move"`); `save_settings` raises `ValueError("library folder is already set; use Move library to change it")` when `library_root` is stored and a different value arrives, unless called with `force=True`.
- Produces: `/api/config` returns `library_root: str | None`, `library_root_source`, `default_operation`; `default_output_dir` is gone.
- `validate_output_dir` in `ui/app.py` becomes a thin wrapper used only by clean export: it calls `validate_library_root` then rejects a path inside the library root with 400.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_library_root.py
def test_settings_file_beats_environment(tmp_db, monkeypatch): ...  # PCUR_LIBRARY_ROOT set, save_settings library_root -> settings wins, source == "settings"
def test_environment_used_when_no_setting(tmp_db, monkeypatch): ...  # source == "environment"
def test_require_raises_when_unset(tmp_db): ...  # LibraryRootMissing
def test_library_root_cannot_change_once_set(tmp_db): ...  # second save_settings with other path raises ValueError; force=True succeeds
def test_default_operation_validated(tmp_db): ...  # "copy" ok, "hardlink" ValueError
def test_config_endpoint_reports_root_and_source(tmp_path): ...  # TestClient; keys present, no default_output_dir
def test_trash_retention_setting_is_gone(): ...  # not hasattr(CuratorConfig(), "trash_retention_days")
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_library_root.py -v`
Expected: FAIL, import error on `photocure.library`.

- [ ] **Step 3: Implement `photocure/library.py`, the settings changes, the `/api/config` change, the Dockerfile env var, and `.gitignore`**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_library_root.py tests/test_ui_api.py::test_config_endpoint tests/test_ui_api.py::test_clean_export_refuses_library_folder -v`
Expected: PASS (update `test_config_endpoint` to the new keys)

- [ ] **Step 5: Commit**

```bash
git add photocure/library.py photocure/services.py photocure/config.py photocure/ui/app.py Dockerfile .gitignore tests
git commit -m "feat: single library root setting with source and lock"
```

---

### Task 3: Plan API and CLI read the root; move or copy only

**Files:**
- Modify: `photocure/ui/app.py:118-124` (PlanReq), `:821-828` (build), `:2506-2520` (rebuild, diff), `:2574-2598` (readiness), `:2695-2700` (apply)
- Modify: `photocure/cli.py:534-580` (plan), `:586-672` (apply)
- Modify: `photocure/engine/action_engine.py:385-399` (_apply rejects hardlink/symlink)
- Modify: `photocure/services.py:229-234` (_effective_op), `:377-392` (build_plan default root)
- Modify: `photocure/ui/static/index.html:525-562` (plan form)
- Modify: `photocure/ui/static/app.js:4877-4883` (planBody), `:5844` (orgroot default)
- Test: `tests/test_plan_root.py`, update `tests/test_ui_api.py` (`plan_body`, `test_output_dir_rejected`, `test_output_dir_inside_library_rejected_unless_in_place`, `test_plan_rebuild_and_diff`, `test_apply_*`), `tests/test_cli.py::test_cli_apply_in_place_keeps_originals` (delete), `tests/test_c2_engine.py` hardlink tests (delete), `tests/test_fix_engine_review.py::test_reapply_hardlink_in_place_is_idempotent` (delete)

**Interfaces:**
- Consumes: `require_library_root`, `LibraryRootMissing`, `library_root` from Task 2.
- Produces: `class PlanReq(BaseModel): operation: Literal["move", "copy"] = "move"; only_category: str | None = None; trash: bool = True; include_duplicates: bool = True`.
- Produces: `/api/plan/rebuild`, `/api/plan/diff`, `/api/plan/apply` respond `409 {"detail": "library folder not set"}` when `require_library_root` raises; the diff response keeps `organize_root`.
- Produces: `/api/plan/readiness` adds `"library_root_set": bool` and the issue string `"library folder not set"`.
- Produces: `Planner.build_plan(organize_root=None, ...)` defaults to `library_root(self.db_path, self.cfg)` and raises `LibraryRootMissing` when none; `mode` parameter stays but only `CATEGORY_FIRST` is accepted.
- Produces: CLI `plan`, `apply`, `trash-purge` lose `--output-dir` and `--mode`, gain `--library-root: Path | None` override; `--operation` accepts `move|copy` only (exit 2 otherwise).
- Produces: `ActionEngine._apply` raises `ValueError("unsupported operation")` for anything but move or copy.
- Plan form keeps `#op` (move, copy), `#onlycat`, `#trash`; `#orgmode`, `#orgroot`, `#btn-browse-out` and their help text are removed; a `<p class="help" id="copy-hint" hidden>` under `#op` reads: "Originals stay where they are, including the ones marked for delete. A copy of those goes to the trash folder." and is shown when copy is selected.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plan_root.py
def test_plan_rebuild_409_without_root(env_no_root): ...  # status 409, detail == "library folder not set"
def test_plan_apply_moves_into_library_root(env): ...  # set root via /api/settings, apply, every executed row path startswith root
def test_hardlink_rejected_by_api(env): ...  # 422
def test_readiness_reports_missing_root(env_no_root): ...  # library_root_set False, issue present
def test_cli_plan_uses_setting(tmp_path): ...  # CliRunner: photocure plan with no --output-dir after settings saved
def test_cli_rejects_symlink_operation(tmp_path): ...  # exit code 2
def test_engine_rejects_hardlink(tmp_path): ...  # ActionEngine.execute with hardlink -> failures == 1, error contains "unsupported operation"
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_plan_root.py -v`
Expected: FAIL (422 on missing output_dir, options still exist).

- [ ] **Step 3: Implement the API, planner, engine, CLI and form changes; delete the hardlink, symlink and in_place tests listed above; update `plan_body` in `tests/test_ui_api.py` to `{"operation": "move", **kw}` and make the `env` fixture save `library_root` via `/api/settings`**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_plan_root.py tests/test_ui_api.py tests/test_cli.py tests/test_c2_engine.py tests/test_fix_engine_review.py tests/test_plans.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add photocure tests
git commit -m "feat(plan): destination comes from the library root, move or copy only"
```

---

### Task 4: Apply and undo set stages; copy repoints and trashes junk

**Files:**
- Modify: `photocure/engine/action_engine.py:196-208` (trash branch), `:249-262` (row update)
- Modify: `photocure/engine/rollback.py:107-145` (_reconcile_pending), `:58-105` (_rollback_transaction row update)
- Test: `tests/test_stage_transitions.py`

**Interfaces:**
- Consumes: `STAGE_*` constants, `trash_dir`.
- Produces: after a committed action the engine sets, for both move and copy: `img.origin_path = img.origin_path or str(action.source)`, `img.path = str(target)`, `img.relative_path = _rel(target, root)`, `img.target_path = str(target)`; `img.stage = "trash"` when `marked and trash`, else `"library"`; `img.route_status` unchanged in meaning (`"executed"`, or `"trashed"` for trash). `img.scan_status` stays `"scanned"` for trashed rows (today it is set to `"missing"`), so thumbnails and faces still render in the Trash stage; the same in `_reconcile_pending`.
- Produces: `marked and trash` applies to `copy` as well: the junk file is copied to `_trash_target(...)`; `marked_left_out` only counts when `trash` is off.
- Produces: `RollbackEngine` sets `img.stage = "new"`, `img.route_status = "unrouted"`, `img.path = entry.source_path`, `img.origin_path = None` for every reversed entry (move and copy alike), and the same mapping in `_reconcile_pending` for recovered commits (`stage` library or trash).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stage_transitions.py  (fixtures: scan 4 images, mark one delete_approved, root = tmp_path/"Library")
def test_move_apply_sets_library_stage_and_origin(...): ...  # stage library, origin_path == old path, path under root
def test_copy_apply_repoints_row_to_library_copy(...): ...  # source still exists, rec.path under root, stage library
def test_copy_with_trash_copies_junk_into_trash(...): ...  # marked row: file at root/.trash/<txn>/..., stage trash, scan_status scanned, source still exists, marked_left_out == 0
def test_move_without_trash_leaves_marked_in_new(...): ...  # trash=False: stage new, marked_left_out == 1
def test_undo_returns_rows_to_new(...): ...  # after rollback: stage new, path == source, origin_path None; copy target removed
def test_cross_device_move_still_lands_in_library(monkeypatch, ...): ...  # force copy fallback like test_c2_engine does; stage library, path under root
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_stage_transitions.py -v`
Expected: FAIL (stage stays new, copy keeps source path).

- [ ] **Step 3: Implement the engine and rollback changes**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_stage_transitions.py tests/test_action_engine.py tests/test_ledger_recovery.py tests/test_fix_engine_review.py tests/test_ui_api.py::test_move_to_output_dir_and_trash_index_removal_with_undo -v`
Expected: PASS (rename that last test to drop `output_dir` from its name and body)

- [ ] **Step 5: Commit**

```bash
git add photocure/engine tests
git commit -m "feat(engine): apply and undo drive the photo stage, copy trashes junk too"
```

---

### Task 5: Rescan skips photos already in Library or Trash

**Files:**
- Modify: `photocure/scanner/file_scanner.py:590-640` (_resolve_dedup, _find_db_original)
- Modify: `photocure/db/repository.py:108-117` (known_quick_hashes)
- Test: `tests/test_scan_skip_library.py`

**Interfaces:**
- Produces: `ScanResult.skipped_library: int`; a file whose sha256 matches a row with `stage in ("library", "trash")` is not inserted and increments `skipped_library`. `known_quick_hashes()` keeps returning hashes of library and trash rows so the lazy sha256 path triggers.
- Produces: `/api/scan` response includes `skipped_library`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scan_skip_library.py
def test_rescan_of_copied_source_adds_no_row(...): ...  # scan, copy-apply, scan same folder again -> total rows unchanged, skipped_library == n
def test_rescan_of_trashed_copy_adds_no_row(...): ...  # copy-apply with trash on marked row, rescan -> no new row for it
def test_different_file_with_same_name_is_still_new(...): ...  # content differs -> new row
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_scan_skip_library.py -v`
Expected: FAIL (duplicate row created).

- [ ] **Step 3: Implement the skip in `_resolve_dedup` (hash when quick hash is known; if the sha256 match has `stage != "new"`, drop the record and count it)**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_scan_skip_library.py tests/test_scanner.py tests/test_fix_scanner_review.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add photocure/scanner photocure/db/repository.py photocure/ui/app.py tests
git commit -m "feat(scan): skip files already in the library or trash"
```

---

### Task 6: Stage filter replaces scope across the read API

**Files:**
- Modify: `photocure/ui/app.py:1028-1040` (images), `:1553-1572` (people), `:1599-1630` (person photos), `:1883-1905` (overview), `:1978-2000` (bursts), `:988-991` (stats)
- Modify: `photocure/people_actions.py:146-220` (list_people), `:311-396` (delete_person, bulk_delete_people)
- Modify: `photocure/ui/static/app.js` every `scope=` fetch (`:661`, `:2656`, `:3080`, `:3543`, `:5486`) and `state.scope` reads
- Test: `tests/test_stage_filter.py`, update `tests/test_spec_pipeline_library.py` and `tests/test_ui_people.py` where they pass `scope=`

**Interfaces:**
- Produces: query parameter `stage: Literal["new", "library", "trash", "all"] | None` on `/api/images`, `/api/overview`, `/api/bursts`, `/api/people`, `/api/people/{id}/photos`, `/api/people/{id}` (delete), `/api/people/bulk-delete`. Default `"all"`. `scope` is removed and returns 422.
- Produces: `/api/overview` returns `stage_counts: {"new": n, "library": n, "trash": n}` instead of `pipeline_count` / `library_count`.
- Produces: `list_people(db_path, sort, stage="all")`, `delete_person(db_path, person_id, stage="all")`, `bulk_delete_people(db_path, ids, stage="all")`; filter is `ImageRecord.stage == stage` joined through faces, and `PersonCluster.stage == ("trash" if stage == "trash" else "library")`.
- Produces: `state.stage` in `app.js` replaces `state.scope`, values `new | library | trash`; a helper `stageQ()` returns `stage=<state.stage>` for fetches.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stage_filter.py  (fixture: rows in each stage, faces on one per stage)
def test_images_filter_by_stage(env): ...  # counts per stage, scope= -> 422
def test_overview_stage_counts(env): ...
def test_bursts_hide_groups_with_no_member_in_stage(env): ...
def test_people_in_trash_stage_lists_only_trash_clusters(env): ...
def test_person_photos_respect_stage(env): ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_stage_filter.py -v`
Expected: FAIL (`stage` ignored, `scope` still accepted).

- [ ] **Step 3: Implement the backend filters, the `people_actions` signatures and the `app.js` fetch changes (`stageQ()`, rename `state.scope`)**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_stage_filter.py tests/test_spec_pipeline_library.py tests/test_ui_people.py tests/test_people_actions.py tests/test_ui_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add photocure tests
git commit -m "feat(api): filter every listing by stage"
```

---

### Task 7: Reanalyse only in Trash; Analyze Now only in New

**Files:**
- Modify: `photocure/inference/pipeline.py:205-232` (row selection), `:236-250` (run signature)
- Modify: `photocure/ui/app.py:152-154` (InferReq), `:2228-2242` (do_infer), `:2829-2850` (remove /api/rescore)
- Modify: `photocure/cli.py:410-415`, `:456-474` (drop --reanalyze)
- Modify: `photocure/ui/static/index.html:246-247`, `:411` (remove buttons)
- Modify: `photocure/ui/static/app.js:784`, `:1955-1982`, `:5108`, `:6020-6021`, `:6160`, `:6172-6205` (remove handlers, fix copy)
- Test: `tests/test_infer_stage.py`; update `tests/test_fix_incremental_analyze.py` (delete `test_reanalyze_reprocesses_everything_and_keeps_rows`, `test_reanalyze_skips_missing_or_trashed_images`, `test_ui_has_reanalyse_buttons_and_skips_when_nothing_new`), `tests/test_cli.py` for `--reanalyze`

**Interfaces:**
- Produces: `InferencePipeline.run(..., stage: str = "new", reanalyze: bool = False)`; row selection is `ImageRecord.stage == stage` and `scan_status != "missing"`; when `reanalyze` is true every row of that stage with `infer_status == "inferred"` is reset to `pending` and `faces_schema_version = 0` first. `reanalyze_stub` is removed.
- Produces: `class InferReq(RunOverrides): batch_size: int = 32; stage: Literal["new", "trash"] = "new"`; `/api/infer` with `stage="trash"` implies `reanalyze=True`; `stage="library"` is 422.
- Produces: `/api/rescore` is removed (404); CLI `infer` has `--stage new|trash` (default new) and no `--reanalyze`.
- Produces: `do_infer` marks rows whose file is gone (`not Path(rec.path).exists()`) as `scan_status = "missing"` before running and reports `missing` in its result.
- UI: `#btn-recheck`, `#btn-reanalyse-all`, `#btn-rescore`, the `data-reanalyze` empty-state button and their handlers are gone; the "Nothing new to analyze" toast no longer mentions Reanalyse.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_infer_stage.py
def test_infer_default_touches_only_new_rows(env): ...  # library row stays inferred, untouched updated_at
def test_infer_trash_stage_reanalyses_trashed_rows(env): ...  # trashed rows processed, new rows untouched
def test_infer_library_stage_rejected(env): ...  # 422
def test_rescore_endpoint_gone(env): ...  # 404 or 405
def test_trash_reanalyse_marks_vanished_file_missing(env): ...  # unlink trash file, run -> scan_status missing, no exception
def test_ui_has_no_reanalyse_buttons(): ...  # index.html / app.js do not contain btn-recheck, btn-reanalyse-all, btn-rescore, data-reanalyze
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_infer_stage.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement pipeline, API, CLI and UI removals**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_infer_stage.py tests/test_fix_incremental_analyze.py tests/test_inference.py tests/test_cli.py tests/test_fix_inference_review.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add photocure tests
git commit -m "feat: reanalyse only in trash, analyze only new"
```

---

### Task 8: Manual edits in Library go through the ledger

**Files:**
- Create: `photocure/library_actions.py`
- Modify: `photocure/ui/app.py:2400-2444` (reassign), `:2446-2490` (delete_image), `:1845-1880` (best-shot promotion), `:2600-2695` (library sync)
- Modify: `photocure/people_actions.py` (add `detach_faces`)
- Test: `tests/test_library_actions.py`

**Interfaces:**
- Produces, in `photocure/library_actions.py`:
  - `def move_to_category(dbp: Path, cfg: CuratorConfig, image_id: int, category: str) -> dict` builds one `PlannedAction(operation="move")` with `target = library_target(...)` and runs `ActionEngine.execute([...], organize_root=root)`; returns `{"ok": bool, "transaction_id": str, "path": str}`; raises `FileNotFoundError` when `rec.path` does not exist; no-op `{"ok": True, "moved": False}` when target equals current path. Only valid for `stage == "library"`.
  - `def trash_image(dbp: Path, cfg: CuratorConfig, image_id: int) -> dict` sets `delete_approved = True`, runs the engine with `trash=True` and `organize_root=root` (so the file lands in `<root>/.trash/<txn>/`), then calls `detach_faces`; returns `{"ok": True, "transaction_id": str}`. Afterwards the row has `stage = "trash"`, `scan_status = "scanned"`, `path` at the trash location.
- Produces, in `photocure/people_actions.py`: `def detach_faces(s, image_id: int) -> int` (the face loop currently inside `delete_image`: `ignored = True`, `person_id = None`, decrement counts, delete empty unnamed clusters).
- `/api/images/{id}/reassign` calls `move_to_category` when `rec.stage == "library"`; `FileNotFoundError` becomes `HTTPException(409, "file is missing on disk")` and the bucket change is rolled back.
- `DELETE /api/images/{id}` calls `trash_image`; a `stage == "new"` row keeps today's behaviour of only marking `delete_approved` (it leaves at apply).
- Best-shot promotion in Library uses `move_to_category` for the promoted frame and leaves alternates where they are (the `Bursts/<name>/alternates` shuffle is dropped).
- `/api/library/sync` only reconciles rows with `stage == "library"` and never changes `stage`; its response adds `"missing_ids": [int, ...]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_library_actions.py  (fixture: move-apply 3 photos into root)
def test_move_to_category_moves_file_and_writes_ledger(...): ...  # new path under root/<folder>, ledger row committed, undo restores
def test_move_to_category_creates_missing_folder(...): ...
def test_move_to_category_missing_file_is_409_and_bucket_unchanged(env): ...  # unlink file, POST reassign -> 409, manual_bucket unchanged
def test_trash_image_moves_into_txn_folder_and_detaches_faces(...): ...  # path matches root/.trash/TX_*/..., stage trash, scan_status scanned, faces ignored, person face_count down
def test_delete_new_photo_only_marks_it(env): ...  # stage new row: delete_approved True, file untouched
def test_library_sync_never_pulls_out_of_library(env): ...  # unlink a library file, sync -> scan_status missing, stage still library, id in missing_ids
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_library_actions.py -v`
Expected: FAIL, import error.

- [ ] **Step 3: Implement `library_actions.py`, `detach_faces`, and rewire the three endpoints and sync**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_library_actions.py tests/test_ui_api.py tests/test_ui_people.py tests/test_people_actions.py tests/test_ui_face_rename.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add photocure tests
git commit -m "feat(library): category moves and deletes go through the ledger"
```

---

### Task 9: Trash operations and endpoints

**Files:**
- Create: `photocure/trash_actions.py`
- Modify: `photocure/ui/app.py` (new endpoints after `/api/library/sync`)
- Modify: `photocure/services.py:753-797` (purge_trash becomes empty_trash), `photocure/cli.py:740-757` (trash-purge)
- Test: `tests/test_trash_actions.py`; update `tests/test_cli.py::test_cli_trash_verify_repair_and_purge`, `tests/test_fix_engine_review.py::test_purge_trash_marks_ledger_with_relative_output_dir`

**Interfaces:**
- Consumes: `library_target`, `require_library_root`, `trash_dir`, `ActionEngine.execute`, `detach_faces`, and `attach_faces_to_library` from Task 10 (call it; Task 10 provides the body, until then it may be a stub returning `{"matched": 0, "created": 0}`).
- Produces, in `photocure/trash_actions.py`, each taking `(dbp: Path, cfg: CuratorConfig, ids: list[int] | None, *, cancel_check=None, on_progress=None)` and returning a dict:
  - `restore(...)` -> `{"ok": True, "restored": int, "skipped": int, "skipped_reasons": {"file_missing": int}, "transaction_id": str}`; per photo: target `library_target(cfg, root, rec, effective_category)`, engine move with collision policy, then `stage = "library"`, `route_status = "executed"`, `delete_approved = False`, `attach_faces_to_library(s, rec.id)`.
  - `return_to_origin(...)` -> `{"ok": True, "returned": int, "skipped": int, "skipped_reasons": {"origin_missing": int, "origin_occupied": int, "file_missing": int}, "transaction_id": str}`; target is `rec.origin_path`; after the move `stage = "new"`, `route_status = "unrouted"`, `infer_status = "pending"`, `origin_path = None`.
  - `delete_permanently(...)` -> `{"ok": True, "deleted": int, "bytes": int}`; unlinks file and sidecar, removes emptied `<txn>` folders, sets `route_status = "purged"`, `scan_status = "missing"`, marks matching ledger rows `purged`.
  - `empty_trash(dbp, cfg)` -> same shape as `delete_permanently`, over every `stage == "trash"` row plus any stray file under `trash_dir(root)`.
  - `ids=None` means every trash row for `return_to_origin`; `restore` and `delete_permanently` require ids.
- Produces endpoints (all `dependencies=auth`, all take `_BUSY` and return `202 {"ok": True, "job_id"}` via `spawn`, 409 when busy): `POST /api/trash/restore {ids}`, `POST /api/trash/return {ids: list | null}`, `POST /api/trash/delete {ids}`, `POST /api/trash/empty {}`. `POST /api/infer {"stage": "trash"}` is the reanalyse.
- Produces: `Planner.empty_trash() -> dict` replaces `purge_trash` and simply delegates to `trash_actions.empty_trash(self.db_path, self.cfg)`; CLI `trash-purge` calls it after `typer.confirm("Permanently delete everything in the trash?")`, no `--older-than-days`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trash_actions.py  (fixture: 4 photos; move-apply with trash so one lands in trash; manual-trash one library photo)
def test_restore_places_photo_by_current_category(...): ...  # path == root/<folder>/<name>, stage library, delete_approved False, ledger committed
def test_restore_respects_collision_policy_and_never_overwrites(...): ...  # pre-create target file with other bytes; restored to "name (1).ext"; original bytes intact
def test_restore_skips_missing_file_and_reports(...): ...
def test_return_to_origin_moves_back_and_resets(...): ...  # move-trashed photo: file at origin, stage new, infer pending
def test_return_skips_occupied_origin(...): ...  # copy-trashed photo: origin still has file -> skipped_reasons.origin_occupied == 1, trash file untouched
def test_return_all_when_ids_none(...): ...
def test_delete_permanently_unlinks_and_marks_purged(...): ...  # file gone, txn folder gone when empty, route_status purged, ledger purged
def test_empty_trash_clears_every_row_and_folder(...): ...
def test_trash_endpoints_409_when_busy(env): ...  # hold _BUSY, each endpoint -> 409
def test_trash_endpoints_run_as_jobs(env): ...  # 202 + job finishes with the result dict
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_trash_actions.py -v`
Expected: FAIL, import error.

- [ ] **Step 3: Implement `trash_actions.py`, the four endpoints, `empty_trash`, and the CLI change; delete `purge_trash` and its retention test**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_trash_actions.py tests/test_cli.py tests/test_fix_engine_review.py tests/test_ui_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add photocure tests
git commit -m "feat(trash): restore, return to origin, delete permanently, empty trash"
```

---

### Task 10: People per stage

**Files:**
- Modify: `photocure/people_actions.py:398-560` (recluster_people), add `attach_faces_to_library`
- Modify: `photocure/ui/app.py:1592-1597` (/api/people/recluster)
- Modify: `photocure/inference/pipeline.py` (clusters created during a trash run get `stage = "trash"`)
- Test: `tests/test_people_stage.py`

**Interfaces:**
- Produces: `recluster_people(db_path, threshold=0.64, stage="new") -> dict`; `stage="new"` behaves as today but filters `ImageRecord.stage == "new"` and anchors on `PersonCluster.stage == "library"` clusters with library faces; `stage="trash"` clusters only `ImageRecord.stage == "trash"` faces into clusters created with `stage = "trash"` and never touches library clusters.
- Produces: `attach_faces_to_library(s, image_id: int, threshold: float = 0.64) -> dict` -> `{"matched": int, "created": int}`; for each non-ignored face with an embedding: cosine against every `PersonCluster.stage == "library"` centroid; best `>= threshold` joins it (face_count +1, running centroid update), otherwise a new library cluster; faces leave their trash cluster, and a trash cluster with `face_count <= 0` is deleted.
- Produces: `POST /api/people/recluster?stage=new|trash` (default new).
- Produces: `InferencePipeline.run(stage="trash")` passes `stage` into whatever creates clusters so trash runs produce `stage = "trash"` clusters.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_people_stage.py  (fixture: seeded embeddings via quantize_embedding as tests/test_ui_api.py does)
def test_trash_recluster_creates_trash_clusters_only(env): ...  # library clusters unchanged, new clusters stage trash
def test_attach_matches_known_library_person(session): ...  # similar embedding -> matched 1, face.person_id == library cluster id, face_count +1
def test_attach_creates_new_library_person_for_unknown_face(session): ...  # created 1, cluster.stage == "library"
def test_attach_removes_empty_trash_cluster(session): ...
def test_people_endpoint_new_stage_hides_trash_clusters(env): ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_people_stage.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the `stage` parameter, `attach_faces_to_library`, and wire `restore` (Task 9) to the real function**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_people_stage.py tests/test_trash_actions.py tests/test_people_actions.py tests/test_face_identity.py tests/test_ui_people.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add photocure tests
git commit -m "feat(people): trash clusters stay apart, restore matches faces back"
```

---

### Task 11: Three-stage switch in the header

**Files:**
- Modify: `photocure/ui/static/index.html:125-134` (switch markup), `:73-96` (nav groups: add `id="navgroup-trash"` with a Categories, People, Bursts set is not needed; reuse existing tabs), `:24-60` (add `i-trash-stage` symbol if none fits)
- Modify: `photocure/ui/static/style.css:466-486` (switch styles)
- Modify: `photocure/ui/static/app.js:5861-5900` (setMode -> setStage), `:5405-5430` (renderCounts), `:24-38` (TITLES)
- Test: `tests/test_ui_stage_switch.py` (static assertions, same style as the removed `test_ui_has_reanalyse_buttons_and_skips_when_nothing_new`)

**Interfaces:**
- Consumes: `/api/overview` `stage_counts` from Task 6.
- Produces markup: `<div class="top-mode-switch" id="stage-switch" role="tablist">` with three `<button class="top-mode-btn" role="tab" data-stage="new|library|trash">` each containing `<svg class="ic">`, `<span>New</span> | <span>Library</span> | <span>Trash</span>`, and `<span class="mode-badge" data-stage-count="new|library|trash">`.
- Produces CSS: `.top-mode-btn` padding `10px 18px`, font 14px/600, icon 16px; `.top-mode-btn.active` background `var(--accent)`, color `var(--accent-ink)`, `box-shadow: 0 6px 16px rgba(250, 59, 145, 0.35)`; inactive badges muted as today. The active colour is pink for every stage.
- Produces JS: `function setStage(stage)` sets `state.stage`, toggles `.active` and `aria-selected`, shows nav groups per stage: New -> `#navgroup-pipeline`, `#navgroup-browse`, `#navgroup-review`; Library -> `#navgroup-browse` only; Trash -> `#navgroup-browse` only (Categories, People, Bursts; Search, Highlights, Events, Utility hidden by `data-stages` attribute on the nav buttons, e.g. `data-stages="new library"`). It calls `resetVisibleGrids()`, `loadStats()`, and `showTab("categories")` when the current tab is not allowed in the new stage. `renderCounts` writes `stage_counts` into the three badges. `document.body.dataset.stage = stage` so CSS can hide per-stage controls.
- TITLES gains `trash: ["Trash", "Photos you removed. Restore them, send them back where they came from, or delete them for good."]` used as the page subtitle while `state.stage === "trash"` on the categories tab.
- Entering the Library stage calls `/api/library/sync` (as today) and, when its `missing` count is above zero, appends "N photos are missing on disk" to the page subtitle instead of hiding them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui_stage_switch.py
def test_switch_has_three_stages_with_badges(): ...  # index.html contains data-stage="new|library|trash" and data-stage-count for each
def test_nav_buttons_declare_stages(): ...  # every .navitem with data-tab in (search, highlights, events, utility) has data-stages without "trash"
def test_js_uses_state_stage_not_scope(): ...  # "state.scope" not in app.js; "function setStage" in app.js
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_ui_stage_switch.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement markup, CSS and `setStage`**

- [ ] **Step 4: Run tests and look at it**

Run: `pytest tests/test_ui_stage_switch.py -v` then `photocure start` and click the three segments in light and dark theme.
Expected: PASS; active segment pink in all three; counts update after a scan.

- [ ] **Step 5: Commit**

```bash
git add photocure/ui/static tests/test_ui_stage_switch.py
git commit -m "feat(ui): three-stage switch with counts"
```

---

### Task 12: Settings page: library folder, default operation, plan screen defaults

**Files:**
- Modify: `photocure/ui/static/index.html:605-660` (new first section "Library folder")
- Modify: `photocure/ui/static/app.js:5212-5320` (renderSettings, saveSettings), `:4877-4883` (planBody default op)
- Test: `tests/test_ui_settings_root.py`

**Interfaces:**
- Consumes: `/api/settings` payload `values.library_root`, `sources.library_root`, `values.default_operation`; `/api/browse` picker (`bindPicker`, already used for scan folder).
- Produces markup, first `.sect` in `#tab-settings`: `<h2>Library folder</h2>`, `<input id="set-library-root" type="text" readonly>`, `<button id="btn-browse-library" class="btn">Browse</button>` (shown only while no root is stored), `<p class="help" id="library-root-source">` reading "Set in Settings" / "Set by PCUR_LIBRARY_ROOT" / "Not set yet", `<button id="btn-move-library" class="btn secondary">Move library to another folder</button>` (Task 15 wires it; hidden when source is environment or no root), and a `<select id="set-default-op">` with Move / Copy.
- `saveSettings()` includes `default_operation`, and `library_root` only when the input is enabled (no root stored yet).
- `#op` on the plan screen is preselected from `default_operation` on load.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui_settings_root.py
def test_settings_markup_has_library_folder_controls(): ...  # ids present in index.html
def test_settings_payload_includes_default_operation(tmp_path): ...  # GET /api/settings values has default_operation == "move"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_ui_settings_root.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement markup and JS**

- [ ] **Step 4: Run tests and look at it**

Run: `pytest tests/test_ui_settings_root.py -v`; in the app set a root, reload, confirm the field is read-only and Browse is gone.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add photocure/ui/static tests/test_ui_settings_root.py
git commit -m "feat(ui): library folder and default operation in settings"
```

---

### Task 13: First-run modal

**Files:**
- Modify: `photocure/ui/static/index.html` (new `<div id="firstrun" class="modal" hidden>` after `#picker`)
- Modify: `photocure/ui/static/app.js` (boot sequence near `:5840`, after `/api/config`)
- Modify: `photocure/ui/static/style.css` (`.firstrun-steps`)
- Test: `tests/test_ui_firstrun.py`

**Interfaces:**
- Consumes: `/api/config` `library_root`, `library_root_source`; `accelChoices()` and `#accel-list` rendering from `renderSettings`; `/api/settings` POST.
- Produces: `async function maybeFirstRun(cfg)` called once at boot; opens `#firstrun` when `cfg.library_root` is null. Step 1 has `#firstrun-root` (readonly input) + `#firstrun-browse` using the existing picker; Next is disabled until a path is chosen. Step 2 renders the accelerator radios into `#firstrun-accel` (same markup as `#accel-list`, generated by a shared `renderAccelList(container)` extracted from `renderSettings`), preselecting the detected device. Finish posts `{library_root, device, device_index}` in one `/api/settings` call, then closes and runs `loadStats()`. The modal has no close button and ignores Escape and backdrop clicks.
- No modal when `library_root_source === "environment"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui_firstrun.py
def test_firstrun_markup_present(): ...  # ids firstrun, firstrun-root, firstrun-browse, firstrun-accel, firstrun-finish
def test_firstrun_js_gated_on_config(): ...  # "maybeFirstRun" in app.js and "library_root_source" referenced
def test_config_source_environment_when_env_set(tmp_path, monkeypatch): ...  # PCUR_LIBRARY_ROOT -> /api/config source == "environment"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_ui_firstrun.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the modal, `renderAccelList`, `maybeFirstRun`**

- [ ] **Step 4: Run tests and look at it**

Run: `pytest tests/test_ui_firstrun.py -v`; start with a fresh db and confirm the modal blocks, both steps work, and it does not return after reload.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add photocure/ui/static tests/test_ui_firstrun.py
git commit -m "feat(ui): first run asks for the library folder and device"
```

---

### Task 14: Trash stage UI

**Files:**
- Modify: `photocure/ui/static/index.html:698-713` (bulkbar), `:279-324` (categories panel header: trash toolbar)
- Modify: `photocure/ui/static/app.js:1080-1100` (applyPicks), `:6052` (bulk handlers), delete flows at the lightbox and card menus
- Modify: `photocure/ui/static/style.css` (`body[data-stage="trash"]` rules)
- Test: `tests/test_ui_trash.py`

**Interfaces:**
- Consumes: `/api/trash/*` endpoints and `/api/infer {"stage": "trash"}` from Tasks 7 and 9; `followJob`, `runTask`, `showConfirmDialog`, `toast`.
- Produces bulkbar buttons with class `trash-only`: `#bulk-restore` ("Restore"), `#bulk-return` ("Return to origin"), `#bulk-purge` ("Delete permanently", danger). `applyPicks` toggles `trash-ctx` on `#bulkbar` when `state.stage === "trash"`; CSS shows `.trash-only` only in `trash-ctx` and hides `.lib-only`, `#bulk-lock`, `#bulk-clean` there.
- Produces a toolbar `#trash-toolbar` in the categories panel, visible only when `body[data-stage="trash"]`: `#btn-trash-reanalyse` ("Reanalyse trash"), `#btn-trash-return-all` ("Return all to origin"), `#btn-trash-empty` ("Empty trash", danger).
- Behaviour: Restore and Return run without a confirm; Delete permanently and Empty trash confirm with `variant: "danger"`; Return all confirms with the count. Every result toast reports counts and, when `skipped > 0`, the reasons in plain words: "2 skipped: original folder missing (1), a file already sits at the origin (1)".
- In Library and New, delete controls keep their current labels; in Trash the card and lightbox delete action is replaced by Delete permanently.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui_trash.py
def test_trash_controls_present(): ...  # ids bulk-restore, bulk-return, bulk-purge, trash-toolbar, btn-trash-reanalyse, btn-trash-return-all, btn-trash-empty
def test_js_calls_trash_endpoints(): ...  # "/api/trash/restore", "/api/trash/return", "/api/trash/delete", "/api/trash/empty" in app.js
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_ui_trash.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement markup, CSS, handlers**

- [ ] **Step 4: Run tests and exercise it**

Run: `pytest tests/test_ui_trash.py -v`; in the app trash a photo from Library, switch to Trash, restore it, trash again, return to origin, empty trash.
Expected: PASS; each action lands in the right stage and the counts in the switch follow.

- [ ] **Step 5: Commit**

```bash
git add photocure/ui/static tests/test_ui_trash.py
git commit -m "feat(ui): trash stage actions"
```

---

### Task 15: Move library to another folder

**Files:**
- Create: `photocure/library_migration.py`
- Modify: `photocure/ui/app.py` (two endpoints), `photocure/ui/static/index.html` (`#movelib` modal), `photocure/ui/static/app.js` (wire `#btn-move-library`)
- Test: `tests/test_library_migration.py`

**Interfaces:**
- Consumes: `library_root`, `validate_library_root`, `save_settings(force=True)`, `file_sha256`.
- Produces, in `photocure/library_migration.py`:
  - `def preview_migration(dbp: Path, cfg: CuratorConfig, new_root: Path) -> dict` -> `{"files": int, "bytes": int, "free_bytes": int, "enough_space": bool, "ok": bool, "error": str | None}`; `ok` is false when `new_root` exists and is not empty, or is inside the current root, or the current root is inside it.
  - `def migrate_library(dbp: Path, cfg: CuratorConfig, new_root: Path, *, on_progress=None, cancel_check=None) -> dict` -> `{"ok": bool, "copied": int, "bytes": int, "failed": list[str], "old_root": str, "new_root": str}`; copies every file under the old root (including `.trash/`) to the same relative path, verifies size and sha256, and only when every copy verified: one database transaction rewriting `images.path`, `images.target_path`, `images.relative_path` for `stage in ("library", "trash")`, `action_ledger.target_path` (and `source_path` where it points under the old root), then `save_settings(dbp, {"library_root": str(new_root)}, force=True)`. On any failure nothing in the database or settings changes and `failed` lists the paths.
- Produces endpoints: `GET /api/library/migrate/preview?path=` (400 on validation error, else the preview dict) and `POST /api/library/migrate {path}` -> 202 job holding `_BUSY`; 403 when `library_root_source == "environment"`.
- UI: `#btn-move-library` opens `#movelib` (picker for the new folder, preview line "N files, X GB; Y GB free", Start button disabled unless `ok && enough_space`), runs the job with `followJob`, and on success reloads settings and shows "Library moved. The old folder at <old> is yours to delete."

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_library_migration.py  (fixture: apply 3 photos + trash 1 into root A)
def test_preview_counts_files_and_rejects_non_empty_target(...): ...
def test_migration_copies_verifies_and_rewrites_paths(...): ...  # every library/trash row path under B, ledger targets under B, setting == B, files still in A
def test_migration_failure_leaves_database_and_setting_untouched(monkeypatch, ...): ...  # make one copy raise -> ok False, paths still A, setting A
def test_migrate_endpoint_forbidden_when_root_from_env(tmp_path, monkeypatch): ...  # 403
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_library_migration.py -v`
Expected: FAIL, import error.

- [ ] **Step 3: Implement module, endpoints and modal**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_library_migration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add photocure tests
git commit -m "feat: move the library to another folder"
```

---

### Task 16: Docs, Docker compose notes, full suite

**Files:**
- Modify: `README.md` (settings section: library folder, first run, Docker env var, trash has no timer), `docker-compose.yml:32-39` (replace the hardlink comment with "set PCUR_LIBRARY_ROOT or the OUTPUT_DIR mount"), `photocure/mcp_server.py` only if its tool docs mention `--output-dir`
- Test: whole suite

- [ ] **Step 1: Grep for leftovers**

Run: `git grep -n "PCUR_DEFAULT_OUTPUT\|default_output_dir\|trash_retention\|hardlink\|symlink\|in_place\|reanalyze\|rescore\|scope=" -- photocure README.md docker-compose*.yml Dockerfile`
Expected: only the rollback engine's historic `hardlink`/`symlink` reversal code and `_effective_op` remain.

- [ ] **Step 2: Update README and compose text**

- [ ] **Step 3: Run the full suite**

Run: `pytest -q`
Expected: all pass. Paste the summary line into the final report.

- [ ] **Step 4: Commit**

```bash
git add README.md docker-compose.yml photocure
git commit -m "docs: library folder, first run and trash behaviour"
```

# Full review findings (develop @ 44d1588)

Consolidated output of five parallel reviews: file-safety core, web/auth/scanner/DB, latest-commit
diff, face-identity algorithms, vision/quality/editor algorithms. Verified claims are marked (v).

Decision taken with the owner: the SigLIP 2 / TOPIQ-LIQE claims are corrected in the docs; the
code keeps OpenAI CLIP ViT-B/16 plus the LAION aesthetic head.

## A. Data loss / file safety

1. (v) `engine/action_engine.py:304` calls `self._resolve_collision()`, which does not exist. Swallowed
   at `:321`, so a companion `.mov`/`.xmp` whose target name is taken is left behind and the report says
   `failures 0`.
2. (v) `library_actions.py:24-40` `_move_sidecar` does an unledgered `shutil.move` onto an existing
   destination. With (1), B's `.MOV` overwrites A's; Undo cannot recover. Callers: `trash_actions.py:323`,
   `library_actions.py:68,97`, `burst_actions.py:241`. The engine already moves companions safely.
3. `engine/rollback.py:211-232`: undoing a copy-apply when the source is gone unlinks the only copy.
4. `trash_actions.py:253-261` `finalize_deletes` with unmarked ids organizes the file into the normal
   library folder but sets the row to `stage=trash`.
5. Companion ledger rows reuse the photo's `image_id` (`action_engine.py:305`); `_reconcile_pending`
   (`rollback.py:136-140`) then points the photo row at its `.mov`.
6. `action_engine.py:46-54` `find_companion_files` dedups by path string; on case-insensitive
   filesystems `IMG.mov` and `IMG.MOV` both match one file.
7. `trash_actions.py:318-319` raises on partial failure before the post-processing at `:325-349`.
8. `cli.py:1369-1371` `db restore` safety copy ignores the WAL.
9. `sidecar_queue.py:72-86` flush can write an orphan `.xmp` at a path the engine just moved away from.

## B. Web API / auth / scanner / DB

Solid: no path traversal, all SQL bound, `esc()` consistent in app.js, XXE guarded, constant-time
token compare, Host/Origin checks, non-root Docker.

1. (v) `ui/app.py:3003-3016` `/api/reset` JSON body bypasses the `Literal["new","all"]` validation; any
   other string wipes every row and the ledger.
2. `scanner/file_scanner.py:636-643`: renaming a photo in a watched folder makes the new path a
   duplicate of a ghost row (`_find_db_original` only excludes `scan_status == "missing"`).
3. `ui/app.py:2806-2882` `do_cluster` deletes groups and nulls `burst_id` before running the CPU-heavy
   detector, holding the write lock for the whole run.
4. `ui/app.py:2529-2530` `/api/bursts` N+1 (`get_by_id` per burst for a filename).
5. `post_fix_dates` (`:1496`), `curate_budget` (`:1378`), `editor_bokeh` (`:1915`) take unvalidated
   bodies; `float()`/`int()` on bad input gives 500.
6. `ui/app.py:893` session cookie is the long-lived master token; no logout.
7. `scanner/xmp.py:92,158` DOCTYPE/ENTITY guard is ASCII-byte based; UTF-16 sidecars pass.
8. `db/models.py` missing indexes: `delete_approved`, `duplicate_of`, `near_dup_of`, `origin_path`.
9. `ui/app.py:731` `exclusive()` reads `label`/`title`; `ProgressManager.state` has `task`.
10. `ui/app.py:781` `watch_ledger` job logs are unbounded.
11. `tests/test_ui_api.py:70-88` `MUTATING` omits many mutating routes.

## C. Face identity

1. (v) `clustering/face_identity.py:53`, `people_actions.py:558,745,954` clamp `min(threshold, 0.35)`
   for 512-D: the knob can only loosen. `recluster_people:789-802` adds +0.08/+0.06/+0.04 so big
   clusters run at ~0.31.
2. `people_actions.py:822-825` exemplars are added only when dissimilar (<0.78) to existing ones:
   outliers become permanent exemplars, matched with `max()`: single-linkage drift.
3. "Chronological medoid" is an EMA (0.7/0.3), never persisted; `pipeline.py:429-438` reloads only
   centroid + image_ids. `people_actions.py:815` double-counts `face_count`.
4. `_find_merge_matches` O(C^2) Python per request; `_arrange_by_similarity` O(C^3) worst case plus
   a DB query per match that is discarded at `:290`; recluster O(F x total_faces) at `:790-802`.
5. Co-occurrence veto missing from merge suggestions (`_find_merge_matches`, `_arrange_by_similarity`).
6. Greedy order-dependent assignment with no reassignment pass.
7. `face_engine.py:71-87` strict landmark ratios run in `_detect_one`, so profile faces vanish from
   blink/group logic too; landmarks clipped to [0,1] before alignment; `"data"` input name hard-coded;
   `estimateAffinePartial2D` defaults to RANSAC on 5 points.
8. Merge pill (latest commit): named target wins at bare threshold even when an unnamed cluster scores
   far higher; `_best_face_id_for` not given `stage`; `face_count=0` named clusters offered; duplicate
   `has_any_face` query; duplicated centroid-decode loop.
9. `scanner/phash.py` `dhash_from_gray`/`compute_dhash` dead code.
10. Tests use 2-D toy vectors; nothing checks drift, boosts, order-invariance.

## D. Vision / quality / editor

1. (v) README:32-37, `models/MANIFEST.json:91-96`, NOTICE item 21 claim SigLIP 2 + TOPIQ/LIQE; code is
   CLIP ViT-B/16 (`config.py:233`) + LAION head. `semantic_engine.py:177-184,204` naflex path is dead.
2. (v) `curate_budget.py:103-104` imports `make_text_encoder` (does not exist; it is
   `search.default_encoder`) and `cfg.backend` (does not exist). Scale balancing never runs.
3. `highlights.py:76-89` `highlight_expr` has no quality term; `highlight_score(quality=)` has no
   caller passing it; `top_of_year` has no diversity.
4. `semantic_engine.py:429-433` correction bonus `0.6*100*(sim-0.60)` flips classes from two examples.
5. `rules/decision_matrix.py:258-272` camera-model guard skips `document`/`slide` and
   `"scenery" in dist` is always true.
6. `quality/blur_detector.py:128-131` anisotropy rule constants not in `QualityThresholds`.
7. `editor_actions.py:685-707` MI-GAN squashes the whole image to 512x512 and upsamples back.
8. `editor_actions.py` depth input 518x518 square resize distorts aspect.
9. `models/__init__.py:62` hard-coded `/home/safiyu/...` path; `editor_actions.py:728` generic
   `depth_anything.onnx` fallback would load a non-commercial Base/Large export.
10. `search.py:85-111` and `clustering/similarity.py` decode JSON embeddings per row per query.

## E. Housekeeping

- `build/` is a stale setuptools artifact (gitignored). `.claude/worktrees/single-library-root` is a
  live worktree for `feat/single-library-root`; leave it.

## Out of scope (accepted)

- Shipping SigLIP 2 (owner chose docs correction).
- Empirical recalibration of zero-shot thresholds (needs a labelled set).
- Blur patch upscaling (contradicts an earlier deliberate fix), composite per-face ECC refinement,
  linear-light bokeh compositing: need real-image validation.
- `post_apply_hook` SSRF: requires the auth token; left as is.

# Single library root, three stages, trash as a pipeline

Date: 2026-10-26
Status: draft, awaiting owner approval

## Intent

Photocure today lets every plan choose its own destination folder. After a few
runs the finished photos are spread across several output folders, the app has
to track all of them, and the Library stage becomes painful to keep in sync.
Trash lands in two different places depending on whether a photo was deleted by
hand or staged as junk during apply. Reanalyse buttons act on photos that are
already finished.

This design gives the app one permanent home for every finished photo, a clear
three stage model (New, Library, Trash) that the whole UI is built around, and
removes automatic AI reruns from everything except Trash.

The owner is the only user right now, so no data migration from the old layout
is needed. Existing rows that point outside the new library root may be dropped
or re-imported by hand.

## Goals

- One library root, set once in Settings. Every apply and every restore writes
  into it. The app never asks for a destination again.
- A finished photo's row always points at the file inside the library root,
  whether it was moved or copied there.
- Three stages, visible as the main switch in the header: New, Library, Trash.
- Finished photos are never re-analysed. Only manual edits are allowed, and a
  manual category change moves the file on disk to match.
- Trash is a real place in the UI: list, reanalyse, restore, return to origin,
  delete permanently.

## Non-goals

- Migrating photos from old output folders. Not needed, nobody else is using it.
- Changing the folder layout inside the library root. `category_first` stays.
- Hardlink and symlink operations. They go away.
- `in_place` ("leave originals, add links") mode. It goes away with symlinks.
- Multi-library support. One root per database.

## Anti-goals

- Any state where a finished photo's `path` points outside the library root.
- Any path where the AI runs on a Library photo, including through the CLI
  default.
- Silently sending a trashed photo somewhere the user did not pick.

## Constraints

- Sources can be read-only, so **copy** must remain alongside **move**.
- Every filesystem change still goes through the action ledger so undo and
  reconcile keep working (`photocure/engine/action_engine.py`,
  `photocure/engine/rollback.py`).
- Trash is only ever emptied by the user. There is no time-based purge.
- The CLI (`photocure plan`, `apply`, `infer`, `trash-purge`) must stay usable
  without the web UI. Where the UI loses an option, the CLI loses it too, so the
  two cannot drift.

## The model

### Stages

Add an explicit `stage` column on `images`: `"new" | "library" | "trash"`.
Today the stage is inferred from `route_status == "executed"` and
`route_status == "trashed"`, which spreads the rule across many queries. The
new column is the single source of truth for which switch section a photo
belongs to. `route_status` stays for plan bookkeeping (`unrouted`, `planned`,
`executed`, `skipped`, `review`) and is no longer used to decide visibility.

Transitions:

| From | To | Trigger |
|------|-----|---------|
| new | library | plan apply (move or copy), file now lives under the library root |
| new | trash | plan apply with trash on, photo was approved for delete (move or copy) |
| library | trash | manual delete in Library |
| library | new | Undo of the last apply transaction |
| trash | library | Restore |
| trash | new | Return to origin (file goes back to its source folder) |
| trash | gone | Delete permanently, Empty trash |

Undo stays as it is today, in the UI and the CLI. It is the only way a finished
photo goes back to New, and for a copy apply it deletes the library copies and
leaves the sources alone.

With a **copy** apply and trash on, photos approved for delete are copied into
the trash folder, not left in the source. Trash then looks the same whichever
operation was used, and "Delete permanently" only ever touches the library side.
The plan screen shows a one-line hint when copy is selected: originals stay
where they are, including the ones marked for delete, and a copy of those goes
to the trash folder.

### One library root

`library_root` already exists as a setting (`services.SETTINGS_KEYS`) but has
no UI and is only used by a few endpoints. It becomes the one destination:

- Settings page gets a "Library folder" field with Browse, using the existing
  `/api/browse` picker and the same validation as today's
  `validate_output_dir` (absolute, not a system dir, not inside a source
  folder, not the photocure data dir). Once a root is saved the field is
  read-only; changing it goes through the migration below.
- The plan screen loses "Destination folder", "Folder layout" and the hardlink
  and symlink options. What remains: Move or Copy, "Apply to" category filter,
  and the trash toggle.
- `/api/plan/rebuild`, `/api/plan/diff` and `/api/plan/apply` stop accepting
  `output_dir`. They read the root from settings and return 409 with a clear
  message if it is not set.
- Clean export (`/api/export/clean`) is not part of this. It builds a share
  bundle for somewhere else and keeps its own folder picker and
  `validate_output_dir`, with the extra rule that the chosen folder must not be
  inside the library root, so library sync never imports the stripped copies.
- The CLI `--output-dir` option on `plan`, `apply`, `trash-purge` is replaced by
  reading the same setting, with `--library-root` as an explicit override for
  scripting. `trash-purge` loses its `--older-than-days` option and becomes the
  CLI form of Empty trash: it deletes everything in the trash after a confirm.
- Plan readiness (`/api/plan/readiness`) reports "library folder not set" as a
  blocker so the UI can point the user to Settings.

The trash folder is `<library_root>/.trash/`. Both the apply path and the
manual delete path write there, and both get their own `<txn>/` subfolder
through the ledger. Manual delete opens a one-action transaction instead of
writing into a shared folder, so undo and `_mark_purged` work the same way for
both. Nothing in the trash is removed on a timer; `PCUR_TRASH_RETENTION_DAYS`
and `trash_retention_days` are dropped.

### Moving the library to another folder

The Settings field is read-only once set. Next to it sits a "Move library to
another folder" button that opens a dedicated migration:

1. Pick the new folder with the same picker and validation. It must be empty
   or not exist yet.
2. Preview: number of files and total size under the current root, including
   `.trash/`, and free space at the destination.
3. Run as a job with progress. For every file under the current root, copy it
   to the same relative path under the new root and verify size and hash.
   Nothing is deleted from the old root.
4. When every copy has verified, update `path`, `target_path` and
   `relative_path` on every `library` and `trash` row, rewrite ledger targets
   that point under the old root, and save the new `library_root` setting,
   all in one database transaction.
5. Report what was copied and tell the user the old folder is theirs to delete.

If any copy fails, nothing in the database changes and the setting stays on
the old root; the partial copy at the destination is reported so the user can
clear it. Other filesystem actions (apply, restore, delete) are blocked while
the migration runs. When `library_root` comes from the environment (Docker),
the button is hidden; the operator changes the mount instead.

### Copy means the library copy is the photo

After a copy apply, the image row's `path` becomes the library path, exactly as
it already does for move (`action_engine.py` around line 258). The source path
is kept in a new `origin_path` column so "Return to origin" knows where the
photo came from. `origin_path` is set on first apply for both move and copy.

On a later scan of the same source folder, a file whose content hash matches a
row that already has `stage = "library"` or `stage = "trash"` is skipped. It
does not create a new row and does not show up in New. The scanner already
keeps `quick_hash` and `sha256` per row; the skip uses those. Trash counts too,
otherwise a copied-then-trashed photo would re-enter New and a later Restore
would produce two. This is my call; the owner can ask for an "already in
library" count in the scan summary instead if silent skipping turns out to
hide problems.

### Finished photos: manual edits only

In Library the following remain, all manual:

- rename a person, merge people, reassign a face
- change a photo's category
- remove a photo from a burst, change best shot
- rating, favourite, keywords
- delete (to trash)

Changing a category in Library moves the file into
`<library_root>/<category>/` (the same `category_first` layout the planner
uses), creating the folder if needed, through the action ledger so it is
undoable. If the file is missing on disk the request fails with 409 and the
database is left unchanged.

The following are removed from the UI entirely:

- "Re-check rules" (`#btn-recheck`)
- "Reanalyse everything" (`#btn-reanalyse-all`)
- "Analyze again" on the empty state (`data-reanalyze`)
- "Re-check my photos" (`#btn-rescore`)

`/api/rescore` and the `reanalyze` flag on `/api/infer` are removed. The CLI
`photocure infer --reanalyze` is removed; `infer` only touches rows with
`infer_status = "pending"` and `stage = "new"`, plus the Trash case below.

"Analyze Now" stays, scoped to `stage = "new"`.

Library sync (`/api/library/sync`) stays. It reconciles rows whose file was
moved by hand inside the library root. It must never pull a row out of Library;
a file it cannot find is marked `scan_status = "missing"` and surfaced as a
count in the Library header, not hidden.

### Trash as its own pipeline

Trash is a third switch section with the same grid, filters, People and Bursts
views as the other two, scoped to `stage = "trash"`.

Actions available in Trash:

- **Reanalyse.** Runs the inference pipeline on trashed photos only
  (`stage = "trash"`, any `infer_status`). Faces, category and quality are
  recomputed. This is the only place the AI reruns on an already analysed photo.
- **Restore** (single, multi-select or all). For each photo: compute the target
  as `<library_root>/<effective category>/<filename>` using the same organizer
  code the planner uses, with the configured collision policy; move the file;
  set `stage = "library"`, `route_status = "executed"`, `path` to the new
  location, `delete_approved = False`; write a ledger entry. Whatever the
  current category is, including a junk one, decides the folder; the user
  reanalyses or changes the category first if they want it elsewhere. Faces
  are matched to existing Library people as described under People below.
  No plan, no preview, no apply step.
- **Return to origin** (multi-select or all). Moves the file back to
  `origin_path`, sets `stage = "new"`, `route_status = "unrouted"`,
  `infer_status = "pending"`, writes a ledger entry. If `origin_path` is empty,
  its parent folder does not exist, or a file already sits at `origin_path`
  (the normal case after a copy apply), that photo is skipped and counted in
  the result; the toast says how many were skipped and why. Nothing is sent to
  a fallback location and nothing is overwritten.

#### People across stages

`PersonCluster` gets a `stage` column: `"library"` or `"trash"`. People created
by analysing New photos and by apply belong to `library` (New and Library share
one set of people, as they do today). Trashing a photo detaches its faces
(`person_id = None`, `ignored = True`, face counts decremented, empty unnamed
people deleted), which is what manual delete already does; apply-trash starts
doing the same. The People view in Trash is therefore empty until Reanalyse
runs; Reanalyse clusters trash faces into `trash` people only, and the New and
Library People lists filter them out. On Restore, each face is compared against
`library` people with the threshold `recluster_people` uses; a match joins that
person, otherwise a new `library` person is created. The `trash` cluster is
deleted once it has no faces left. Trash rows stay `scan_status = "missing"` with
`path` at the trash location, so every feature that skips missing photos
(search, duplicates, highlights, events) keeps skipping them; the Trash stage
lists them by `stage = "trash"` instead, ignoring `scan_status`.
- **Delete permanently** (multi-select) and **Empty trash**. Both confirm first.
  They unlink the files, remove emptied `<txn>/` folders, and set
  `route_status = "purged"`, `scan_status = "missing"`. This is the only way
  anything leaves the trash for good; there is no retention timer.

### The switch

Option A from the mockups: a larger segmented pill in the header, three
segments, each with an icon, label and count. The active segment is filled with
`--accent` (pink) whatever the stage, matching today's active colour. Inactive
segments stay muted. Counts come from `/api/stats` grouped by `stage`.

Switching stage resets the visible grids, reloads stats and shows the right
nav groups:

- New: Pipeline and Review groups (today's "pipeline" mode)
- Library: Categories, People, Bursts, Highlights, Search
- Trash: Categories, People, Bursts, plus a Trash action bar with Reanalyse,
  Restore, Return to origin, Delete permanently, Empty trash

The `scope` query parameter on existing endpoints becomes `stage` with values
`new | library | trash`. `all` is kept for internal callers.

### First run

When the UI loads and no library root is set (neither in the settings file
nor through the `PCUR_LIBRARY_ROOT` environment variable), a modal opens before
anything else. It has two steps and cannot be dismissed until both are done;
the rest of the app is blocked behind it.

1. **Library folder.** Same Browse picker and validation as the Settings
   field.
2. **Device.** The same accelerator radio list the Settings page shows today
   (`#accel-list`, choices from `DEVICE_CHOICES`), with the detected device
   preselected and "Auto" as the safe default. The GPU index field appears
   only when the chosen device needs one, as it does in Settings.

Saving writes both `library_root` and `device` through `/api/settings` in one
call. Once saved, the popup never shows again unless `library_root` is cleared.
The device alone being unset does not trigger it; `auto` is a fine default and
Settings remains the place to change it later.

Precedence follows the existing rule: settings file beats environment beats
default. The Settings page shows the effective value and says where it came
from, so a Docker user can see that the env var is in charge.

Docker: the Dockerfile currently sets `PCUR_DEFAULT_OUTPUT=/output/Organized`.
That becomes `PCUR_LIBRARY_ROOT=/output/Organized`, so containers start with a
root already set and the popup never appears. The compose files keep mounting
`${OUTPUT_DIR:-./output}` at `/output`. `PCUR_LIBRARY_ROOT` already works
because `library_root` lives on `CuratorConfig`, which is environment-driven.

### Settings

New or changed keys in `SETTINGS_KEYS`:

- `library_root` (existing, now required for apply and restore)
- `default_operation`: `"move" | "copy"`, default `"move"`, pre-selects the
  plan screen

`PCUR_DEFAULT_OUTPUT` and `default_output_dir` in `/api/config` go away.

## What the builder decides

- Exact column migration strategy for `stage` and `origin_path` (the project
  already has a schema version mechanism in `photocure/db`). Backfill:
  `executed -> library`, `trashed -> trash`, everything else `new`.
- Whether Restore reuses `Planner.build_plan` with a filter, or a smaller
  function that calls the organizer directly. Either is fine as long as the
  ledger is written.
- Icon choice for the three segments.
- How the Trash action bar fits with the existing selection bar (`#bulkbar`).
  It should not be a second floating bar.
- Test layout. Existing tests under `tests/` that pass `output_dir` or use
  hardlink, symlink or `in_place` will need updating or removal.

## Owner decisions recorded

- Move or copy only; copy kept because sources can be read-only.
- The library root is the final home regardless of move or copy.
- No migration of old output folders.
- Switch style A, active segment pink.
- Re-check and Reanalyse removed from New and Library, kept only in Trash.
- Restore goes to the library root, matched against existing people and
  categories.
- Return to origin is a bulk action, all or multi-select.
- Category change in Library moves the file, creating the folder if needed.
- First run shows a blocking popup to choose the library folder and the
  device; Docker sets the root through the environment so the popup never
  appears there.
- A copy apply copies approved junk into the trash folder; the plan screen
  hints at this when copy is chosen.
- No 30-day retention. Trash is emptied only from the UI (or the CLI
  equivalent).
- The library folder is read-only once set. Changing it is a dedicated
  migration that copies everything to the new folder and updates the database.

## Calls I made (owner can overrule)

- Restored photos go to Library directly, not back through New.
- Return to origin skips photos whose origin is missing or already occupied
  and reports the count.
- Duplicates of Library or Trash photos found on rescan are skipped silently.
- Manual delete writes its own ledger transaction and trash subfolder, so undo
  treats it like any other action.
- Undo of the last apply stays and is the only `library -> new` path.
- Clean export is untouched except that it refuses a folder inside the root.
- Restore places a photo by its current category, junk included, and clears
  the delete approval.
- People get a `stage`; trash faces are detached on trash and matched back on
  Restore.
- Migration copies rather than moves and leaves the old folder for the user.
- Delete permanently and Empty trash are offered in the UI with a confirm.
- `default_operation` setting added so the plan screen remembers move vs copy.

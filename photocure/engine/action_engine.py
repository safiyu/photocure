"""Transactional filesystem mutation engine.

Safety guarantees (spec section 7.1):
- **Default dry-run**: no disk mutation unless ``execute`` is called.
- **No overwrites**: name collisions are renamed with a ``_dup<SHA[:6]>`` suffix.
- **Move verification**: before unlinking a source after a cross-device move,
  the target's size + SHA-256 are verified to match.
- **Transactional ledger**: every mutation is recorded ``PENDING`` (and the row
  is committed to SQLite) first, then flipped to ``COMMITTED`` (committed again)
  only after the disk operation succeeds, so a crash never loses the journal.

The engine is fully reversible via :class:`RollbackEngine`.
"""
from __future__ import annotations

import errno
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Iterable, Sequence

from ..config import CuratorConfig
from ..db.models import ActionLedger, ImageRecord
from ..db.repository import PhotoRepository
from ..db.session import get_session
from ..rules.organizer import PlannedAction
from .hash import file_sha256, file_size
from .locks import LEDGER_WRITING

log = logging.getLogger("photocure.action")


def new_transaction_id() -> str:
    """Human-friendly, sortable transaction id: ``TX_YYYYMMDD_HHMMSS_<rand>``."""
    from datetime import datetime
    now = datetime.now()
    return f"TX_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


@dataclass
class ExecutionReport:
    transaction_id: str
    executed: int = 0
    skipped: int = 0
    failures: int = 0
    collisions: int = 0
    dry_run: bool = False
    duration_s: float = 0.0
    errors: list = field(default_factory=list)
    copy_fallbacks: int = 0  # hardlinks that fell back to a copy
    skipped_collisions: int = 0  # collision_policy=skip leftovers (not errors)
    filtered: int = 0  # skipped by the only_category filter
    in_place: int = 0  # target already is this photo (re-apply no-op)
    unapproved_junk: int = 0  # junk kept out of the trash because deletion was not approved
    marked_left_out: int = 0  # marked for delete but not staged (no trash option or not a move)
    # image_id -> final on-disk target actually used (after collision rename).
    resolved_targets: dict = field(default_factory=dict)
    # Dry-run diff: what an apply would cost.
    cross_device: int = 0
    bytes_total: int = 0
    bytes_needed: int = 0
    missing_sources: int = 0
    free_bytes: dict = field(default_factory=dict)
    space_ok: bool = True

    def diff_summary(self) -> dict:
        return {
            "actions": self.executed, "collisions": self.collisions,
            "cross_device": self.cross_device, "missing_sources": self.missing_sources,
            "bytes_total": self.bytes_total, "bytes_needed": self.bytes_needed,
            "free_bytes": dict(self.free_bytes), "space_ok": self.space_ok,
        }


def _nearest_existing(p: Path) -> Path:
    """Closest existing ancestor of ``p`` (``p`` itself when it exists)."""
    cur = Path(p)
    while not cur.exists() and cur != cur.parent:
        cur = cur.parent
    return cur


def _first_missing_dir(d: Path) -> Path | None:
    """Topmost directory in ``d``'s ancestry that does not exist yet."""
    top = None
    cur = Path(d)
    while not cur.exists() and cur != cur.parent:
        top = cur
        cur = cur.parent
    return top


def _bytes_needed(op: str, size: int, cross: bool) -> int:
    if op == "copy":
        return size
    if op in ("move", "hardlink") and cross:
        return size
    return 0


class ActionEngine:
    """Execute a :class:`PlannedAction` list transactionally."""

    def __init__(self, config: CuratorConfig | None = None,
                 db_path: Path | None = None):
        self.cfg = config or CuratorConfig()
        self.db_path = db_path or self.cfg.db_path

    # -------------------------------------------------------------- dry-run
    def dry_run(self, plan: Sequence[PlannedAction]) -> ExecutionReport:
        """Report what *would* happen without touching the filesystem."""
        rep = ExecutionReport(transaction_id=new_transaction_id(), dry_run=True)
        needed_by_dev: dict[int, int] = {}
        free_by_dev: dict[int, tuple[str, int]] = {}
        seen_targets: set[Path] = set()
        for a in plan:
            if a.target.exists() or a.target in seen_targets:
                a.collides = True
                rep.collisions += 1
            seen_targets.add(a.target)
            rep.executed += 1  # "would execute"
            if not os.path.lexists(a.source):
                rep.missing_sources += 1
                continue
            size = file_size(a.source)
            rep.bytes_total += size
            probe = _nearest_existing(a.target.parent)
            cross = not self._same_device(a.source, probe)
            if cross and a.operation in ("move", "hardlink"):
                rep.cross_device += 1
            need = _bytes_needed(a.operation, size, cross)
            rep.bytes_needed += need
            try:
                dev = os.stat(probe).st_dev
                if dev not in free_by_dev:
                    free_by_dev[dev] = (str(probe), shutil.disk_usage(probe).free)
                needed_by_dev[dev] = needed_by_dev.get(dev, 0) + need
            except OSError:
                continue
        for dev, (where, free) in free_by_dev.items():
            rep.free_bytes[where] = free
            if needed_by_dev.get(dev, 0) > free:
                rep.space_ok = False
        rep.duration_s = 0.0
        return rep

    # --------------------------------------------------------------- execute
    def execute(self, plan: Sequence[PlannedAction], *args, **kwargs) -> ExecutionReport:
        with LEDGER_WRITING:
            return self._execute(plan, *args, **kwargs)

    def _execute(self, plan: Sequence[PlannedAction],
                 transaction_id: str | None = None,
                 organize_root: Path | None = None,
                 only_category: Iterable[str] | str | None = None,
                 trash: bool = False,
                 cancel_check: Callable[[], bool] | None = None,
                 ) -> ExecutionReport:
        """Apply every planned mutation, recording the ledger as we go.

        For each action the ledger row is written ``PENDING`` and committed
        *before* the disk mutation, then flipped to ``COMMITTED`` (and committed)
        on success or ``FAILED`` on error.
        """
        txn = transaction_id or new_transaction_id()
        rep = ExecutionReport(transaction_id=txn, dry_run=False)
        self._rep = rep  # read by _apply; safe because execute() holds LEDGER_WRITING
        t0 = time.monotonic()
        wanted = _norm_categories(only_category)
        root = Path(organize_root) if organize_root else None
        with get_session(self.db_path) as session:
            repo = PhotoRepository(session)
            for action in plan:
                if cancel_check and cancel_check():
                    break
                if wanted is not None and _cat_value(action) not in wanted:
                    rep.skipped += 1
                    rep.filtered += 1
                    continue
                entry = None
                mutated = False
                action.collides = False  # drop flags left by dry_run
                sha = action.sha256 or ""
                size = action.file_size or 0
                try:
                    marked = False
                    if action.image_id:
                        rec0 = repo.get_by_id(action.image_id)
                        marked = bool(rec0 is not None and rec0.delete_approved)
                    if marked:
                        if trash and action.operation == "move":
                            action.target = self._trash_target(action, root, txn)
                        else:  # marked for delete: never placed in the output, the original stays put
                            rep.skipped += 1
                            rep.marked_left_out += 1
                            continue
                    elif trash and action.operation == "move" and _is_junk(action):
                        rep.unapproved_junk += 1  # not approved: stays in the junk review folder
                    if self._is_in_place(action, repo):
                        self._mark_in_place(action, repo, root)
                        session.commit()
                        rep.skipped += 1
                        rep.in_place += 1
                        continue
                    if not os.path.lexists(action.source):
                        raise FileNotFoundError(f"source missing: {action.source}")
                    if not sha:
                        sha = file_sha256(action.source)
                        size = file_size(action.source)
                    action.sha256, action.file_size = sha, size

                    # 1. Resolve collision first so the ledger names the real target.
                    created = _first_missing_dir(action.target.parent)
                    target = self._prepare_target(action, repo)
                    if target is None:  # collision_policy == "skip"
                        rep.skipped += 1
                        rep.skipped_collisions += 1
                        continue

                    # 2. PENDING ledger row, durable before any mutation.
                    img = repo.get_by_id(action.image_id) if action.image_id else None
                    entry = repo.ledger_add(
                        txn, action.image_id, str(action.source),
                        str(target), action.operation, sha, size)
                    entry.created_dir = str(created) if created else ""
                    entry.prev_relative_path = (img.relative_path if img else "") or ""
                    session.commit()

                    # 3. Filesystem mutation.
                    entry.actual_operation = self._apply(action, target, sha)
                    mutated = True
                    st = os.lstat(target)
                    entry.target_dev, entry.target_ino = st.st_dev, st.st_ino

                    # 4. COMMIT ledger row (post-mutation success), durable.
                    action.target = target
                    entry_id = entry.id
                    try:
                        repo.ledger_commit(entry)
                        if img is not None:
                            img.target_path = str(target)
                            img.route_status = "executed"
                            if action.operation == "move":
                                self._retire_stale_row(session, str(target), img.id)
                                img.path = str(target)
                                img.relative_path = _rel(target, root)
                        session.commit()
                    except Exception as e:
                        # The file already moved: record that so rollback can reverse it.
                        session.rollback()
                        try:
                            done = session.get(ActionLedger, entry_id)
                            if done is not None:
                                repo.ledger_commit(done)
                                done.error = f"moved but image row not updated: {e}"
                                session.commit()
                        except Exception:
                            log.exception("could not record moved-but-unrecorded ledger entry")
                            session.rollback()
                        raise

                    rep.executed += 1
                    rep.resolved_targets[action.image_id] = str(target)
                    if action.collides:
                        rep.collisions += 1
                except Exception as e:
                    log.error("action failed for %s: %s", action.source, e)
                    rep.failures += 1
                    rep.errors.append(f"{action.source}: {e}")
                    session.rollback()
                    # After a successful mutation leave the row pending so reconcile recovers it.
                    if entry is not None and not mutated:
                        try:
                            repo.ledger_fail(entry, str(e))
                            session.commit()
                        except Exception:
                            log.exception("could not mark ledger entry failed")
                            session.rollback()
        rep.duration_s = time.monotonic() - t0
        return rep

    # ------------------------------------------------------------- helpers
    def _trash_target(self, action: PlannedAction, root: Path | None, txn: str) -> Path:
        base = Path(root) if root else action.target.parent
        trash_root = base / self.cfg.trash_dir_name / txn
        try:
            rel = action.target.relative_to(base)
        except ValueError:
            rel = Path(action.target.name)
        return trash_root / rel

    @staticmethod
    def _retire_stale_row(session, target: str, keep_id: int | None) -> None:
        """Free ``target`` from a stale missing row so UNIQUE(images.path) holds."""
        from sqlmodel import select
        for row in session.exec(select(ImageRecord).where(
                ImageRecord.path == target, ImageRecord.id != keep_id)).all():
            if row.scan_status == "missing":
                row.path = f"{target}#retired-{row.id}"
        session.flush()

    @staticmethod
    def _is_in_place(action: PlannedAction, repo) -> bool:
        """True when the target already is this photo (same file, or its own earlier apply)."""
        target = action.target
        if not os.path.lexists(target):
            return False
        try:
            if os.path.samefile(action.source, target):
                return True
        except OSError:
            pass  # source gone: fall back to the ledger
        if not action.image_id:
            return False
        from sqlmodel import select
        from ..db.models import ActionLedger
        prior = repo.s.exec(select(ActionLedger).where(
            ActionLedger.image_id == action.image_id,
            ActionLedger.status == "committed",
            ActionLedger.target_path == str(target))).first()
        if prior is None or not prior.sha256:
            return False
        return file_sha256(target) == prior.sha256

    @staticmethod
    def _mark_in_place(action: PlannedAction, repo, root: Path | None) -> None:
        img = repo.get_by_id(action.image_id) if action.image_id else None
        if img is None:
            return
        img.target_path = str(action.target)
        img.route_status = "executed"
        if action.operation == "move":
            img.path = str(action.target)
            img.relative_path = _rel(action.target, root)

    def _prepare_target(self, action: PlannedAction, repo) -> Path | None:
        """Create the destination dir and resolve a name collision per ``collision_policy``.

        Returns the final target, or None when the policy is ``skip`` and the name is taken.
        """
        action.target.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.lexists(action.target):
            return action.target
        action.collides = True
        policy = self.cfg.collision_policy
        if policy == "skip":
            return None
        stem, suffix = action.target.stem, action.target.suffix
        if policy == "date_suffix":
            from datetime import datetime
            img = repo.get_by_id(action.image_id) if action.image_id else None
            when = img.taken_at if img is not None and img.taken_at else None
            if when is None:
                try:
                    when = datetime.fromtimestamp(os.stat(action.source).st_mtime)
                except OSError:
                    when = datetime.now()
            base = f"{stem}_{when:%Y%m%d}"
            candidate = action.target.with_name(f"{base}{suffix}")
        else:  # sequence: photo (1).jpg, photo (2).jpg
            base, candidate = stem, action.target
        n = 0
        while os.path.lexists(candidate):
            n += 1
            candidate = action.target.with_name(f"{base} ({n}){suffix}")
        return candidate

    def _apply(self, action: PlannedAction, target: Path, sha: str = "") -> str:
        """Perform the mutation; return the operation that actually happened."""
        src = action.source
        op = action.operation
        if op == "move":
            self._move(src, target, sha)
        elif op == "copy":
            self._copy(src, target)
        elif op == "hardlink":
            return self._link_or_copy(src, target, self._rep)
        elif op == "symlink":
            self._symlink(src, target)
        else:
            raise ValueError(f"unknown operation: {op}")
        return op

    @staticmethod
    def _same_device(a: Path, b: Path) -> bool:
        try:
            return os.stat(a).st_dev == os.stat(b).st_dev
        except OSError:
            return False

    @staticmethod
    def _check_space(src: Path, target: Path) -> None:
        need = file_size(src)
        free = shutil.disk_usage(_nearest_existing(target.parent)).free
        if need > free:
            raise OSError(f"not enough free space on target volume ({need} > {free})")

    @staticmethod
    def _move(src: Path, target: Path, expected_sha: str = "") -> None:
        """Move with same-device ``os.replace`` when possible, else copy+verify+delete."""
        # ``os.path.samefile`` raises if the target does not exist yet (the
        # normal case for a fresh move), so only consult it when it is present.
        if target.exists() and os.path.samefile(src, target):
            return  # already in place
        if ActionEngine._same_device(src, target.parent):
            os.replace(src, target)
            return
        # Cross-device: copy, verify against the ledger hash, then remove the source.
        if src.is_symlink():
            raise RuntimeError("refusing cross-device move of a symlink source")
        ActionEngine._check_space(src, target)
        _copy_or_cleanup(src, target, follow_symlinks=False)
        want = expected_sha or file_sha256(src)
        if file_sha256(target) != want or file_sha256(src) != want:
            target.unlink(missing_ok=True)
            raise RuntimeError("post-move hash mismatch (source preserved)")
        src.unlink()

    @staticmethod
    def _copy(src: Path, target: Path) -> None:
        ActionEngine._check_space(src, target)
        _copy_or_cleanup(src, target)

    @staticmethod
    def _link_or_copy(src: Path, target: Path, rep: ExecutionReport) -> str:
        """Hardlink, or copy when the filesystem refuses. Returns "hardlink" or "copy"."""
        try:
            os.link(src, target)
            return "hardlink"
        except FileExistsError:
            raise
        except OSError as e:
            if e.errno not in {errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise  # e.g. ENOSPC: do not mask a real failure with a copy attempt
            ActionEngine._check_space(src, target)
            _copy_or_cleanup(src, target)
            rep.copy_fallbacks += 1
            why = {errno.EXDEV: "cross-filesystem (EXDEV)",
                   errno.EPERM: "refused (EPERM, protected_hardlinks or ACL?)"}.get(
                       e.errno, f"unsupported (errno {e.errno})")
            log.warning("hardlink %s; copied %s -> %s", why, src, target)
            return "copy"

    @staticmethod
    def _symlink(src: Path, target: Path) -> None:
        target.symlink_to(src.resolve())


def _norm_categories(only: Iterable[str] | str | None) -> set[str] | None:
    if only is None:
        return None
    if isinstance(only, str):
        only = [only]
    vals = {str(getattr(c, "value", c)).lower() for c in only}
    return vals or None


def _cat_value(action: PlannedAction) -> str:
    c = action.category
    return str(getattr(c, "value", c)).lower()


def _is_junk(action: PlannedAction) -> bool:
    return bool(getattr(action.category, "is_junk", False))


def _copy_or_cleanup(src: Path, target: Path, follow_symlinks: bool = True) -> None:
    """copy2, removing a truncated target on failure (never one that pre-existed)."""
    existed = os.path.lexists(target)
    try:
        shutil.copy2(str(src), str(target), follow_symlinks=follow_symlinks)
    except BaseException:
        if not existed:
            Path(target).unlink(missing_ok=True)
        raise


def _rel(target: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return str(target.relative_to(root))
        except ValueError:
            pass
    return target.name

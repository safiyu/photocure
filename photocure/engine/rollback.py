"""Atomic rollback of committed transactions.

Reverses every ``COMMITTED`` action in a transaction in **reverse
chronological order** (spec section 2.E), verifying the destination file's
SHA-256 (and size) before relocating it back to its exact original path.

Per-operation reversal:
- ``move``    : move target -> source; remove the target if it is now a
                leftover empty directory.
- ``copy``    : delete the target copy (source is untouched by copy).
- ``hardlink``: unlink the target link (the inode/source data is preserved).
- ``symlink`` : remove the symlink (never follows it).

Rollback is idempotent: actions already ``ROLLED_BACK`` are skipped. Rows left
``PENDING`` by a crash are reconciled against the filesystem first.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CuratorConfig
from ..db.repository import PhotoRepository
from ..db.session import get_session
from .hash import file_sha256, file_size

log = logging.getLogger("photocure.rollback")


@dataclass
class RollbackReport:
    transaction_id: str
    reversed: int = 0
    skipped: int = 0
    failures: int = 0
    duration_s: float = 0.0
    errors: list = field(default_factory=list)
    notes: list = field(default_factory=list)


class RollbackEngine:
    """Reverse a committed transaction using the action ledger."""

    def __init__(self, config: CuratorConfig | None = None,
                 db_path: Path | None = None):
        self.cfg = config or CuratorConfig()
        self.db_path = db_path or self.cfg.db_path

    def rollback_transaction(self, transaction_id: str,
                             verify: bool = True,
                             cancel_check: Callable[[], bool] | None = None) -> RollbackReport:
        rep = RollbackReport(transaction_id=transaction_id)
        import time
        t0 = time.monotonic()
        with get_session(self.db_path) as session:
            repo = PhotoRepository(session)
            self._reconcile_pending(transaction_id, repo, rep)
            session.commit()
            entries = repo.ledger_committed_desc(transaction_id)
            if not entries:
                rep.notes.append("no committed actions found for transaction")
                return rep

            for entry in entries:  # already reverse-chronological
                if cancel_check and cancel_check():
                    break
                try:
                    ok, note = self._reverse_one(entry, repo, verify)
                    if ok:
                        repo.mark_rolled_back(entry)
                        img = repo.get_by_id(entry.image_id) if entry.image_id else None
                        if img is not None:
                            img.route_status = "unrouted"
                            img.target_path = None
                            if entry.operation_type == "move":
                                img.path = entry.source_path
                                if entry.prev_relative_path:
                                    img.relative_path = entry.prev_relative_path
                        rep.reversed += 1
                    elif note == "already_reverted":
                        repo.mark_rolled_back(entry)
                        rep.skipped += 1
                        rep.notes.append(note)
                    else:
                        rep.failures += 1
                        rep.errors.append(f"{entry.target_path}: {note}")
                    session.commit()
                except Exception as e:
                    log.error("rollback failed for %s: %s", entry.target_path, e)
                    session.rollback()
                    rep.failures += 1
                    rep.errors.append(f"{entry.target_path}: {e}")
        rep.duration_s = time.monotonic() - t0
        return rep

    # ------------------------------------------------------------- recovery
    def _reconcile_pending(self, transaction_id: str, repo, rep: RollbackReport) -> None:
        """Resolve ``PENDING`` rows left by a crash by inspecting the disk."""
        for entry in repo.ledger_pending(transaction_id):
            target, source = Path(entry.target_path), Path(entry.source_path)
            op = entry.operation_type
            if not os.path.lexists(target):
                repo.ledger_fail(entry, "crash before mutation (nothing on disk)")
                rep.notes.append(f"pending without effect: {target}")
                continue
            intact = self._target_matches(target, entry, op)
            if op == "move":
                complete = intact and not os.path.lexists(source)
            else:
                complete = intact
            if complete:
                repo.ledger_commit(entry)
                img = repo.get_by_id(entry.image_id) if entry.image_id else None
                if img is not None and op == "move":
                    img.path = str(target)
                    img.target_path = str(target)
                    img.route_status = "executed"
                rep.notes.append(f"pending recovered as committed: {target}")
            else:
                # Partial copy or interrupted cross-device move: source still holds the data.
                try:
                    target.unlink()
                except OSError:
                    pass
                repo.ledger_fail(entry, "crash mid-operation (partial target removed)")
                rep.notes.append(f"pending partial target removed: {target}")

    @staticmethod
    def _target_matches(target: Path, entry, op: str) -> bool:
        if op == "symlink":
            return target.is_symlink()
        if not entry.sha256:
            return True
        try:
            return file_sha256(target) == entry.sha256
        except OSError:
            return False

    # ------------------------------------------------------------- helpers
    def _reverse_one(self, entry, repo, verify: bool):
        """Reverse a single committed action. Returns (ok, note)."""
        target = Path(entry.target_path)
        source = Path(entry.source_path)
        op = entry.operation_type

        # Pre-flight verification: the destination must exist and match the
        # recorded hash before we touch anything.
        if not target.exists() and not target.is_symlink():
            return False, "destination missing (nothing to revert)"
        if verify and entry.sha256:
            try:
                actual = file_sha256(target) if not target.is_symlink() \
                    else file_sha256(target.resolve())
            except OSError:
                actual = ""
            if actual != entry.sha256:
                return False, ("hash mismatch (refusing to revert; "
                               f"expected {entry.sha256[:12]}.. got {actual[:12]}..)")
        if verify and entry.file_size and not target.is_symlink():
            if file_size(target) != entry.file_size:
                return False, "size mismatch (refusing to revert)"

        # If the source slot is already occupied (file or symlink), never clobber it.
        if os.path.lexists(source):
            if op in ("copy", "hardlink", "symlink"):
                # Source still there; just drop the redundant target.
                self._remove_target(target, op)
                self._prune_empty_dirs(target.parent, entry.created_dir)
                return True, "source already present; removed target copy"
            return False, "source path already occupied; manual review needed"

        source.parent.mkdir(parents=True, exist_ok=True)

        if op == "move":
            if self._same_device(target, source.parent):
                os.replace(str(target), str(source))
            else:
                self._verified_move(target, source, entry.sha256, verify)
            self._prune_empty_dirs(target.parent, entry.created_dir)
            return True, "moved back"

        if op in ("copy", "hardlink", "symlink"):
            self._remove_target(target, op)
            self._prune_empty_dirs(target.parent, entry.created_dir)
            return True, f"removed {op} target"

        return False, f"unknown operation '{op}'"

    @staticmethod
    def _verified_move(target: Path, source: Path, sha: str, verify: bool) -> None:
        if target.is_symlink():
            raise RuntimeError("refusing cross-device move of a symlink")
        shutil.copy2(str(target), str(source), follow_symlinks=False)
        if verify and sha and file_sha256(source) != sha:
            source.unlink(missing_ok=True)
            raise RuntimeError("hash mismatch after copy back (target preserved)")
        target.unlink()

    @staticmethod
    def _remove_target(target: Path, op: str) -> None:
        try:
            # unlink removes only this link/copy and never follows symlinks.
            if target.is_symlink() or target.exists():
                target.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _same_device(a: Path, b: Path) -> bool:
        try:
            return os.stat(a).st_dev == os.stat(b).st_dev
        except OSError:
            return False

    @staticmethod
    def _prune_empty_dirs(start: Path, top: str = "") -> None:
        """Remove empty dirs from ``start`` up to and including ``top``.

        ``top`` is the topmost directory the original action created, so
        pre-existing parents are never touched. An empty ``top`` prunes nothing.
        """
        if not top:
            return
        stop = Path(top)
        cur = Path(start)
        while True:
            if cur != stop and stop not in cur.parents:
                break
            try:
                if any(cur.iterdir()):
                    break
                cur.rmdir()
            except (OSError, FileNotFoundError):
                break
            if cur == stop:
                break
            cur = cur.parent

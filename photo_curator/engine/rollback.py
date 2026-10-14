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

Rollback is idempotent: actions already ``ROLLED_BACK`` are skipped.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CuratorConfig
from ..db.repository import PhotoRepository
from ..db.session import get_session
from .hash import file_sha256, file_size

log = logging.getLogger("photo_curator.rollback")


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
                             verify: bool = True) -> RollbackReport:
        rep = RollbackReport(transaction_id=transaction_id)
        import time
        t0 = time.monotonic()
        with get_session(self.db_path) as session:
            repo = PhotoRepository(session)
            entries = repo.ledger_committed_desc(transaction_id)
            if not entries:
                rep.notes.append("no committed actions found for transaction")
                return rep

            for entry in entries:  # already reverse-chronological
                try:
                    ok, note = self._reverse_one(entry, repo, verify)
                    if ok:
                        repo.mark_rolled_back(entry)
                        img = repo.get_by_id(entry.image_id) if entry.image_id else None
                        if img is not None:
                            img.route_status = "unrouted"
                            img.target_path = None
                        rep.reversed += 1
                    elif note == "already_reverted":
                        repo.mark_rolled_back(entry)
                        rep.skipped += 1
                        rep.notes.append(note)
                    else:
                        rep.failures += 1
                        rep.errors.append(f"{entry.target_path}: {note}")
                except Exception as e:
                    log.error("rollback failed for %s: %s", entry.target_path, e)
                    rep.failures += 1
                    rep.errors.append(f"{entry.target_path}: {e}")
        rep.duration_s = time.monotonic() - t0
        return rep

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

        # If the source slot is already occupied (e.g. user restored a file),
        # do not clobber it.
        if source.exists() and not source.is_symlink():
            if op in ("copy", "hardlink", "symlink"):
                # Source still there; just drop the redundant target.
                self._remove_target(target, op)
                return True, "source already present; removed target copy"
            return False, "source path already occupied; manual review needed"

        source.parent.mkdir(parents=True, exist_ok=True)

        if op == "move":
            if target.exists() or target.is_symlink():
                # Move it back (os.replace on same device, else copy+delete).
                if self._same_device(target, source.parent):
                    import os
                    os.replace(str(target), str(source))
                else:
                    shutil.move(str(target), str(source))
            self._prune_empty_dirs(target.parent)
            return True, "moved back"

        if op in ("copy", "hardlink", "symlink"):
            self._remove_target(target, op)
            self._prune_empty_dirs(target.parent)
            return True, f"removed {op} target"

        return False, f"unknown operation '{op}'"

    @staticmethod
    def _remove_target(target: Path, op: str) -> None:
        try:
            if target.is_symlink():
                target.unlink()
            elif target.exists():
                if op == "hardlink":
                    target.unlink()  # removes just this link, preserves inode
                else:
                    target.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _same_device(a: Path, b: Path) -> bool:
        import os
        try:
            return os.stat(a).st_dev == os.stat(b).st_dev
        except OSError:
            return False

    @staticmethod
    def _prune_empty_dirs(start: Path) -> None:
        """Remove ``start`` and any empty parent dirs up to (not including) the
        organize root, so an undo leaves no empty scaffolding behind."""
        cur = start
        while cur != cur.parent:
            try:
                if any(cur.iterdir()):
                    break
                cur.rmdir()
            except (OSError, FileNotFoundError):
                break
            cur = cur.parent

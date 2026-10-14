"""Transactional filesystem mutation engine.

Safety guarantees (spec section 7.1):
- **Default dry-run**: no disk mutation unless ``execute`` is called.
- **No overwrites**: name collisions are renamed with a ``_dup<SHA[:6]>`` suffix.
- **Move verification**: before unlinking a source after a cross-device move,
  the target's size + SHA-256 are verified to match.
- **Transactional ledger**: every mutation is recorded ``PENDING`` first, then
  flipped to ``COMMITTED`` only after the disk operation succeeds.

The engine is fully reversible via :class:`RollbackEngine`.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from ..config import CuratorConfig
from ..db.repository import PhotoRepository
from ..db.session import get_session
from ..rules.organizer import PlannedAction
from .hash import file_sha256, file_size

log = logging.getLogger("photo_curator.action")


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
    # image_id -> final on-disk target actually used (after collision rename).
    resolved_targets: dict = field(default_factory=dict)


class ActionEngine:
    """Execute a :class:`PlannedAction` list transactionally."""

    def __init__(self, config: CuratorConfig | None = None,
                 db_path: Optional[Path] = None):
        self.cfg = config or CuratorConfig()
        self.db_path = db_path or self.cfg.db_path

    # -------------------------------------------------------------- dry-run
    def dry_run(self, plan: Sequence[PlannedAction]) -> ExecutionReport:
        """Report what *would* happen without touching the filesystem."""
        rep = ExecutionReport(transaction_id=new_transaction_id(), dry_run=True)
        for a in plan:
            # Dry-run collision detection (target already exists on disk).
            if a.target.exists():
                a.collides = True
                rep.collisions += 1
            rep.executed += 1  # "would execute"
        rep.duration_s = 0.0
        return rep

    # --------------------------------------------------------------- execute
    def execute(self, plan: Sequence[PlannedAction],
                transaction_id: Optional[str] = None,
                ) -> ExecutionReport:
        """Apply every planned mutation, recording the ledger as we go.

        For each action the ledger row is written ``PENDING`` *before* the disk
        mutation, then flipped to ``COMMITTED`` on success or ``FAILED`` on
        error -- matching the spec's transactional contract.
        """
        txn = transaction_id or new_transaction_id()
        rep = ExecutionReport(transaction_id=txn, dry_run=False)
        t0 = time.monotonic()
        with get_session(self.db_path) as session:
            repo = PhotoRepository(session)
            for action in plan:
                sha = action.sha256 or ""
                size = action.file_size or 0
                try:
                    if not action.source.exists():
                        raise FileNotFoundError(f"source missing: {action.source}")
                    if not sha:
                        sha = file_sha256(action.source)
                        size = file_size(action.source)
                    action.sha256, action.file_size = sha, size

                    # 1. PENDING ledger row (pre-mutation).
                    entry = repo.ledger_add(
                        txn, action.image_id, str(action.source),
                        str(action.target), action.operation, sha, size)

                    # 2. Resolve collision + apply mutation.
                    target = self._prepare_target(action, repo)
                    self._apply(action, target)

                    # 3. COMMIT ledger row (post-mutation success).
                    action.target = target
                    repo.ledger_commit(entry)
                    img = repo.get_by_id(action.image_id)
                    if img is not None:
                        img.target_path = str(target)
                        img.route_status = "executed"

                    rep.executed += 1
                    rep.resolved_targets[action.image_id] = str(target)
                    if action.collides:
                        rep.collisions += 1
                except Exception as e:
                    log.error("action failed for %s: %s", action.source, e)
                    rep.failures += 1
                    rep.errors.append(f"{action.source}: {e}")
                    try:
                        repo.ledger_fail(entry, str(e))
                    except Exception:
                        pass
        rep.duration_s = time.monotonic() - t0
        return rep

    # ------------------------------------------------------------- helpers
    def _prepare_target(self, action: PlannedAction, repo) -> Path:
        """Ensure the destination directory exists and resolve collisions.

        Returns the final target path (possibly renamed with ``_dup<sha6>``).
        """
        action.target.parent.mkdir(parents=True, exist_ok=True)
        final = action.target
        if final.exists():
            action.collides = True
            stem = action.target.stem
            sha6 = (action.sha256 or file_sha256(action.source))[:6]
            candidate = action.target.with_name(f"{stem}_dup{sha6}{action.target.suffix}")
            n = 1
            while candidate.exists():
                candidate = action.target.with_name(
                    f"{stem}_dup{sha6}_{n}{action.target.suffix}")
                n += 1
            final = candidate
        return final

    def _apply(self, action: PlannedAction, target: Path) -> None:
        src = action.source
        op = action.operation
        if op == "move":
            self._move(src, target)
        elif op == "copy":
            self._copy(src, target)
        elif op == "hardlink":
            self._hardlink(src, target)
        elif op == "symlink":
            self._symlink(src, target)
        else:
            raise ValueError(f"unknown operation: {op}")

    @staticmethod
    def _same_device(a: Path, b: Path) -> bool:
        try:
            return os.stat(a).st_dev == os.stat(b).st_dev
        except OSError:
            return False

    @staticmethod
    def _move(src: Path, target: Path) -> None:
        """Move with same-device ``os.replace`` when possible, else copy+verify+delete."""
        # ``os.path.samefile`` raises if the target does not exist yet (the
        # normal case for a fresh move), so only consult it when it is present.
        if target.exists() and os.path.samefile(src, target):
            return  # already in place
        if ActionEngine._same_device(src, target.parent):
            os.replace(src, target)
            return
        # Cross-device: copy, verify hash, then remove the source.
        shutil.copy2(str(src), str(target))
        if file_sha256(target) != file_sha256(src):
            target.unlink(missing_ok=True)
            raise RuntimeError("post-move hash mismatch (source preserved)")
        src.unlink()

    @staticmethod
    def _copy(src: Path, target: Path) -> None:
        shutil.copy2(str(src), str(target))

    @staticmethod
    def _hardlink(src: Path, target: Path) -> None:
        try:
            os.link(src, target)
        except FileExistsError:
            raise
        except OSError as e:
            # Cross-device or filesystem without hardlink support -> copy.
            log.warning("hardlink failed (%s); using copy", e)
            shutil.copy2(str(src), str(target))

    @staticmethod
    def _symlink(src: Path, target: Path) -> None:
        try:
            target.symlink_to(src.resolve())
        except FileExistsError:
            raise

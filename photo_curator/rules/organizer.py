"""Folder organizer: resolve destination paths for a routing plan.

Option 1 (default, category-first date hierarchy)::

    Organized/
    ├── People/2024-05-12/IMG_1024.JPG
    ├── Non-Human/Food/2024-05-12/IMG_1025.JPG
    ├── Non-Human/Scenery/2024-06-01/IMG_1050.JPG
    ├── Junk_Review/{Screenshots,Documents,Graphics,Accidental,Duplicates}/...
    ├── Uncertain_Review/...
    ├── Bursts/Burst_<id>/alternates/...
    └── Unknown_Date/...

Option 2 (in-place, non-destructive): originals stay put; a hardlink or
symlink is created in the same target tree. ``build_destination_plan`` returns
the identical target layout; the *operation* (hardlink/symlink vs move) is a
concern of the ActionEngine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections.abc import Sequence

from ..config import CuratorConfig
from ..db.models import BurstGroup, ImageRecord
from ..rules.decision_matrix import Decision, _date_components
from ..rules.taxonomy import CategoryType


class OrganizeMode(str, Enum):
    CATEGORY_FIRST = "category_first"   # Option 1
    IN_PLACE = "in_place"               # Option 2


@dataclass
class PlannedAction:
    """A single source -> target filesystem mutation."""

    image_id: int
    source: Path
    target: Path
    category: CategoryType
    operation: str = "move"   # move | copy | hardlink | symlink
    reason: str = ""
    is_alternate: bool = False
    burst_name: str | None = None
    # Collision flag set by the ActionEngine at execution time.
    collides: bool = False
    # Filled by the ActionEngine at execution time (ledger provenance).
    sha256: str = ""
    file_size: int = 0


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(name: str) -> str:
    """Strip characters that are unsafe in cross-platform filenames."""
    cleaned = _SAFE_RE.sub("_", name).strip("._")
    return cleaned or "unnamed"


def _safe_date(datestr: str) -> str:
    return _SAFE_RE.sub("_", datestr).strip("_") or "Unknown_Date"


class FolderOrganizer:
    """Build the destination path graph from a set of decisions."""

    def __init__(self, config: CuratorConfig | None = None,
                 organize_root: Path | None = None,
                 mode: OrganizeMode = OrganizeMode.CATEGORY_FIRST):
        self.cfg = config or CuratorConfig()
        self.root = Path(organize_root or self.cfg.organize_root)
        self.mode = mode
        # burst id -> stable burst name (set via set_bursts before planning).
        self._burst_id_to_name: dict[int, str] = {}

    # --------------------------------------------------------------- bursts
    def set_bursts(self, bursts: Sequence[BurstGroup]) -> None:
        for b in bursts:
            if b.id is not None:
                self._burst_id_to_name[b.id] = b.name

    def burst_name_for(self, rec: ImageRecord) -> str | None:
        if rec.burst_id is not None:
            return self._burst_id_to_name.get(rec.burst_id)
        return None

    # -------------------------------------------------------------- resolve
    def resolve_target(self, rec: ImageRecord, decision: Decision) -> Path:
        """Compute the target path (without the filename) for one image."""
        if decision.is_alternate:
            return self._alternate_path(rec)

        cat = decision.category
        base = self.root / cat.folder
        date = _date_components(rec.taken_at) if decision.needs_date else None
        if date is None and decision.needs_date:
            # Face/food/scenery/animal but no usable date -> Unknown_Date.
            date = (self.cfg.unknown_date_dir, "00", "00")
        if date is not None and decision.needs_date:
            base = base / _safe_date(date[0]) / _safe_date(date[1]) / _safe_date(date[2])
        elif cat == CategoryType.UNKNOWN or (not decision.needs_date and date is None and rec.taken_at is None and cat not in (
            CategoryType.SCREENSHOT, CategoryType.DOCUMENT, CategoryType.GRAPHIC,
            CategoryType.DUPLICATE, CategoryType.ACCIDENTAL, CategoryType.UNCERTAIN)):
            base = self.root / self.cfg.unknown_date_dir
        return base

    def _alternate_path(self, rec: ImageRecord) -> Path:
        name = self.burst_name_for(rec) or f"Burst_{rec.id:06d}"
        return self.root / "Bursts" / _safe_name(name) / "alternates"

    def full_target(self, rec: ImageRecord, decision: Decision) -> Path:
        d = self.resolve_target(rec, decision)
        return d / _safe_name(rec.filename)

    # --------------------------------------------------------------- plan
    def build_destination_plan(
            self,
            records: Sequence[ImageRecord],
            decisions: Sequence[Decision],
            operation: str = "move",
            ) -> list[PlannedAction]:
        """Produce the ordered list of :class:`PlannedAction` for the plan."""
        plan: list[PlannedAction] = []
        # Option 2 uses hardlinks by default (non-destructive, no extra space).
        op = operation
        if self.mode == OrganizeMode.IN_PLACE and op == "move":
            op = "hardlink"
        for rec, dec in zip(records, decisions):
            target = self.full_target(rec, dec)
            plan.append(PlannedAction(
                image_id=rec.id,
                source=Path(rec.path),
                target=target,
                operation=op,
                category=dec.category,
                reason=dec.reason,
                is_alternate=dec.is_alternate,
                burst_name=self.burst_name_for(rec) if dec.is_alternate else None,
            ))
        return plan

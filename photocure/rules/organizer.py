"""Folder organizer: resolve destination paths for a routing plan.

Option 1 (default, category-first date hierarchy)::

    Organized/
    ├── People/2024/05/12/IMG_1024.JPG
    ├── People/Unknown_Date/IMG_0001.JPG          (no usable date)
    ├── Non-Human/Food/2024/05/12/IMG_1025.JPG
    ├── Non-Human/Scenery/2024/06/01/IMG_1050.JPG
    ├── Utility_Reference/{Screenshots,Documents,Notes_and_Whiteboards,
    │                      Labels_and_Signs}/...
    ├── Junk_Review/{Graphics,Accidental,Duplicates,Near_Duplicates,Unreadable}/...
    ├── Uncertain_Review/...
    ├── Bursts/Burst_<id>/alternates/...
    └── Unknown_Date/...

With ``use_events`` the day folder carries the event label, e.g.
``People/2024/05/12_Event3/``.

Option 2 (in-place, non-destructive): originals stay put; a hardlink or
symlink is created in the same target tree. ``build_destination_plan`` returns
the identical target layout; the *operation* (hardlink/symlink vs move) is a
concern of the ActionEngine.
"""
from __future__ import annotations

import re
import unicodedata
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
# Only characters that are illegal on common filesystems; Unicode is kept.
_RESERVED_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')
_MAX_NAME_BYTES = 255


def _safe_name(name: str) -> str:
    """Make a filename safe across filesystems while keeping Unicode letters."""
    cleaned = _RESERVED_RE.sub("_", unicodedata.normalize("NFC", name)).strip(" .")
    if not cleaned:
        return "unnamed"
    if len(cleaned.encode("utf-8")) > _MAX_NAME_BYTES:
        stem, dot, ext = cleaned.rpartition(".")
        if not dot or len(ext.encode("utf-8")) > 16:
            stem, ext = cleaned, ""
        else:
            ext = "." + ext
        budget = _MAX_NAME_BYTES - len(ext.encode("utf-8"))
        stem = stem.encode("utf-8")[:budget].decode("utf-8", "ignore")
        cleaned = (stem + ext).strip(" .") or "unnamed"
    return cleaned


def _safe_date(datestr: str) -> str:
    return _SAFE_RE.sub("_", datestr).strip("_") or "Unknown_Date"


class FolderOrganizer:
    """Build the destination path graph from a set of decisions."""

    def __init__(self, config: CuratorConfig | None = None,
                 organize_root: Path | None = None,
                 mode: OrganizeMode = OrganizeMode.CATEGORY_FIRST,
                 use_events: bool = False):
        self.cfg = config or CuratorConfig()
        self.root = Path(organize_root or self.cfg.organize_root)
        self.mode = mode
        self.use_events = use_events
        # burst id -> stable burst name (set via set_bursts before planning).
        self._burst_id_to_name: dict[int, str] = {}
        # event id -> label (set via set_events when ``use_events`` is on).
        self._event_labels: dict[int, str] = {}

    # --------------------------------------------------------------- bursts
    def set_bursts(self, bursts: Sequence[BurstGroup]) -> None:
        for b in bursts:
            if b.id is not None:
                self._burst_id_to_name[b.id] = b.name

    def burst_name_for(self, rec: ImageRecord) -> str | None:
        if rec.burst_id is not None:
            return self._burst_id_to_name.get(rec.burst_id)
        return None

    # --------------------------------------------------------------- events
    def set_events(self, events: Sequence) -> None:
        """Register EventGroup rows so day folders can carry the event label."""
        for e in events:
            if e.id is not None:
                self._event_labels[e.id] = e.label or f"Event{e.id}"

    def event_label_for(self, rec: ImageRecord) -> str | None:
        eid = getattr(rec, "event_id", None)
        if not self.use_events or eid is None:
            return None
        return self._event_labels.get(eid, f"Event{eid}")

    # -------------------------------------------------------------- resolve
    def resolve_target(self, rec: ImageRecord, decision: Decision) -> Path:
        """Compute the target path (without the filename) for one image."""
        if decision.is_alternate:
            return self._alternate_path(rec)

        cat = decision.category
        base = self.root / self._folder_for(decision)
        if decision.needs_date:
            date = _date_components(rec.taken_at)
            if date is None:
                # Date-grouped class without a usable date (never 1970).
                return base / self.cfg.unknown_date_dir
            day = _safe_date(date[2])
            label = self.event_label_for(rec)
            if label:
                day = _safe_name(f"{day}_{label}")
            return base / _safe_date(date[0]) / _safe_date(date[1]) / day
        if cat == CategoryType.UNKNOWN:
            return self.root / self.cfg.unknown_date_dir
        return base

    @staticmethod
    def _folder_for(decision: Decision) -> str:
        if decision.folder:
            return decision.folder
        return decision.category.folder

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

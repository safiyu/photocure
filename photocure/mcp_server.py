"""MCP server: lets an AI agent search, look at and act on the photo library.

The agent finds photos by what they show (CLIP text search), by date, or by category, then
asks for the picture itself and reads it, for example a router label or a receipt. Two tools
write NEW files (an open-eye composite, a clean export copy); nothing ever changes, moves or
deletes an existing photo or a database row. stdout carries the protocol, so everything that
logs goes to stderr.
"""
from __future__ import annotations

import functools
import logging
import os
import sys
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func
from sqlmodel import col, select

from .config import CuratorConfig
from .db import models as M
from .db.session import get_session, init_db
from .rules.taxonomy import CategoryType

log = logging.getLogger("photocure.mcp")

# Module level so the tools' string annotations (``-> Image``) resolve; the SDK is optional.
try:
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as MCPServer, Image
        from mcp.server.fastmcp.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError:
        Image = MCPServer = ToolError = ToolAnnotations = None


def _guard(fn):
    """Show the agent the real reason a call failed instead of a generic server error."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            raise ToolError(str(e)) from None
    return wrapper

MAX_RESULTS = 50
VIEW_TIERS = ("medium", "preview")
_CATEGORIES = sorted(c.value for c in CategoryType)

INSTRUCTIONS = (
    "Read-only access to the user's Photocure photo library. To answer a question about a "
    "photo (for example a label, receipt or sign): 1) call search_photos with a short "
    "description and an ISO date range (you know today's date, so work out phrases like "
    "'last month' yourself); 2) call view_photo on the best matches and read the text in "
    "the image; 3) answer from what you can actually see, and say so if it is unreadable. "
    "Labels, serial numbers and signs are usually in the 'utility' category. "
    "Group shots with a blinking person: list_bursts(only_fixable=true) finds bursts where two "
    "frames together cover everyone; create_composite then saves a fixed photo (or a layered "
    "PSD) next to the original. To share photos safely, find their ids and call clean_export; "
    "it strips location and camera details and can blur or leave out photos showing license "
    "plates, cards, screens, house numbers or ID documents (that check can miss small details, "
    "so tell the user to look the result over). Only create_composite and clean_export write "
    "files, and only new ones; no tool can modify, move or delete an existing photo."
)


def _day(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise ValueError(f"{name} must be an ISO date like 2026-09-30, got {value!r}") from None


class LibraryQueries:
    """The queries behind the MCP tools. Plain Python so it can be tested without MCP."""

    def __init__(self, cfg: CuratorConfig):
        self.cfg = cfg
        self.db = cfg.db_path
        self._encoder = None

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _eff():
        return func.coalesce(M.ImageRecord.manual_bucket, M.ImageRecord.category)

    def _filters(self, date_from, date_to, category) -> list:
        I = M.ImageRecord
        conds = [I.scan_status != "missing"]
        d0, d1 = _day(date_from, "date_from"), _day(date_to, "date_to")
        if d0 is not None:
            conds.append(I.taken_at >= datetime.combine(d0, dtime.min, tzinfo=timezone.utc))
        if d1 is not None:
            conds.append(I.taken_at < datetime.combine(d1, dtime.min, tzinfo=timezone.utc)
                         + timedelta(days=1))
        if category:
            if category not in _CATEGORIES:
                raise ValueError(f"unknown category {category!r}; choose one of {_CATEGORIES}")
            conds.append(self._eff() == category)
        return conds

    @staticmethod
    def _row(r: M.ImageRecord, score: float | None = None) -> dict:
        out = {
            "id": r.id, "filename": r.filename, "path": r.path,
            "taken_at": r.taken_at.isoformat() if r.taken_at else None,
            "category": r.manual_bucket or r.category,
        }
        if score is not None:
            out["score"] = round(score, 4)
        return out

    # ------------------------------------------------------------------ tools
    def overview(self) -> dict:
        I = M.ImageRecord
        with get_session(self.db) as s:
            total = s.exec(select(func.count()).select_from(I)).one()
            first, last = s.exec(select(func.min(I.taken_at), func.max(I.taken_at))).one()
            rows = s.exec(select(self._eff(), func.count()).group_by(self._eff())).all()
            analysed = s.exec(select(func.count()).select_from(I).where(
                I.infer_status == "inferred")).one()
        return {
            "photos": int(total), "analysed": int(analysed),
            "earliest": first.isoformat() if first else None,
            "latest": last.isoformat() if last else None,
            "by_category": {(k or "unsorted"): int(n) for k, n in rows},
            "categories": _CATEGORIES,
        }

    def list_photos(self, date_from=None, date_to=None, category=None,
                    filename_contains=None, limit=20, offset=0) -> dict:
        I = M.ImageRecord
        limit = max(1, min(int(limit), MAX_RESULTS))
        conds = self._filters(date_from, date_to, category)
        if filename_contains:
            like = "%" + filename_contains.replace("\\", "\\\\").replace("%", "\\%") \
                .replace("_", "\\_") + "%"
            conds.append(I.filename.ilike(like, escape="\\"))
        with get_session(self.db) as s:
            total = s.exec(select(func.count()).select_from(I).where(*conds)).one()
            rows = s.exec(select(I).where(*conds).order_by(col(I.taken_at).desc(), col(I.id))
                          .limit(limit).offset(max(0, int(offset)))).all()
            return {"total": int(total), "items": [self._row(r) for r in rows]}

    def search(self, query, date_from=None, date_to=None, category=None, limit=10) -> dict:
        from . import search as search_mod
        from .clustering.similarity import _unit, top_k
        text = " ".join(str(query).split())
        if not text:
            raise ValueError("query is empty")
        limit = max(1, min(int(limit), MAX_RESULTS))
        if self._encoder is None:
            self._encoder = search_mod.default_encoder(self.cfg)
        try:
            vec = self._encoder.encode(text)
        except search_mod.SearchUnavailable as e:
            raise RuntimeError(f"{e} Use list_photos with a date or category instead.") from None
        I, S = M.ImageRecord, M.SemanticScore
        conds = [col(S.backend) != "stub", col(I.is_duplicate) == False,  # noqa: E712
                 col(I.is_alternate) == False, *self._filters(date_from, date_to, category)]  # noqa: E712
        with get_session(self.db) as s:
            pairs = top_k(s, _unit(vec), limit, filters=tuple(conds), join_images=True)
            ids = [i for i, _ in pairs]
            rows = s.exec(select(I).where(col(I.id).in_(ids))).all() if ids else []
            by_id = {r.id: r for r in rows}
            return {"items": [self._row(by_id[i], sc) for i, sc in pairs if i in by_id]}

    def list_bursts(self, date_from=None, date_to=None, only_fixable=False, limit=20) -> dict:
        from .clustering.group_shots import group_matrix_payload
        from sqlalchemy.orm import selectinload
        limit = max(1, min(int(limit), MAX_RESULTS))
        d0, d1 = _day(date_from, "date_from"), _day(date_to, "date_to")
        I, B = M.ImageRecord, M.BurstGroup
        out = []
        with get_session(self.db) as s:
            for b in s.exec(select(B).order_by(col(B.start_time).desc(), col(B.id).desc())).all():
                day = b.start_time.date() if b.start_time else None
                if (d0 or d1) and (day is None or (d0 and day < d0) or (d1 and day > d1)):
                    continue
                members = s.exec(select(I).where(
                    I.burst_id == b.id, I.infer_status == "inferred"
                ).options(selectinload(I.quality), selectinload(I.faces))).all()
                grp = group_matrix_payload(members)
                if only_fixable and grp["status"] != "composite":
                    continue
                out.append({
                    "id": b.id, "name": b.name, "photos": len(members),
                    "start_time": b.start_time.isoformat() if b.start_time else None,
                    "best_shot_id": b.best_shot_id, "group_status": grp["status"],
                    "partner_id": grp["partner_id"],
                    "frames_with_blinks": sum(1 for m in members if m.has_blinks),
                    "can_composite": grp["status"] == "composite",
                })
                if len(out) >= limit:
                    break
        return {"items": out}

    def composite(self, burst_id: int, mode: str = "fixed") -> dict:
        from .burst_actions import composite_burst
        return composite_burst(self.db, int(burst_id), mode)

    def clean_export(self, ids, output_dir, policy="blur", html=True, rename=True) -> dict:
        from .burst_actions import clean_export_selection
        return clean_export_selection(self.cfg, [int(i) for i in ids], output_dir,
                                      policy=policy, html=html, rename=rename)

    def info(self, image_id: int) -> dict:
        I = M.ImageRecord
        with get_session(self.db) as s:
            r = s.get(I, int(image_id))
            if r is None:
                raise ValueError(f"no photo with id {image_id}")
            sem = r.semantic
            out = self._row(r)
            out.update({
                "relative_path": r.relative_path, "width": r.width, "height": r.height,
                "camera_model": r.camera_model, "rating": r.rating,
                "favorite": bool(r.is_favorite), "city": r.city, "region": r.region,
                "country_code": r.country_code, "burst_id": r.burst_id,
                "is_best_shot": bool(r.is_best_shot), "has_blinks": bool(r.has_blinks),
                "faces": [f.eye_status for f in r.faces or []], "file_exists": any(
                    p and os.path.exists(p) for p in (r.path, r.target_path)),
            })
            if sem is not None:
                out["ai_guess"] = {"category": sem.top1_category,
                                   "confidence": round(sem.top1_score, 3)}
            return out

    def image_bytes(self, image_id: int, size: str = "preview") -> tuple[bytes, str]:
        """WebP bytes for one photo. The file comes from the database row, never from the caller."""
        if size not in VIEW_TIERS:
            raise ValueError(f"size must be one of {VIEW_TIERS}")
        from .thumbs import render_tier
        with get_session(self.db) as s:
            r = s.get(M.ImageRecord, int(image_id))
            if r is None:
                raise ValueError(f"no photo with id {image_id}")
            path = next((p for p in (r.path, r.target_path) if p and os.path.exists(p)), None)
            ext, orientation = r.extension, r.orientation
        if path is None:
            raise FileNotFoundError("the photo file is no longer on disk")
        return render_tier(path, ext or Path(path).suffix, orientation, size), "webp"

    def close(self) -> None:
        if self._encoder is not None:
            self._encoder.close()


def build_server(queries: LibraryQueries):
    """Register the tools on an MCP server. Needs the optional ``mcp`` package."""
    if MCPServer is None:
        raise ImportError(
            "The MCP server needs the 'mcp' package. Install it with: pip install 'photocure[mcp]'")
    server = MCPServer("photocure", instructions=INSTRUCTIONS)
    read = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
    write = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False,
                            open_world_hint=False)

    @server.tool(annotations=read)
    @_guard
    def library_overview() -> dict:
        """Photo count, date range, and how many photos are in each category."""
        return queries.overview()

    @server.tool(annotations=read)
    @_guard
    def search_photos(query: str, date_from: str | None = None, date_to: str | None = None,
                      category: str | None = None, limit: int = 10) -> dict:
        """Find photos by what they show, e.g. 'router label with wifi password'.

        Dates are ISO (YYYY-MM-DD, inclusive). Results are best match first with a score;
        view the top few with view_photo to confirm. Photos without a date are skipped when
        a date range is given.
        """
        return queries.search(query, date_from, date_to, category, limit)

    @server.tool(annotations=read)
    @_guard
    def list_photos(date_from: str | None = None, date_to: str | None = None,
                    category: str | None = None, filename_contains: str | None = None,
                    limit: int = 20, offset: int = 0) -> dict:
        """Browse photos by date, category or file name, newest first."""
        return queries.list_photos(date_from, date_to, category, filename_contains, limit, offset)

    @server.tool(annotations=read)
    @_guard
    def photo_info(image_id: int) -> dict:
        """Details for one photo: date, size, camera, place, category and the AI's guess."""
        return queries.info(image_id)

    @server.tool(annotations=read)
    @_guard
    def view_photo(image_id: int, size: str = "preview") -> Image:
        """Return the photo so you can read text in it. size: preview (1600px) or medium (800px)."""
        data, fmt = queries.image_bytes(image_id, size)
        return Image(data=data, format=fmt)

    @server.tool(annotations=read)
    @_guard
    def list_bursts(date_from: str | None = None, date_to: str | None = None,
                    only_fixable: bool = False, limit: int = 20) -> dict:
        """List burst groups, newest first. only_fixable=true keeps bursts where no single frame
        has everyone's eyes open but two frames together do, so create_composite can fix them."""
        return queries.list_bursts(date_from, date_to, only_fixable, limit)

    @server.tool(annotations=write)
    @_guard
    def create_composite(burst_id: int, mode: str = "fixed") -> dict:
        """Blend the open eyes of one burst frame into the best frame. mode 'fixed' saves
        IMG_xxxx_fixed.jpg next to the original; 'layered' saves a PSD with the aligned frame as
        a masked layer. Only new files are written. Fails clearly if the frames cannot be aligned."""
        return queries.composite(burst_id, mode)

    @server.tool(annotations=write)
    @_guard
    def clean_export(image_ids: list[int], output_dir: str, policy: str = "blur",
                     web_gallery: bool = True, neutral_names: bool = True) -> dict:
        """Write metadata-free copies (no GPS, camera serial or other EXIF) into a NEW folder
        under output_dir, which must be an absolute path outside the photo library. policy:
        'blur' blurs areas flagged as plates, cards, screens, house numbers or ID documents,
        'skip' leaves flagged photos out, 'include' exports without the check. The check needs
        the AI model and can miss small details. Returns the folder and what was flagged."""
        return queries.clean_export(image_ids, output_dir, policy, web_gallery, neutral_names)

    return server


def run_stdio(db: Path | str) -> None:
    """Serve the library at ``db`` over stdio until the client disconnects."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    from .services import apply_settings, load_settings
    cfg = CuratorConfig.load(Path(db).expanduser().resolve())
    cfg = apply_settings(cfg, load_settings(cfg.db_path))
    cfg.device.device = "cpu"  # the text search tower is small; never hold the GPU for it
    init_db(cfg.db_path)
    queries = LibraryQueries(cfg)
    try:
        build_server(queries).run("stdio")
    finally:
        queries.close()

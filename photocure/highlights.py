"""Highlights: rank photos by taste, pick the best of each event, and export a year in review.

The ranking starts from the model's aesthetic score. A learned personal preference can be
blended in, stars lift or lower a photo around three, and a favourite adds a fixed bonus. The same
formula exists in Python and as a SQL expression so listings and exports always agree.
"""
from __future__ import annotations

import base64
import html
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import and_, case, func, not_, or_
from sqlmodel import col, select

from .clustering.events import UNRELIABLE_TIME_SOURCES
from .db.models import EventGroup, ImageRecord, SemanticScore

log = logging.getLogger("photocure.highlights")


@dataclass(frozen=True)
class HighlightWeights:
    rating: float = 0.10       # added at five stars, subtracted at one, nothing at three
    favorite: float = 0.15     # added for a favourite
    preference: float = 0.35   # share of the learned taste score in the base score

    @classmethod
    def from_features(cls, features: dict) -> HighlightWeights:
        return cls(rating=float(features.get("highlight_rating_weight", cls.rating)),
                   favorite=float(features.get("highlight_favorite_weight", cls.favorite)),
                   preference=float(features.get("highlight_preference_weight", cls.preference)))


BAD_STARS = (-1, 1, 2)


def label_for(rating: int | None, favorite: bool) -> int | None:
    """1 liked, 0 disliked, None when the signal is neutral or contradicts itself."""
    liked = bool(favorite) or (rating is not None and rating >= 4)
    disliked = rating in BAD_STARS
    if liked == disliked:
        return None
    return 1 if liked else 0


def highlight_score(aesthetic: float | None, preference: float | None, rating: int | None,
                    favorite: bool, w: HighlightWeights) -> float | None:
    """The ranking score, or None when the photo cannot be a highlight (rejected or unscored).

    A photo with its own clear signal (a favourite or a good or bad rating) ignores the learned
    taste, because that signal is what the taste model was trained on and would count twice.
    """
    if aesthetic is None or rating == -1:
        return None
    use_taste = preference is not None and label_for(rating, favorite) is None
    base = (1.0 - w.preference) * aesthetic + w.preference * preference if use_taste else aesthetic
    if rating and 1 <= rating <= 5:
        base += w.rating * (rating - 3) / 2.0
    if favorite:
        base += w.favorite
    return float(base)


def highlight_expr(w: HighlightWeights):
    """SQL twin of :func:`highlight_score` over ``SemanticScore`` joined to ``ImageRecord``."""
    S, I = SemanticScore, ImageRecord
    stars_in = func.coalesce(col(I.rating), 0)
    liked = or_(col(I.is_favorite) == True, stars_in >= 4)  # noqa: E712
    disliked = stars_in.in_(BAD_STARS)
    labelled = or_(and_(liked, not_(disliked)), and_(disliked, not_(liked)))
    base = case((and_(col(S.preference).is_not(None), not_(labelled)),
                 (1.0 - w.preference) * col(S.aesthetic) + w.preference * col(S.preference)),
                else_=col(S.aesthetic))
    stars = case((and_(col(I.rating) >= 1, col(I.rating) <= 5),
                  (col(I.rating) - 3) * (w.rating / 2.0)), else_=0.0)
    fav = case((col(I.is_favorite) == True, w.favorite), else_=0.0)  # noqa: E712
    return base + stars + fav


def candidate_filters() -> list:
    """Conditions every highlight shares: scored, not a duplicate or alternate, not rejected."""
    I, S = ImageRecord, SemanticScore
    return [col(S.aesthetic).is_not(None),
            col(I.is_duplicate) == False,  # noqa: E712
            col(I.is_alternate) == False,  # noqa: E712
            col(I.scan_status) != "missing",
            or_(col(I.rating).is_(None), col(I.rating) != -1)]


def _joined(stmt):
    return stmt.join(SemanticScore, SemanticScore.image_id == ImageRecord.id)


def best_of_events(session, w: HighlightWeights, per_event: int = 1,
                   min_score: float | None = None) -> list[dict]:
    """The top ``per_event`` photos of every event, events in time order."""
    expr = highlight_expr(w)
    stmt = _joined(select(ImageRecord.id, ImageRecord.event_id, expr)).where(
        col(ImageRecord.event_id).is_not(None), *candidate_filters())
    if min_score is not None:
        stmt = stmt.where(col(SemanticScore.aesthetic) >= min_score)  # the model score, as in the photo grid
    by_event: dict[int, list[tuple[int, float]]] = {}
    for image_id, event_id, score in session.exec(stmt).all():
        by_event.setdefault(int(event_id), []).append((int(image_id), float(score)))
    events = {e.id: e for e in session.exec(select(EventGroup)).all()}
    out = []
    for event_id, rows in by_event.items():
        ev = events.get(event_id)
        if ev is None:
            continue
        rows.sort(key=lambda r: (-r[1], r[0]))
        out.append({"event_id": event_id, "label": ev.label or f"Event{event_id}",
                    "start": ev.start, "end": ev.end, "total": len(rows),
                    "count": min(per_event, len(rows)), "items": rows[:per_event]})
    out.sort(key=lambda e: (e["start"] is None, e["start"] or datetime.min, e["event_id"]))
    return out


def _year_bounds(year: int) -> tuple[datetime, datetime]:
    end = (datetime(year + 1, 1, 1, tzinfo=timezone.utc) if year < 9999
           else datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc))
    return datetime(year, 1, 1, tzinfo=timezone.utc), end


def _reliable_time():
    return [col(ImageRecord.taken_at).is_not(None),
            col(ImageRecord.taken_at_source).not_in(tuple(UNRELIABLE_TIME_SOURCES))]


def top_of_year(session, w: HighlightWeights, year: int, n: int = 50) -> list[tuple[int, float]]:
    """The ``n`` highest ranked photos taken in ``year`` as ``(image_id, score)``."""
    lo, hi = _year_bounds(year)
    expr = highlight_expr(w)
    stmt = (_joined(select(ImageRecord.id, expr))
            .where(*candidate_filters(), *_reliable_time(),
                   col(ImageRecord.taken_at) >= lo, col(ImageRecord.taken_at) < hi)
            .order_by(expr.desc(), col(ImageRecord.id)).limit(n))
    return [(int(i), float(s)) for i, s in session.exec(stmt).all()]


def years_available(session) -> list[tuple[int, int]]:
    """Years that have scored photos, as ``(year, count)`` newest first."""
    year = func.strftime("%Y", ImageRecord.taken_at)
    stmt = (_joined(select(year, func.count()))
            .where(*candidate_filters(), *_reliable_time()).group_by(year).order_by(year.desc()))
    return [(int(y), int(c)) for y, c in session.exec(stmt).all() if y]


_PAGE_CSS = """
body{margin:0;padding:32px;background:#14161a;color:#e8e8ea;font:15px/1.45 system-ui,sans-serif}
h1{font-size:28px;margin:0 0 4px}p.sub{margin:0 0 26px;color:#9aa0a6}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:18px}
figure{margin:0;background:#1d2026;border-radius:12px;overflow:hidden}
figure img{display:block;width:100%;aspect-ratio:4/3;object-fit:cover;background:#0e0f12}
.noimg{display:flex;align-items:center;justify-content:center;aspect-ratio:4/3;color:#777;background:#0e0f12}
figcaption{padding:10px 12px;font-size:13px;color:#b8bcc2}
figcaption b{display:block;color:#fff;font-size:14px;overflow-wrap:anywhere}
.rank{float:right;color:#f5c542;font-weight:600}
"""


def _thumb_data_uri(path: str | None, ext: str, orientation: int, tier: str) -> str | None:
    if not path or not Path(path).exists():
        return None
    from .thumbs import render_tier
    try:
        data = render_tier(path, ext or Path(path).suffix, orientation, tier)
    except Exception as e:
        log.warning("no thumbnail for %s: %s", path, e)
        return None
    return "data:image/webp;base64," + base64.b64encode(data).decode("ascii")


def export_html(session, ranked: list[tuple[int, float]], title: str, tier: str = "medium") -> str:
    """A self-contained contact sheet (thumbnails inlined, no scripts) for ``ranked`` photos."""
    cards = []
    for rank, (image_id, score) in enumerate(ranked, 1):
        rec = session.get(ImageRecord, image_id)
        if rec is None:
            continue
        uri = _thumb_data_uri(rec.path, rec.extension, rec.orientation, tier)
        picture = (f'<img src="{uri}" alt="{html.escape(rec.filename, quote=True)}">'
                   if uri else '<div class="noimg">Preview unavailable</div>')
        when = rec.taken_at.strftime("%d %b %Y") if rec.taken_at else ""
        cat = (rec.manual_bucket or rec.category or "").replace("_", " ")
        meta = ", ".join(x for x in (when, cat.capitalize()) if x)
        cards.append(f'<figure>{picture}<figcaption><span class="rank">#{rank}</span>'
                     f"<b>{html.escape(rec.filename)}</b>{html.escape(meta)}"
                     f" &middot; score {score:.2f}</figcaption></figure>")
    safe = html.escape(title)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{safe}</title><style>{_PAGE_CSS}</style></head><body>"
            f'<h1>{safe}</h1><p class="sub">{len(cards)} photos, best first. Made with Photocure.</p>'
            f'<div class="grid">{"".join(cards)}</div></body></html>')


def copy_originals(session, ranked: list[tuple[int, float]], out_dir: Path) -> int:
    """Copy the original files into ``out_dir`` as ``01_name.jpg`` and so on. Returns the count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rank, (image_id, _score) in enumerate(ranked, 1):
        rec = session.get(ImageRecord, image_id)
        if rec is None or not Path(rec.path).exists():
            continue
        shutil.copy2(rec.path, out_dir / f"{rank:02d}_{Path(rec.filename).name}")
        copied += 1
    return copied

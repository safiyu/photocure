"""Blink classification: a pure classifier, a greedy IoU face tracker, and the glue to the DB rows.

Flag, never cull: a blink only sets badges and a score penalty. Nothing here touches a file.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

import numpy as np

from ..config import FaceSettings

Box = tuple[float, float, float, float]
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def classify_blinks(tracks: list[list[tuple[int, float | None]]], *,
                    closed_hi: float = 0.75,
                    possible: float = 0.50,
                    rel_delta: float = 0.35,
                    rel_floor: float = 0.40,
                    pct: float = 15.0) -> dict[tuple[int, int], str]:
    """Label every (track_idx, frame_idx) as 'open', 'possible', 'closed' or 'unknown'.

    ``tracks`` holds one list per tracked face, each entry ``(frame_idx, eye_closed_score)``.
    A score of None is always 'unknown'. A score at or above ``closed_hi`` is 'closed' on its own.
    With at least three scores for a face, a score is also 'closed' when it reaches ``rel_floor``
    and sits ``rel_delta`` above that face's own low-percentile baseline, so a narrow-eyed
    subject is not flagged and one wide-eyed frame cannot move the baseline much. With fewer
    than three scores only the absolute thresholds apply.
    """
    out: dict[tuple[int, int], str] = {}
    for ti, track in enumerate(tracks):
        scores = [s for _, s in track if s is not None]
        base = float(np.percentile(scores, pct)) if len(scores) >= 3 else None
        for fi, s in track:
            if s is None:
                out[(ti, fi)] = "unknown"
            elif s >= closed_hi or (base is not None and s >= rel_floor and (s - base) >= rel_delta):
                out[(ti, fi)] = "closed"
            elif s >= possible:
                out[(ti, fi)] = "possible"
            else:
                out[(ti, fi)] = "open"
    return out


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def track_faces(frames: Sequence[Sequence[Box]], min_iou: float = 0.45) -> list[list[tuple[int, int]]]:
    """Link faces across consecutive frames by greedy best-overlap matching.

    ``frames`` is in capture order. Returns tracks as lists of ``(frame_idx, face_idx)``.
    A track only continues into the very next frame; a face with no match of at least
    ``min_iou`` starts a new track, so a camera pan or a person leaving breaks the link.
    """
    tracks: list[list[tuple[int, int]]] = []
    last_box: dict[int, Box] = {}
    for t, boxes in enumerate(frames):
        live = [ti for ti, tr in enumerate(tracks) if tr[-1][0] == t - 1]
        pairs = sorted(((iou(last_box[ti], b), ti, fi) for ti in live for fi, b in enumerate(boxes)),
                       key=lambda p: (-p[0], p[1], p[2]))
        used_t: set[int] = set()
        used_f: set[int] = set()
        for score, ti, fi in pairs:
            if score < min_iou:
                break
            if ti in used_t or fi in used_f:
                continue
            tracks[ti].append((t, fi))
            last_box[ti] = boxes[fi]
            used_t.add(ti)
            used_f.add(fi)
        for fi, b in enumerate(boxes):
            if fi not in used_f:
                tracks.append([(t, fi)])
                last_box[len(tracks) - 1] = b
    return tracks


def _box(face) -> Box | None:
    vals = (face.x1, face.y1, face.x2, face.y2)
    return None if any(v is None for v in vals) else (float(vals[0]), float(vals[1]),
                                                       float(vals[2]), float(vals[3]))


def classify_group(frames: Sequence[Sequence], face_cfg: FaceSettings) -> list[int]:
    """Classify the faces of one burst (or one lone photo) and update them in place.

    ``frames`` is a capture-ordered list of per-image face lists. Sets ``eye_status`` and
    ``is_blink`` on every face and returns the closed-face count per frame.
    """
    boxed: list[list] = []
    for faces in frames:
        boxed.append([f for f in faces if _box(f) is not None])
        for f in faces:
            if _box(f) is None:  # a row from an older version: no box, so nothing to track
                f.eye_status, f.is_blink = "unknown", False
    tracks = track_faces([[_box(f) for f in fs] for fs in boxed], face_cfg.track_iou)
    scored = [[(fi, boxed[fi][k].eye_closed_score) for fi, k in tr] for tr in tracks]
    labels = classify_blinks(scored, closed_hi=face_cfg.closed_hi, possible=face_cfg.possible,
                             rel_delta=face_cfg.rel_delta, rel_floor=face_cfg.rel_floor,
                             pct=face_cfg.baseline_pct)
    for ti, tr in enumerate(tracks):
        for fi, k in tr:
            face = boxed[fi][k]
            face.eye_status = labels[(ti, fi)]
            face.is_blink = face.eye_status == "closed"
    return [sum(1 for f in faces if f.is_blink) for faces in frames]


def apply_blinks(records: Iterable, burst_members: Iterable[Sequence], face_cfg: FaceSettings) -> int:
    """Set eye_status/is_blink on faces and blink_count/has_blinks on images.

    Bursts are classified together, so a subject's own frames define their baseline. Every
    other photo is its own group. Tracks never cross a burst boundary. Returns how many
    images have a blink.
    """
    def key(r):
        t = r.taken_at
        if t is not None and t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (t or _EPOCH, r.id or 0)

    seen: set[int] = set()
    groups: list[list] = []
    for members in burst_members:
        ordered = sorted(members, key=key)
        groups.append(ordered)
        seen.update(id(m) for m in ordered)
    groups.extend([r] for r in records if id(r) not in seen)
    flagged = 0
    for group in groups:
        counts = classify_group([list(getattr(r, "faces", None) or []) for r in group], face_cfg)
        for rec, n in zip(group, counts):
            rec.blink_count = n
            rec.has_blinks = n > 0
            flagged += n > 0
    return flagged

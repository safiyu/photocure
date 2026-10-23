"""Group shots: find the burst frame where the weakest face is strongest.

Flag, never cull: this only scores and suggests. Nothing here touches a file.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from math import ceil

from .blinks import Box, iou, track_faces

_EYES = {"possible": 0.5, "unknown": 0.6, "closed": 0.0}


@dataclass(frozen=True)
class FaceObs:
    """One detected face in one frame."""

    box: Box | None
    eye_status: str = "unknown"
    eye_closed: float | None = None
    sharpness: float | None = None
    person_id: int | None = None
    face_id: int | None = None
    is_blink: bool | None = None
    emb: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SubjectFrame:
    state: str  # open | possible | closed | unknown | missing
    quality: float


@dataclass(frozen=True)
class FrameScore:
    index: int
    min_quality: float | None
    mean_quality: float | None
    subjects: tuple[SubjectFrame, ...]


@dataclass(frozen=True)
class GroupMatrix:
    status: str  # not_group | perfect | composite | best_effort
    subject_count: int
    frames: tuple[FrameScore, ...]
    best: int | None
    composite: tuple[int, int] | None


def _bridge(groups: list[dict[int, FaceObs]], max_gap: int = 2,
            min_iou: float = 0.45) -> list[dict[int, FaceObs]]:
    """Rejoin a track lost for a frame or two when the face reappears in the same spot."""
    out: list[dict[int, FaceObs]] = []
    for g in sorted(groups, key=min):
        start = min(g)
        best, best_iou = None, min_iou
        for h in out:
            end = max(h)
            if h.keys() & g.keys() or not 2 <= start - end <= max_gap + 1:
                continue
            a, b = h[end].box, g[start].box
            overlap = iou(a, b) if a is not None and b is not None else 0.0
            if overlap >= best_iou and (best is None or overlap > best_iou):
                best, best_iou = h, overlap
        if best is None:
            out.append(dict(g))
        else:
            best.update(g)
    return out


# Same-person embedding similarity across frames of one burst: at or above SAME is one person,
# at or below DIFFERENT is two people, in between falls back to position.
_SAME = 0.50
_DIFFERENT = 0.35


def _emb_of(face) -> tuple[float, ...] | None:
    raw = getattr(face, "embedding_json", None)
    if not raw:
        return None
    try:
        v = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return tuple(float(x) for x in v) if isinstance(v, list) and v else None


def _cos(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) != len(b):
        return None
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return None
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _verdict(sim: float | None) -> bool | None:
    if sim is None:
        return None
    if sim >= _SAME:
        return True
    return False if sim <= _DIFFERENT else None


def _same_person(a: FaceObs, b: FaceObs) -> bool | None:
    """True when two faces are the same person, False when they are not, None when unknown."""
    if a.person_id is not None and b.person_id is not None:
        return a.person_id == b.person_id
    if a.emb is not None and b.emb is not None:
        return _verdict(_cos(a.emb, b.emb))
    return None


def _track_same_person(tr1: dict[int, FaceObs], tr2: dict[int, FaceObs]) -> bool | None:
    pids1 = {f.person_id for f in tr1.values() if f.person_id is not None}
    pids2 = {f.person_id for f in tr2.values() if f.person_id is not None}
    if pids1 and pids2:
        return bool(pids1 & pids2)
    e1 = [f.emb for f in tr1.values() if f.emb is not None]
    e2 = [f.emb for f in tr2.values() if f.emb is not None]
    if e1 and e2:
        mean1 = [sum(c) / len(e1) for c in zip(*e1)]
        mean2 = [sum(c) / len(e2) for c in zip(*e2)]
        return _verdict(_cos(mean1, mean2))
    return None


def _subjects(frames: Sequence[Sequence[FaceObs]]) -> list[dict[int, FaceObs]]:
    """Link faces into subjects: tracked by spatial position, scale, and overlap, bridged across gaps."""
    tracks: list[dict[int, FaceObs]] = []
    for t, fs in enumerate(frames):
        boxed = [f for f in fs if f.box is not None]
        if not boxed:
            continue
        if not tracks:
            for f in boxed:
                tracks.append({t: f})
            continue

        cand_matches = []
        for ti, tr in enumerate(tracks):
            last_t = max(tr.keys())
            last_f = tr[last_t]
            gap = t - last_t
            if gap > 8:
                continue
            b1 = last_f.box
            cx1, cy1 = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
            w1, h1 = max(0.01, b1[2] - b1[0]), max(0.01, b1[3] - b1[1])

            for fi, f2 in enumerate(boxed):
                b2 = f2.box
                cx2, cy2 = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
                w2, h2 = max(0.01, b2[2] - b2[0]), max(0.01, b2[3] - b2[1])

                d = math.hypot(cx1 - cx2, cy1 - cy2)
                same = _same_person(last_f, f2)
                if same is False:
                    continue  # known to be someone else, however close they stand
                if same is True:
                    cand_matches.append((0.02 * d, ti, fi))  # identity wins; position breaks ties
                    continue
                rw = min(w1, w2) / max(w1, w2)
                rh = min(h1, h2) / max(h1, h2)
                ov = iou(b1, b2)

                cost = 1.8 * d + 0.4 * (1.0 - (rw + rh) / 2.0) - 0.5 * ov + (gap - 1) * 0.04
                if cost < 0.48:
                    cand_matches.append((cost, ti, fi))

        cand_matches.sort(key=lambda x: x[0])
        used_t = set()
        used_f = set()
        for cost, ti, fi in cand_matches:
            if ti in used_t or fi in used_f:
                continue
            tracks[ti][t] = boxed[fi]
            used_t.add(ti)
            used_f.add(fi)

        for fi, f in enumerate(boxed):
            if fi not in used_f:
                tracks.append({t: f})

    changed = True
    while changed:
        changed = False
        best_merge = None
        best_cost = 0.42
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                tr1 = tracks[i]
                tr2 = tracks[j]
                if set(tr1.keys()) & set(tr2.keys()):
                    continue

                same = _track_same_person(tr1, tr2)
                if same is False:
                    continue
                if same is True:
                    best_merge = (i, j)
                    best_cost = -1.0
                    break

                min_gap = min(abs(t1 - t2) for t1 in tr1 for t2 in tr2)
                cx1 = sum((f.box[0] + f.box[2]) / 2.0 for f in tr1.values() if f.box) / len(tr1)
                cy1 = sum((f.box[1] + f.box[3]) / 2.0 for f in tr1.values() if f.box) / len(tr1)
                cx2 = sum((f.box[0] + f.box[2]) / 2.0 for f in tr2.values() if f.box) / len(tr2)
                cy2 = sum((f.box[1] + f.box[3]) / 2.0 for f in tr2.values() if f.box) / len(tr2)
                d_avg = math.hypot(cx1 - cx2, cy1 - cy2)

                t1_end = max(tr1.keys()) if min(tr2.keys()) > max(tr1.keys()) else min(tr1.keys())
                t2_start = min(tr2.keys()) if min(tr2.keys()) > max(tr1.keys()) else max(tr2.keys())
                f1_end = tr1[t1_end]
                f2_start = tr2[t2_start]
                if f1_end.box and f2_start.box:
                    d_end = math.hypot((f1_end.box[0] + f1_end.box[2]) / 2.0 - (f2_start.box[0] + f2_start.box[2]) / 2.0,
                                       (f1_end.box[1] + f1_end.box[3]) / 2.0 - (f2_start.box[1] + f2_start.box[3]) / 2.0)
                else:
                    d_end = d_avg

                cost = min(d_avg, d_end) + min_gap * 0.02
                if cost < best_cost:
                    best_cost = cost
                    best_merge = (i, j)
            if best_cost < 0:
                break

        if best_merge:
            i, j = best_merge
            tracks[i].update(tracks[j])
            tracks.pop(j)
            changed = True

    for t, fs in enumerate(frames):
        for f in fs:
            if f.box is None and f.person_id is not None:
                matched = False
                for tr in tracks:
                    if t not in tr and any(f_other.person_id == f.person_id for f_other in tr.values()):
                        tr[t] = f
                        matched = True
                        break
                if not matched:
                    tracks.append({t: f})

    return tracks


def _quality(face: FaceObs, sharp_ref: float | None) -> float:
    if face.eye_status == "open":
        eyes = 1.0 - 0.3 * min(max(face.eye_closed or 0.0, 0.0), 1.0)
    else:
        eyes = _EYES.get(face.eye_status, 0.6)
    rel = 1.0 if face.sharpness is None or not sharp_ref else min(face.sharpness / sharp_ref, 1.0)
    return eyes * (0.7 + 0.3 * rel)


def score_group(frames: Sequence[Sequence[FaceObs]], *, perfect_at: float = 0.8,
                missing_q: float = 0.3, tiebreak: Sequence[float] | None = None) -> GroupMatrix:
    """Score every frame by its weakest subject and pick the best, or a pair to composite."""
    n = len(frames)
    if n < 2:
        return GroupMatrix("not_group", 0, (), None, None)
    need = max(2, ceil(n / 2))
    core = [s for s in _subjects(frames) if len(s) >= need]
    if len(core) < 2:
        return GroupMatrix("not_group", len(core), (), None, None)
    refs = [max((f.sharpness for f in s.values() if f.sharpness is not None), default=None)
            for s in core]
    scores: list[FrameScore] = []
    for t in range(n):
        subs = []
        for s, ref in zip(core, refs, strict=True):
            f = s.get(t)
            if f is None:
                subs.append(SubjectFrame("missing", missing_q))
            else:
                known = ("open", "possible", "closed")
                state = f.eye_status if f.eye_status in known else "unknown"
                subs.append(SubjectFrame(state, _quality(f, ref)))
        qs = [x.quality for x in subs]
        scores.append(FrameScore(t, min(qs), sum(qs) / len(qs), tuple(subs)))
    tb = (list(tiebreak) if tiebreak is not None else []) + [0.0] * n  # pad a short list with zeros
    best = max(range(n), key=lambda t: (scores[t].min_quality, scores[t].mean_quality, tb[t], -t))
    if scores[best].min_quality >= perfect_at:
        return GroupMatrix("perfect", len(core), tuple(scores), best, None)
    weak = [i for i, x in enumerate(scores[best].subjects) if x.quality < perfect_at]
    partners = [(min(scores[t].subjects[i].quality for i in weak), -t, t) for t in range(n)
                if t != best and all(scores[t].subjects[i].quality >= perfect_at for i in weak)]
    if partners:
        partner = max(partners)[2]
        return GroupMatrix("composite", len(core), tuple(scores), best, (best, partner))
    return GroupMatrix("best_effort", len(core), tuple(scores), best, None)


def group_matrix_payload(members) -> dict:
    """The group matrix of one burst's frames, in capture order, as plain JSON."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def when(r):
        t = r.taken_at
        if t is not None and t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t or epoch

    ordered = sorted(members, key=lambda r: (when(r), r.id or 0))
    frames = []
    for r in ordered:
        faces = []
        for f in r.faces or []:
            box = (f.x1, f.y1, f.x2, f.y2)
            has_box = all(v is not None for v in box)
            faces.append(FaceObs(box=tuple(float(v) for v in box) if has_box else None,
                                 eye_status=f.eye_status, eye_closed=f.eye_closed_score,
                                 sharpness=f.face_sharpness, person_id=f.person_id,
                                 emb=_emb_of(f), face_id=f.id))
        frames.append(faces)
    tiebreak = [float(r.quality.sharpness_norm) if r.quality is not None else 0.0 for r in ordered]
    m = score_group(frames, tiebreak=tiebreak)
    return {
        "status": m.status, "subject_count": m.subject_count,
        "best_id": ordered[m.best].id if m.best is not None else None,
        "partner_id": ordered[m.composite[1]].id if m.composite else None,
        "frames": [{"id": ordered[f.index].id, "min_quality": round(f.min_quality, 3),
                    "subjects": [{"state": x.state, "q": round(x.quality, 3)} for x in f.subjects]}
                   for f in m.frames],
    }


def burst_face_matrix(members) -> dict:
    """Matrix comparing all tracked faces across all burst frames in capture order."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def when(r):
        t = r.taken_at
        if t is not None and t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t or epoch

    ordered = sorted(members, key=lambda r: (when(r), r.id or 0))
    n = len(ordered)
    if n == 0:
        return {"subjects": [], "rows": []}

    frames: list[list[FaceObs]] = []
    for r in ordered:
        faces = []
        for f in r.faces or []:
            box = (f.x1, f.y1, f.x2, f.y2)
            has_box = all(v is not None for v in box)
            faces.append(FaceObs(box=tuple(float(v) for v in box) if has_box else None,
                                 eye_status=f.eye_status, eye_closed=f.eye_closed_score,
                                 sharpness=f.face_sharpness, person_id=f.person_id,
                                 emb=_emb_of(f), face_id=f.id,
                                 is_blink=bool(getattr(f, "is_blink", False)) or (f.eye_status == "closed")))
        frames.append(faces)

    raw_subjects = _subjects(frames)
    if not raw_subjects:
        return {"subjects": [], "rows": []}

    def subject_mean_x(subj: dict[int, FaceObs]) -> float:
        centers = []
        for obs in subj.values():
            if obs.box is not None:
                centers.append((obs.box[0] + obs.box[2]) / 2.0)
        return sum(centers) / len(centers) if centers else 0.5

    # Sort subjects horizontally (left to right across the camera frame)
    sorted_subjects = sorted(raw_subjects, key=subject_mean_x)

    subjects_meta = [
        {
            "id": idx,
            "name": f"Person {idx + 1}",
            "avg_x": round(subject_mean_x(s), 3),
        }
        for idx, s in enumerate(sorted_subjects)
    ]

    rows = []
    for t, r in enumerate(ordered):
        qm = r.quality
        row_faces = []
        for subj_idx, s in enumerate(sorted_subjects):
            obs = s.get(t)
            if obs is not None and obs.face_id is not None:
                is_blink = bool(obs.is_blink) or (obs.eye_status == "closed")
                row_faces.append({
                    "subject_idx": subj_idx,
                    "face_id": obs.face_id,
                    "eye_status": obs.eye_status or "unknown",
                    "is_blink": is_blink,
                    "eye_closed_score": round(obs.eye_closed, 3) if obs.eye_closed is not None else None,
                    "sharpness": round(obs.sharpness, 1) if obs.sharpness is not None else None,
                    "box": list(obs.box) if obs.box else None,
                    "url": f"/api/faces/{obs.face_id}.jpg"
                })
            else:
                row_faces.append(None)

        rows.append({
            "id": r.id,
            "filename": r.filename,
            "is_best_shot": bool(r.is_best_shot),
            "delete_approved": bool(r.delete_approved),
            "has_blinks": bool(r.has_blinks),
            "taken_at": r.taken_at.isoformat() if r.taken_at else None,
            "width": r.width,
            "height": r.height,
            "sharpness_norm": round(qm.sharpness_norm, 3) if qm and qm.sharpness_norm is not None else None,
            "faces": row_faces,
        })

    return {
        "subjects": subjects_meta,
        "rows": rows,
    }

"""Group shots: find the burst frame where the weakest face is strongest.

Flag, never cull: this only scores and suggests. Nothing here touches a file.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
        for h in out:
            end = max(h)
            if h.keys() & g.keys() or not 2 <= start - end <= max_gap + 1:
                continue
            a, b = h[end].box, g[start].box
            if a is not None and b is not None and iou(a, b) >= min_iou:
                h.update(g)
                break
        else:
            out.append(dict(g))
    return out


def _subjects(frames: Sequence[Sequence[FaceObs]]) -> list[dict[int, FaceObs]]:
    """Link faces into subjects: tracked by overlap, bridged across gaps by person id."""
    boxed = [[f for f in fs if f.box is not None] for fs in frames]
    tracks = track_faces([[f.box for f in fs] for fs in boxed])
    groups = _bridge([{t: boxed[t][k] for t, k in tr} for tr in tracks])
    for t, fs in enumerate(frames):
        for f in fs:
            if f.box is None and f.person_id is not None:
                groups.append({t: f})
    merged: list[dict[int, FaceObs]] = []
    owner: dict[int, int] = {}
    for g in groups:
        pids = {f.person_id for f in g.values() if f.person_id is not None}
        target = None
        for pid in pids:
            idx = owner.get(pid)
            # Two faces in one frame are two people, whatever the cluster id says.
            if idx is not None and not (merged[idx].keys() & g.keys()):
                target = idx
                break
        if target is None:
            merged.append(dict(g))
            target = len(merged) - 1
        else:
            merged[target].update(g)
        for pid in pids:
            owner.setdefault(pid, target)
    return merged


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
    tb = list(tiebreak) if tiebreak is not None else [0.0] * n
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

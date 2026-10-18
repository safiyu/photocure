"""Event stories: tag the scale of a shot with zero-shot prompts and pick a balanced selection.

Flag, never cull: this only labels and suggests. Nothing here touches a file.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SCALES = ("wide", "closeup", "action", "ambient", "medium")

SCALE_LABELS = {
    "wide": "Establishing wide", "closeup": "Close-up detail", "action": "Action and motion",
    "ambient": "Ambient and architecture", "medium": "Medium shot",
}

SCALE_PROMPTS: dict[str, tuple[str, ...]] = {
    "wide": (
        "an establishing wide shot showing the whole scene or venue",
        "a wide angle photo of an entire place with people small in the frame",
        "a panoramic view of the surroundings",
    ),
    "closeup": (
        "a close-up detail shot of a small object",
        "a macro photo of a single detail",
        "a tight close-up of hands or a face filling the frame",
    ),
    "action": (
        "an action shot with motion and movement",
        "a candid photo of people dancing, running or playing",
        "a photo with motion blur from sport or activity",
    ),
    "ambient": (
        "an ambient photo of the atmosphere, decor and architecture of a place",
        "a quiet photo of a room, building or setting with no people",
        "a moody photo of lights, tables and surroundings",
    ),
    "medium": (
        "a medium shot of a few people posing together",
        "a photo of people at a table or standing in a group",
        "an ordinary snapshot of people from the waist up",
    ),
}

# Share of a story each scale should get, when the event has that scale at all.
DEFAULT_MIX = {"wide": 0.20, "closeup": 0.20, "action": 0.20, "ambient": 0.15, "medium": 0.25}

Item = tuple[int, float, str]


def classify_scales(embeddings, text, min_margin: float = 0.0) -> list[str]:
    """Scale of each image: the nearest of five unit prompt vectors, medium when unsure."""
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] == 0:
        return []
    prompts = np.asarray(text, dtype=np.float32)
    prompts = prompts / np.maximum(np.linalg.norm(prompts, axis=1, keepdims=True), 1e-9)
    norms = np.linalg.norm(emb, axis=1)
    sims = (emb / np.maximum(norms, 1e-9)[:, None]) @ prompts.T
    out: list[str] = []
    for row, norm in zip(sims, norms, strict=True):
        if norm < 1e-9:
            out.append("medium")
            continue
        order = np.argsort(row)[::-1]
        confident = row[order[0]] - row[order[1]] >= min_margin
        out.append(SCALES[int(order[0])] if confident else "medium")
    return out


def balance_selection(items: Sequence[Item], n: int,
                      mix: dict[str, float] | None = None) -> list[Item]:
    """Pick ``n`` photos so every scale present is covered, then fill by mix and score."""
    if n <= 0 or not items:
        return []
    weights = mix or DEFAULT_MIX
    ranked = sorted(items, key=lambda it: (-it[1], it[0]))
    if n >= len(ranked):
        return ranked
    buckets: dict[str, list[Item]] = {s: [] for s in SCALES}
    for it in ranked:
        buckets[it[2] if it[2] in buckets else "medium"].append(it)
    present = [s for s in SCALES if buckets[s]]
    taken = dict.fromkeys(SCALES, 0)
    chosen: list[Item] = []

    def take(scale: str) -> None:
        chosen.append(buckets[scale][taken[scale]])
        taken[scale] += 1

    if n >= len(present):
        for s in present:
            take(s)
    else:
        heaviest = sorted(present, key=lambda s: (-weights.get(s, 0), -buckets[s][0][1],
                                                  SCALES.index(s)))
        for s in heaviest[:n]:
            take(s)
    total = sum(weights.get(s, 0) for s in present) or 1.0
    target = {s: n * weights.get(s, 0) / total for s in present}
    while len(chosen) < n:
        room = [s for s in present if taken[s] < len(buckets[s])]
        take(max(room, key=lambda s: (target[s] - taken[s], buckets[s][taken[s]][1],
                                       -SCALES.index(s))))
    return sorted(chosen, key=lambda it: (-it[1], it[0]))


def coverage(items: Sequence[Item]) -> dict[str, int]:
    """How many chosen photos each scale has, listing only scales that appear."""
    counts = {s: sum(1 for it in items if it[2] == s) for s in SCALES}
    return {s: c for s, c in counts.items() if c}

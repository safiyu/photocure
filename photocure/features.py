"""Feature settings kept beside the database in ``<stem>.features.json``.

These are the knobs for highlights ranking, keyword export and the watched folder. They live apart
from the device settings so one feature never changes the other's file.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

log = logging.getLogger("photocure.features")

# key -> (kind, low, high); numbers are floats in [low, high], strings are capped at ``high`` chars.
_SPEC: dict[str, tuple[str, float, float]] = {
    "highlight_rating_weight": ("float", 0.0, 0.5),
    "highlight_favorite_weight": ("float", 0.0, 0.5),
    "highlight_preference_weight": ("float", 0.0, 1.0),
    "write_keywords": ("bool", 0, 0),
    "watch_enabled": ("bool", 0, 0),
    "watch_path": ("str", 0, 4096),
}
DEFAULTS: dict = {
    "highlight_rating_weight": 0.10,
    "highlight_favorite_weight": 0.15,
    "highlight_preference_weight": 0.35,
    "write_keywords": False,
    "watch_enabled": False,
    "watch_path": "",
}


def features_path(db_path: Path | str) -> Path:
    p = Path(db_path)
    return p.with_name(f"{p.stem}.features.json")


def _check(key: str, value):
    kind, lo, hi = _SPEC[key]
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"'{key}' must be true or false.")
        return value
    if kind == "str":
        if not isinstance(value, str):
            raise ValueError(f"'{key}' must be text.")
        if len(value) > hi:
            raise ValueError(f"'{key}' is too long.")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must be a number.")
    if not lo <= float(value) <= hi:
        raise ValueError(f"'{key}' must be between {lo:g} and {hi:g}.")
    return float(value)


def _read(db_path: Path | str) -> dict:
    try:
        raw = json.loads(features_path(db_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    good: dict = {}
    for key, value in raw.items():
        if key not in _SPEC:
            continue
        try:
            good[key] = _check(key, value)
        except ValueError as e:
            log.warning("ignoring bad stored feature setting: %s", e)
    return good


def load_features(db_path: Path | str) -> dict:
    """Every feature setting, with defaults for anything not stored or not valid."""
    return {**DEFAULTS, **_read(db_path)}


def save_features(db_path: Path | str, values: dict) -> dict:
    """Validate and store settings; ``None`` resets a key. Returns the full effective settings."""
    checked: dict = {}
    for key, value in values.items():
        if key not in _SPEC:
            raise ValueError(f"Unknown setting '{key}'.")
        checked[key] = None if value is None else _check(key, value)
    stored = _read(db_path)
    for key, value in checked.items():
        if value is None:
            stored.pop(key, None)
        else:
            stored[key] = value
    path = features_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(stored, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return {**DEFAULTS, **stored}

"""Taste learning: a personal 0..1 score trained from the user's favourites, stars and rejects.

A logistic model over the unit CLIP embedding separates the photos the user likes (favourites,
four or five stars) from the ones they do not (rejects, one or two stars). It is only applied when
cross-validation shows it generalises, and it never replaces the model's own aesthetic score: the
highlights ranking blends the two.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy import and_, func, not_, or_
from sqlmodel import col, select

from ..db.models import ImageRecord, PreferenceModel, SemanticScore, utcnow
from ..db.repository import dequantize_embedding
from ..highlights import BAD_STARS, label_for  # noqa: F401  (label_for is part of this module's API)
from .semantic_engine import EMBED_DIM

log = logging.getLogger("photocure.preference")

KIND = "aesthetic"
MIN_PER_CLASS = 8
MIN_CV_ACCURACY = 0.65
_CHUNK = 2000


@dataclass
class TrainResult:
    status: str                 # trained | not_enough | weak
    n_positive: int = 0
    n_negative: int = 0
    cv_accuracy: float = 0.0
    scored: int = 0


def _unit(blob: str) -> np.ndarray | None:
    v = dequantize_embedding(blob)
    if not v or len(v) != EMBED_DIM:
        return None
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-9 else None


def _features(unit: np.ndarray) -> np.ndarray:
    """Scale unit vectors so each coordinate is about one, which keeps regularization sensible."""
    return unit * np.float32(np.sqrt(EMBED_DIM))


def collect_examples(session) -> tuple[list[int], np.ndarray, np.ndarray]:
    """``(image_ids, X, y)`` for every labelled photo with a real embedding."""
    rows = session.exec(
        select(ImageRecord.id, ImageRecord.rating, ImageRecord.is_favorite, SemanticScore.embedding)
        .join(SemanticScore, SemanticScore.image_id == ImageRecord.id)
        .where(col(SemanticScore.backend) != "stub")).all()
    ids, vecs, ys = [], [], []
    for image_id, rating, fav, blob in rows:
        y = label_for(rating, fav)
        if y is None:
            continue
        u = _unit(blob)
        if u is None:
            continue
        ids.append(int(image_id))
        vecs.append(_features(u))
        ys.append(y)
    X = np.stack(vecs) if vecs else np.empty((0, EMBED_DIM), dtype=np.float32)
    return ids, X, np.asarray(ys, dtype=np.int64)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float = 1.0,
                 iters: int = 30) -> tuple[np.ndarray, float]:
    """Class-balanced, ridge-regularized logistic regression by Newton steps. Returns ``(w, b)``."""
    n, d = X.shape
    n_pos, n_neg = int(y.sum()), int(n - y.sum())
    sw = np.where(y == 1, n / (2.0 * max(n_pos, 1)), n / (2.0 * max(n_neg, 1))).astype(np.float64)
    Xa = np.hstack([X.astype(np.float64), np.ones((n, 1))])
    reg = np.diag(np.r_[np.full(d, lam), 0.0])
    theta = np.zeros(d + 1)
    for _ in range(iters):
        p = _sigmoid(Xa @ theta)
        grad = Xa.T @ (sw * (p - y)) + reg @ theta
        h = (Xa * (sw * p * (1 - p) + 1e-6)[:, None]).T @ Xa + reg + 1e-8 * np.eye(d + 1)
        step = np.linalg.solve(h, grad)
        theta -= step
        if float(np.linalg.norm(step)) < 1e-6:
            break
    return theta[:d].astype(np.float32), float(theta[d])


def cross_validate(X: np.ndarray, y: np.ndarray, lam: float = 1.0, folds: int = 5,
                   seed: int = 0) -> float:
    """Mean balanced accuracy over stratified folds (0.5 is chance)."""
    rng = np.random.default_rng(seed)
    k = max(2, min(folds, int(y.sum()), int(len(y) - y.sum())))
    assign = np.empty(len(y), dtype=np.int64)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        assign[idx] = np.arange(len(idx)) % k
    scores = []
    for f in range(k):
        test = assign == f
        w, b = fit_logistic(X[~test], y[~test], lam)
        pred = (X[test] @ w + b) > 0
        yt = y[test]
        recalls = [float((pred[yt == c] == bool(c)).mean()) for c in (0, 1) if (yt == c).any()]
        scores.append(float(np.mean(recalls)))
    return float(np.mean(scores))


def _active_model(session) -> PreferenceModel | None:
    return session.exec(select(PreferenceModel).where(PreferenceModel.kind == KIND)
                        .order_by(col(PreferenceModel.id).desc())).first()


def apply_preferences(session, only_missing: bool = False) -> int:
    """Write ``SemanticScore.preference`` from the active model; returns rows changed.

    Without a model every stored preference is cleared. Placeholder embeddings and vectors of
    another size are never scored. The caller owns the commit.
    """
    model = _active_model(session)
    S = SemanticScore
    if model is None:
        stale = session.exec(select(S).where(col(S.preference).is_not(None))).all()
        for row in stale:
            row.preference = None
        session.flush()
        return len(stale)
    w = np.asarray(json.loads(model.weights), dtype=np.float32)
    changed, last = 0, 0
    while True:
        stmt = select(S).where(col(S.id) > last).order_by(col(S.id)).limit(_CHUNK)
        if only_missing:
            stmt = stmt.where(col(S.preference).is_(None))
        rows = session.exec(stmt).all()
        if not rows:
            break
        last = rows[-1].id
        for row in rows:
            u = _unit(row.embedding) if row.backend != "stub" and len(w) == EMBED_DIM else None
            value = None if u is None else float(_sigmoid(np.dot(_features(u), w) + model.bias))
            if value != row.preference:
                row.preference = value
                changed += 1
        session.flush()
        for row in rows:
            session.expunge(row)
    return changed


def train_preferences(session, min_per_class: int = MIN_PER_CLASS,
                      min_accuracy: float = MIN_CV_ACCURACY) -> TrainResult:
    """Train from the current ratings and apply the model if it generalises."""
    _, X, y = collect_examples(session)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos < min_per_class or n_neg < min_per_class:
        return TrainResult("not_enough", n_pos, n_neg)
    acc = cross_validate(X, y)
    if acc < min_accuracy:
        return TrainResult("weak", n_pos, n_neg, acc)
    w, b = fit_logistic(X, y)
    for old in session.exec(select(PreferenceModel).where(PreferenceModel.kind == KIND)).all():
        session.delete(old)
    session.add(PreferenceModel(kind=KIND, weights=json.dumps([float(v) for v in w]), bias=b,
                                n_positive=n_pos, n_negative=n_neg, cv_accuracy=acc,
                                trained_at=utcnow()))
    session.flush()
    return TrainResult("trained", n_pos, n_neg, acc, apply_preferences(session))


def reset_preferences(session) -> None:
    """Forget the model and every personal score."""
    for old in session.exec(select(PreferenceModel).where(PreferenceModel.kind == KIND)).all():
        session.delete(old)
    session.flush()
    apply_preferences(session)


def status(session) -> dict:
    """What a settings panel needs: label counts, whether training is possible, the active model."""
    I, S = ImageRecord, SemanticScore
    bad = func.coalesce(col(I.rating), 0).in_(BAD_STARS)
    good = or_(col(I.is_favorite) == True, func.coalesce(col(I.rating), 0) >= 4)  # noqa: E712
    base = (select(func.count()).select_from(I).join(S, S.image_id == I.id)
            .where(col(S.backend) != "stub"))
    n_pos = session.exec(base.where(and_(good, not_(bad)))).one()
    n_neg = session.exec(base.where(and_(bad, not_(good)))).one()
    model = _active_model(session)
    scored = session.exec(select(func.count()).select_from(S)
                          .where(col(S.preference).is_not(None))).one()
    return {"n_positive": int(n_pos), "n_negative": int(n_neg), "min_per_class": MIN_PER_CLASS,
            "can_train": int(n_pos) >= MIN_PER_CLASS and int(n_neg) >= MIN_PER_CLASS,
            "trained": model is not None, "scored": int(scored),
            "cv_accuracy": round(model.cv_accuracy, 3) if model else None,
            "trained_at": model.trained_at.isoformat() if model else None,
            "model_positive": model.n_positive if model else None,
            "model_negative": model.n_negative if model else None}

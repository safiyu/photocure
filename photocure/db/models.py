"""SQLModel relational tables for the Photocure state store (Schema v1).

Design notes:
- ``ImageRecord`` is the single source of truth for every ingested file.
- ``QualityMetric``, ``FaceDetection`` and ``SemanticScore`` are child rows
  linked by ``image_id`` and cascade-deleted with their parent.
- Embeddings are stored as compact JSON (512-dim float16 -> ~1 KB) to keep
  SQLite portable and avoid a binary BLOB migration.
- ``ActionLedger`` is the append-only transaction journal backing the
  RollbackEngine.

Note: do NOT enable ``from __future__ import annotations`` here. PEP 563 turns
the ``Relationship`` target annotations into raw strings, which SQLAlchemy 2.x
cannot resolve (it raises ``InvalidRequestError``). Forward references below are
already written as quoted strings, so they resolve without PEP 563.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, Relationship, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ImageRecord(SQLModel, table=True):
    __tablename__ = "images"

    id: int | None = Field(default=None, primary_key=True)
    # Absolute path on disk (unique per library).
    path: str = Field(index=True, unique=True)
    relative_path: str = Field(default="")
    filename: str
    extension: str = Field(default="", index=True)
    mime_type: str = Field(default="")

    file_size: int = 0
    # st_mtime seen at scan time; None on rows scanned before this column existed.
    file_mtime: float | None = None
    # Tier-1 quick fingerprint (size + head/tail xxh64) as hex.
    quick_hash: str = Field(default="", index=True)
    # Tier-2 definitive hash.
    sha256: str = Field(default="", index=True)

    width: int = 0
    height: int = 0
    orientation: int = 1

    # Effective timestamp after the 4-tier cascade.
    taken_at: datetime | None = Field(default=None, index=True)
    # Which tier resolved the timestamp: exif:dtoriginal, exif:create,
    # filename, mtime, or unknown.
    taken_at_source: str = Field(default="")
    gps_lat: float | None = None
    gps_lon: float | None = None
    camera_model: str = Field(default="", index=True)
    # Filled from the GPS position by the optional offline geocoder (photocure[geo]).
    city: str = Field(default="")
    region: str = Field(default="")
    country_code: str = Field(default="", index=True)
    # Local favourite. Never written to a sidecar.
    is_favorite: bool = Field(default=False, index=True)

    # Scan state machine.
    scan_status: str = Field(default="scanned", index=True)
    # "pending", "inferred", "failed".
    infer_status: str = Field(default="pending", index=True)
    # Exact duplicate flag (sha256 seen before).
    is_duplicate: bool = Field(default=False, index=True)
    duplicate_of: int | None = Field(default=None, foreign_key="images.id")

    # Clustering.
    burst_id: int | None = Field(default=None, index=True)
    is_best_shot: bool = Field(default=False)
    is_alternate: bool = Field(default=False)

    # Final routing (set by DecisionMatrix / manual review).
    category: str | None = Field(default=None, index=True)
    subcategory: str | None = Field(default=None)
    # "unrouted", "planned", "executed", "skipped", "review".
    route_status: str = Field(default="unrouted", index=True)
    target_path: str | None = Field(default=None)
    # Manual review bucket overrides from the Web UI.
    manual_bucket: str | None = Field(default=None)

    # Perceptual hash (hex) and the image it near-duplicates.
    phash: str | None = Field(default=None, index=True)
    near_dup_of: int | None = Field(default=None)
    # Minutes east of UTC for the camera-local taken_at, when known.
    tz_offset_minutes: int | None = Field(default=None)
    # Star rating: 1-5 stars, -1 rejected, None or 0 unrated (from XMP sidecars or set in the UI).
    rating: int | None = Field(default=None, index=True)
    # Human-locked rows are never re-routed by automatic passes.
    locked: bool = Field(default=False)
    event_id: int | None = Field(default=None, index=True)

    # Blink flags (set by photocure.clustering.blinks); faces_schema_version tells which
    # face pipeline produced the face rows (older rows are re-detected).
    has_blinks: bool = Field(default=False, index=True)
    blink_count: int = Field(default=0)
    faces_schema_version: int = Field(default=0)

    # XMP sidecar tracking (schema v3, for C2).
    xmp_written_rating: int | None = Field(default=None)
    xmp_written_tags: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    quality: Optional["QualityMetric"] = Relationship(
        back_populates="image", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    faces: list["FaceDetection"] = Relationship(
        back_populates="image",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    semantic: Optional["SemanticScore"] = Relationship(
        back_populates="image", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class QualityMetric(SQLModel, table=True):
    __tablename__ = "quality_metrics"

    id: int | None = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="images.id", unique=True, index=True)
    image: Optional["ImageRecord"] = Relationship(back_populates="quality")

    laplacian_variance: float = 0.0
    sharpness_norm: float = 0.0  # 0..1
    # Luminance metrics (0..255).
    mean_luminance: float = 0.0
    rms_contrast: float = 0.0
    black_clip_pct: float = 0.0
    white_clip_pct: float = 0.0
    # Composite verdicts.
    is_blurry: bool = False
    is_too_dark: bool = False
    is_overexposed: bool = False
    quality_score: float = 0.0  # 0..1 composite
    is_low_contrast: bool = False
    computed_at: datetime = Field(default_factory=utcnow)


class FaceDetection(SQLModel, table=True):
    __tablename__ = "face_detections"

    id: int | None = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="images.id", index=True)
    image: Optional["ImageRecord"] = Relationship(back_populates="faces")

    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    confidence: float
    # 5 landmark points "[x1,y1,...,x5,y5]" in pixels of the downscaled frame.
    landmarks: str = Field(default="")
    # Downscale factor (decoded side / original side) the pixel coords refer to.
    scale: float = 1.0
    # Normalized face embedding as JSON and its person cluster.
    embedding_json: str | None = Field(default=None)
    person_id: int | None = Field(default=None, index=True)
    eye_aspect_ratio: float | None = None  # legacy, no longer written
    # Laplacian variance of the face crop (proxy for focus).
    face_sharpness: float | None = None
    # Box as fractions of the EXIF-oriented image (0..1). Null on rows from older versions.
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    # (eyeBlinkLeft + eyeBlinkRight) / 2 from MediaPipe, 0..1; null when it could not be measured.
    eye_closed_score: float | None = None
    # open | possible | closed | unknown. Unknown is never scored as open or closed.
    eye_status: str = Field(default="unknown")
    is_blink: bool = Field(default=False)


class SemanticScore(SQLModel, table=True):
    __tablename__ = "semantic_scores"

    id: int | None = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="images.id", unique=True, index=True)
    image: Optional["ImageRecord"] = Relationship(back_populates="semantic")

    model_name: str = Field(default="")
    top1_category: str = Field(default="")
    top1_score: float = 0.0
    top2_category: str = Field(default="")
    top2_score: float = 0.0
    score_margin: float = 0.0
    # Full softmax distribution serialized as JSON {category: score}.
    distribution: str = Field(default="{}")
    # Normalized 512-dim image embedding (JSON, float16 quantized).
    embedding: str = Field(default="")
    aesthetic: float | None = Field(default=None)
    # Name of the head that produced ``aesthetic``; "" when no score is stored.
    aesthetic_model: str = Field(default="")
    # Personal taste score (0..1) learned from the user's ratings; None until a model is trained.
    preference: float | None = Field(default=None)
    # Semantic backend that produced the row ("clip", "stub"); "" means unknown.
    backend: str = Field(default="")
    computed_at: datetime = Field(default_factory=utcnow)


class PreferenceModel(SQLModel, table=True):
    """The latest taste model trained from the user's ratings (one active row per kind)."""

    __tablename__ = "preference_models"

    id: int | None = Field(default=None, primary_key=True)
    kind: str = Field(default="aesthetic", index=True)
    # JSON list of weights over the unit CLIP embedding, plus the bias.
    weights: str = Field(default="[]")
    bias: float = 0.0
    n_positive: int = 0
    n_negative: int = 0
    # Held-out accuracy from cross-validation; the model is only applied when it beats chance.
    cv_accuracy: float = 0.0
    trained_at: datetime = Field(default_factory=utcnow)


class BurstGroup(SQLModel, table=True):
    __tablename__ = "burst_groups"

    id: int | None = Field(default=None, primary_key=True)
    # Stable human-friendly id, e.g. "Burst_20240512_0001".
    name: str = Field(unique=True, index=True)
    image_count: int = 0
    best_shot_id: int | None = Field(default=None, foreign_key="images.id")
    start_time: datetime | None = None
    end_time: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)


class ActionLedger(SQLModel, table=True):
    """Append-only transaction journal for every filesystem mutation."""

    __tablename__ = "action_ledger"

    id: int | None = Field(default=None, primary_key=True)
    transaction_id: str = Field(index=True)
    image_id: int | None = Field(default=None, foreign_key="images.id", index=True)
    source_path: str
    target_path: str
    operation_type: str = Field(default="move")  # move | copy | hardlink | symlink
    # Topmost directory this action created (undo prunes up to it); "" if none.
    created_dir: str = Field(default="")
    # images.relative_path before a move, restored on rollback.
    prev_relative_path: str = Field(default="")
    sha256: str = Field(default="")
    file_size: int = 0
    # What really happened on disk (a requested hardlink may have become a copy).
    actual_operation: str = Field(default="")
    target_dev: int | None = Field(default=None)
    target_ino: int | None = Field(default=None)
    # "pending", "committed", "rolled_back", "failed".
    status: str = Field(default="pending", index=True)
    error: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)
    committed_at: datetime | None = None


class PersonCluster(SQLModel, table=True):
    __tablename__ = "person_clusters"

    id: int | None = Field(default=None, primary_key=True)
    name: str | None = Field(default=None)
    face_count: int = 0


class EventGroup(SQLModel, table=True):
    __tablename__ = "event_groups"

    id: int | None = Field(default=None, primary_key=True)
    start: datetime | None = None
    end: datetime | None = None
    label: str | None = Field(default=None)
    image_count: int = 0


class PlanRecord(SQLModel, table=True):
    """A persisted, content-hashed routing plan."""

    __tablename__ = "plans"

    id: int | None = Field(default=None, primary_key=True)
    plan_id: str = Field(unique=True, index=True)
    mode: str = Field(default="category_first")
    operation: str = Field(default="move")
    organize_root: str = Field(default="")
    include_duplicates: bool = True
    # Hash of the routing inputs; apply refuses when it no longer matches.
    input_hash: str = Field(default="")
    burst_count: int = 0
    # Transaction that executed this plan; a plan is applied at most once.
    applied_transaction: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)


class PlanActionRow(SQLModel, table=True):
    __tablename__ = "plan_actions"

    id: int | None = Field(default=None, primary_key=True)
    plan_id: str = Field(index=True)
    seq: int = 0
    image_id: int | None = Field(default=None, index=True)
    source: str
    target: str
    operation: str = Field(default="move")
    category: str = Field(default="")
    reason: str = Field(default="")
    is_alternate: bool = False
    burst_name: str | None = Field(default=None)

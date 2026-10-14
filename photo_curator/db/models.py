"""SQLModel relational tables for the Photo Curator state store (Schema v1).

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
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ImageRecord(SQLModel, table=True):
    __tablename__ = "images"

    id: Optional[int] = Field(default=None, primary_key=True)
    # Absolute path on disk (unique per library).
    path: str = Field(index=True, unique=True)
    relative_path: str = Field(default="")
    filename: str
    extension: str = Field(default="", index=True)
    mime_type: str = Field(default="")

    file_size: int = 0
    # Tier-1 quick fingerprint (size + head/tail xxh64) as hex.
    quick_hash: str = Field(default="", index=True)
    # Tier-2 definitive hash.
    sha256: str = Field(default="", index=True)

    width: int = 0
    height: int = 0
    orientation: int = 1

    # Effective timestamp after the 4-tier cascade.
    taken_at: Optional[datetime] = Field(default=None, index=True)
    # Which tier resolved the timestamp: exif:dtoriginal, exif:create,
    # filename, mtime, or unknown.
    taken_at_source: str = Field(default="")
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    camera_model: str = Field(default="")

    # Scan state machine.
    scan_status: str = Field(default="scanned", index=True)
    # "pending", "inferred", "failed".
    infer_status: str = Field(default="pending", index=True)
    # Exact duplicate flag (sha256 seen before).
    is_duplicate: bool = Field(default=False, index=True)
    duplicate_of: Optional[int] = Field(default=None, foreign_key="images.id")

    # Clustering.
    burst_id: Optional[int] = Field(default=None, index=True)
    is_best_shot: bool = Field(default=False)
    is_alternate: bool = Field(default=False)

    # Final routing (set by DecisionMatrix / manual review).
    category: Optional[str] = Field(default=None, index=True)
    subcategory: Optional[str] = Field(default=None)
    # "unrouted", "planned", "executed", "skipped", "review".
    route_status: str = Field(default="unrouted", index=True)
    target_path: Optional[str] = Field(default=None)
    # Manual review bucket overrides from the Web UI.
    manual_bucket: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    quality: Optional["QualityMetric"] = Relationship(
        back_populates="image", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    faces: List["FaceDetection"] = Relationship(
        back_populates="image",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    semantic: Optional["SemanticScore"] = Relationship(
        back_populates="image", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class QualityMetric(SQLModel, table=True):
    __tablename__ = "quality_metrics"

    id: Optional[int] = Field(default=None, primary_key=True)
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
    computed_at: datetime = Field(default_factory=utcnow)


class FaceDetection(SQLModel, table=True):
    __tablename__ = "face_detections"

    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="images.id", index=True)
    image: Optional["ImageRecord"] = Relationship(back_populates="faces")

    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    confidence: float
    # 5 landmark points serialized as "[x1,y1,...,x5,y5]" (normalized 0..1).
    landmarks: str = Field(default="")
    eye_aspect_ratio: Optional[float] = None
    # Laplacian variance of the face crop (proxy for focus).
    face_sharpness: Optional[float] = None


class SemanticScore(SQLModel, table=True):
    __tablename__ = "semantic_scores"

    id: Optional[int] = Field(default=None, primary_key=True)
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
    computed_at: datetime = Field(default_factory=utcnow)


class BurstGroup(SQLModel, table=True):
    __tablename__ = "burst_groups"

    id: Optional[int] = Field(default=None, primary_key=True)
    # Stable human-friendly id, e.g. "Burst_20240512_0001".
    name: str = Field(unique=True, index=True)
    image_count: int = 0
    best_shot_id: Optional[int] = Field(default=None, foreign_key="images.id")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)


class ActionLedger(SQLModel, table=True):
    """Append-only transaction journal for every filesystem mutation."""

    __tablename__ = "action_ledger"

    id: Optional[int] = Field(default=None, primary_key=True)
    transaction_id: str = Field(index=True)
    image_id: Optional[int] = Field(default=None, foreign_key="images.id", index=True)
    source_path: str
    target_path: str
    operation_type: str = Field(default="move")  # move | copy | hardlink | symlink
    sha256: str = Field(default="")
    file_size: int = 0
    # "pending", "committed", "rolled_back", "failed".
    status: str = Field(default="pending", index=True)
    error: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)
    committed_at: Optional[datetime] = None

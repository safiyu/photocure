"""Centralized application configuration (Pydantic v2)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("photocure.config")

# "integrated" means whichever integrated or entry level GPU the machine has.
DEVICE_CHOICES = ("auto", "cuda", "rocm", "xpu", "mps", "directml", "integrated", "cpu")
MAX_WORKERS = 128


def normalize_device(value: object) -> str:
    """Return the canonical device name, or raise ValueError listing the choices."""
    v = str(value).strip().lower()
    if v not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device '{value}'. Choose one of: {', '.join(DEVICE_CHOICES)}.")
    return v


def _cpus(cpu_count: int | None = None) -> int:
    return max(1, int(cpu_count or os.cpu_count() or 1))


def auto_scan_workers(cpu_count: int | None = None) -> int:
    """Scanning is IO bound, so more threads than cores helps."""
    return min(32, _cpus(cpu_count) * 2)


def auto_decode_workers(cpu_count: int | None = None) -> int:
    return min(8, _cpus(cpu_count))


def auto_torch_threads(on_cpu: bool, cpu_count: int | None = None) -> int:
    """All cores for CPU inference, a couple of helper threads when a GPU does the work."""
    return min(_cpus(cpu_count), 8) if on_cpu else 2


LEGACY_DB_NAME = "photo_curator.db"  # the app was called photo-curator before photocure


def default_db_path() -> Path:
    """photocure.db, unless only a database from the old name exists in this folder."""
    new, legacy = Path("photocure.db"), Path(LEGACY_DB_NAME)
    return legacy if (not new.exists() and legacy.exists()) else new


DEFAULT_DB = default_db_path()


class DeviceSettings(BaseSettings):
    """GPU / device settings tuned for an 8 GiB RTX 4060 shared with the desktop."""

    model_config = SettingsConfigDict(env_prefix="PCUR_DEVICE_")

    # One of DEVICE_CHOICES. An unknown value falls back to "auto" with a warning.
    device: str = "auto"
    # Which GPU to use when the machine has more than one.
    device_index: int = Field(default=0, ge=0, le=MAX_WORKERS)
    # Threads decoding images ahead of the model. 0 means auto.
    decode_workers: int = Field(default=0, ge=0, le=MAX_WORKERS)
    # CPU threads used by torch. 0 means auto.
    torch_threads: int = Field(default=0, ge=0, le=MAX_WORKERS)
    # Cap PyTorch fraction of VRAM (0.75 of 8 GiB ~= 6 GiB, keeping headroom for Xorg/Jellyfin).
    cuda_memory_fraction: float = 0.75
    use_fp16: bool = True
    face_batch_size: int = 32
    clip_batch_size: int = 64
    # Downscale longest edge before model inference (saves VRAM on 48MP frames).
    face_max_side: int = 1280
    clip_max_side: int = 512
    # How many decoded frames to hold in RAM at once while running inference.
    # Bounding this (instead of the whole library) keeps peak memory ~O(chunk)
    # on 100k+ image sweeps; each phase decodes its chunks and discards them.
    infer_chunk_size: int = 512

    @field_validator("device", mode="before")
    @classmethod
    def _check_device(cls, v: object) -> str:
        try:
            return normalize_device(v)
        except ValueError as e:
            log.warning("%s Using 'auto'.", e)
            return "auto"


class QualityThresholds(BaseSettings):
    """Quality scoring thresholds (Laplacian variance & luminance)."""

    model_config = SettingsConfigDict(env_prefix="PCUR_QUALITY_")

    # Laplacian variance of a center-cropped, normalized patch.
    blur_low: float = 50.0      # below this -> "blurry"
    blur_high: float = 250.0    # at/above this -> "sharp"
    # Blurry also needs this share of edge energy to survive a re-blur (sharp is 0.2 to 0.45).
    blur_retained_min: float = 0.58
    # Luminance (0-255 grayscale mean).
    brightness_min: float = 18.0   # below -> pitch-black (accidental/lens cap)
    brightness_max: float = 245.0  # above -> blown-out
    # Percent of pixels clipped in the extreme tails.
    black_clip_pct: float = 0.60   # >60% near-black pixels
    white_clip_pct: float = 0.50   # >50% near-white pixels
    # Quality verdicts that force review routing.
    force_review_quality: bool = True
    # RMS contrast below this is flagged low-contrast.
    low_contrast_rms: float = 6.0


class BurstThresholds(BaseSettings):
    """Burst detection & best-shot selection thresholds."""

    model_config = SettingsConfigDict(env_prefix="PCUR_BURST_")

    window_seconds: int = 30
    similarity_threshold: float = 0.88
    # Burst group must contain >= 2 members to be a burst.
    min_group_size: int = 2
    # Best-shot composite weights.
    weight_blur: float = 0.40
    weight_face_sharp: float = 0.30
    weight_eye: float = 0.20
    weight_resolution: float = 0.10
    # EAR (eye aspect ratio) below this is considered "closed eyes".
    ear_closed: float = 0.16
    ear_open: float = 0.24
    # Longest time span a single burst may cover.
    max_burst_duration_s: float = 120.0
    weight_aesthetic: float = 0.10


class TaxonomySettings(BaseSettings):
    """Zero-shot classification gating."""

    model_config = SettingsConfigDict(env_prefix="PCUR_TAXONOMY_")

    min_confidence: float = 0.40   # below -> Uncertain_Review
    min_margin: float = 0.08       # top-1 minus top-2 below this -> Uncertain_Review
    face_confidence_min: float = 0.65
    # A perceptual-hash near-duplicate must also have CLIP embeddings at least this similar.
    near_dup_min_cosine: float = 0.93
    # Optional YAML or JSON file with a custom taxonomy.
    taxonomy_file: str | None = None
    # Top-1 below this never routes to a class, even when its group agrees.
    min_class_floor: float = 0.25
    # Weight of the learned (corrections) prototype scores versus zero-shot.
    feedback_weight: float = 0.6
    # Corrected photos needed in a class before it is learned from.
    feedback_min_examples: int = 2
    clip_model_name: str = "ViT-B-16"
    clip_pretrained: str = "openai"


class CuratorConfig(BaseSettings):
    """Top-level application settings.

    All sub-settings are overridable via environment variables with the
    ``PCUR_`` prefix (e.g. ``PCUR_BURST_SIMILARITY_THRESHOLD=0.9``).
    """

    model_config = SettingsConfigDict(env_prefix="PCUR_", env_file=".env", extra="ignore")

    db_path: Path = Field(default_factory=default_db_path)
    cache_dir: Path = Field(default=Path(".cache"))

    # Supported photo extensions (lowercase, with dot).
    extensions: tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
        ".cr2", ".nef", ".arw", ".dng", ".tif", ".tiff",
    )

    # Scan threads. 0 means auto (min(32, cpu_count * 2)), resolved when the config is built.
    scan_workers: int = Field(default=0, ge=0, le=MAX_WORKERS)
    hash_chunk_size: int = 1024 * 1024
    quick_hash_head_bytes: int = 64 * 1024
    quick_hash_tail_bytes: int = 64 * 1024

    device: DeviceSettings = Field(default_factory=DeviceSettings)
    quality: QualityThresholds = Field(default_factory=QualityThresholds)
    burst: BurstThresholds = Field(default_factory=BurstThresholds)
    taxonomy: TaxonomySettings = Field(default_factory=TaxonomySettings)

    # Trash staging: junk moves into <root>/<trash_dir_name>/<txn>/ and is purged later.
    trash_dir_name: str = ".trash"
    trash_retention_days: int = 30

    # Output organization root (Option 1).
    organize_root: Path = Path("Organized")
    unknown_date_dir: str = "Unknown_Date"

    # Server & auth (C1).
    host: str = "127.0.0.1"
    port: int = 8080
    allowed_hosts: str = ""  # PCUR_ALLOWED_HOSTS: comma-separated override list
    collision_policy: str = "sequence"  # sequence, date_suffix, skip
    auto_db_backup: bool = True
    burst_gap_seconds: float = 2.0

    @model_validator(mode="after")
    def _resolve_scan_workers(self) -> CuratorConfig:
        # The scanner hands this straight to a thread pool, so never leave 0 in place.
        if self.scan_workers <= 0:
            self.scan_workers = auto_scan_workers()
        return self

    @classmethod
    def load(cls, db_path: Path | None = None) -> CuratorConfig:
        cfg = cls()
        if db_path is not None:
            cfg.db_path = Path(db_path)
        return cfg


def resolve_workers(cfg: CuratorConfig, resolved_device: str | None = None,
                    cpu_count: int | None = None) -> dict:
    """Concrete worker counts: explicit values win, 0 means auto.

    ``resolved_device`` is the device that will really run (after auto detection);
    without it the configured device is used and "auto" is treated as a GPU.
    """
    device = (resolved_device or cfg.device.device or "auto").lower()
    on_cpu = device == "cpu"
    scan = cfg.scan_workers if cfg.scan_workers > 0 else auto_scan_workers(cpu_count)
    decode = (cfg.device.decode_workers if cfg.device.decode_workers > 0
              else auto_decode_workers(cpu_count))
    threads = (cfg.device.torch_threads if cfg.device.torch_threads > 0
               else auto_torch_threads(on_cpu, cpu_count))
    clamp = lambda n: max(1, min(MAX_WORKERS, int(n)))  # noqa: E731
    return {"scan_workers": clamp(scan), "decode_workers": clamp(decode),
            "torch_threads": clamp(threads)}

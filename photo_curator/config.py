"""Centralized application configuration (Pydantic v2)."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeviceSettings(BaseSettings):
    """GPU / device settings tuned for an 8 GiB RTX 4060 shared with the desktop."""

    model_config = SettingsConfigDict(env_prefix="PCUR_DEVICE_")

    device: str = "cuda"  # "cuda", "cpu", "auto"
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


class QualityThresholds(BaseSettings):
    """Quality scoring thresholds (Laplacian variance & luminance)."""

    model_config = SettingsConfigDict(env_prefix="PCUR_QUALITY_")

    # Laplacian variance of a center-cropped, normalized patch.
    blur_low: float = 50.0      # below this -> "blurry"
    blur_high: float = 250.0    # at/above this -> "sharp"
    # Luminance (0-255 grayscale mean).
    brightness_min: float = 18.0   # below -> pitch-black (accidental/lens cap)
    brightness_max: float = 245.0  # above -> blown-out
    # Percent of pixels clipped in the extreme tails.
    black_clip_pct: float = 0.60   # >60% near-black pixels
    white_clip_pct: float = 0.50   # >50% near-white pixels
    # Quality verdicts that force review routing.
    force_review_quality: bool = True


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


class TaxonomySettings(BaseSettings):
    """Zero-shot classification gating."""

    model_config = SettingsConfigDict(env_prefix="PCUR_TAXONOMY_")

    min_confidence: float = 0.40   # below -> Uncertain_Review
    min_margin: float = 0.08       # top-1 minus top-2 below this -> Uncertain_Review
    face_confidence_min: float = 0.65
    clip_model_name: str = "ViT-B-16"
    clip_pretrained: str = "openai"


class CuratorConfig(BaseSettings):
    """Top-level application settings.

    All sub-settings are overridable via environment variables with the
    ``PCUR_`` prefix (e.g. ``PCUR_BURST_SIMILARITY_THRESHOLD=0.9``).
    """

    model_config = SettingsConfigDict(env_prefix="PCUR_", env_file=".env", extra="ignore")

    db_path: Path = Field(default=Path("photo_curator.db"))
    cache_dir: Path = Field(default=Path(".cache"))

    # Supported photo extensions (lowercase, with dot).
    extensions: tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
        ".cr2", ".nef", ".arw", ".dng", ".tif", ".tiff",
    )

    scan_workers: int = 8
    hash_chunk_size: int = 1024 * 1024
    quick_hash_head_bytes: int = 64 * 1024
    quick_hash_tail_bytes: int = 64 * 1024

    device: DeviceSettings = DeviceSettings()
    quality: QualityThresholds = QualityThresholds()
    burst: BurstThresholds = BurstThresholds()
    taxonomy: TaxonomySettings = TaxonomySettings()

    # Output organization root (Option 1).
    organize_root: Path = Path("Organized")
    unknown_date_dir: str = "Unknown_Date"

    @classmethod
    def load(cls, db_path: Path | None = None) -> CuratorConfig:
        cfg = cls()
        if db_path is not None:
            cfg.db_path = Path(db_path)
        return cfg

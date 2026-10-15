"""Shared builder for engine/plan tests: a scanned library with distinct images."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from photocure.config import CuratorConfig
from photocure.db.repository import PhotoRepository
from photocure.db.session import get_session, init_db
from photocure.services import Planner


def write_noise_jpeg(path: Path, seed: int, size=(96, 128)) -> Path:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


@dataclass
class World:
    cfg: CuratorConfig
    db: Path
    root: Path
    out: Path
    planner: Planner
    ids: list[int]
    paths: list[Path]


def build_world(tmp_path: Path, n: int = 4) -> World:
    from photocure.scanner.file_scanner import FileScanner

    db = tmp_path / "w.db"
    cfg = CuratorConfig()
    cfg.db_path = db
    cfg.device.device = "cpu"
    root = tmp_path / "lib"
    paths = [write_noise_jpeg(root / f"IMG_2024050{i + 1}_120000.jpg", seed=100 + i)
             for i in range(n)]
    init_db(db, drop=True)
    FileScanner(cfg).scan_directory(root, db_path=db, show_progress=False)
    with get_session(db) as s:
        repo = PhotoRepository(s)
        recs = sorted(repo.all_images(), key=lambda r: r.filename)
        for r in recs:
            r.infer_status = "inferred"
        ids = [r.id for r in recs]
    return World(cfg, db, root, tmp_path / "out", Planner(cfg, db_path=db), ids, paths)


def set_bucket(world: World, image_id: int, bucket: str | None) -> None:
    with get_session(world.db) as s:
        PhotoRepository(s).set_manual_bucket(image_id, bucket)


def get_img(world: World, image_id: int):
    with get_session(world.db) as s:
        r = PhotoRepository(s).get_by_id(image_id)
        s.expunge(r)
        return r

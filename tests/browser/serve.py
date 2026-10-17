"""Seed a library and serve it, for the browser tests.

python tests/browser/serve.py --dir DIR --port 8765 --n 5000
Writes DIR/meta.json when ready, then serves until killed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CAMERAS = ["Canon R5", "Pixel 8", "Nikon Z6"]
COUNTRIES = ["MA", "FR", "", "ES"]
FOREIGN = 10          # this photo's sidecar already carries a rating written by "another tool"
BURST = (100, 101, 102, 103)


def seed(root: Path, n: int) -> dict:
    import cv2
    import numpy as np

    from photocure.db import models as M
    from photocure.db.session import get_session, init_db
    from photocure.scanner.xmp import update_xmp_sidecar

    lib = root / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    db = root / "browser.db"
    init_db(db, drop=True)
    start = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    step = timedelta(days=730) / n
    paths = []
    for i in range(n):
        p = lib / f"IMG_{i:05d}.jpg"
        shade = (i * 37) % 256
        img = np.full((72, 96, 3), (shade, (shade + 80) % 256, (shade + 160) % 256), np.uint8)
        cv2.putText(img, str(i), (4, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        paths.append(p)
    update_xmp_sidecar(paths[FOREIGN], rating=2)
    with get_session(db) as s:
        for lo in range(0, n, 500):
            for i in range(lo, min(n, lo + 500)):
                s.add(M.ImageRecord(
                    path=str(paths[i]), relative_path=paths[i].name, filename=paths[i].name, extension=".jpg",
                    mime_type="image/jpeg", file_size=paths[i].stat().st_size, width=96, height=72,
                    taken_at=start + step * i, taken_at_source="exif:dtoriginal",
                    camera_model=CAMERAS[i % 3], country_code=COUNTRIES[i % 4],
                    rating=3 if i % 50 == 0 else None, scan_status="scanned", infer_status="inferred",
                    category="scenery", quick_hash=f"q{i:06d}", sha256=f"s{i:06d}"))
            s.flush()
        bg = M.BurstGroup(name="Burst_test", image_count=len(BURST))
        s.add(bg)
        s.flush()
        ids = {}
        for k, i in enumerate(BURST):
            rec = s.exec(M.ImageRecord.__table__.select().where(M.ImageRecord.filename == paths[i].name)).first()
            ids[i] = rec.id
        for k, i in enumerate(BURST):
            r = s.get(M.ImageRecord, ids[i])
            r.burst_id, r.is_best_shot, r.is_alternate = bg.id, k == 0, k != 0
            blink = i in (BURST[1], BURST[2])
            r.has_blinks, r.blink_count = blink, 1 if blink else 0
            s.add(M.FaceDetection(image_id=r.id, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1, confidence=0.95,
                                  x1=0.2, y1=0.2, x2=0.8, y2=0.8, eye_closed_score=0.9 if blink else 0.05,
                                  eye_status="closed" if blink else "open", is_blink=blink, face_sharpness=300))
        bg.best_shot_id = ids[BURST[0]]
        _seed_features(s, paths)
    return {"n": n, "lib": str(lib), "foreign": paths[FOREIGN].name, "burst_photos": [paths[i].name for i in BURST],
            "burst_id": 1}


def _seed_features(s, paths) -> None:
    """Data for the search, highlights, duplicates and event views."""
    import numpy as np
    from sqlmodel import select

    from photocure.db import models as M
    from photocure.db.repository import quantize_embedding

    def rec(i):
        return s.exec(select(M.ImageRecord).where(M.ImageRecord.filename == paths[i].name)).one()

    exact_a, exact_b = rec(300), rec(301)           # identical files
    exact_a.sha256 = exact_b.sha256 = "sha-identical-pair"
    keeper, lookalike = rec(400), rec(401)          # a look-alike pair
    lookalike.near_dup_of = keeper.id
    ev = M.EventGroup(label="Seaside", start=rec(500).taken_at, end=rec(509).taken_at, image_count=10)
    s.add(ev)
    s.flush()
    for k, i in enumerate(range(500, 510)):         # ten scored photos, best first
        r = rec(i)
        r.event_id = ev.id
        vec = np.zeros(512, dtype=np.float32)
        vec[k] = 1.0
        s.add(M.SemanticScore(image_id=r.id, backend="clip", top1_category="scenery",
                              aesthetic=round(0.90 - 0.04 * k, 2), embedding=quantize_embedding(vec.tolist())))
    s.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--n", type=int, default=5000)
    a = ap.parse_args()
    root = Path(a.dir)
    root.mkdir(parents=True, exist_ok=True)
    meta = seed(root, a.n)
    otc, token = "otc-for-the-browser-test-0123456789", "token-for-the-browser-test-0123456789abcdef"
    os.environ.update({"PCUR_PORT": str(a.port), "PCUR_HOST": "127.0.0.1", "PCUR_OTC": otc, "PCUR_AUTH_TOKEN": token,
                       "PCUR_CACHE_DIR": str(root / "cache"), "PCUR_SIDECAR_DEBOUNCE_SECONDS": "0.1",
                       "PCUR_BACKEND": "stub"})
    import uvicorn

    from photocure.ui.app import create_app
    app = create_app(str(root / "browser.db"))
    meta.update(base=f"http://127.0.0.1:{a.port}", otc=otc, token=token, db=str(root / "browser.db"))
    (root / "meta.json").write_text(json.dumps(meta))
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

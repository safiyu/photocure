"""Build a realistic demo photo library for Photocure.

Features included in the demo set:
- ~110 distinct real-world photos (people, scenery, animals, food, architecture, vehicles)
- 3 burst sequences (4 frames each, 1s apart, with sharpness variations)
- 5 exact duplicate pairs (same hash)
- 4 near duplicate pairs (slight scale / recompression)
- 8 utility images (store receipts, screenshots, whiteboard notes)
- Realistic EXIF dates & camera models
"""
import concurrent.futures
from datetime import datetime, timedelta
import io
import os
from pathlib import Path
import random
import sys
import urllib.request

from PIL import Image, ImageDraw, ImageFilter, ExifTags

OUTPUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/safiyu/demo_photos")

CAMERAS = [
    ("Apple", "iPhone 15 Pro"),
    ("Sony", "ILCE-7RM4"),
    ("Fujifilm", "X-T5"),
    ("Canon", "EOS R6"),
    ("Google", "Pixel 8 Pro"),
]

BASE_EVENTS = [
    (datetime(2025, 5, 12, 10, 0), "Spring Weekend Hike"),
    (datetime(2025, 6, 20, 14, 30), "Summer Beach Vacation"),
    (datetime(2025, 7, 4, 18, 45), "City Evening & Architecture"),
    (datetime(2025, 8, 15, 12, 15), "Road Trip & Scenery"),
    (datetime(2025, 9, 28, 19, 0), "Dinner & Street Food"),
    (datetime(2025, 10, 10, 11, 20), "Family & Portrait Session"),
]


def attach_exif(img: Image.Image, dt: datetime, make: str, model: str) -> Image.Image:
    exif = img.getexif()
    dt_str = dt.strftime("%Y:%m:%d %H:%M:%S")
    exif[0x010F] = make
    exif[0x0110] = model
    exif[0x0132] = dt_str
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    exif_ifd[0x9003] = dt_str  # DateTimeOriginal
    exif_ifd[0x9004] = dt_str  # DateTimeDigitized
    return exif


def download_image(img_id: int) -> Image.Image | None:
    # 1200x800 resolution is sharp and fast to download
    url = f"https://picsum.photos/id/{img_id}/1200/800.jpg"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhotocureDemo/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def generate_receipt(idx: int) -> Image.Image:
    w, h = 600, 900
    img = Image.new("RGB", (w, h), color=(248, 248, 246))
    d = ImageDraw.Draw(img)

    # Header
    d.rectangle([30, 20, w - 30, 80], fill=(235, 235, 230))
    d.text((w // 2 - 80, 40), f"STORE #{1000 + idx} RECEIPT", fill=(30, 30, 30))

    # Items
    y = 120
    d.line([(40, y), (w - 40, y)], fill=(180, 180, 180), width=2)
    y += 20
    items = [
        ("Organic Whole Milk 1 Gal", "$4.89"),
        ("Sourdough Artisan Bread", "$5.49"),
        ("Cold Brew Coffee 32oz", "$6.25"),
        ("Avocados (Bag of 4)", "$4.99"),
        ("Dark Roast Espresso Beans", "$14.50"),
        ("Sparkling Water 12pk", "$7.99"),
        ("Reusable Grocery Bag", "$0.99"),
    ]
    for name, price in items:
        d.text((50, y), name, fill=(50, 50, 50))
        d.text((w - 120, y), price, fill=(30, 30, 30))
        y += 40

    d.line([(40, y + 10), (w - 40, y + 10)], fill=(180, 180, 180), width=2)
    y += 30
    d.text((50, y), "SUBTOTAL", fill=(80, 80, 80))
    d.text((w - 120, y), "$45.10", fill=(50, 50, 50))
    y += 30
    d.text((50, y), "TAX (8.25%)", fill=(80, 80, 80))
    d.text((w - 120, y), "$3.72", fill=(50, 50, 50))
    y += 35
    d.text((50, y), "TOTAL PAID", fill=(10, 10, 10))
    d.text((w - 120, y), "$48.82", fill=(10, 10, 10))

    # Barcode representation
    y += 90
    for bx in range(80, w - 80, 8):
        bw = random.choice([2, 4, 6])
        d.rectangle([bx, y, bx + bw, y + 60], fill=(20, 20, 20))
    d.text((w // 2 - 90, y + 70), f"TXN-84920384-{idx}", fill=(100, 100, 100))
    return img


def generate_screenshot(idx: int) -> Image.Image:
    w, h = 1000, 700
    img = Image.new("RGB", (w, h), color=(30, 30, 36))
    d = ImageDraw.Draw(img)

    # Window title bar
    d.rectangle([0, 0, w, 36], fill=(45, 45, 52))
    # Window buttons
    d.ellipse([12, 12, 24, 24], fill=(255, 95, 86))
    d.ellipse([32, 12, 44, 24], fill=(255, 189, 46))
    d.ellipse([52, 12, 64, 24], fill=(39, 201, 63))
    d.text((w // 2 - 80, 10), f"Terminal - Server Logs #{idx}", fill=(180, 180, 180))

    # Code / log text
    y = 60
    lines = [
        "$ photocure analyze --dir /photos",
        "[INFO] Initializing vision backbone: ViT-B-32 (OpenCLIP)",
        "[INFO] Scanning 124 photos across 6 events...",
        "[INFO] Processing burst groups (window=3.0s, sim>=0.85)...",
        "[INFO] Found 3 burst clusters (12 frames total)",
        "[INFO] Eye Landmarker: 0 blinks detected in primary frames",
        "[SUCCESS] Analysis complete in 4.2s. 0 errors, 14 categorized.",
        "$ systemctl status photocure.service",
        "Active: active (running) since Sun 2026-10-18 20:06:33 UTC",
    ]
    for line in lines:
        col = (100, 220, 100) if line.startswith("$") or "[SUCCESS]" in line else (200, 200, 200)
        d.text((40, y), line, fill=col)
        y += 45
    return img


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Creating demo photo library in: {OUTPUT_DIR}")

    # 1. Download base catalog of ~100 distinct photos from Picsum/Unsplash
    # We sample a diverse list of known stable IDs
    # (ids 10 through 150 have great variety: nature, people, animals, buildings, food)
    img_ids = [
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
        70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 87, 88, 89, 90,
        91, 92, 93, 94, 96, 98, 99, 100, 101, 102,
        103, 104, 106, 107, 108, 109, 110, 111, 112, 113,
        114, 115, 116, 117, 118, 119, 120, 121, 122, 123
    ]

    print(f"Downloading {len(img_ids)} diverse photos...")
    images_downloaded = []

    def fetch_one(iid):
        im = download_image(iid)
        return (iid, im) if im else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for res in pool.map(fetch_one, img_ids):
            if res and res[1]:
                images_downloaded.append(res)
                if len(images_downloaded) % 20 == 0:
                    print(f"  Downloaded {len(images_downloaded)} / {len(img_ids)}...")

    print(f"Successfully downloaded {len(images_downloaded)} photos.")

    saved_count = 0

    # 2. Save base photos with distributed event timestamps
    base_saved_images = []
    for idx, (iid, img) in enumerate(images_downloaded):
        event_base_dt, event_name = BASE_EVENTS[idx % len(BASE_EVENTS)]
        # Offset slightly within the event day (e.g. 5-30 mins apart)
        dt = event_base_dt + timedelta(minutes=(idx * 7) % 360, seconds=random.randint(10, 50))
        make, model = CAMERAS[idx % len(CAMERAS)]
        exif = attach_exif(img, dt, make, model)

        filename = f"IMG_{2000 + idx:04d}.JPG"
        path = OUTPUT_DIR / filename
        img.save(path, "JPEG", quality=90, exif=exif)
        base_saved_images.append((path, img, dt, make, model))
        saved_count += 1

    # 3. Create 3 Burst sequences (4 frames each = 12 burst photos)
    # Burst condition: timestamps < 3.0s apart, high visual similarity
    print("Generating 3 burst sequences (4 frames each)...")
    burst_sources = [base_saved_images[5], base_saved_images[25], base_saved_images[50]]
    for b_idx, (src_path, src_img, src_dt, make, model) in enumerate(burst_sources):
        burst_base_dt = src_dt + timedelta(hours=2)
        for frame_idx in range(4):
            # Frame 0 is sharpest/base; later frames have slight crop/jitter or slight blur
            f_img = src_img.copy()
            w, h = f_img.size
            if frame_idx == 0:
                pass  # best shot!
            elif frame_idx == 1:
                # slight shift
                f_img = f_img.crop((5, 5, w - 5, h - 5)).resize((w, h), Image.Resampling.LANCZOS)
            elif frame_idx == 2:
                # slightly blurred (alternate shot)
                f_img = f_img.filter(ImageFilter.GaussianBlur(radius=1.2))
            elif frame_idx == 3:
                # slightly darker / shift
                f_img = f_img.crop((10, 0, w, h - 10)).resize((w, h), Image.Resampling.LANCZOS)

            frame_dt = burst_base_dt + timedelta(seconds=frame_idx)
            exif = attach_exif(f_img, frame_dt, make, model)
            f_path = OUTPUT_DIR / f"BURST_{b_idx + 1:02d}_FRAME_{frame_idx + 1:02d}.JPG"
            f_img.save(f_path, "JPEG", quality=90, exif=exif)
            saved_count += 1

    # 4. Create 6 Exact Duplicate pairs (identical bytes, different names / subfolders)
    print("Generating 6 exact duplicate pairs...")
    dup_sub = OUTPUT_DIR / "Downloads_Backup"
    dup_sub.mkdir(exist_ok=True)
    for d_idx in range(6):
        src_path, src_img, _, _, _ = base_saved_images[10 + d_idx]
        dup_name = f"COPY_{src_path.stem}.JPG"
        (dup_sub / dup_name).write_bytes(src_path.read_bytes())
        saved_count += 1

    # 5. Create 4 Near-Duplicate pairs (slight scale or recompressed)
    print("Generating 4 near-duplicate pairs...")
    for n_idx in range(4):
        src_path, src_img, src_dt, make, model = base_saved_images[40 + n_idx]
        w, h = src_img.size
        near_img = src_img.resize((int(w * 0.96), int(h * 0.96)), Image.Resampling.LANCZOS)
        exif = attach_exif(near_img, src_dt + timedelta(minutes=1), make, model)
        near_path = OUTPUT_DIR / f"EDIT_{src_path.stem}.JPG"
        near_img.save(near_path, "JPEG", quality=82, exif=exif)
        saved_count += 1

    # 6. Create 8 Utility images (store receipts & software screenshots)
    print("Generating 8 utility shots (receipts and screenshots)...")
    util_dt = datetime(2025, 9, 10, 14, 0)
    for u_idx in range(4):
        rec_img = generate_receipt(u_idx + 1)
        exif = attach_exif(rec_img, util_dt + timedelta(days=u_idx * 3), "Apple", "iPhone 15 Pro")
        rec_path = OUTPUT_DIR / f"RECEIPT_202509{10 + u_idx}_00{u_idx}.JPG"
        rec_img.save(rec_path, "JPEG", quality=88, exif=exif)
        saved_count += 1

    for s_idx in range(4):
        scr_img = generate_screenshot(s_idx + 1)
        exif = attach_exif(scr_img, util_dt + timedelta(days=s_idx * 4), "Apple", "MacBook Pro")
        scr_path = OUTPUT_DIR / f"SCREENSHOT_202509{15 + s_idx}_00{s_idx}.JPG"
        scr_img.save(scr_path, "JPEG", quality=88, exif=exif)
        saved_count += 1

    print(f"\n[DONE] Successfully generated {saved_count} photos in {OUTPUT_DIR}!")


if __name__ == "__main__":
    main()

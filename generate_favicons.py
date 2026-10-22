# generate_favicons.py
"""Render the logo SVG into a multi-size favicon set."""
from pathlib import Path
import io

from PIL import Image

# Adjust these paths if you move the logo or static folder.
SVG_FILE = Path("photocure/ui/static/logo.svg")
OUT_DIR = Path("photocure/ui/static")  # static assets folder

def render_svg_to_pil(svg_path: Path, size: int = 1024) -> Image.Image:
    """Render an SVG to a square RGBA Pillow image."""
    svg_bytes = svg_path.read_bytes()

    # Try the lightweight, pure-Python renderer first.
    try:
        import resvg_py
        png_bytes = resvg_py.svg_to_bytes(svg_string=svg_bytes.decode("utf-8"), width=size, height=size)
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        # Fallback to cairosvg if resvg-py isn’t available.
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=size, output_height=size)
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")

def save_multisize_ico(master: Image.Image, path: Path) -> None:
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]
    master.save(path, format="ICO", sizes=sizes)

def main() -> None:
    master = render_svg_to_pil(SVG_FILE, size=1024)

    # 1️⃣ Favicon (ICO with several sizes)
    save_multisize_ico(master, OUT_DIR / "favicon.ico")

    # 2️⃣ Apple touch icon – 180 × 180 (PNG)
    master.resize((180, 180), Image.LANCZOS).save(
        OUT_DIR / "apple-touch-icon.png", format="PNG"
    )

    # 3️⃣ PWA icons – 192 × 192 and 512 × 512 (PNG)
    master.resize((192, 192), Image.LANCZOS).save(
        OUT_DIR / "icon-192.png", format="PNG"
    )
    master.resize((512, 512), Image.LANCZOS).save(
        OUT_DIR / "icon-512.png", format="PNG"
    )

    print(f"Created {OUT_DIR / 'favicon.ico'} (16‑64 px)")
    print(f"Created {OUT_DIR / 'apple-touch-icon.png'} (180 px)")
    print(f"Created {OUT_DIR / 'icon-192.png'} (192 px)")
    print(f"Created {OUT_DIR / 'icon-512.png'} (512 px)")

if __name__ == "__main__":
    main()

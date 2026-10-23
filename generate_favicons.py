"""Render the logo SVG into the favicon and app icon set (needs resvg-py or cairosvg)."""
import io
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "photocure" / "ui" / "static"
SVG_FILE = OUT_DIR / "logo.svg"
MASTER = 1024


def render_svg(svg_path: Path, height: int) -> Image.Image:
    svg_bytes = svg_path.read_bytes()
    try:
        import resvg_py
        png = resvg_py.svg_to_bytes(svg_string=svg_bytes.decode("utf-8"), height=height)
    except ImportError:
        import cairosvg
        png = cairosvg.svg2png(bytestring=svg_bytes, output_height=height)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def square(img: Image.Image, size: int, background=(0, 0, 0, 0), pad: float = 0.0) -> Image.Image:
    """Fit img inside a size x size canvas without distorting it."""
    inner = int(size * (1 - 2 * pad))
    scale = min(inner / img.width, inner / img.height)
    fitted = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), background)
    canvas.alpha_composite(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return canvas


def main() -> None:
    logo = render_svg(SVG_FILE, MASTER)
    master = square(logo, MASTER)

    master.save(OUT_DIR / "favicon.ico", format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])

    # iOS fills transparency with black, so the touch icon is a flat white tile
    touch = square(logo, 180, background=(255, 255, 255, 255), pad=0.12).convert("RGB")
    touch.save(OUT_DIR / "apple-touch-icon.png", format="PNG")

    for size in (192, 512):
        master.resize((size, size), Image.LANCZOS).save(OUT_DIR / f"icon-{size}.png", format="PNG")


if __name__ == "__main__":
    main()

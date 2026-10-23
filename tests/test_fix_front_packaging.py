import tomllib
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_static_fonts_covered_by_package_data():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    patterns = cfg["tool"]["setuptools"]["package-data"]["photocure"]
    pkg = ROOT / "photocure"
    fonts = [p for p in (pkg / "ui/static/fonts").rglob("*") if p.is_file()]
    assert fonts
    for f in fonts:
        rel = f.relative_to(pkg).as_posix()
        assert any(
            fnmatch(rel, pat) and rel.count("/") == pat.count("/") for pat in patterns
        ), rel

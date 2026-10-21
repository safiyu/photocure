#!/usr/bin/env python3
"""Bump Photocure version in the single canonical location: photocure/__init__.py.

Usage:
    python scripts/bump_version.py <new_version>
    python scripts/bump_version.py [patch|minor|major]

Examples:
    python scripts/bump_version.py patch      # 1.2.1 -> 1.2.2
    python scripts/bump_version.py minor      # 1.2.1 -> 1.3.0
    python scripts/bump_version.py major      # 1.2.1 -> 2.0.0
    python scripts/bump_version.py 1.3.0      # Explicit version
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
INIT_FILE = ROOT / "photocure" / "__init__.py"
METAINFO_FILE = ROOT / "packaging" / "deb" / "photocure.metainfo.xml"


def get_current_version() -> str:
    content = INIT_FILE.read_text(encoding="utf-8")
    match = re.search(r'(?m)^__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise RuntimeError(f"Could not find __version__ in {INIT_FILE}")
    return match.group(1)


def compute_next_version(current: str, bump_type: str) -> str:
    parts = current.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Current version '{current}' is not valid X.Y.Z semver.")
    major, minor, patch = (int(p) for p in parts)

    if bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    if bump_type == "major":
        return f"{major + 1}.0.0"

    # Assume explicit version string provided
    if not re.match(r"^\d+\.\d+\.\d+", bump_type):
        raise ValueError(f"Invalid bump type or version string: '{bump_type}'")
    return bump_type.lstrip("v")


def update_init_file(new_version: str) -> None:
    content = INIT_FILE.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        content,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Failed to update __version__ in photocure/__init__.py")
    INIT_FILE.write_text(updated, encoding="utf-8")


def update_metainfo(new_version: str) -> None:
    if not METAINFO_FILE.exists():
        return
    tree = ET.parse(METAINFO_FILE)
    root = tree.getroot()
    releases = root.find("releases")
    if releases is None:
        return
    if any(r.get("version") == new_version for r in releases):
        return

    rel = ET.Element("release", {"version": new_version, "type": "stable", "date": str(date.today())})
    desc = ET.SubElement(rel, "description")
    p = ET.SubElement(desc, "p")
    p.text = f"Photocure release {new_version}."
    releases.insert(0, rel)

    # Format XML nicely
    ET.indent(tree, space="  ", level=0)
    tree.write(METAINFO_FILE, encoding="utf-8", xml_declaration=True)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        current = get_current_version()
        print(f"Current version: {current}")
        return 1

    arg = sys.argv[1].strip()
    current = get_current_version()
    new_version = compute_next_version(current, arg)

    if new_version == current:
        print(f"Version is already {current}. Nothing to do.")
        return 0

    print(f"Bumping Photocure version: {current} -> {new_version}")
    update_init_file(new_version)
    update_metainfo(new_version)

    print("✔ Updated photocure/__init__.py")
    print(f"All components automatically derive version {new_version} from photocure.__version__:")
    print("  • pyproject.toml (dynamic setuptools versioning)")
    print("  • UI web app (/ and /api/about dynamic injection)")
    print("  • packaging/deb/build_deb.sh")
    print("  • .github/workflows/ci-cd.yml")
    return 0


if __name__ == "__main__":
    sys.exit(main())

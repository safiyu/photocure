#!/usr/bin/env bash
# Build Photocure .deb package
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"
STAGING_DIR="$DIST_DIR/deb_staging"

# Extract version (can be passed as $1, VERSION_OVERRIDE, or dynamically from photocure)
VERSION="${1:-${VERSION_OVERRIDE:-}}"
if [ -z "$VERSION" ]; then
    VERSION="$(python3 -c 'import photocure; print(photocure.__version__)' 2>/dev/null || true)"
fi
if [ -z "$VERSION" ]; then
    VERSION="$(python3 -c 'from pathlib import Path; import re; m = re.search(r"__version__\s*=\s*[\x22\x27]([^\x22\x27]+)", (Path("photocure/__init__.py")).read_text()); print(m.group(1) if m else "")' 2>/dev/null || true)"
fi
VERSION="${VERSION#v}"
if [ -z "$VERSION" ]; then
    echo "Error: Could not determine version from photocure package." >&2
    exit 1
fi
ARCH="amd64"
DEB_NAME="photocure_${VERSION}_${ARCH}.deb"

echo "=== Building Photocure v${VERSION} Debian Package (${ARCH}) ==="

# Clean staging area
rm -rf "$STAGING_DIR"
mkdir -p "$DIST_DIR"
mkdir -p "$STAGING_DIR/DEBIAN"
mkdir -p "$STAGING_DIR/opt/photocure/wheels"
mkdir -p "$STAGING_DIR/usr/bin"
mkdir -p "$STAGING_DIR/usr/share/applications"
mkdir -p "$STAGING_DIR/usr/share/metainfo"
mkdir -p "$STAGING_DIR/usr/lib/systemd/user"
mkdir -p "$STAGING_DIR/usr/share/doc/photocure"
for size in 16x16 24x24 32x32 48x48 64x64 128x128 256x256 512x512; do
  mkdir -p "$STAGING_DIR/usr/share/icons/hicolor/$size/apps"
done
mkdir -p "$STAGING_DIR/usr/share/pixmaps"

# 1. Package metadata and maintainer scripts (no pip, no process killing)
sed "s/Version: .*/Version: $VERSION/" "$SCRIPT_DIR/control" > "$STAGING_DIR/DEBIAN/control"
cp "$SCRIPT_DIR/postinst" "$STAGING_DIR/DEBIAN/postinst"
cp "$SCRIPT_DIR/postrm" "$STAGING_DIR/DEBIAN/postrm"

# 2. Build the wheel from a copy stamped with the package version, so that
#    "pip install --upgrade" on the user's venv always sees a new version.
WHEEL_SRC="$DIST_DIR/wheel_src"
rm -rf "$WHEEL_SRC"
mkdir -p "$WHEEL_SRC"
cp -r "$PROJECT_ROOT/photocure" "$PROJECT_ROOT/pyproject.toml" "$PROJECT_ROOT/README.md" "$WHEEL_SRC/"
find "$WHEEL_SRC" -type d -name "__pycache__" -prune -exec rm -rf {} +
python3 - "$WHEEL_SRC/photocure/__init__.py" "$VERSION" <<'PY'
import re, sys
path, version = sys.argv[1:3]
text = open(path).read()
text, n = re.subn(r'(?m)^__version__\s*=\s*".*"$', f'__version__ = "{version}"', text, count=1)
assert n == 1, "no __version__ line found in photocure/__init__.py"
open(path, "w").write(text)
PY
python3 -m pip wheel --no-deps --quiet -w "$STAGING_DIR/opt/photocure/wheels" "$WHEEL_SRC"
rm -rf "$WHEEL_SRC"

# 3. Bootstrap, version marker and launcher
cp "$SCRIPT_DIR/bootstrap.py" "$STAGING_DIR/opt/photocure/bootstrap.py"
printf '%s\n' "$VERSION" > "$STAGING_DIR/opt/photocure/VERSION"
cp "$SCRIPT_DIR/photocure-wrapper" "$STAGING_DIR/usr/bin/photocure"
cp "$SCRIPT_DIR/photocure.desktop" "$STAGING_DIR/usr/share/applications/io.github.safiyu.photocure.desktop"
cp "$SCRIPT_DIR/photocure.metainfo.xml" "$STAGING_DIR/usr/share/metainfo/io.github.safiyu.photocure.metainfo.xml"
python3 - "$STAGING_DIR/usr/share/metainfo/io.github.safiyu.photocure.metainfo.xml" "$VERSION" <<'PY'
import sys, xml.etree.ElementTree as ET
from datetime import date
path, version = sys.argv[1:3]
tree = ET.parse(path)
root = tree.getroot()
releases = root.find("releases")
if releases is not None and not any(r.get("version") == version for r in releases):
    rel = ET.Element("release", {"version": version, "type": "stable", "date": str(date.today())})
    desc = ET.SubElement(rel, "description")
    p = ET.SubElement(desc, "p")
    p.text = f"Photocure release {version}"
    releases.insert(0, rel)
    tree.write(path, encoding="utf-8", xml_declaration=True)
PY
cp "$SCRIPT_DIR/photocure.service" "$STAGING_DIR/usr/lib/systemd/user/photocure.service"

# 4. Icons - generate multi-resolution from source or use source directly
LOGO="$PROJECT_ROOT/docs/logo.png"
if [ -f "$LOGO" ]; then
    if command -v convert >/dev/null 2>&1; then
        # Generate resized versions with ImageMagick
        for size in 16 24 32 48 64 128 256 512; do
            convert "$LOGO" -resize "${size}x${size}!" \
                "$STAGING_DIR/usr/share/icons/hicolor/${size}x${size}/apps/io.github.safiyu.photocure.png"
        done
    else
        # Fallback: copy source to all standard sizes
        for size in 16 24 32 48 64 128 256 512; do
            cp "$LOGO" "$STAGING_DIR/usr/share/icons/hicolor/${size}x${size}/apps/io.github.safiyu.photocure.png"
        done
    fi
    cp "$LOGO" "$STAGING_DIR/usr/share/pixmaps/io.github.safiyu.photocure.png"
fi

# 5. Documentation
cp "$PROJECT_ROOT/NOTICE" "$STAGING_DIR/usr/share/doc/photocure/NOTICE"
cp "$PROJECT_ROOT/LICENSE" "$STAGING_DIR/usr/share/doc/photocure/LICENSE"

# Permissions: root-owned (dpkg-deb --root-owner-group), dirs 0755, files 0644,
# executables 0755. Nothing is world-writable.
find "$STAGING_DIR" -type d -exec chmod 755 {} +
find "$STAGING_DIR" -type f -exec chmod 644 {} +
chmod 755 "$STAGING_DIR/DEBIAN/postinst" "$STAGING_DIR/DEBIAN/postrm" \
    "$STAGING_DIR/opt/photocure/bootstrap.py" "$STAGING_DIR/usr/bin/photocure"

# 6. Build .deb package
echo "Running dpkg-deb..."
dpkg-deb --build --root-owner-group "$STAGING_DIR" "$DIST_DIR/$DEB_NAME"
cp "$DIST_DIR/$DEB_NAME" "$DIST_DIR/photocure_amd64.deb"

echo ""
echo "=== Package built successfully! ==="
echo "Location: $DIST_DIR/$DEB_NAME"
echo "Size: $(du -h "$DIST_DIR/$DEB_NAME" | cut -f1)"
echo ""
echo "=== Package Information ==="
dpkg-deb -I "$DIST_DIR/$DEB_NAME"

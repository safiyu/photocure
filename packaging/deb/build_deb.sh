#!/usr/bin/env bash
# Build Photocure .deb package
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"
STAGING_DIR="$DIST_DIR/deb_staging"

# Extract version (can be passed as $1 or from pyproject.toml)
VERSION="${1:-${VERSION_OVERRIDE:-$(grep -m1 '^version = ' "$PROJECT_ROOT/pyproject.toml" | cut -d '"' -f2)}}"
VERSION="${VERSION#v}"
if [ -z "$VERSION" ]; then
    VERSION="0.1.0"
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
mkdir -p "$STAGING_DIR/usr/share/icons/hicolor/256x256/apps"
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
python3 - "$WHEEL_SRC/pyproject.toml" "$VERSION" <<'PY'
import re, sys
path, version = sys.argv[1:3]
text = open(path).read()
text, n = re.subn(r'(?m)^version = ".*"$', f'version = "{version}"', text, count=1)
assert n == 1, "no version line found in pyproject.toml"
open(path, "w").write(text)
PY
python3 -m pip wheel --no-deps --quiet -w "$STAGING_DIR/opt/photocure/wheels" "$WHEEL_SRC"
rm -rf "$WHEEL_SRC"

# 3. Bootstrap, version marker and launcher
cp "$SCRIPT_DIR/bootstrap.py" "$STAGING_DIR/opt/photocure/bootstrap.py"
printf '%s\n' "$VERSION" > "$STAGING_DIR/opt/photocure/VERSION"
cp "$SCRIPT_DIR/photocure-wrapper" "$STAGING_DIR/usr/bin/photocure"
cp "$SCRIPT_DIR/photocure.desktop" "$STAGING_DIR/usr/share/applications/photocure.desktop"

# 4. Icons
if [ -f "$PROJECT_ROOT/docs/logo.png" ]; then
    cp "$PROJECT_ROOT/docs/logo.png" "$STAGING_DIR/usr/share/icons/hicolor/256x256/apps/photocure.png"
    cp "$PROJECT_ROOT/docs/logo.png" "$STAGING_DIR/usr/share/pixmaps/photocure.png"
fi

# Permissions: root-owned (dpkg-deb --root-owner-group), dirs 0755, files 0644,
# executables 0755. Nothing is world-writable.
find "$STAGING_DIR" -type d -exec chmod 755 {} +
find "$STAGING_DIR" -type f -exec chmod 644 {} +
chmod 755 "$STAGING_DIR/DEBIAN/postinst" "$STAGING_DIR/DEBIAN/postrm" \
    "$STAGING_DIR/opt/photocure/bootstrap.py" "$STAGING_DIR/usr/bin/photocure"

# 5. Build .deb package
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

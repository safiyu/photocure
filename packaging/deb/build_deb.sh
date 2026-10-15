#!/usr/bin/env bash
# Build Photocure .deb package
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"
STAGING_DIR="$DIST_DIR/deb_staging"

# Extract version from pyproject.toml
VERSION=$(grep -m1 '^version = ' "$PROJECT_ROOT/pyproject.toml" | cut -d '"' -f2)
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
mkdir -p "$STAGING_DIR/opt/photocure"
mkdir -p "$STAGING_DIR/usr/bin"
mkdir -p "$STAGING_DIR/usr/share/applications"
mkdir -p "$STAGING_DIR/usr/share/icons/hicolor/256x256/apps"
mkdir -p "$STAGING_DIR/usr/share/pixmaps"

# 1. Copy package metadata and scripts
sed "s/Version: .*/Version: $VERSION/" "$SCRIPT_DIR/control" > "$STAGING_DIR/DEBIAN/control"
cp "$SCRIPT_DIR/postinst" "$STAGING_DIR/DEBIAN/postinst"
cp "$SCRIPT_DIR/prerm" "$STAGING_DIR/DEBIAN/prerm"
cp "$SCRIPT_DIR/postrm" "$STAGING_DIR/DEBIAN/postrm"
chmod 755 "$STAGING_DIR/DEBIAN/postinst" "$STAGING_DIR/DEBIAN/prerm" "$STAGING_DIR/DEBIAN/postrm"

# 2. Copy source code into /opt/photocure
cp -r "$PROJECT_ROOT/photocure" "$STAGING_DIR/opt/photocure/"
cp "$PROJECT_ROOT/pyproject.toml" "$STAGING_DIR/opt/photocure/"
cp "$PROJECT_ROOT/README.md" "$STAGING_DIR/opt/photocure/"

# Remove Python cache directories from staging
find "$STAGING_DIR/opt/photocure" -type d -name "__pycache__" -exec rm -rf {} +
find "$STAGING_DIR/opt/photocure" -type f -name "*.pyc" -delete

# 3. Copy launcher and desktop shortcuts
cp "$SCRIPT_DIR/photocure-wrapper" "$STAGING_DIR/usr/bin/photocure"
chmod 755 "$STAGING_DIR/usr/bin/photocure"

cp "$SCRIPT_DIR/photocure.desktop" "$STAGING_DIR/usr/share/applications/photocure.desktop"
chmod 644 "$STAGING_DIR/usr/share/applications/photocure.desktop"

# 4. Copy icons
if [ -f "$PROJECT_ROOT/docs/logo.png" ]; then
    cp "$PROJECT_ROOT/docs/logo.png" "$STAGING_DIR/usr/share/icons/hicolor/256x256/apps/photocure.png"
    cp "$PROJECT_ROOT/docs/logo.png" "$STAGING_DIR/usr/share/pixmaps/photocure.png"
    chmod 644 "$STAGING_DIR/usr/share/icons/hicolor/256x256/apps/photocure.png" "$STAGING_DIR/usr/share/pixmaps/photocure.png"
fi

# 5. Build .deb package
echo "Running dpkg-deb..."
dpkg-deb --build --root-owner-group "$STAGING_DIR" "$DIST_DIR/$DEB_NAME"

echo ""
echo "=== Package built successfully! ==="
echo "Location: $DIST_DIR/$DEB_NAME"
echo "Size: $(du -h "$DIST_DIR/$DEB_NAME" | cut -f1)"
echo ""
echo "=== Package Information ==="
dpkg-deb -I "$DIST_DIR/$DEB_NAME"

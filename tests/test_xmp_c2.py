"""Tests for C2: XMP sidecar writer and hardlink undo (§3.1-3.3)."""
import tempfile
from pathlib import Path

import pytest

from photocure.scanner.xmp import update_xmp_sidecar, get_on_disk_rating, _find_sidecar_path


class TestXMPSidecarWriter:
    """C2 §3.1: non-destructive XMP sidecar writing."""

    def test_create_new_sidecar(self):
        """Acceptance: create <filename>.<ext>.xmp when none exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            photo = Path(tmpdir) / "IMG_0001.JPG"
            photo.touch()
            success, msg = update_xmp_sidecar(photo, rating=3)
            assert success, msg
            sidecar = Path(tmpdir) / "IMG_0001.JPG.xmp"
            assert sidecar.exists()
            assert b"<xmp:Rating>3</xmp:Rating>" in sidecar.read_bytes()

    def test_prefer_ext_sidecar(self):
        """Acceptance: use <filename>.<ext>.xmp if it exists (preferred form)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            photo = Path(tmpdir) / "IMG_0001.JPG"
            photo.touch()
            ext_sidecar = Path(tmpdir) / "IMG_0001.JPG.xmp"
            ext_sidecar.write_text('<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about=""/></rdf:RDF>')
            success, msg = update_xmp_sidecar(photo, rating=2)
            assert success
            assert b"<xmp:Rating>2</xmp:Rating>" in ext_sidecar.read_bytes()

    def test_use_legacy_sidecar(self):
        """Acceptance: use <filename>.xmp (legacy) if it's the only one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            photo = Path(tmpdir) / "IMG_0001.JPG"
            photo.touch()
            legacy = Path(tmpdir) / "IMG_0001.xmp"
            legacy.write_text('<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about=""/></rdf:RDF>')
            success, msg = update_xmp_sidecar(photo, rating=1)
            assert success
            assert not Path(tmpdir / "IMG_0001.JPG.xmp").exists()
            assert b"<xmp:Rating>1</xmp:Rating>" in legacy.read_bytes()

    def test_dual_sidecar_guard(self):
        """Acceptance: refuse to write if both .xmp and .<ext>.xmp exist (spec §3.1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            photo = Path(tmpdir) / "IMG_0001.JPG"
            photo.touch()
            (Path(tmpdir) / "IMG_0001.xmp").touch()
            (Path(tmpdir) / "IMG_0001.JPG.xmp").touch()
            success, msg = update_xmp_sidecar(photo, rating=4)
            assert not success
            assert "dual sidecar" in msg.lower()

    def test_element_preservation(self):
        """Acceptance: round-trip preserves non-Photocure elements (spec §3.1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            photo = Path(tmpdir) / "IMG_0001.JPG"
            photo.touch()
            sidecar = Path(tmpdir) / "IMG_0001.JPG.xmp"
            # Sidecar with Adobe CRS and LR elements.
            original_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:xmp="http://ns.adobe.com/xap/1.0/"
         xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
  <rdf:Description rdf:about="">
    <crs:WhiteBalance>Daylight</crs:WhiteBalance>
    <xmp:Label>Test Label</xmp:Label>
  </rdf:Description>
</rdf:RDF>'''
            sidecar.write_text(original_xml)
            success, msg = update_xmp_sidecar(photo, rating=5, label="Updated")
            assert success
            result = sidecar.read_text()
            assert "WhiteBalance" in result, "CRS element was lost"
            assert "Updated" in result, "Label was not updated"
            assert "Daylight" in result, "CRS value was lost"

    def test_photocure_tags(self):
        """Acceptance: add/remove flat photocure-* tags only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            photo = Path(tmpdir) / "IMG_0001.JPG"
            photo.touch()
            success, msg = update_xmp_sidecar(
                photo,
                rating=3,
                add_tags=("photocure-best-shot", "photocure-blink", "other-tag")
            )
            assert success
            sidecar = Path(tmpdir) / "IMG_0001.JPG.xmp"
            content = sidecar.read_text()
            assert "photocure-best-shot" in content
            assert "photocure-blink" in content
            assert "other-tag" not in content, "Non-photocure tags should not be added"


class TestDivergenceTracking:
    """C2 §3.2: first-run baseline & foreign rating detection."""

    def test_get_on_disk_rating(self):
        """Acceptance: extract xmp:Rating from sidecar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            photo = Path(tmpdir) / "IMG_0001.JPG"
            photo.touch()
            # Create sidecar with existing rating.
            update_xmp_sidecar(photo, rating=4)
            assert get_on_disk_rating(photo) == 4
            # Update rating.
            update_xmp_sidecar(photo, rating=2)
            assert get_on_disk_rating(photo) == 2


class TestHardlinkFallback:
    """C2 §3.3: hardlink with EXDEV fallback & undo (stubs)."""

    def test_hardlink_same_filesystem(self):
        """Acceptance: hardlink succeeds on same filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src.txt"
            src.write_text("test content")
            dst = Path(tmpdir) / "dst.txt"
            from photocure.engine.action_engine import ExecutionReport, ActionEngine
            rep = ExecutionReport(transaction_id="TX_test")
            op, dev, ino = ActionEngine._link_or_copy(src, dst, rep)
            assert op == "hardlink"
            assert dst.exists()
            assert src.stat().st_ino == dst.stat().st_ino, "Hardlink should have same inode"
            assert rep.copy_fallbacks == 0

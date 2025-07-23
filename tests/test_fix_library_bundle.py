from pathlib import Path

from photocure.config import CuratorConfig
from photocure.scanner.file_scanner import FileScanner


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_photos_library_bundle_is_not_walked(tmp_path):
    _touch(tmp_path / "keep.jpg")
    _touch(tmp_path / "Photos Library.photoslibrary/resources/derivatives/4/A_1_102_o.jpeg")
    _touch(tmp_path / "Photos Library.photoslibrary/originals/4/A.heic")
    _touch(tmp_path / "Old.aplibrary/Masters/B.jpg")
    _touch(tmp_path / "sub/also.jpg")
    scanner = FileScanner(CuratorConfig())
    names = sorted(p.name for p in scanner.iter_files(
        tmp_path, extensions=[".jpg", ".jpeg", ".heic"]))
    assert names == ["also.jpg", "keep.jpg"]


def test_watch_mode_ignores_library_bundle(tmp_path):
    f = tmp_path / "Photos Library.photoslibrary/resources/derivatives/4/A_1_102_o.jpeg"
    _touch(f)
    res = FileScanner(CuratorConfig()).scan_paths(tmp_path, [f], db_path=tmp_path / "t.db")
    assert res.scanned == 0

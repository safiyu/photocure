"""EXIF reader: filename patterns, tag keys, GPS, camera, time zones."""
from __future__ import annotations

import datetime as dt
import os

import exifread
import pytest
from PIL import Image

from photocure.scanner import exif_reader
from photocure.scanner.exif_reader import (
    ExifInfo,
    _parse_exif_date,
    _parse_offset,
    extract_exif,
    extract_from_filename,
    taken_at_utc,
)


def _jpeg_with_exif(path, *, dto=None, digitized=None, modified=None, offset=None,
                    model=None, gps=None):
    im = Image.new("RGB", (64, 48), (120, 60, 30))
    ex = Image.Exif()
    if model:
        ex[272] = model
    if modified:
        ex[306] = modified
    sub = ex.get_ifd(0x8769)
    if dto:
        sub[36867] = dto
    if digitized:
        sub[36868] = digitized
    if offset:
        sub[36881] = offset
    if gps:
        g = ex.get_ifd(0x8825)
        g[1], g[2], g[3], g[4] = gps
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, exif=ex)
    return path


@pytest.mark.parametrize("name,expected", [
    ("PXL_20240512_153045567.jpg", dt.datetime(2024, 5, 12, 15, 30, 45)),
    ("PXL_20240512T153045.jpg", dt.datetime(2024, 5, 12, 15, 30, 45)),
    ("IMG_20240512_153045.jpg", dt.datetime(2024, 5, 12, 15, 30, 45)),
    ("IMG-20240512-WA0001.jpg", dt.datetime(2024, 5, 12)),
    ("Screenshot_2024-05-12_15-30-45.png", dt.datetime(2024, 5, 12, 15, 30, 45)),
    ("DSC_20240512.jpg", dt.datetime(2024, 5, 12)),
])
def test_filename_patterns(name, expected):
    assert extract_from_filename(name) == expected


@pytest.mark.parametrize("name", ["IMG_30001231.jpg", "IMG_19700101.jpg",
                                  "IMG_20241399.jpg", "holiday.jpg"])
def test_filename_garbage_rejected(name):
    assert extract_from_filename(name) is None


def test_exif_date_cascade_prefers_original(tmp_path):
    p = _jpeg_with_exif(tmp_path / "a.jpg", dto="2021:03:04 05:06:07",
                        digitized="2022:01:01 00:00:00", modified="2023:01:01 00:00:00")
    info = extract_exif(p)
    assert info.taken_at == dt.datetime(2021, 3, 4, 5, 6, 7)
    assert info.taken_at_source == "exif:dtoriginal"


def test_exif_digitized_beats_modified(tmp_path):
    p = _jpeg_with_exif(tmp_path / "a.jpg", digitized="2022:01:01 00:00:00",
                        modified="2023:01:01 00:00:00")
    info = extract_exif(p)
    assert info.taken_at == dt.datetime(2022, 1, 1)
    assert info.taken_at_source == "exif:create"


def test_exif_modified_is_last_exif_fallback(tmp_path):
    p = _jpeg_with_exif(tmp_path / "IMG_20200101_000000.jpg", modified="2023:05:06 07:08:09")
    assert extract_exif(p).taken_at == dt.datetime(2023, 5, 6, 7, 8, 9)


def test_exifread_uses_current_tag_keys(tmp_path):
    p = _jpeg_with_exif(tmp_path / "a.jpg", dto="2021:03:04 05:06:07",
                        digitized="2021:03:04 05:06:08", modified="2021:03:04 05:06:09",
                        offset="+02:00")
    with open(p, "rb") as f:
        tags = exifread.process_file(f, details=False)
    for key in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime",
                "EXIF OffsetTimeOriginal"):
        assert key in tags


def test_exifread_path_dates_files_pillow_cannot(tmp_path, monkeypatch):
    p = _jpeg_with_exif(tmp_path / "a.jpg", dto="2021:03:04 05:06:07", offset="-05:00")
    monkeypatch.setattr(exif_reader, "Image", None)  # force the exifread pass
    info = extract_exif(p)
    assert info.taken_at == dt.datetime(2021, 3, 4, 5, 6, 7)
    assert info.taken_at_source == "exif:dtoriginal"
    assert info.tz_offset_minutes == -300


def test_exifread_pass_reads_digitized_and_modified(tmp_path, monkeypatch):
    monkeypatch.setattr(exif_reader, "Image", None)
    p = _jpeg_with_exif(tmp_path / "a.jpg", digitized="2022:01:01 00:00:00")
    assert extract_exif(p).taken_at == dt.datetime(2022, 1, 1)
    p2 = _jpeg_with_exif(tmp_path / "b.jpg", modified="2023:02:03 04:05:06")
    assert extract_exif(p2).taken_at == dt.datetime(2023, 2, 3, 4, 5, 6)


def test_camera_model_from_pillow_tag_272(tmp_path):
    p = _jpeg_with_exif(tmp_path / "a.jpg", model="Pixel 8 Pro")
    assert extract_exif(p).camera_model == "Pixel 8 Pro"


def test_gps_from_pillow_ifd(tmp_path):
    p = _jpeg_with_exif(tmp_path / "a.jpg", gps=("S", (33.0, 30.0, 0.0), "E", (151.0, 12.0, 36.0)))
    info = extract_exif(p)
    assert info.gps_lat == pytest.approx(-33.5)
    assert info.gps_lon == pytest.approx(151.21)


def test_gps_out_of_range_is_dropped(tmp_path):
    p = _jpeg_with_exif(tmp_path / "a.jpg", gps=("N", (95.0, 0.0, 0.0), "E", (10.0, 0.0, 0.0)))
    assert extract_exif(p).gps_lat is None


def test_offset_time_normalizes_to_utc(tmp_path):
    p = _jpeg_with_exif(tmp_path / "a.jpg", dto="2024:05:12 01:30:00", offset="+02:00")
    info = extract_exif(p)
    assert info.tz_offset_minutes == 120
    u = taken_at_utc(info)
    assert u == dt.datetime(2024, 5, 11, 23, 30, tzinfo=dt.timezone.utc)


def test_no_offset_keeps_wall_clock_tagged_utc(tmp_path):
    p = _jpeg_with_exif(tmp_path / "a.jpg", dto="2024:05:12 01:30:00")
    info = extract_exif(p)
    assert info.tz_offset_minutes is None
    assert taken_at_utc(info) == dt.datetime(2024, 5, 12, 1, 30, tzinfo=dt.timezone.utc)


def test_mtime_tier_is_converted_from_local_time(tmp_path):
    p = tmp_path / "plain.jpg"
    Image.new("RGB", (8, 8)).save(p)
    target = dt.datetime(2022, 6, 15, 12, 0, 0, tzinfo=dt.timezone.utc).timestamp()
    os.utime(p, (target, target))
    info = extract_exif(p)
    assert info.taken_at_source == "mtime"
    assert info.tz_offset_minutes is not None
    assert taken_at_utc(info) == dt.datetime(2022, 6, 15, 12, 0, tzinfo=dt.timezone.utc)


def test_unknown_when_nothing_dates_the_file():
    assert taken_at_utc(ExifInfo()) is None


def test_parse_helpers():
    assert _parse_exif_date("2024:05:12 15:30:45\x00") == dt.datetime(2024, 5, 12, 15, 30, 45)
    assert _parse_exif_date("0000:00:00 00:00:00") is None
    assert _parse_exif_date("1970:01:01 00:00:00") is None
    assert _parse_exif_date("2999:01:01 00:00:00") is None
    assert _parse_offset("+05:30") == 330
    assert _parse_offset("-0800") == -480
    assert _parse_offset("Z") == 0
    assert _parse_offset("bogus") is None


def test_decompression_bomb_guard_is_explicit():
    assert Image.MAX_IMAGE_PIXELS is not None
    assert Image.MAX_IMAGE_PIXELS >= exif_reader.MAX_IMAGE_PIXELS_CAP

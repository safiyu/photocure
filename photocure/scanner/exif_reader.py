"""EXIF metadata extraction with a deterministic date fallback cascade.

Cascade order (first success wins):
    1. EXIF ``DateTimeOriginal``
    2. EXIF ``DateTimeDigitized`` (exiftool "CreateDate")
    3. EXIF ``DateTime`` (last modified, weakest EXIF source)
    4. XMP sidecar ``DateTimeOriginal`` (``xmp:dtoriginal``)
    5. Strict filename regex (IMG_ / PXL_ / Screenshot_ / DSC_ patterns)
    6. Filesystem ``stat.st_mtime``
    7. ``unknown`` -> routed to ``Unknown_Date/`` (never the 1970 epoch)

Time bases: ``ExifInfo.taken_at`` is always a naive wall-clock value.
``tz_offset_minutes`` is set when the zone is known (EXIF ``OffsetTime*``, an
XMP offset, or the machine-local offset for mtime). :func:`taken_at_utc`
returns a true UTC instant in that case, and otherwise the naive wall clock
tagged UTC so the calendar date is preserved.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

from .xmp import read_xmp

# Explicit decompression-bomb cap (pixels); Pillow errors out above twice this.
MAX_IMAGE_PIXELS_CAP = 500_000_000

from .. import heif as _heif  # noqa: F401  (registers the HEIF opener so Pillow can read HEIC EXIF)

try:
    from PIL import Image
    _cur = Image.MAX_IMAGE_PIXELS
    if _cur is None or _cur < MAX_IMAGE_PIXELS_CAP:
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS_CAP
except ImportError:  # pragma: no cover
    Image = None

try:
    import exifread
    import logging as _logging

    # exifread logs a warning for every PNG without EXIF, which floods the console.
    _logging.getLogger("exifread").setLevel(_logging.ERROR)
except ImportError:  # pragma: no cover
    exifread = None

try:
    import rawpy
except ImportError:  # pragma: no cover
    rawpy = None

# --------------------------------------------------------------------------
# Filename patterns (strict, per spec).
# --------------------------------------------------------------------------
_FILENAME_PATTERNS = [
    # IMG_20240512_153045 / IMG_20240512153045 / IMG_2024-05-12_15.30.45
    re.compile(r"(?i)IMG[_\-]?(\d{4})[._\-]?(\d{2})[._\-]?(\d{2})"
               r"(?:[_\-]?(\d{2})[._\-]?(\d{2})[._\-]?(\d{2}))?"),
    # PXL_20240512_153045567 / PXL_20240512T153045
    re.compile(r"(?i)PXL[_\-]?(20\d{2})(\d{2})(\d{2})"
               r"[T_\-]?(\d{2})[._\-]?(\d{2})[._\-]?(\d{2})"),
    # Screenshot_2024-05-12_15-30-45 / Screenshot 20240512-153045
    re.compile(r"(?i)Screenshot[_ \-](\d{4})[._\-]?(\d{2})[._\-]?(\d{2})"
               r"[._\-]?(\d{2})[._\-]?(\d{2})[._\-]?(\d{2})"),
    # DSC_20240512 / 2024-05-12_15-30-45 generic camera
    re.compile(r"(?i)DSC[_\-]?(\d{4})(\d{2})(\d{2})"
               r"(?:[_\-]?(\d{2})[._\-]?(\d{2})[._\-]?(\d{2}))?"),
]

MIN_VALID_DATE = dt.datetime(1980, 1, 1)  # pre-1980 treated as garbage/epoch
_MAX_FUTURE = dt.timedelta(days=366)  # dates further ahead are treated as garbage

_OFFSET_RE = re.compile(r"^\s*(Z|[+-]\d{2}:?\d{2})\s*$")


@dataclass
class ExifInfo:
    width: int = 0
    height: int = 0
    orientation: int = 1
    taken_at: dt.datetime | None = None
    taken_at_source: str = "unknown"
    tz_offset_minutes: int | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    camera_model: str = ""
    rating: int | None = None
    xmp_label: str = ""
    xmp_keywords: list = field(default_factory=list)
    error: str = field(default="")

    @property
    def has_date(self) -> bool:
        return self.taken_at is not None


def _plausible(d: dt.datetime) -> bool:
    return MIN_VALID_DATE <= d <= dt.datetime.now() + _MAX_FUTURE


def _parse_exif_date(s: str | None) -> dt.datetime | None:
    """Parse EXIF date strings such as ``2024:05:12 15:30:45``."""
    if not s:
        return None
    s = str(s).strip("\x00 \t\r\n")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H.%M.%S",
                "%Y-%m-%dT%H:%M:%S", "%Y:%m:%d", "%Y-%m-%d"):
        try:
            parsed = dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
        return parsed if _plausible(parsed) else None
    return None


def _parse_offset(s) -> int | None:
    """Parse an EXIF ``OffsetTime*`` value (``+02:00``) into minutes east of UTC."""
    if s is None:
        return None
    m = _OFFSET_RE.match(str(s).strip("\x00 "))
    if not m:
        return None
    tz = m.group(1)
    if tz == "Z":
        return 0
    digits = tz[1:].replace(":", "")
    minutes = int(digits[:2]) * 60 + int(digits[2:])
    if minutes > 14 * 60:
        return None
    return -minutes if tz[0] == "-" else minutes


def extract_from_filename(filename: str) -> dt.datetime | None:
    """Tier-3: strict filename regex extraction. Naive local time."""
    for pat in _FILENAME_PATTERNS:
        m = pat.search(filename)
        if not m:
            continue
        g = m.groups()
        try:
            year, month, day = int(g[0]), int(g[1]), int(g[2])
            hour = int(g[3]) if g[3] else 0
            minute = int(g[4]) if g[4] else 0
            second = int(g[5]) if g[5] else 0
            parsed = dt.datetime(year, month, day, hour, minute, second)
        except (ValueError, IndexError):
            continue
        if _plausible(parsed):
            return parsed
    return None


def taken_at_utc(info: ExifInfo) -> dt.datetime | None:
    """UTC-aware ``taken_at``: a true instant when the offset is known, else
    the naive wall clock tagged UTC (calendar date preserved)."""
    d = info.taken_at
    if d is None:
        return None
    if info.tz_offset_minutes is not None:
        return (d - dt.timedelta(minutes=info.tz_offset_minutes)).replace(
            tzinfo=dt.timezone.utc)
    return d.replace(tzinfo=dt.timezone.utc)


def _valid_gps(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _dms(v) -> float:
    deg, mn = float(v[0]), float(v[1])
    sec = float(v[2]) if len(v) > 2 else 0.0
    return deg + mn / 60.0 + sec / 3600.0


def _gps_from_exif(tags) -> tuple[float, float] | None:
    """Read GPS from exifread tags; returns (lat, lon) or None."""
    try:
        lat_ref = str(tags.get("GPS GPSLatitudeRef", "N"))
        lat = tuple(float(x) for x in tags["GPS GPSLatitude"].values)
        lon_ref = str(tags.get("GPS GPSLongitudeRef", "E"))
        lon = tuple(float(x) for x in tags["GPS GPSLongitude"].values)
        lat_v = _dms(lat) * (-1.0 if lat_ref.upper().startswith("S") else 1.0)
        lon_v = _dms(lon) * (-1.0 if lon_ref.upper().startswith("W") else 1.0)
    except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError):
        return None
    return (lat_v, lon_v) if _valid_gps(lat_v, lon_v) else None


def _gps_from_pil(gps_ifd) -> tuple[float, float] | None:
    """Read GPS from a Pillow GPS IFD (tags 1-4); returns (lat, lon) or None."""
    try:
        lat_ref = str(gps_ifd.get(1, "N"))
        lon_ref = str(gps_ifd.get(3, "E"))
        lat_v = _dms(gps_ifd[2]) * (-1.0 if lat_ref.upper().startswith("S") else 1.0)
        lon_v = _dms(gps_ifd[4]) * (-1.0 if lon_ref.upper().startswith("W") else 1.0)
    except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError):
        return None
    return (lat_v, lon_v) if _valid_gps(lat_v, lon_v) else None


# (Pillow tag, exifread key, source label, Pillow offset tag, exifread offset key)
_DATE_TAGS = (
    (36867, "EXIF DateTimeOriginal", "exif:dtoriginal", 36881, "EXIF OffsetTimeOriginal"),
    (36868, "EXIF DateTimeDigitized", "exif:create", 36882, "EXIF OffsetTimeDigitized"),
    (306, "Image DateTime", "exif:create", 36880, "EXIF OffsetTime"),
)


class ExifReader:
    """Extracts dimensions, orientation, camera, GPS and the cascaded date."""

    def read(self, path: Path | str) -> ExifInfo:
        path = Path(path)
        info = ExifInfo()
        try:
            st = path.stat()
        except OSError as e:
            info.error = f"stat failed: {e}"
            return info

        # ---- Pillow pass (JPEG/PNG/WebP/HEIC/TIFF and many RAW via plugins)
        exif_tags: dict = {}
        try:
            if Image is not None:
                with Image.open(path) as im:
                    # ``im.size`` parses only the header (no pixel decode), so
                    # this is cheap even for large RAW/JPEG frames. (Pillow 12
                    # removed ``ImageFile.load(exif_only=...)``; do not use it.)
                    info.width, info.height = im.size
                    exif = im.getexif() if hasattr(im, "getexif") else None
                    if exif:
                        exif_tags = self._normalize_pil_exif(exif)
                        info.orientation = int(exif.get(274, 1) or 1)
                        info.camera_model = str(exif_tags.get(272, "")).strip("\x00 ")
                        gps = exif_tags.get("gps")
                        if gps:
                            info.gps_lat, info.gps_lon = gps
        except Exception as e:  # corrupt / unsupported
            info.error = f"pillow: {e.__class__.__name__}: {e}"
        pillow_ok = bool(info.width)

        # ---- rawpy pass for camera RAW (CR2/NEF/ARW/DNG)
        if not info.width and rawpy is not None:
            try:
                with rawpy.imread(str(path)) as rp:
                    info.width = rp.width
                    info.height = rp.height
                    desc = rp.descr
                    if desc.camera_make or desc.camera_model:
                        info.camera_model = " ".join(
                            x for x in (desc.camera_make, desc.camera_model) if x
                        ).strip()
            except Exception as e:
                if not info.error:
                    info.error = f"rawpy: {e.__class__.__name__}: {e}"

        # ---- EXIF dates from Pillow's parsed tags (offset tags follow the date)
        any_offset = _parse_offset(exif_tags.get(36881))
        if any_offset is None:
            any_offset = _parse_offset(exif_tags.get(36880))
        for p_tag, _k, source, p_off, _ko in _DATE_TAGS:
            d = _parse_exif_date(exif_tags.get(p_tag))
            if d:
                info.taken_at, info.taken_at_source = d, source
                off = _parse_offset(exif_tags.get(p_off))
                info.tz_offset_minutes = off if off is not None else any_offset
                break

        # ---- exifread pass: RAW and other files Pillow could not date
        if exifread is not None and not info.taken_at:
            try:
                with open(path, "rb") as f:
                    raw_tags = exifread.process_file(f, details=False)
                for _p, key, source, _po, off_key in _DATE_TAGS:
                    tag = raw_tags.get(key)
                    d = _parse_exif_date(str(tag)) if tag else None
                    if d:
                        info.taken_at, info.taken_at_source = d, source
                        off = _parse_offset(str(raw_tags[off_key])) \
                            if raw_tags.get(off_key) else None
                        if off is None and raw_tags.get("EXIF OffsetTimeOriginal"):
                            off = _parse_offset(str(raw_tags["EXIF OffsetTimeOriginal"]))
                        info.tz_offset_minutes = off
                        break
                if info.gps_lat is None:
                    gps = _gps_from_exif(raw_tags)
                    if gps:
                        info.gps_lat, info.gps_lon = gps
                if not info.camera_model and raw_tags.get("Image Model"):
                    info.camera_model = str(raw_tags["Image Model"]).strip("\x00 ")
                if not pillow_ok and raw_tags.get("Image Orientation"):
                    try:
                        info.orientation = int(raw_tags["Image Orientation"].values[0])
                    except (ValueError, TypeError, IndexError, AttributeError):
                        pass
            except Exception as e:
                if not info.error:
                    info.error = f"exifread: {e.__class__.__name__}: {e}"

        # ---- XMP sidecar: rating always, date only when EXIF had none
        xmp = read_xmp(path)
        if xmp is not None:
            info.rating = xmp.rating
            info.xmp_label = xmp.label
            info.xmp_keywords = list(xmp.keywords)
            if not info.taken_at and xmp.taken_at and _plausible(xmp.taken_at):
                info.taken_at, info.taken_at_source = xmp.taken_at, "xmp:dtoriginal"
                info.tz_offset_minutes = xmp.tz_offset_minutes

        # ---- Tier: filename
        if not info.taken_at:
            d = extract_from_filename(path.name)
            if d:
                info.taken_at, info.taken_at_source = d, "filename"

        # ---- Tier: mtime (machine-local wall clock; offset recorded)
        if not info.taken_at:
            try:
                local = dt.datetime.fromtimestamp(st.st_mtime).astimezone()
                m = local.replace(tzinfo=None)
                if m >= MIN_VALID_DATE:
                    info.taken_at, info.taken_at_source = m, "mtime"
                    off = local.utcoffset()
                    info.tz_offset_minutes = (
                        int(off.total_seconds() // 60) if off is not None else None)
            except (OverflowError, OSError, ValueError):
                pass

        if not info.taken_at:
            info.taken_at_source = "unknown"
            info.tz_offset_minutes = None
        return info

    @staticmethod
    def _normalize_pil_exif(exif) -> dict:
        """Flatten a PIL Exif into {int-tag: value} plus a parsed ``"gps"`` pair."""
        out: dict = {}
        try:
            for k, v in exif.items():
                try:
                    out[int(k)] = v
                except (ValueError, TypeError):
                    continue
            # Exif IFD (0x8769) holds DateTimeOriginal (36867) & OffsetTime*.
            try:
                for k, v in exif.get_ifd(0x8769).items():
                    try:
                        out[int(k)] = v
                    except (ValueError, TypeError):
                        continue
            except Exception:
                pass
            # GPS IFD (0x8825) is parsed here so the exifread pass is optional.
            try:
                gps = _gps_from_pil(exif.get_ifd(0x8825))
                if gps:
                    out["gps"] = gps
            except Exception:
                pass
        except Exception:
            pass
        return out


def extract_exif(path: Path | str) -> ExifInfo:
    """Module-level convenience wrapper around :class:`ExifReader`."""
    return ExifReader().read(path)


def effective_date(info: ExifInfo) -> tuple[dt.datetime | None, str]:
    """Return ``(date, source)`` where source is one of the cascade tiers."""
    if info.taken_at is not None:
        return info.taken_at, info.taken_at_source
    return None, info.taken_at_source or "unknown"

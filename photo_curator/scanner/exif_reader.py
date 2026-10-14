"""EXIF metadata extraction with a deterministic 4-tier date fallback cascade.

Cascade order (first success wins):
    1. EXIF ``DateTimeOriginal``
    2. EXIF ``CreateDate``
    3. Strict filename regex (IMG_ / PXL_ / Screenshot_ / DSC_ patterns)
    4. Filesystem ``stat.st_mtime``
    5. ``unknown`` -> routed to ``Unknown_Date/`` (never the 1970 epoch)
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import exifread
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
    re.compile(r"(?i)PXL[_\-]?(?:20)?(\d{2})(\d{2})(\d{2})"
               r"[T_\-]?(\d{2})[._\-]?(\d{2})[._\-]?(\d{2})"),
    # Screenshot_2024-05-12_15-30-45 / Screenshot 20240512-153045
    re.compile(r"(?i)Screenshot[_ \-](\d{4})[._\-]?(\d{2})[._\-]?(\d{2})"
               r"[._\-]?(\d{2})[._\-]?(\d{2})[._\-]?(\d{2})"),
    # DSC_20240512 / 2024-05-12_15-30-45 generic camera
    re.compile(r"(?i)DSC[_\-]?(\d{4})(\d{2})(\d{2})"
               r"(?:[_\-]?(\d{2})[._\-]?(\d{2})[._\-]?(\d{2}))?"),
]

MIN_VALID_DATE = dt.datetime(1980, 1, 1)  # pre-1980 treated as garbage/epoch


@dataclass
class ExifInfo:
    width: int = 0
    height: int = 0
    orientation: int = 1
    taken_at: dt.datetime | None = None
    taken_at_source: str = "unknown"
    gps_lat: float | None = None
    gps_lon: float | None = None
    camera_model: str = ""
    error: str = field(default="")

    @property
    def has_date(self) -> bool:
        return self.taken_at is not None


def _parse_exif_date(s: str | None) -> dt.datetime | None:
    """Parse EXIF date strings such as ``2024:05:12 15:30:45``."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H.%M.%S",
                "%Y-%m-%dT%H:%M:%S", "%Y:%m:%d", "%Y-%m-%d"):
        try:
            parsed = dt.datetime.strptime(s, fmt)
            if parsed < MIN_VALID_DATE:
                return None
            return parsed
        except ValueError:
            continue
    return None


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
            if parsed < MIN_VALID_DATE:
                continue
            return parsed
        except (ValueError, IndexError):
            continue
    return None


def _gps_from_exif(tags) -> tuple[float, float] | None:
    """Read GPS from exifread tags; returns (lat, lon) or None."""
    try:
        lat_ref = str(tags.get("GPS GPSLatitudeRef", "N"))
        lat = tuple(float(x) for x in tags["GPS GPSLatitude"].values)
        lon_ref = str(tags.get("GPS GPSLongitudeRef", "E"))
        lon = tuple(float(x) for x in tags["GPS GPSLongitude"].values)

        def _dms(v):
            deg, mn, sec = v[0], v[1], v[2] if len(v) > 2 else 0.0
            return deg + mn / 60.0 + sec / 3600.0

        lat_v = _dms(lat) * (-1.0 if lat_ref.upper().startswith("S") else 1.0)
        lon_v = _dms(lon) * (-1.0 if lon_ref.upper().startswith("W") else 1.0)
        return lat_v, lon_v
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


class ExifReader:
    """Extracts dimensions, orientation, camera, GPS and the cascaded date."""

    def read(self, path: Path | str) -> ExifInfo:
        path = Path(path)
        info = ExifInfo()
        st = None
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
                        exif_tags = self._normalize_pil_exif(exif, im)
                        info.orientation = int(exif.get(274, 1) or 1)
                        info.camera_model = str(exif_tags.get("Model", ""))
        except Exception as e:  # corrupt / unsupported
            info.error = f"pillow: {e.__class__.__name__}: {e}"

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

        # ---- exifread pass for date/GPS (reads APP1 without full decode)
        if exifread is not None and not info.taken_at and not info.gps_lat:
            try:
                with open(path, "rb") as f:
                    raw_tags = exifread.process_file(f, details=False)
                dt_orig = raw_tags.get("Exif DateTimeOriginal")
                dt_create = raw_tags.get("Exif DateTime")
                dt_digit = raw_tags.get("Exif DigitizationTime")
                if dt_orig:
                    d = _parse_exif_date(str(dt_orig))
                    if d:
                        info.taken_at, info.taken_at_source = d, "exif:dtoriginal"
                if not info.taken_at and dt_digit:
                    d = _parse_exif_date(str(dt_digit))
                    if d:
                        info.taken_at, info.taken_at_source = d, "exif:create"
                if not info.taken_at and dt_create:
                    d = _parse_exif_date(str(dt_create))
                    if d:
                        info.taken_at, info.taken_at_source = d, "exif:create"
                if not info.taken_at and info.taken_at_source == "unknown":
                    # also check Pillow's exif values
                    pass
                if not info.gps_lat:
                    gps = _gps_from_exif(raw_tags)
                    if gps:
                        info.gps_lat, info.gps_lon = gps
                if not info.camera_model and raw_tags.get("Image Model"):
                    info.camera_model = str(raw_tags["Image Model"])
                if not info.orientation and raw_tags.get("Image Orientation"):
                    try:
                        info.orientation = int(str(raw_tags["Image Orientation"]))
                    except ValueError:
                        pass
            except Exception as e:
                if not info.error:
                    info.error = f"exifread: {e.__class__.__name__}: {e}"

        # ---- Pillow exif date fallback (PIL exposes EXIF directly)
        if not info.taken_at:
            for key, source in ((36867, "exif:dtoriginal"),   # DateTimeOriginal
                                (306, "exif:create"),          # DateTime (CreateDate)
                                (36868, "exif:create")):       # DigitizationTime
                v = exif_tags.get(key)
                if v:
                    d = _parse_exif_date(str(v))
                    if d:
                        info.taken_at, info.taken_at_source = d, source
                        break

        # ---- Tier 3: filename
        if not info.taken_at:
            d = extract_from_filename(path.name)
            if d:
                info.taken_at, info.taken_at_source = d, "filename"

        # ---- Tier 4: mtime
        if not info.taken_at and st is not None:
            m = dt.datetime.fromtimestamp(st.st_mtime)
            if m >= MIN_VALID_DATE:
                info.taken_at, info.taken_at_source = m, "mtime"

        if not info.taken_at:
            info.taken_at_source = "unknown"
        return info

    @staticmethod
    def _normalize_pil_exif(exif, im) -> dict:
        """Flatten a PIL Exif into {int-tag: value} including IFD pointers."""
        out: dict = {}
        try:
            for k, v in exif.items():
                try:
                    out[int(k)] = v
                except (ValueError, TypeError):
                    continue
            # Exif IFD (0x8769) holds DateTimeOriginal (36867) & DateTime (306).
            try:
                exif_ifd = exif.get_ifd(0x8769)
                for k, v in exif_ifd.items():
                    try:
                        out[int(k)] = v
                    except (ValueError, TypeError):
                        continue
            except Exception:
                pass
            # GPS IFD (0x8825)
            try:
                gps_ifd = exif.get_ifd(0x8825)
                for k, v in gps_ifd.items():
                    out[f"GPS {k}"] = v
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

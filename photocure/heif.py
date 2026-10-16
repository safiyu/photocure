"""Registers Pillow's HEIC/HEIF opener, once, for every module that opens images.

Import this module (not pillow_heif directly) before calling Image.open. Registration happens
at import, and Python imports a module only once.
"""
from __future__ import annotations

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # HEIC support becomes a no-op
    pillow_heif = None

__all__ = ["pillow_heif"]

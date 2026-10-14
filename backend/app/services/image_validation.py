"""Validation of untrusted uploaded image files.

Uploaded bytes are never executed or written using client-supplied paths. A
file is accepted only if (1) its extension is allowed, (2) its declared MIME
type (when provided) is an image type, (3) its magic bytes match a supported
format, (4) it fully decodes, and (5) its dimensions are within limits.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/x-ms-bmp"}
PIL_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not an acceptable image."""


@dataclass
class ValidatedImage:
    rgb: np.ndarray  # HxWx3 uint8, EXIF orientation applied
    sha256: str
    format: str
    original_filename: str | None
    raw_bytes: bytes

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


def _sniff_format(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data.startswith(b"BM"):
        return "BMP"
    return None


def validate_image_bytes(
    data: bytes,
    filename: str | None = None,
    content_type: str | None = None,
    *,
    max_bytes: int,
    min_side: int = 64,
    max_pixels: int = 40_000_000,
) -> ValidatedImage:
    if not data:
        raise ImageValidationError("Empty file.")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"File too large ({len(data) / 1e6:.1f} MB); limit is {max_bytes / 1e6:.1f} MB."
        )
    if filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise ImageValidationError(
                f"Unsupported file extension '{ext or '(none)'}'. Allowed: "
                + ", ".join(sorted(ALLOWED_EXTENSIONS))
            )
    if content_type and content_type not in ALLOWED_MIME and content_type != "application/octet-stream":
        raise ImageValidationError(f"Unsupported content type '{content_type}'.")

    sniffed = _sniff_format(data)
    if sniffed is None:
        raise ImageValidationError("File content is not a supported image format.")

    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format not in PIL_FORMATS or probe.format != sniffed:
                raise ImageValidationError("Image format does not match its content.")
            w, h = probe.size
            if w * h > max_pixels:
                raise ImageValidationError("Image has too many pixels.")
            probe.verify()
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            rgb = np.asarray(img, dtype=np.uint8).copy()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError(f"Image could not be decoded: {exc}") from exc

    if min(rgb.shape[:2]) < min_side:
        raise ImageValidationError(f"Image is too small; minimum side is {min_side}px.")

    return ValidatedImage(
        rgb=rgb,
        sha256=hashlib.sha256(data).hexdigest(),
        format=sniffed,
        original_filename=filename,
        raw_bytes=data,
    )

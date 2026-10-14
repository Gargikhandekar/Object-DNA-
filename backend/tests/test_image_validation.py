import io

import numpy as np
import pytest
from PIL import Image

from app.services.image_validation import ImageValidationError, validate_image_bytes
from tests.conftest import encode

MAX = 5 * 1024 * 1024


def rgb(h=100, w=120):
    return (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)


def test_accepts_valid_jpeg_and_png():
    for ext in (".jpg", ".png"):
        v = validate_image_bytes(encode(rgb(), ext), f"a{ext}", None, max_bytes=MAX)
        assert v.rgb.shape == (100, 120, 3)
        assert len(v.sha256) == 64


def test_rejects_bad_extension():
    with pytest.raises(ImageValidationError, match="extension"):
        validate_image_bytes(encode(rgb()), "evil.exe", None, max_bytes=MAX)


def test_rejects_non_image_content_with_image_extension():
    with pytest.raises(ImageValidationError):
        validate_image_bytes(b"#!/bin/sh\nrm -rf /\n" * 10, "x.jpg", "image/jpeg", max_bytes=MAX)


def test_rejects_mismatched_mime():
    with pytest.raises(ImageValidationError, match="content type"):
        validate_image_bytes(encode(rgb()), "a.jpg", "text/html", max_bytes=MAX)


def test_rejects_oversized_and_empty():
    with pytest.raises(ImageValidationError, match="too large"):
        validate_image_bytes(encode(rgb()), "a.jpg", None, max_bytes=100)
    with pytest.raises(ImageValidationError, match="Empty"):
        validate_image_bytes(b"", "a.jpg", None, max_bytes=MAX)


def test_rejects_truncated_image():
    data = encode(rgb(300, 300), ".png")
    with pytest.raises(ImageValidationError):
        validate_image_bytes(data[: len(data) // 2], "a.png", None, max_bytes=MAX)


def test_rejects_too_small():
    with pytest.raises(ImageValidationError, match="too small"):
        validate_image_bytes(encode(rgb(20, 20)), "a.jpg", None, max_bytes=MAX, min_side=64)


def test_applies_exif_orientation():
    img = Image.fromarray(rgb(80, 160))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 CW on display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    v = validate_image_bytes(buf.getvalue(), "a.jpg", None, max_bytes=MAX)
    assert v.rgb.shape[:2] == (160, 80)

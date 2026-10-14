"""Texture, colour and shape descriptors.

* LBP: multi-scale rotation-invariant uniform Local Binary Patterns
  (``riu2`` mapping, Ojala et al. 2002) with bilinear circular sampling,
  computed on the CLAHE-normalised grayscale image and restricted to the
  foreground mask when it is reliable.
* Gradient/edge statistics: Canny edge density and a gradient-orientation
  histogram weighted by magnitude.
* Colour: 2-D Hue x Saturation histogram. The value (brightness) channel is
  deliberately excluded to reduce sensitivity to lighting intensity.
* Shape: log-scaled Hu moments plus normalised geometric measurements of the
  largest foreground contour (only when a reliable mask exists).
"""

from __future__ import annotations

import cv2
import numpy as np

LBP_VERSION = "lbp-riu2-v1"
COLOR_VERSION = "hsv-hs-v1"
SHAPE_VERSION = "shape-hu-v1"
GRADIENT_VERSION = "grad-v1"

LBP_SCALES: tuple[tuple[int, float], ...] = ((8, 1.0), (16, 2.0), (24, 3.0))
COLOR_BINS = (18, 8)  # hue, saturation


def _lbp_riu2(gray: np.ndarray, P: int, R: float) -> np.ndarray:
    """Return riu2 LBP codes in [0, P+1] for every interior pixel."""
    img = gray.astype(np.float32)
    h, w = img.shape
    pad = int(np.ceil(R)) + 1
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    center = img - 1e-3
    ones = np.zeros((h, w), np.uint8)
    transitions = np.zeros((h, w), np.uint8)
    first = prev = None
    for p in range(P):
        angle = 2.0 * np.pi * p / P
        dy, dx = -R * np.sin(angle), R * np.cos(angle)
        y0, x0 = pad + dy, pad + dx
        fy, fx = int(np.floor(y0)), int(np.floor(x0))
        ty, tx = y0 - fy, x0 - fx
        if ty < 1e-6 and tx < 1e-6:
            val = padded[fy:fy + h, fx:fx + w]
        else:
            val = cv2.addWeighted(
                cv2.addWeighted(padded[fy:fy + h, fx:fx + w], (1 - ty) * (1 - tx),
                                padded[fy:fy + h, fx + 1:fx + 1 + w], (1 - ty) * tx, 0),
                1.0,
                cv2.addWeighted(padded[fy + 1:fy + 1 + h, fx:fx + w], ty * (1 - tx),
                                padded[fy + 1:fy + 1 + h, fx + 1:fx + 1 + w], ty * tx, 0),
                1.0, 0)
        bit = (val >= center).view(np.uint8)
        ones += bit
        if prev is None:
            first = bit
        else:
            transitions += prev ^ bit
        prev = bit
    transitions += prev ^ first
    return np.where(transitions <= 2, ones, P + 1).astype(np.int32)


def lbp_histogram(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Concatenated, per-scale L1-normalised riu2 histograms."""
    parts = []
    sel = None if mask is None else mask > 0
    for P, R in LBP_SCALES:
        codes = _lbp_riu2(gray, P, R)
        values = codes[sel] if sel is not None else codes.ravel()
        hist = np.bincount(values, minlength=P + 2).astype(np.float32)
        hist /= max(hist.sum(), 1.0)
        parts.append(hist)
    return np.concatenate(parts)


def gradient_features(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """[edge_density, mean_magnitude/255, 16-bin orientation histogram]."""
    edges = cv2.Canny(gray, 60, 150)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy)
    sel = np.ones(gray.shape, bool) if mask is None else mask > 0
    edge_density = float((edges[sel] > 0).mean()) if sel.any() else 0.0
    m = mag[sel]
    a = (ang[sel] % np.pi) / np.pi  # orientation (unsigned)
    hist, _ = np.histogram(a, bins=16, range=(0.0, 1.0), weights=m)
    hist = hist.astype(np.float32)
    hist /= max(hist.sum(), 1e-6)
    return np.concatenate([[edge_density, float(m.mean() / 255.0) if m.size else 0.0], hist]).astype(np.float32)


def color_histogram(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, list(COLOR_BINS), [0, 180, 0, 256])
    hist = hist.astype(np.float32).ravel()
    hist /= max(hist.sum(), 1e-6)
    return hist


def shape_features(mask: np.ndarray | None, reliable: bool) -> np.ndarray | None:
    """[7 log-Hu moments, aspect_ratio, extent, solidity, circularity] or None."""
    if mask is None or not reliable:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 100:
        return None
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    aspect = min(rw, rh) / max(rw, rh, 1e-6)
    x, y, w, h = cv2.boundingRect(c)
    extent = area / max(w * h, 1)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    solidity = area / max(hull_area, 1e-6)
    perim = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / max(perim * perim, 1e-6)
    return np.concatenate([log_hu, [aspect, extent, solidity, circularity]]).astype(np.float32)


# --------------------------------------------------------------------------
# Raw similarity functions (each documented range noted).
# --------------------------------------------------------------------------

def histogram_intersection(a: np.ndarray, b: np.ndarray, groups: int = 1) -> float:
    """Mean histogram intersection over ``groups`` equally-normalised blocks; in [0, 1]."""
    return float(np.minimum(a, b).sum() / groups)


def lbp_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Average per-scale histogram intersection in [0, 1]."""
    return histogram_intersection(a, b, groups=len(LBP_SCALES))


def is_achromatic(color_hist: np.ndarray, threshold: float = 0.9) -> bool:
    """True if >= ``threshold`` of the H-S histogram mass has saturation < 32 (greyscale-like image)."""
    h = color_hist.reshape(COLOR_BINS)
    return float(h[:, 0].sum()) >= threshold


def bhattacharyya_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 - Bhattacharyya distance of two L1-normalised histograms; in [0, 1]."""
    bc = float(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None)).sum())
    return float(1.0 - np.sqrt(max(0.0, 1.0 - min(bc, 1.0))))


def gradient_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a[2:], b[2:]).sum())


def shape_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """exp(-mean |difference|) over log-Hu and geometric terms; in (0, 1]. None if unavailable."""
    if a is None or b is None:
        return None
    hu_d = float(np.abs(a[:7] - b[:7]).mean()) / 3.0
    geo_d = float(np.abs(a[7:] - b[7:]).mean())
    return float(np.exp(-(hu_d + 2.0 * geo_d)))

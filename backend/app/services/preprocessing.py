"""Image normalisation, optional foreground segmentation and quality metrics.

Every image (reference or query) goes through the same deterministic pipeline
so that features are comparable:

1. RGB uint8 input (EXIF orientation already applied by validation).
2. Resize so the longest side equals ``max_side`` (area interpolation).
3. Optional GrabCut foreground mask initialised from a centred rectangle.
   The mask is only used if it passes reliability checks; otherwise every
   pixel is used and ``mask_reliable`` is False.
4. Optional crop to the foreground bounding box.
5. Grayscale + CLAHE (contrast-limited adaptive histogram equalisation) as the
   input for ORB and LBP, which reduces the effect of global illumination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

PREPROCESSING_VERSION = "prep-v2"


@dataclass
class PreprocessConfig:
    max_side: int = 640
    use_foreground_mask: bool = True
    crop_to_foreground: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    def as_dict(self) -> dict:
        return {
            "version": PREPROCESSING_VERSION,
            "max_side": self.max_side,
            "use_foreground_mask": self.use_foreground_mask,
            "crop_to_foreground": self.crop_to_foreground,
            "clahe_clip": self.clahe_clip,
            "clahe_grid": self.clahe_grid,
        }


@dataclass
class PreprocessedImage:
    rgb: np.ndarray  # resized (and optionally cropped) RGB uint8
    gray: np.ndarray  # CLAHE-normalised grayscale uint8
    mask: np.ndarray | None  # uint8 {0,255} foreground mask or None
    mask_reliable: bool
    scale: float  # processed / original
    crop_box: tuple[int, int, int, int] | None  # x, y, w, h in resized coords
    quality: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def feature_mask(self) -> np.ndarray | None:
        return self.mask if self.mask_reliable else None

    @property
    def keypoint_mask(self) -> np.ndarray | None:
        """Foreground mask widened by ~25 % of the image size.

        GrabCut often trims parts of objects that fill the frame; the widened mask keeps their keypoints
        while still excluding distant background clutter.
        """
        m = self.feature_mask
        if m is None:
            return None
        k = max(1, int(0.25 * max(m.shape)))
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))


def resize_max_side(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    scale = max_side / float(max(h, w))
    if abs(scale - 1.0) < 1e-6:
        return rgb.copy(), 1.0
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    out = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp)
    return out, scale


def foreground_mask(rgb: np.ndarray, work_side: int = 256, iterations: int = 3) -> tuple[np.ndarray, bool]:
    """Estimate a foreground mask with GrabCut. Returns (mask, reliable)."""
    h, w = rgb.shape[:2]
    small, s = resize_max_side(rgb, work_side)
    sh, sw = small.shape[:2]
    mx, my = max(2, int(sw * 0.06)), max(2, int(sh * 0.06))
    rect = (mx, my, sw - 2 * mx, sh - 2 * my)
    gc_mask = np.zeros((sh, sw), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.setRNGSeed(0)
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc_mask, rect, bgd, fgd,
                    iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.full((h, w), 255, np.uint8), False
    fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # Keep the largest connected component.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    reliable = False
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        total_fg = float(areas.sum())
        fg = np.where(labels == largest, 255, 0).astype(np.uint8)
        frac = areas.max() / float(sh * sw)
        dominance = areas.max() / max(total_fg, 1.0)
        reliable = 0.05 <= frac <= 0.92 and dominance >= 0.7
        # If the segment runs into the initialisation rectangle on two or more sides, the object
        # most likely fills the frame and GrabCut has cut part of it away: do not trust the mask.
        bx, by, bw, bh = stats[largest, :4]
        touches = sum((bx <= mx + 2, by <= my + 2, bx + bw >= sw - mx - 2, by + bh >= sh - my - 2))
        if touches >= 2:
            reliable = False
    mask = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
    if not reliable:
        mask = np.full((h, w), 255, np.uint8)
    return mask, reliable


def image_quality(rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    warnings = []
    if lap_var < 40:
        warnings.append("Image appears blurry (low Laplacian variance).")
    if brightness < 40:
        warnings.append("Image is very dark.")
    if brightness > 220:
        warnings.append("Image is very bright / overexposed.")
    if contrast < 20:
        warnings.append("Image has low contrast.")
    return {
        "sharpness_laplacian_var": round(lap_var, 2),
        "brightness_mean": round(brightness, 2),
        "contrast_std": round(contrast, 2),
        "warnings": warnings,
    }


def preprocess(rgb: np.ndarray, config: PreprocessConfig | None = None) -> PreprocessedImage:
    config = config or PreprocessConfig()
    resized, scale = resize_max_side(rgb, config.max_side)
    mask, reliable = (None, False)
    if config.use_foreground_mask or config.crop_to_foreground:
        mask, reliable = foreground_mask(resized)

    crop_box = None
    if config.crop_to_foreground and reliable and mask is not None:
        x, y, w, h = cv2.boundingRect(mask)
        pad = int(0.05 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(resized.shape[1], x + w + pad), min(resized.shape[0], y + h + pad)
        crop_box = (x0, y0, x1 - x0, y1 - y0)
        resized = resized[y0:y1, x0:x1].copy()
        mask = mask[y0:y1, x0:x1].copy()

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=config.clahe_clip, tileGridSize=(config.clahe_grid, config.clahe_grid))
    gray = clahe.apply(gray)

    quality = image_quality(resized)
    quality["foreground_mask_reliable"] = bool(reliable)
    if mask is not None and reliable:
        quality["foreground_fraction"] = round(float((mask > 0).mean()), 4)

    return PreprocessedImage(
        rgb=resized,
        gray=gray,
        mask=mask if config.use_foreground_mask else None,
        mask_reliable=bool(reliable and config.use_foreground_mask),
        scale=scale,
        crop_box=crop_box,
        quality=quality,
        config=config.as_dict(),
    )


def dhash(rgb: np.ndarray, size: int = 8) -> int:
    """Difference hash, used to detect near-duplicate images."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))

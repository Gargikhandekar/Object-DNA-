"""Object DNA generation: runs every extractor on a preprocessed image."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.services import texture_features as tf
from app.services.embedding_extractor import EmbeddingExtractor, EmbeddingUnavailable
from app.services.orb_matcher import ORB_VERSION, OrbFeatures, extract_orb
from app.services.preprocessing import PreprocessConfig, PreprocessedImage, preprocess

FINGERPRINT_SCHEMA = "odna-fingerprint-v1"


@dataclass
class FeatureConfig:
    orb_nfeatures: int = 1500

    def as_dict(self) -> dict:
        return {"orb_nfeatures": self.orb_nfeatures}


@dataclass
class FeatureSet:
    orb: OrbFeatures
    lbp: np.ndarray
    gradient: np.ndarray
    color: np.ndarray
    shape: np.ndarray | None
    embedding: np.ndarray | None = None
    embedding_backbone: str | None = None
    embedding_version: str | None = None
    meta: dict = field(default_factory=dict)

    def extractor_versions(self) -> dict[str, str]:
        out = {
            "orb": ORB_VERSION,
            "lbp": tf.LBP_VERSION,
            "gradient": tf.GRADIENT_VERSION,
            "color": tf.COLOR_VERSION,
        }
        if self.shape is not None:
            out["shape"] = tf.SHAPE_VERSION
        if self.embedding is not None and self.embedding_backbone:
            out[f"embedding:{self.embedding_backbone}"] = self.embedding_version or ""
        return out

    # ----------------------------------------------------------- persistence
    def save(self, directory: Path) -> dict[str, Path]:
        """Write artifacts; return {extractor: path}."""
        directory.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        orb_path = directory / "orb.npz"
        np.savez_compressed(orb_path, keypoints=self.orb.keypoints, descriptors=self.orb.descriptors,
                            schema=FINGERPRINT_SCHEMA, version=ORB_VERSION)
        paths["orb"] = orb_path
        global_path = directory / "global.npz"
        arrays = {"lbp": self.lbp, "gradient": self.gradient, "color": self.color,
                  "schema": FINGERPRINT_SCHEMA}
        if self.shape is not None:
            arrays["shape"] = self.shape
        np.savez_compressed(global_path, **arrays)
        for key in ("lbp", "gradient", "color"):
            paths[key] = global_path
        if self.shape is not None:
            paths["shape"] = global_path
        if self.embedding is not None and self.embedding_backbone:
            emb_path = directory / f"embedding_{self.embedding_backbone}.npz"
            np.savez_compressed(emb_path, embedding=self.embedding.astype(np.float32),
                                version=self.embedding_version or "", schema=FINGERPRINT_SCHEMA)
            paths[f"embedding:{self.embedding_backbone}"] = emb_path
        return paths

    @staticmethod
    def load(directory: Path, backbone: str | None = None) -> FeatureSet:
        with np.load(directory / "orb.npz") as z:
            orb = OrbFeatures(z["keypoints"].astype(np.float32), z["descriptors"].astype(np.uint8))
        with np.load(directory / "global.npz") as z:
            lbp, grad, color = z["lbp"], z["gradient"], z["color"]
            shape = z["shape"] if "shape" in z.files else None
        emb = emb_version = None
        if backbone and backbone != "none":
            p = directory / f"embedding_{backbone}.npz"
            if p.exists():
                with np.load(p) as z:
                    emb = z["embedding"].astype(np.float32)
                    emb_version = str(z["version"])
        return FeatureSet(orb=orb, lbp=lbp, gradient=grad, color=color, shape=shape,
                          embedding=emb, embedding_backbone=backbone if emb is not None else None,
                          embedding_version=emb_version)


def extract_features(
    prep: PreprocessedImage,
    config: FeatureConfig | None = None,
    embedder: EmbeddingExtractor | None = None,
    timings: dict | None = None,
) -> FeatureSet:
    """Run every extractor. If ``timings`` is given, per-extractor ms are added to it."""
    config = config or FeatureConfig()
    mask = prep.feature_mask
    t = time.perf_counter()
    orb = extract_orb(prep.gray, prep.keypoint_mask, config.orb_nfeatures)
    t_orb = time.perf_counter()
    lbp = tf.lbp_histogram(prep.gray, mask)
    grad = tf.gradient_features(prep.gray, mask)
    t_tex = time.perf_counter()
    color = tf.color_histogram(prep.rgb, mask)
    t_col = time.perf_counter()
    shape = tf.shape_features(prep.mask, prep.mask_reliable)

    emb = None
    emb_backbone = emb_version = None
    emb_message = None
    if embedder is not None and embedder.available:
        try:
            emb = embedder.embed_one(_embedding_input(prep))
            emb_backbone, emb_version = embedder.backbone, embedder.version
        except EmbeddingUnavailable as exc:
            emb_message = str(exc)
    elif embedder is not None:
        emb_message = embedder.status().get("message")
    t_emb = time.perf_counter()
    if timings is not None:
        timings["orb"] = (t_orb - t) * 1000
        timings["texture"] = (t_tex - t_orb) * 1000
        timings["color"] = (t_col - t_tex) * 1000
        timings["deep"] = (t_emb - t_col) * 1000

    meta = {
        "keypoints": orb.count,
        "mask_reliable": prep.mask_reliable,
        "embedding_message": emb_message,
        "feature_config": config.as_dict(),
        "preprocessing": prep.config,
    }
    return FeatureSet(orb=orb, lbp=lbp, gradient=grad, color=color, shape=shape, embedding=emb,
                      embedding_backbone=emb_backbone, embedding_version=emb_version, meta=meta)


def _embedding_input(prep: PreprocessedImage) -> np.ndarray:
    """Crop to the foreground bounding box (if reliable) before embedding."""
    import cv2

    if prep.mask_reliable and prep.mask is not None and prep.crop_box is None:
        x, y, w, h = cv2.boundingRect(prep.mask)
        pad = int(0.08 * max(w, h))
        H, W = prep.rgb.shape[:2]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        if (x1 - x0) >= 32 and (y1 - y0) >= 32:
            return prep.rgb[y0:y1, x0:x1]
    return prep.rgb


def process_image(rgb: np.ndarray, prep_config: PreprocessConfig, feat_config: FeatureConfig,
                  embedder: EmbeddingExtractor | None) -> tuple[PreprocessedImage, FeatureSet]:
    prep = preprocess(rgb, prep_config)
    return prep, extract_features(prep, feat_config, embedder)

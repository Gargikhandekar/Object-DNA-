"""Deep global embeddings from pretrained image encoders.

Weights are NEVER downloaded implicitly. They must be fetched once with
``python scripts/download_weights.py`` (see README). If they are missing, the
extractor reports itself as unavailable and the rest of the pipeline keeps
working with the classical features only (fusion re-normalises weights over the
available components).

Supported backbones (registry below):

* ``resnet50``  - torchvision ResNet-50 (ImageNet-1k, IMAGENET1K_V2 weights);
  global-average-pooled 2048-d feature.
* ``dinov2_vits14`` - DINOv2 ViT-S/14 (self-supervised); 384-d CLS token.
  Needs the hub repository source and weights placed locally by the download
  script.

All embeddings are L2-normalised, so cosine similarity is a dot product.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    weights_file: str
    dim: int
    input_size: int
    version: str
    description: str


BACKBONES: dict[str, BackboneSpec] = {
    "resnet50": BackboneSpec(
        name="resnet50",
        weights_file="resnet50-11ad3fa6.pth",
        dim=2048,
        input_size=224,
        version="torchvision-resnet50-imagenet1k-v2",
        description="ResNet-50, ImageNet-1k supervised (torchvision IMAGENET1K_V2), GAP 2048-d",
    ),
    "dinov2_vits14": BackboneSpec(
        name="dinov2_vits14",
        weights_file="dinov2_vits14_pretrain.pth",
        dim=384,
        input_size=224,
        version="dinov2-vits14-pretrain",
        description="DINOv2 ViT-S/14 self-supervised, CLS token 384-d",
    ),
}


class EmbeddingUnavailable(RuntimeError):
    pass


class EmbeddingExtractor:
    """Lazily-initialised, thread-safe embedding extractor."""

    def __init__(self, backbone: str, weights_dir: Path, device: str = "cpu",
                 allow_random_init: bool = False):
        self.backbone = backbone
        self.weights_dir = Path(weights_dir)
        self.device = device
        self.allow_random_init = allow_random_init
        self._model = None
        self._lock = threading.Lock()
        self._error: str | None = None
        self._random_init = False

    # -- status ---------------------------------------------------------
    @property
    def spec(self) -> BackboneSpec | None:
        return BACKBONES.get(self.backbone)

    @property
    def version(self) -> str:
        spec = self.spec
        base = spec.version if spec else self.backbone
        return base + ("-RANDOM-INIT-TEST-ONLY" if self._random_init else "")

    @property
    def extractor_name(self) -> str:
        return f"embedding:{self.backbone}"

    def weights_path(self) -> Path | None:
        spec = self.spec
        return self.weights_dir / spec.weights_file if spec else None

    def status(self) -> dict:
        spec = self.spec
        if self.backbone == "none":
            return {"available": False, "loaded": False, "backbone": "none",
                    "message": "Deep embeddings disabled (ODNA_EMBEDDING_BACKBONE=none)."}
        if spec is None:
            return {"available": False, "loaded": False, "backbone": self.backbone,
                    "message": f"Unknown backbone '{self.backbone}'."}
        wp = self.weights_path()
        present = wp is not None and wp.exists()
        if self.backbone == "dinov2_vits14":
            present = present and self._dinov2_repo().exists()
        msg = None
        if self._error:
            msg = self._error
        elif not present and not self.allow_random_init:
            msg = (f"Weights for '{self.backbone}' not found in {self.weights_dir}. "
                   f"Run: python scripts/download_weights.py --backbone {self.backbone}")
        return {
            "available": bool((present or self.allow_random_init) and not self._error),
            "loaded": self._model is not None,
            "backbone": self.backbone,
            "version": self.version,
            "dimension": spec.dim,
            "description": spec.description,
            "weights_path": str(wp) if wp else None,
            "message": msg,
        }

    @property
    def available(self) -> bool:
        return bool(self.status()["available"])

    def _dinov2_repo(self) -> Path:
        return self.weights_dir / "dinov2_repo"

    # -- loading --------------------------------------------------------
    def _build(self):
        import torch
        import torchvision

        spec = self.spec
        if spec is None:
            raise EmbeddingUnavailable(f"Unknown backbone '{self.backbone}'.")
        wp = self.weights_path()
        have_weights = wp is not None and wp.exists()
        if self.backbone == "resnet50":
            model = torchvision.models.resnet50(weights=None)
            if have_weights:
                state = torch.load(wp, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            elif self.allow_random_init:
                self._random_init = True
            else:
                raise EmbeddingUnavailable(self.status()["message"])
            model.fc = torch.nn.Identity()
        elif self.backbone == "dinov2_vits14":
            repo = self._dinov2_repo()
            if not (have_weights and repo.exists()):
                raise EmbeddingUnavailable(self.status()["message"])
            model = torch.hub.load(str(repo), "dinov2_vits14", source="local", pretrained=False)
            model.load_state_dict(torch.load(wp, map_location="cpu", weights_only=True))
        else:  # pragma: no cover
            raise EmbeddingUnavailable(f"Unsupported backbone '{self.backbone}'.")
        model.eval().to(self.device)
        return model

    def load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    self._model = self._build()
                    logger.info("Loaded embedding backbone %s", self.version)
                except EmbeddingUnavailable:
                    raise
                except Exception as exc:  # pragma: no cover - environment specific
                    self._error = f"Failed to load '{self.backbone}': {exc}"
                    raise EmbeddingUnavailable(self._error) from exc
        return self._model

    # -- inference ------------------------------------------------------
    def _to_tensor(self, rgb_images: list[np.ndarray]):
        import cv2
        import torch

        size = self.spec.input_size if self.spec else 224
        batch = []
        for rgb in rgb_images:
            h, w = rgb.shape[:2]
            # Resize shorter side to `size`, then centre-crop (standard eval transform).
            s = size / min(h, w)
            resized = cv2.resize(rgb, (max(size, round(w * s)), max(size, round(h * s))),
                                 interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            rh, rw = resized.shape[:2]
            y0, x0 = (rh - size) // 2, (rw - size) // 2
            crop = resized[y0:y0 + size, x0:x0 + size].astype(np.float32) / 255.0
            crop = (crop - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
            batch.append(torch.from_numpy(crop.transpose(2, 0, 1)))
        return torch.stack(batch).to(self.device)

    def embed(self, rgb_images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        import torch

        model = self.load()
        outs = []
        with torch.inference_mode():
            for i in range(0, len(rgb_images), batch_size):
                feats = model(self._to_tensor(rgb_images[i:i + batch_size]))
                outs.append(feats.float().cpu().numpy())
        emb = np.concatenate(outs, axis=0) if outs else np.zeros((0, self.spec.dim), np.float32)
        return l2_normalize(emb)

    def embed_one(self, rgb: np.ndarray) -> np.ndarray:
        return self.embed([rgb])[0]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


_extractors: dict[tuple, EmbeddingExtractor] = {}
_extractors_lock = threading.Lock()


def get_extractor(backbone: str, weights_dir: Path, device: str = "cpu") -> EmbeddingExtractor:
    """Process-wide cached extractor so model initialisation happens once."""
    key = (backbone, str(weights_dir), device)
    with _extractors_lock:
        if key not in _extractors:
            _extractors[key] = EmbeddingExtractor(backbone, weights_dir, device)
        return _extractors[key]

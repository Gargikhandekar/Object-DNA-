"""Process-wide runtime objects (pipeline configuration, store, embedder)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.services.embedding_extractor import EmbeddingExtractor, get_extractor
from app.services.feature_extraction import FeatureConfig
from app.services.fingerprint_store import FingerprintStore
from app.services.preprocessing import PreprocessConfig
from app.services.similarity import FusionConfig


@dataclass
class Runtime:
    settings: Settings
    prep_config: PreprocessConfig
    feat_config: FeatureConfig
    embedder: EmbeddingExtractor | None
    store: FingerprintStore
    fusion: FusionConfig

    def reload_fusion(self) -> FusionConfig:
        self.fusion = load_fusion(self.settings)
        return self.fusion


def load_fusion(settings: Settings) -> FusionConfig:
    cfg = FusionConfig.load(settings.active_fusion_path)
    if cfg is None:
        cfg = FusionConfig(accept_threshold=settings.accept_threshold,
                           uncertain_threshold=settings.uncertain_threshold,
                           margin_threshold=settings.margin_threshold)
    return cfg


_runtime: Runtime | None = None
_lock = threading.Lock()


def build_runtime(settings: Settings, embedder: EmbeddingExtractor | None = None) -> Runtime:
    settings.ensure_dirs()
    prep = PreprocessConfig(max_side=settings.processing_max_side,
                            use_foreground_mask=settings.use_foreground_mask,
                            crop_to_foreground=settings.crop_to_foreground)
    feat = FeatureConfig(orb_nfeatures=settings.orb_nfeatures)
    if embedder is None and settings.embedding_backbone != "none":
        embedder = get_extractor(settings.embedding_backbone, settings.resolved_weights_dir, settings.device)
    store = FingerprintStore(settings.resolved_storage_dir, prep, feat, embedder)
    return Runtime(settings, prep, feat, embedder, store, load_fusion(settings))


def get_runtime() -> Runtime:
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = build_runtime(get_settings())
        return _runtime


def set_runtime(rt: Runtime | None) -> None:
    global _runtime
    with _lock:
        _runtime = rt

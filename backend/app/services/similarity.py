"""Pairwise evidence between two fingerprints and transparent score fusion.

Raw component scores
--------------------
=============  ==========================================  ==============
component      raw value                                   raw range
=============  ==========================================  ==============
deep           cosine similarity of L2-normalised embeds   [-1, 1]
geometric      # RANSAC-consistent ORB matches (inliers)   [0, inf)
texture        mean multi-scale LBP histogram intersection  [0, 1]
color          1 - Bhattacharyya distance (H-S histogram)  [0, 1]
shape          exp(-weighted mean |log-Hu / geometry| diff) (0, 1]
=============  ==========================================  ==============

Normalisation
-------------
Every component is mapped to [0, 1] with a clipped linear ramp
``s = clip((raw - lo) / (hi - lo), 0, 1)``. The default ``lo``/``hi`` values
below are hand-set heuristics; the Research Lab can replace them with values
fitted on a *validation* split (``lo`` = 5th percentile of different-instance
pairs, ``hi`` = 95th percentile of same-instance pairs).

Fusion
------
``score = sum_i w_i * s_i / sum_i w_i`` over the components that are
*available* for this pair (e.g. the deep embedding is missing if weights are
not installed; shape is missing without a reliable mask on both sides).
Missing evidence is therefore excluded, not treated as zero. Each
component's contribution ``w_i * s_i / sum_i w_i`` sums exactly to the fused
score, which makes the evidence breakdown additive and auditable.

The fused value is a *similarity score*, not a calibrated probability.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.services import texture_features as tf
from app.services.feature_extraction import FeatureSet
from app.services.orb_matcher import OrbMatchResult, match_orb

COMPONENTS = ("deep", "geometric", "texture", "color", "shape")


@dataclass
class PairEvidence:
    raw: dict[str, float | None]
    orb: OrbMatchResult | None = None
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"raw": {k: (None if v is None else round(float(v), 5)) for k, v in self.raw.items()},
             "extras": {k: round(float(v), 5) for k, v in self.extras.items()}}
        if self.orb is not None:
            d["orb"] = self.orb.summary()
        return d


def global_evidence(q: FeatureSet, r: FeatureSet) -> dict[str, float | None]:
    """Cheap, vectorisable components (everything except ORB)."""
    deep = None
    if q.embedding is not None and r.embedding is not None and q.embedding_backbone == r.embedding_backbone:
        deep = float(np.dot(q.embedding, r.embedding))
    return {
        "deep": deep,
        "texture": tf.lbp_similarity(q.lbp, r.lbp),
        "color": tf.bhattacharyya_similarity(q.color, r.color),
        "shape": tf.shape_similarity(q.shape, r.shape),
    }


def pair_evidence(q: FeatureSet, r: FeatureSet, *, ratio: float = 0.75, reproj: float = 6.0,
                  keep_matches: bool = False, with_orb: bool = True) -> PairEvidence:
    raw = global_evidence(q, r)
    orb = None
    if with_orb:
        orb = match_orb(q.orb, r.orb, ratio=ratio, reproj_threshold=reproj, keep_matches=keep_matches)
        raw["geometric"] = float(orb.inliers)
    else:
        raw["geometric"] = None
    extras = {"gradient_orientation": tf.gradient_similarity(q.gradient, r.gradient),
              "edge_density_delta": float(q.gradient[0] - r.gradient[0])}
    return PairEvidence(raw=raw, orb=orb, extras=extras)


# ---------------------------------------------------------------------------
# Fusion configuration
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {"deep": 0.35, "geometric": 0.35, "texture": 0.15, "color": 0.15, "shape": 0.0}
DEFAULT_RANGES = {
    "deep": (0.5, 0.95),
    "geometric": (8.0, 150.0),
    "texture": (0.80, 0.98),
    "color": (0.40, 0.90),
    "shape": (0.30, 0.95),
}


@dataclass
class FusionConfig:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    ranges: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_RANGES))
    accept_threshold: float = 0.55
    uncertain_threshold: float = 0.35
    margin_threshold: float = 0.05
    source: str = "default-heuristic"
    notes: str = "Hand-set defaults; not tuned on data."

    def normalize(self, component: str, raw: float | None) -> float | None:
        if raw is None:
            return None
        lo, hi = self.ranges[component]
        return float(np.clip((raw - lo) / max(hi - lo, 1e-9), 0.0, 1.0))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ranges"] = {k: list(v) for k, v in self.ranges.items()}
        return d

    @staticmethod
    def from_dict(d: dict) -> FusionConfig:
        cfg = FusionConfig()
        cfg.weights.update({k: float(v) for k, v in d.get("weights", {}).items() if k in COMPONENTS})
        cfg.ranges.update({k: (float(v[0]), float(v[1])) for k, v in d.get("ranges", {}).items()
                           if k in COMPONENTS})
        for key in ("accept_threshold", "uncertain_threshold", "margin_threshold"):
            if key in d and d[key] is not None:
                setattr(cfg, key, float(d[key]))
        cfg.source = d.get("source", cfg.source)
        cfg.notes = d.get("notes", cfg.notes)
        return cfg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: Path) -> FusionConfig | None:
        if not path.exists():
            return None
        return FusionConfig.from_dict(json.loads(path.read_text()))


@dataclass
class FusedScore:
    score: float | None
    normalized: dict[str, float | None]
    contributions: dict[str, float]
    used_weights: dict[str, float]
    missing: list[str]

    def to_dict(self) -> dict:
        return {
            "score": None if self.score is None else round(self.score, 5),
            "normalized": {k: None if v is None else round(v, 5) for k, v in self.normalized.items()},
            "contributions": {k: round(v, 5) for k, v in self.contributions.items()},
            "used_weights": {k: round(v, 5) for k, v in self.used_weights.items()},
            "missing": self.missing,
        }


def fuse(raw: dict[str, float | None], cfg: FusionConfig, components: tuple[str, ...] = COMPONENTS) -> FusedScore:
    normalized = {c: cfg.normalize(c, raw.get(c)) for c in components}
    active = {c: cfg.weights.get(c, 0.0) for c in components
              if normalized[c] is not None and cfg.weights.get(c, 0.0) > 0}
    missing = [c for c in components if normalized[c] is None and cfg.weights.get(c, 0.0) > 0]
    total = sum(active.values())
    if total <= 0:
        return FusedScore(None, normalized, {}, {}, missing)
    used = {c: w / total for c, w in active.items()}
    contrib = {c: used[c] * normalized[c] for c in active}  # type: ignore[operator]
    return FusedScore(float(sum(contrib.values())), normalized, contrib, used, missing)

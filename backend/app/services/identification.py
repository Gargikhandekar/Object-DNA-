"""Open-set instance identification: ranking + accept / uncertain / reject decision."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.services.feature_extraction import FeatureSet
from app.services.similarity import COMPONENTS, FusionConfig, fuse, global_evidence, pair_evidence

# OpenCV matching releases the GIL, so references are compared in parallel.
_POOL = ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 2)), thread_name_prefix="match")

MATCH_FOUND = "MATCH_FOUND"
UNCERTAIN = "UNCERTAIN"
NO_RELIABLE_MATCH = "NO_RELIABLE_MATCH"


@dataclass
class ReferenceEntry:
    object_db_id: int
    object_public_id: str
    image_id: int
    features: FeatureSet


@dataclass
class Decision:
    status: str
    reason: str
    best_score: float | None
    margin: float | None


def decide(ranked_scores: list[float], accept: float, uncertain: float, margin: float) -> Decision:
    """Decision rule shared by the application and the experiments.

    * MATCH_FOUND  if best >= accept and (best - second_best) >= margin
    * UNCERTAIN    if best >= uncertain (but the conditions above fail)
    * NO_RELIABLE_MATCH otherwise.
    ``ranked_scores`` are per-object scores sorted in descending order.
    """
    if not ranked_scores:
        return Decision(NO_RELIABLE_MATCH, "No registered objects with comparable fingerprints.", None, None)
    best = ranked_scores[0]
    gap = best - ranked_scores[1] if len(ranked_scores) > 1 else None
    if best >= accept and (gap is None or gap >= margin):
        return Decision(MATCH_FOUND, f"Best score {best:.3f} >= accept threshold {accept:.3f}"
                        + ("" if gap is None else f" and margin {gap:.3f} >= {margin:.3f}") + ".", best, gap)
    if best >= accept:
        return Decision(UNCERTAIN, f"Best score {best:.3f} passes the accept threshold but the margin to the "
                        f"next object ({gap:.3f}) is below {margin:.3f}; candidates are too close.", best, gap)
    if best >= uncertain:
        return Decision(UNCERTAIN, f"Best score {best:.3f} is between the uncertain ({uncertain:.3f}) and accept "
                        f"({accept:.3f}) thresholds.", best, gap)
    return Decision(NO_RELIABLE_MATCH, f"Best score {best:.3f} is below the uncertain threshold {uncertain:.3f}.",
                    best, gap)


@dataclass
class Candidate:
    object_db_id: int
    object_public_id: str
    score: float
    best_image_id: int
    supporting_references: int
    references_compared: int
    fused: dict
    evidence: dict
    matches: list = field(default_factory=list)
    reference_scores: list = field(default_factory=list)


@dataclass
class IdentificationResult:
    decision: Decision
    candidates: list[Candidate]
    inference_ms: float
    timings: dict
    references_compared: int


def identify(
    query: FeatureSet,
    references: list[ReferenceEntry],
    cfg: FusionConfig,
    *,
    top_k: int = 5,
    orb_shortlist: int = 0,
    ratio: float = 0.75,
    reproj: float = 6.0,
    keep_matches_top: int = 3,
    accept: float | None = None,
    uncertain: float | None = None,
    margin: float | None = None,
) -> IdentificationResult:
    t0 = time.perf_counter()
    accept = cfg.accept_threshold if accept is None else accept
    uncertain = cfg.uncertain_threshold if uncertain is None else uncertain
    margin = cfg.margin_threshold if margin is None else margin

    # Stage 1: cheap global evidence for every reference.
    globals_ = [global_evidence(query, r.features) for r in references]
    t_global = time.perf_counter()

    # Optional shortlist: objects ranked by fused global-only evidence.
    orb_objects: set[str] | None = None
    if orb_shortlist and orb_shortlist > 0:
        per_obj: dict[str, float] = {}
        global_components = tuple(c for c in COMPONENTS if c != "geometric")
        for ref, g in zip(references, globals_):
            s = fuse(g, cfg, global_components).score or 0.0
            per_obj[ref.object_public_id] = max(per_obj.get(ref.object_public_id, 0.0), s)
        orb_objects = set(sorted(per_obj, key=per_obj.get, reverse=True)[:orb_shortlist])

    # Stage 2: ORB + RANSAC and fusion.
    def _evidence(ref: ReferenceEntry):
        use_orb = orb_objects is None or ref.object_public_id in orb_objects
        return pair_evidence(query, ref.features, ratio=ratio, reproj=reproj, with_orb=use_orb)

    evidences = list(_POOL.map(_evidence, references)) if len(references) > 4 else [_evidence(r) for r in references]
    per_object: dict[str, list[tuple[float, ReferenceEntry, dict, object]]] = {}
    for ref, ev in zip(references, evidences):
        fused = fuse(ev.raw, cfg)
        score = fused.score if fused.score is not None else 0.0
        per_object.setdefault(ref.object_public_id, []).append((score, ref, fused.to_dict(), ev))
    t_orb = time.perf_counter()

    candidates: list[Candidate] = []
    for pid, items in per_object.items():
        items.sort(key=lambda x: x[0], reverse=True)
        best_score, best_ref, best_fused, best_ev = items[0]
        candidates.append(Candidate(
            object_db_id=best_ref.object_db_id,
            object_public_id=pid,
            score=float(best_score),
            best_image_id=best_ref.image_id,
            supporting_references=sum(1 for s, *_ in items if s >= uncertain),
            references_compared=len(items),
            fused=best_fused,
            evidence=best_ev.to_dict(),
            reference_scores=[{"image_id": r.image_id, "score": round(s, 5)} for s, r, *_ in items],
        ))
    # Aggregation rule: object score = max over its reference images.
    candidates.sort(key=lambda c: c.score, reverse=True)

    # Keep ORB correspondences for visualisation of the top candidates.
    for cand in candidates[:keep_matches_top]:
        ref = next(r for r in references if r.image_id == cand.best_image_id)
        res = pair_evidence(query, ref.features, ratio=ratio, reproj=reproj, keep_matches=True).orb
        cand.matches = res.matches if res else []

    decision = decide([c.score for c in candidates], accept, uncertain, margin)
    t_end = time.perf_counter()
    return IdentificationResult(
        decision=decision,
        candidates=candidates[:top_k],
        inference_ms=(t_end - t0) * 1000,
        timings={"global_ms": (t_global - t0) * 1000, "matching_ms": (t_orb - t_global) * 1000,
                 "visualisation_ms": (t_end - t_orb) * 1000},
        references_compared=len(references),
    )

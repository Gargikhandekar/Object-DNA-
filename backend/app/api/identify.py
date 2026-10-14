"""Identification (1:N) and pairwise comparison (1:1)."""

from __future__ import annotations

import base64
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import image_url, read_and_validate, runtime
from app.core.runtime import Runtime
from app.database import get_db
from app.models.orm import Identification, PhysicalObject
from app.schemas import CandidateOut, CompareResponse, IdentifyResponse, OrbMatch
from app.services.identification import MATCH_FOUND, decide, identify
from app.services.similarity import fuse, pair_evidence

router = APIRouter(prefix="/api", tags=["identification"])

SCORE_NOTE = ("Scores are fused similarity scores in [0, 1] computed from normalised component evidence. They are "
              "not calibrated probabilities. The evidence breakdown is a summary of what the score is made of, "
              "not a causal explanation.")


def _embedding_status(rt: Runtime) -> dict:
    return rt.embedder.status() if rt.embedder else {"available": False, "backbone": "none",
                                                     "message": "Deep embeddings disabled."}


def _matches(raw) -> list[OrbMatch]:
    return [OrbMatch(qx=m[0], qy=m[1], rx=m[2], ry=m[3], inlier=m[4]) for m in raw]


@router.post("/identify", response_model=IdentifyResponse, summary="Identify a physical object from one image")
async def identify_object(
    file: UploadFile = File(...),
    top_k: int = Form(5, ge=1, le=50),
    accept_threshold: float | None = Form(None, ge=0, le=1),
    uncertain_threshold: float | None = Form(None, ge=0, le=1),
    margin_threshold: float | None = Form(None, ge=0, le=1),
    db: Session = Depends(get_db),
    rt: Runtime = Depends(runtime),
):
    (img,) = await read_and_validate([file], rt, field="file")
    t0 = time.perf_counter()
    prep, feats = rt.store.process(img.rgb)
    t_extract = (time.perf_counter() - t0) * 1000
    refs = rt.store.references(db)
    result = identify(feats, refs, rt.fusion, top_k=top_k, orb_shortlist=rt.settings.orb_shortlist,
                      ratio=rt.settings.orb_ratio, reproj=rt.settings.ransac_reproj_threshold,
                      accept=accept_threshold, uncertain=uncertain_threshold, margin=margin_threshold)
    total_ms = t_extract + result.inference_ms

    objs = {o.public_id: o for o in db.scalars(select(PhysicalObject).where(
        PhysicalObject.public_id.in_([c.object_public_id for c in result.candidates]))).all()}
    candidates = []
    for rank, c in enumerate(result.candidates, start=1):
        o = objs.get(c.object_public_id)
        candidates.append(CandidateOut(
            rank=rank, object_public_id=c.object_public_id, name=o.name if o else "?", category=o.category if o else "?",
            score=round(c.score, 5), best_reference_image_id=c.best_image_id,
            best_reference_url=image_url(c.best_image_id), supporting_references=c.supporting_references,
            references_compared=c.references_compared, fused=c.fused, evidence=c.evidence,
            reference_scores=c.reference_scores, matches=_matches(c.matches)))

    month = datetime.now(timezone.utc).strftime("%Y-%m")
    original, processed = rt.store.save_upload(rt.store.storage_dir / "queries" / month, img, prep)
    status = result.decision.status
    predicted = candidates[0] if (status == MATCH_FOUND and candidates) else None
    thresholds = {
        "accept": accept_threshold if accept_threshold is not None else rt.fusion.accept_threshold,
        "uncertain": uncertain_threshold if uncertain_threshold is not None else rt.fusion.uncertain_threshold,
        "margin": margin_threshold if margin_threshold is not None else rt.fusion.margin_threshold,
    }
    rec = Identification(
        query_image_path=rt.store.rel(processed), status=status,
        selected_object_public_id=predicted.object_public_id if predicted else None,
        best_score=result.decision.best_score,
        candidates=[{"object_public_id": c.object_public_id, "name": c.name, "score": c.score,
                     "fused": c.fused} for c in candidates],
        decision_config={**thresholds, "fusion_source": rt.fusion.source}, inference_ms=total_ms,
    )
    db.add(rec)
    db.commit()
    # The original upload is not needed once the processed copy is stored.
    original.unlink(missing_ok=True)

    return IdentifyResponse(
        identification_id=rec.id, status=status, reason=result.decision.reason,
        predicted_object_id=predicted.object_public_id if predicted else None,
        predicted_name=predicted.name if predicted else None,
        predicted_category=predicted.category if predicted else None,
        best_score=result.decision.best_score, margin=result.decision.margin, candidates=candidates,
        thresholds=thresholds, fusion=rt.fusion.to_dict(), query_image_url=f"/api/history/{rec.id}/image",
        query_processed_size=(int(prep.rgb.shape[1]), int(prep.rgb.shape[0])), query_quality=prep.quality,
        inference_ms=round(total_ms, 2),
        timings={"feature_extraction_ms": round(t_extract, 2),
                 **{k: round(v, 2) for k, v in result.timings.items()}},
        references_compared=result.references_compared, embedding=_embedding_status(rt), note=SCORE_NOTE,
    )


def _data_url(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def raw_pixel_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation of 64x64 grayscale thumbnails, in [-1, 1]."""
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    ga -= ga.mean()
    gb -= gb.mean()
    return float((ga * gb).sum() / max(np.sqrt((ga ** 2).sum() * (gb ** 2).sum()), 1e-6))


@router.post("/compare", response_model=CompareResponse, summary="Same-instance verification of two images")
async def compare_images(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    rt: Runtime = Depends(runtime),
):
    a, b = await read_and_validate([file_a, file_b], rt, field="file_a/file_b")
    t0 = time.perf_counter()
    prep_a, fa = rt.store.process(a.rgb)
    prep_b, fb = rt.store.process(b.rgb)
    ev = pair_evidence(fa, fb, ratio=rt.settings.orb_ratio, reproj=rt.settings.ransac_reproj_threshold,
                       keep_matches=True)
    fused = fuse(ev.raw, rt.fusion)
    ms = (time.perf_counter() - t0) * 1000
    cfg = rt.fusion
    d = decide([fused.score] if fused.score is not None else [], cfg.accept_threshold, cfg.uncertain_threshold, 0.0)
    decision = {"MATCH_FOUND": "SAME_INSTANCE", "UNCERTAIN": "INCONCLUSIVE"}.get(d.status, "DIFFERENT_INSTANCE")
    reason = {
        "SAME_INSTANCE": f"Fused score {fused.score:.3f} >= accept threshold {cfg.accept_threshold:.3f}."
        if fused.score is not None else "",
        "INCONCLUSIVE": f"Fused score is between the uncertain ({cfg.uncertain_threshold:.3f}) and accept "
                        f"({cfg.accept_threshold:.3f}) thresholds.",
        "DIFFERENT_INSTANCE": "Fused score is below the uncertain threshold "
                              f"({cfg.uncertain_threshold:.3f}); the evidence does not support the same instance.",
    }[decision]
    return CompareResponse(
        decision=decision, reason=reason, score=fused.score,
        thresholds={"accept": cfg.accept_threshold, "uncertain": cfg.uncertain_threshold},
        fused=fused.to_dict(), evidence=ev.to_dict(), matches=_matches(ev.orb.matches if ev.orb else []),
        image_a=_data_url(prep_a.rgb), image_b=_data_url(prep_b.rgb),
        size_a=(int(prep_a.rgb.shape[1]), int(prep_a.rgb.shape[0])),
        size_b=(int(prep_b.rgb.shape[1]), int(prep_b.rgb.shape[0])),
        quality_a=prep_a.quality, quality_b=prep_b.quality,
        raw_pixel_similarity=round(raw_pixel_similarity(prep_a.rgb, prep_b.rgb), 4), inference_ms=round(ms, 2),
        embedding=_embedding_status(rt),
        note=SCORE_NOTE + " Two images of the same object can have low raw pixel similarity because of "
                          "viewpoint, scale, background or lighting changes; that is why local, texture and "
                          "embedding evidence are used instead of pixel comparison.",
    )

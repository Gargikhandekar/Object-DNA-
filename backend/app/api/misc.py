"""Images, identification history, dashboard, health and decision settings."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.api.deps import runtime, serialize_history
from app.core.runtime import Runtime
from app.database import get_db
from app.models.orm import ExperimentRun, Identification, Observation, PhysicalObject, ReferenceImage
from app.schemas import DashboardOut, HealthOut, HistoryList

router = APIRouter(prefix="/api", tags=["system"])
VERSION = "1.0.0"


def _file(rt: Runtime, rel: str) -> FileResponse:
    try:
        p = rt.store.abs(rel)
    except ValueError as exc:
        raise HTTPException(404, detail="File not found.") from exc
    if not p.exists():
        raise HTTPException(404, detail="File not found.")
    return FileResponse(p, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/images/{image_id}", summary="Serve a reference image", tags=["objects"])
def get_image(image_id: int, variant: str = Query("processed", pattern="^(processed|original)$"),
              db: Session = Depends(get_db), rt: Runtime = Depends(runtime)):
    img = db.get(ReferenceImage, image_id)
    if img is None:
        raise HTTPException(404, detail="Image not found.")
    return _file(rt, img.processed_path if variant == "processed" else img.original_path)


@router.get("/observations/{obs_id}/image", summary="Serve an observation image", tags=["objects"])
def get_observation_image(obs_id: int, db: Session = Depends(get_db), rt: Runtime = Depends(runtime)):
    o = db.get(Observation, obs_id)
    if o is None:
        raise HTTPException(404, detail="Observation not found.")
    return _file(rt, o.processed_path)


@router.get("/history", response_model=HistoryList, summary="Identification attempt history")
def history(page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=200),
            status: str | None = Query(None, pattern="^(MATCH_FOUND|UNCERTAIN|NO_RELIABLE_MATCH)$"),
            object_id: str | None = None, db: Session = Depends(get_db)):
    stmt = select(Identification)
    if status:
        stmt = stmt.where(Identification.status == status)
    if object_id:
        stmt = stmt.where(Identification.selected_object_public_id == object_id)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(stmt.order_by(Identification.created_at.desc(), Identification.id.desc())
                      .offset((page - 1) * page_size).limit(page_size)).all()
    return HistoryList(items=[serialize_history(r) for r in rows], total=total, page=page, page_size=page_size)


@router.get("/history/{identification_id}/image", summary="Serve a stored query image")
def history_image(identification_id: int, db: Session = Depends(get_db), rt: Runtime = Depends(runtime)):
    rec = db.get(Identification, identification_id)
    if rec is None:
        raise HTTPException(404, detail="Identification not found.")
    return _file(rt, rec.query_image_path)


def _experiment_summaries(db: Session, limit: int = 5) -> list[dict]:
    runs = db.scalars(select(ExperimentRun).where(ExperimentRun.status == "completed")
                      .order_by(ExperimentRun.created_at.desc()).limit(limit)).all()
    out = []
    for r in runs:
        ident = (r.metrics or {}).get("identification", {}).get("methods", {})
        out.append({
            "run_uid": r.run_uid, "experiment": r.experiment, "dataset_id": r.dataset_id,
            "is_synthetic": r.is_synthetic, "created_at": r.created_at.isoformat(),
            "identification_top1": {m: v.get("top1_accuracy") for m, v in ident.items()},
            "verification_auc": {m: v.get("test", {}).get("roc_auc") for m, v in
                                 (r.metrics or {}).get("verification", {}).get("methods", {}).items()},
        })
    return out


@router.get("/dashboard", response_model=DashboardOut, summary="Usage statistics and experiment summaries")
def dashboard(db: Session = Depends(get_db), rt: Runtime = Depends(runtime)):
    n_obj = db.scalar(select(func.count(PhysicalObject.id))) or 0
    n_img = db.scalar(select(func.count(ReferenceImage.id))) or 0
    n_obs = db.scalar(select(func.count(Observation.id))) or 0
    by_status = dict(db.execute(select(Identification.status, func.count()).group_by(Identification.status)).all())
    n_q = sum(by_status.values())
    avg_ms = db.scalar(select(func.avg(Identification.inference_ms)))
    recent = db.scalars(select(Identification).order_by(Identification.created_at.desc()).limit(8)).all()
    cats = db.execute(select(PhysicalObject.category, func.count()).group_by(PhysicalObject.category)).all()
    return DashboardOut(
        usage={
            "registered_objects": n_obj,
            "reference_images": n_img,
            "observations": n_obs,
            "query_attempts": n_q,
            "accepted": by_status.get("MATCH_FOUND", 0),
            "uncertain": by_status.get("UNCERTAIN", 0),
            "rejected": by_status.get("NO_RELIABLE_MATCH", 0),
            "avg_inference_ms": round(avg_ms, 1) if avg_ms is not None else None,
        },
        recent=[serialize_history(r) for r in recent],
        categories=[{"category": c, "count": n} for c, n in cats],
        model_status=rt.embedder.status() if rt.embedder else {"available": False, "backbone": "none"},
        fusion=rt.fusion.to_dict(),
        experiments=_experiment_summaries(db),
        note=("Usage counts describe what users submitted. They are NOT accuracy: production queries are "
              "unlabelled. Model performance is only reported from labelled experiment runs (see Research Lab)."),
    )


@router.get("/health", response_model=HealthOut, summary="API and model readiness")
def health(db: Session = Depends(get_db), rt: Runtime = Depends(runtime)):
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
        n = db.scalar(select(func.count(PhysicalObject.id))) or 0
    except Exception:  # pragma: no cover
        db_ok, n = False, 0
    storage_ok = rt.store.storage_dir.exists()
    emb = rt.embedder.status() if rt.embedder else {"available": False, "backbone": "none",
                                                    "message": "Deep embeddings disabled."}
    return HealthOut(status="ok" if db_ok and storage_ok else "degraded", database=db_ok, storage=storage_ok,
                     embedding=emb, fusion_source=rt.fusion.source, registered_objects=n, version=VERSION)


class DecisionSettings(BaseModel):
    accept_threshold: float = Field(ge=0, le=1)
    uncertain_threshold: float = Field(ge=0, le=1)
    margin_threshold: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _order(self):
        if self.uncertain_threshold > self.accept_threshold:
            raise ValueError("uncertain_threshold must be <= accept_threshold")
        return self


@router.get("/settings/fusion", summary="Active fusion weights, normalisation ranges and thresholds")
def get_fusion(rt: Runtime = Depends(runtime)):
    return rt.fusion.to_dict()


@router.put("/settings/decision", summary="Update decision thresholds")
def set_decision(body: DecisionSettings, rt: Runtime = Depends(runtime)):
    cfg = rt.fusion
    cfg.accept_threshold = body.accept_threshold
    cfg.uncertain_threshold = body.uncertain_threshold
    cfg.margin_threshold = body.margin_threshold
    if "manual thresholds" not in cfg.notes:
        cfg.notes = (cfg.notes + " Thresholds set manually via API.").strip()
    cfg.save(rt.settings.active_fusion_path)
    return cfg.to_dict()


@router.post("/settings/fusion/reset", summary="Revert to the default heuristic fusion configuration")
def reset_fusion(rt: Runtime = Depends(runtime)):
    rt.settings.active_fusion_path.unlink(missing_ok=True)
    return rt.reload_fusion().to_dict()


@router.post("/settings/backfill-embeddings", summary="Compute missing deep embeddings for existing references")
def backfill(db: Session = Depends(get_db), rt: Runtime = Depends(runtime)):
    if rt.embedder is None or not rt.embedder.available:
        raise HTTPException(409, detail=(rt.embedder.status()["message"] if rt.embedder
                                         else "Deep embeddings disabled."))
    return {"computed": rt.store.backfill_embeddings(db)}

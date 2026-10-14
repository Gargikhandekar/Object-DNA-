"""Research experiment execution and results.

Experiments run in a single background worker thread so they never block the
API's request handling. Only one experiment executes at a time; additional
requests are queued. Runs interrupted by a server restart are marked failed
at the next startup.
"""

from __future__ import annotations

import csv
import io
import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import runtime
from app.core.runtime import Runtime
from app.database import get_db, get_sessionmaker
from app.models.orm import ExperimentRun
from app.research import experiments as exp
from app.research.dataset import load_manifest
from app.research.synthetic import generate_dataset
from app.schemas import ExperimentOut, ExperimentRequest
from app.services.similarity import FusionConfig

router = APIRouter(prefix="/api/experiments", tags=["experiments"])
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="experiment")
_gen_lock = threading.Lock()


def datasets_dir(rt: Runtime) -> Path:
    return rt.settings.data_dir / "datasets"


def resolve_dataset(rt: Runtime, name: str) -> Path:
    base = datasets_dir(rt).resolve()
    p = Path(name)
    if not p.is_absolute():
        p = (base / name).resolve()
        if not str(p).startswith(str(base)):
            raise HTTPException(400, detail="Invalid dataset name.")
    if not (p / "manifest.csv").exists():
        raise HTTPException(404, detail=f"Dataset '{name}' has no manifest.csv. Import it with "
                                        "scripts/import_dataset.py or generate the synthetic demo dataset.")
    return p


def _out(r: ExperimentRun) -> ExperimentOut:
    return ExperimentOut.model_validate(r, from_attributes=True)


def _execute(run_id: int, dataset_root: Path, cfg: exp.ExperimentConfig, rt: Runtime) -> None:
    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        run = db.get(ExperimentRun, run_id)
        run.status, run.progress, run.message = "running", 0.0, "Starting"
        db.commit()
        last = [0.0]

        def progress(p: float, msg: str) -> None:
            if p - last[0] >= 0.02 or p >= 1.0:
                last[0] = p
                run.progress, run.message = round(p, 3), msg
                db.commit()
        try:
            metrics = exp.run(dataset_root, cfg, rt.prep_config, rt.feat_config, rt.embedder,
                              rt.settings.experiments_dir / "results",
                              rt.settings.experiments_dir / "cache", progress, run.run_uid)
            run.metrics = json.loads(json.dumps(metrics, default=exp._json_default))
            run.artifact_paths = metrics.get("artifacts", {})
            run.split_id = metrics.get("split", {}).get("split_id")
            run.status, run.progress, run.message = "completed", 1.0, f"Completed in {metrics.get('runtime_s')} s"
        except Exception as exc:
            run.status = "failed"
            run.message = f"{type(exc).__name__}: {exc}"
            run.metrics = {"error": run.message, "traceback": traceback.format_exc()[-4000:]}
        run.finished_at = datetime.now(timezone.utc)
        db.commit()


@router.post("/run", response_model=ExperimentOut, status_code=202, summary="Queue an evaluation run")
def run_experiment(req: ExperimentRequest, db: Session = Depends(get_db), rt: Runtime = Depends(runtime)):
    root = resolve_dataset(rt, req.dataset)
    ds = load_manifest(root)
    cfg = exp.ExperimentConfig(experiment=req.experiment, methods=req.methods, seed=req.seed,
                               val_fraction=req.val_fraction, max_pairs=req.max_pairs,
                               perturbations=req.perturbations)
    run = ExperimentRun(run_uid=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
                        experiment=req.experiment, dataset_id=ds.dataset_id, config=req.model_dump(),
                        status="queued", is_synthetic=ds.is_synthetic, message="Queued")
    db.add(run)
    db.commit()
    _executor.submit(_execute, run.id, root, cfg, rt)
    return _out(run)


@router.get("", response_model=list[ExperimentOut], summary="List experiment runs")
def list_runs(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.scalars(select(ExperimentRun).order_by(ExperimentRun.created_at.desc()).limit(limit)).all()
    out = []
    for r in rows:
        o = _out(r)
        o.metrics = {k: v for k, v in (r.metrics or {}).items()
                     if k in ("identification", "verification", "methods_evaluated", "is_synthetic", "error")}
        if "identification" in o.metrics:  # strip heavy confusion matrices from list view
            o.metrics = json.loads(json.dumps(o.metrics))
            for m in o.metrics["identification"].get("methods", {}).values():
                m.pop("confusion", None)
        if "verification" in o.metrics:
            o.metrics = json.loads(json.dumps(o.metrics))
            for m in o.metrics["verification"].get("methods", {}).values():
                m.get("test", {}).pop("curve", None)
        out.append(o)
    return out


class DatasetInfo(BaseModel):
    name: str
    images: int
    objects: int
    synthetic: bool
    dataset_id: str


@router.get("/datasets", response_model=list[DatasetInfo], summary="List importable datasets")
def list_datasets(rt: Runtime = Depends(runtime)):
    base = datasets_dir(rt)
    out = []
    if base.exists():
        for d in sorted(base.iterdir()):
            if (d / "manifest.csv").exists():
                try:
                    ds = load_manifest(d)
                except Exception:
                    continue
                out.append(DatasetInfo(name=d.name, images=len(ds.samples), objects=len(ds.objects()),
                                       synthetic=ds.is_synthetic, dataset_id=ds.dataset_id))
    return out


class SyntheticRequest(BaseModel):
    name: str = Field("synthetic_demo", pattern=r"^[A-Za-z0-9_\-]{1,60}$")
    n_models: int = Field(5, ge=1, le=12)
    instances_per_model: int = Field(4, ge=1, le=10)
    unregistered_per_model: int = Field(1, ge=0, le=4)
    seed: int = 7


@router.post("/datasets/synthetic", response_model=DatasetInfo, status_code=201,
             summary="Generate the procedural synthetic demo dataset (sanity checks only)")
def make_synthetic(req: SyntheticRequest, rt: Runtime = Depends(runtime)):
    out = datasets_dir(rt) / req.name
    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(409, detail="A dataset is already being generated.")
    try:
        generate_dataset(out, req.n_models, req.instances_per_model, req.unregistered_per_model, seed=req.seed)
    finally:
        _gen_lock.release()
    ds = load_manifest(out)
    return DatasetInfo(name=req.name, images=len(ds.samples), objects=len(ds.objects()), synthetic=True,
                       dataset_id=ds.dataset_id)


@router.get("/{run_uid}", response_model=ExperimentOut, summary="Full results of one run")
def get_run(run_uid: str, db: Session = Depends(get_db)):
    r = db.scalar(select(ExperimentRun).where(ExperimentRun.run_uid == run_uid))
    if r is None:
        raise HTTPException(404, detail="Experiment run not found.")
    return _out(r)


@router.get("/{run_uid}/artifacts/{name}", summary="Download a run artifact (charts, CSV, JSON)")
def get_artifact(run_uid: str, name: str, db: Session = Depends(get_db), rt: Runtime = Depends(runtime)):
    r = db.scalar(select(ExperimentRun).where(ExperimentRun.run_uid == run_uid))
    if r is None or name not in (r.artifact_paths or {}).values():
        raise HTTPException(404, detail="Artifact not found.")
    p = rt.settings.experiments_dir / "results" / run_uid / name
    if not p.exists():
        raise HTTPException(404, detail="Artifact file missing.")
    return FileResponse(p, filename=f"{run_uid}_{name}")


@router.get("/{run_uid}/export.csv", summary="Export the summary metrics of a run as CSV")
def export_csv(run_uid: str, db: Session = Depends(get_db)):
    r = db.scalar(select(ExperimentRun).where(ExperimentRun.run_uid == run_uid))
    if r is None or r.status != "completed":
        raise HTTPException(404, detail="Completed run not found.")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["section", "method", "metric", "value"])
    m = r.metrics or {}
    for meth, v in m.get("verification", {}).get("methods", {}).items():
        for k, val in v.get("test", {}).items():
            if k != "curve":
                w.writerow(["verification_test", meth, k, val])
    for meth, v in m.get("identification", {}).get("methods", {}).items():
        for k, val in v.items():
            if not isinstance(val, (dict, list)):
                w.writerow(["identification_test", meth, k, val])
    for c in m.get("robustness", {}).get("conditions", []):
        for meth, v in c.get("methods", {}).items():
            for k, val in v.items():
                w.writerow([f"robustness:{c['condition']}", meth, k, val])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{run_uid}_summary.csv"'})


@router.post("/{run_uid}/apply-fusion", summary="Use this run's validation-tuned fusion in the application")
def apply_fusion(run_uid: str, db: Session = Depends(get_db), rt: Runtime = Depends(runtime)):
    r = db.scalar(select(ExperimentRun).where(ExperimentRun.run_uid == run_uid))
    if r is None:
        raise HTTPException(404, detail="Experiment run not found.")
    p = rt.settings.experiments_dir / "results" / run_uid / "tuned_fusion.json"
    cfg = FusionConfig.load(p)
    if cfg is None:
        raise HTTPException(409, detail="This run has no tuned fusion configuration.")
    cfg.save(rt.settings.active_fusion_path)
    return rt.reload_fusion().to_dict()


@router.delete("/{run_uid}", status_code=204, summary="Delete an experiment run record and its artifacts")
def delete_run(run_uid: str, db: Session = Depends(get_db), rt: Runtime = Depends(runtime)):
    import shutil
    r = db.scalar(select(ExperimentRun).where(ExperimentRun.run_uid == run_uid))
    if r is None:
        raise HTTPException(404, detail="Experiment run not found.")
    if r.status in ("queued", "running"):
        raise HTTPException(409, detail="Cannot delete a queued or running experiment.")
    db.delete(r)
    db.commit()
    shutil.rmtree(rt.settings.experiments_dir / "results" / run_uid, ignore_errors=True)


def mark_interrupted_runs() -> None:
    with get_sessionmaker()() as db:
        for r in db.scalars(select(ExperimentRun).where(ExperimentRun.status.in_(("queued", "running")))).all():
            r.status, r.message = "failed", "Interrupted by server restart."
        db.commit()

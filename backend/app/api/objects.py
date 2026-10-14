"""Object registration, library, reference management, history and observations."""

from __future__ import annotations

import json
import secrets

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import (
    read_and_validate,
    runtime,
    serialize_detail,
    serialize_image,
    serialize_observation,
    serialize_summary,
)
from app.core.runtime import Runtime
from app.database import get_db
from app.models.orm import Observation, PhysicalObject, ReferenceImage
from app.schemas import (
    ObjectDetail,
    ObjectHistory,
    ObjectList,
    ObservationResult,
    RegistrationResult,
    RegistrationWarning,
)
from app.services import texture_features as tf
from app.services.identification import MATCH_FOUND, identify
from app.services.image_validation import ValidatedImage

router = APIRouter(prefix="/api/objects", tags=["objects"])


def _new_public_id(db: Session) -> str:
    while True:
        pid = "ODNA-" + secrets.token_hex(4).upper()
        if not db.scalar(select(PhysicalObject.id).where(PhysicalObject.public_id == pid)):
            return pid


def _load_object(db: Session, public_id: str) -> PhysicalObject:
    obj = db.scalar(
        select(PhysicalObject)
        .where(PhysicalObject.public_id == public_id)
        .options(selectinload(PhysicalObject.images).selectinload(ReferenceImage.fingerprints),
                 selectinload(PhysicalObject.observations))
    )
    if obj is None:
        raise HTTPException(404, detail=f"Object '{public_id}' not found.")
    return obj


def _check_duplicates(db: Session, images: list[ValidatedImage]) -> None:
    seen: dict[str, str | None] = {}
    for img in images:
        if img.sha256 in seen:
            raise HTTPException(409, detail={"message": "The same file was uploaded more than once in this request.",
                                             "files": [seen[img.sha256], img.original_filename]})
        seen[img.sha256] = img.original_filename
    existing = db.execute(
        select(ReferenceImage.content_sha256, PhysicalObject.public_id, PhysicalObject.name)
        .join(PhysicalObject, PhysicalObject.id == ReferenceImage.object_id)
        .where(ReferenceImage.content_sha256.in_(list(seen)))
    ).all()
    if existing:
        raise HTTPException(409, detail={
            "message": "One or more images are already registered as reference images. An identical file cannot "
                       "be a reference for two identities.",
            "conflicts": [{"filename": seen[sha], "object_id": pid, "object_name": name} for sha, pid, name in existing],
        })


def _parse_metadata(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, detail=f"capture_metadata is not valid JSON: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise HTTPException(422, detail="capture_metadata must be a JSON list of objects (one per file).")
    return data


def _add_images(db: Session, rt: Runtime, obj: PhysicalObject, images: list[ValidatedImage],
                metadata: list[dict]) -> list[RegistrationWarning]:
    warnings: list[RegistrationWarning] = []
    for i, img in enumerate(images):
        meta = metadata[i] if i < len(metadata) else {}
        ref = rt.store.add_reference(db, obj, img, {k: str(v)[:200] for k, v in meta.items()})
        for w in ref.quality.get("warnings", []):
            warnings.append(RegistrationWarning(filename=img.original_filename, message=w))
        if ref.quality.get("embedding_message"):
            warnings.append(RegistrationWarning(filename=img.original_filename,
                                                message="Deep embedding not computed: " + ref.quality["embedding_message"]))
    return warnings


@router.post("", response_model=RegistrationResult, status_code=201,
             summary="Register a physical object with one or more reference images")
async def register_object(
    name: str = Form(..., min_length=1, max_length=200),
    category: str = Form(..., min_length=1, max_length=100),
    description: str | None = Form(None, max_length=5000),
    capture_metadata: str | None = Form(None, description="Optional JSON list, one dict per file"),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    rt: Runtime = Depends(runtime),
):
    images = await read_and_validate(files, rt)
    _check_duplicates(db, images)
    meta = _parse_metadata(capture_metadata)
    obj = PhysicalObject(public_id=_new_public_id(db), name=name.strip(), category=category.strip().lower(),
                         description=(description or "").strip() or None, fingerprint_status="processing")
    db.add(obj)
    db.flush()
    try:
        warnings = _add_images(db, rt, obj, images, meta)
        obj.fingerprint_status = "complete"
        db.commit()
    except Exception:
        db.rollback()
        rt.store.delete_object_files(obj.public_id)
        raise
    if len(images) < 3:
        warnings.append(RegistrationWarning(filename=None, message=(
            "Only %d reference image(s). Register 3+ images from different angles, distances and lighting "
            "for more reliable identification." % len(images))))
    rt.store.invalidate()
    return RegistrationResult(object=serialize_detail(_load_object(db, obj.public_id)), warnings=warnings)


@router.get("", response_model=ObjectList, summary="List registered objects")
def list_objects(
    q: str | None = Query(None, description="Search name, ID, category or description"),
    category: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=200),
    db: Session = Depends(get_db),
):
    stmt = select(PhysicalObject)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(PhysicalObject.name.ilike(like), PhysicalObject.public_id.ilike(like),
                              PhysicalObject.category.ilike(like), PhysicalObject.description.ilike(like)))
    if category:
        stmt = stmt.where(PhysicalObject.category == category.lower())
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    items = db.scalars(stmt.options(selectinload(PhysicalObject.images), selectinload(PhysicalObject.observations))
                       .order_by(PhysicalObject.created_at.desc(), PhysicalObject.id.desc())
                       .offset((page - 1) * page_size).limit(page_size)).all()
    cats = sorted(c for c in db.scalars(select(PhysicalObject.category).distinct()).all())
    return ObjectList(items=[serialize_summary(o) for o in items], total=total, page=page, page_size=page_size,
                      categories=cats)


@router.get("/{object_id}", response_model=ObjectDetail, summary="Object details with reference images")
def get_object(object_id: str, db: Session = Depends(get_db)):
    return serialize_detail(_load_object(db, object_id))


@router.post("/{object_id}/images", response_model=RegistrationResult, status_code=201,
             summary="Add reference images to an existing object")
async def add_images(
    object_id: str,
    files: list[UploadFile] = File(...),
    capture_metadata: str | None = Form(None),
    db: Session = Depends(get_db),
    rt: Runtime = Depends(runtime),
):
    obj = _load_object(db, object_id)
    images = await read_and_validate(files, rt)
    _check_duplicates(db, images)
    try:
        warnings = _add_images(db, rt, obj, images, _parse_metadata(capture_metadata))
        db.commit()
    except Exception:
        db.rollback()
        rt.store.invalidate()
        raise
    db.expire_all()
    return RegistrationResult(object=serialize_detail(_load_object(db, object_id)), warnings=warnings)


@router.delete("/{object_id}/images/{image_id}", status_code=204, summary="Remove one reference image")
def delete_image(object_id: str, image_id: int, db: Session = Depends(get_db), rt: Runtime = Depends(runtime)):
    obj = _load_object(db, object_id)
    img = next((i for i in obj.images if i.id == image_id), None)
    if img is None:
        raise HTTPException(404, detail="Reference image not found for this object.")
    if len(obj.images) <= 1:
        raise HTTPException(409, detail="Cannot remove the last reference image; delete the object instead.")
    db.delete(img)
    db.commit()
    rt.store.delete_image_files(obj.public_id, img)


@router.delete("/{object_id}", status_code=204, summary="Delete an object, its images and fingerprints")
def delete_object(object_id: str, db: Session = Depends(get_db), rt: Runtime = Depends(runtime)):
    obj = _load_object(db, object_id)
    public_id = obj.public_id
    db.delete(obj)  # cascades to images, fingerprints and observations
    db.commit()
    rt.store.delete_object_files(public_id)  # files removed only after the DB commit succeeds


# --------------------------------------------------------------------------- history

@router.get("/{object_id}/history", response_model=ObjectHistory, summary="Registration and observation timeline")
def object_history(object_id: str, db: Session = Depends(get_db)):
    obj = _load_object(db, object_id)
    return ObjectHistory(
        object=serialize_summary(obj),
        first_registered=obj.created_at,
        references=[serialize_image(i) for i in obj.images],
        observations=[serialize_observation(o) for o in obj.observations],
        note=("Measured differences (texture, edges, colour, sharpness) are descriptive statistics. They can be "
              "caused by viewpoint, illumination, camera quality or preprocessing and are not by themselves "
              "evidence of damage or ageing."),
    )


def _surface_changes(query_feats, ref_feats: list, query_quality: dict, ref_qualities: list[dict]) -> dict:
    lbp = [tf.lbp_similarity(query_feats.lbp, r.lbp) for r in ref_feats]
    col = [tf.bhattacharyya_similarity(query_feats.color, r.color) for r in ref_feats]
    edge_ref = float(np.mean([r.gradient[0] for r in ref_feats]))
    sharp_ref = float(np.mean([q.get("sharpness_laplacian_var", 0) for q in ref_qualities])) if ref_qualities else 0
    bright_ref = float(np.mean([q.get("brightness_mean", 0) for q in ref_qualities])) if ref_qualities else 0
    return {
        "texture_similarity_max": round(max(lbp), 4),
        "color_similarity_max": round(max(col), 4),
        "edge_density": round(float(query_feats.gradient[0]), 4),
        "edge_density_reference_mean": round(edge_ref, 4),
        "edge_density_change": round(float(query_feats.gradient[0]) - edge_ref, 4),
        "sharpness": query_quality.get("sharpness_laplacian_var"),
        "sharpness_reference_mean": round(sharp_ref, 2),
        "brightness": query_quality.get("brightness_mean"),
        "brightness_reference_mean": round(bright_ref, 2),
        "keypoints": int(query_feats.orb.count),
        "keypoints_reference_mean": round(float(np.mean([r.orb.count for r in ref_feats])), 1),
    }


@router.post("/{object_id}/observations", response_model=ObservationResult,
             summary="Submit a later image of a registered object for its history")
async def add_observation(
    object_id: str,
    file: UploadFile = File(...),
    note: str | None = Form(None, max_length=2000),
    force: bool = Form(False, description="Store even if the system is only UNCERTAIN (user-confirmed)"),
    db: Session = Depends(get_db),
    rt: Runtime = Depends(runtime),
):
    obj = _load_object(db, object_id)
    (img,) = await read_and_validate([file], rt, field="file")
    prep, feats = rt.store.process(img.rgb)
    refs = rt.store.references(db)
    result = identify(feats, refs, rt.fusion, top_k=5, orb_shortlist=rt.settings.orb_shortlist,
                      ratio=rt.settings.orb_ratio, reproj=rt.settings.ransac_reproj_threshold, keep_matches_top=0)
    own = next((c for c in result.candidates if c.object_public_id == obj.public_id), None)
    if own is None:  # not in the top-k; score it directly against its own references
        own_refs = [r for r in refs if r.object_public_id == obj.public_id]
        own_res = identify(feats, own_refs, rt.fusion, keep_matches_top=0)
        own = own_res.candidates[0] if own_res.candidates else None
    top = result.candidates[0] if result.candidates else None
    status = result.decision.status
    if top is None or own is None:
        return ObservationResult(accepted=False, status="NO_RELIABLE_MATCH", reason="No comparable references.",
                                 observation=None, score=None)
    is_top = top.object_public_id == obj.public_id
    accepted = is_top and status == MATCH_FOUND
    reason = result.decision.reason
    obs_status = "ACCEPTED"
    if not is_top:
        reason = (f"The image is more similar to {top.object_public_id} (score {top.score:.3f}) than to "
                  f"{obj.public_id} (score {own.score:.3f}).")
    if not accepted and force and is_top and own.score >= rt.fusion.uncertain_threshold:
        accepted, obs_status = True, "USER_CONFIRMED_UNCERTAIN"
    if not accepted:
        return ObservationResult(accepted=False, status=status if is_top else "NO_RELIABLE_MATCH", reason=reason,
                                 observation=None, score=own.score)

    ref_ids = {i.id for i in obj.images}
    ref_feats = [r.features for r in refs if r.image_id in ref_ids]
    changes = _surface_changes(feats, ref_feats, prep.quality, [i.quality or {} for i in obj.images])
    original, processed = rt.store.save_upload(rt.store.object_dir(obj.public_id) / "observations", img, prep)
    o = Observation(object_id=obj.id, image_path=rt.store.rel(original), processed_path=rt.store.rel(processed),
                    similarity=own.score, status=obs_status,
                    evidence={"fused": own.fused, "evidence": own.evidence, "decision": status,
                              "reason": result.decision.reason},
                    surface_changes=changes, note=(note or "").strip() or None)
    db.add(o)
    db.commit()
    return ObservationResult(accepted=True, status=status, reason=reason, observation=serialize_observation(o),
                             score=own.score)

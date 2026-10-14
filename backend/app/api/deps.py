"""Shared API dependencies and serialisers."""

from __future__ import annotations

from fastapi import HTTPException, UploadFile

from app.core.runtime import Runtime, get_runtime
from app.models.orm import Identification, Observation, PhysicalObject, ReferenceImage
from app.schemas import (
    FingerprintOut,
    HistoryItem,
    ObjectDetail,
    ObjectSummary,
    ObservationOut,
    ReferenceImageOut,
)
from app.services.image_validation import ImageValidationError, ValidatedImage, validate_image_bytes


def runtime() -> Runtime:
    return get_runtime()


async def read_and_validate(files: list[UploadFile], rt: Runtime, *, field: str = "files") -> list[ValidatedImage]:
    s = rt.settings
    if not files:
        raise HTTPException(422, detail=f"At least one image is required in '{field}'.")
    if len(files) > s.max_images_per_request:
        raise HTTPException(413, detail=f"Too many files; at most {s.max_images_per_request} per request.")
    out, errors = [], []
    for f in files:
        data = await f.read(s.max_upload_bytes + 1)
        try:
            out.append(validate_image_bytes(data, f.filename, f.content_type, max_bytes=s.max_upload_bytes,
                                            min_side=s.min_image_side, max_pixels=s.max_image_pixels))
        except ImageValidationError as exc:
            errors.append({"filename": f.filename, "error": str(exc)})
    if errors:
        raise HTTPException(422, detail={"message": "One or more files were rejected.", "files": errors})
    return out


def image_url(image_id: int, variant: str = "processed") -> str:
    return f"/api/images/{image_id}?variant={variant}"


def serialize_image(img: ReferenceImage, with_fingerprints: bool = False) -> ReferenceImageOut:
    return ReferenceImageOut(
        id=img.id,
        url=image_url(img.id),
        original_url=image_url(img.id, "original"),
        original_filename=img.original_filename,
        width=img.width,
        height=img.height,
        capture_metadata=img.capture_metadata or {},
        quality=img.quality or {},
        uploaded_at=img.uploaded_at,
        fingerprints=[FingerprintOut.model_validate(fp) for fp in img.fingerprints] if with_fingerprints else [],
    )


def serialize_summary(obj: PhysicalObject) -> ObjectSummary:
    return ObjectSummary(
        public_id=obj.public_id,
        name=obj.name,
        category=obj.category,
        description=obj.description,
        created_at=obj.created_at,
        fingerprint_status=obj.fingerprint_status,
        image_count=len(obj.images),
        observation_count=len(obj.observations),
        thumbnail_url=image_url(obj.images[0].id) if obj.images else None,
    )


def serialize_detail(obj: PhysicalObject) -> ObjectDetail:
    return ObjectDetail(**serialize_summary(obj).model_dump(),
                        images=[serialize_image(i, with_fingerprints=True) for i in obj.images])


def serialize_history(item: Identification) -> HistoryItem:
    return HistoryItem(
        id=item.id,
        created_at=item.created_at,
        status=item.status,
        selected_object_public_id=item.selected_object_public_id,
        best_score=item.best_score,
        inference_ms=item.inference_ms,
        query_image_url=f"/api/history/{item.id}/image",
        top_candidates=[{k: c.get(k) for k in ("object_public_id", "name", "score")} for c in (item.candidates or [])[:3]],
        source=item.source,
    )


def serialize_observation(o: Observation) -> ObservationOut:
    return ObservationOut(
        id=o.id,
        observed_at=o.observed_at,
        similarity=o.similarity,
        status=o.status,
        image_url=f"/api/observations/{o.id}/image",
        evidence=o.evidence or {},
        surface_changes=o.surface_changes or {},
        note=o.note,
    )

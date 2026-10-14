"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["MATCH_FOUND", "UNCERTAIN", "NO_RELIABLE_MATCH"]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class FingerprintOut(ORMModel):
    id: int
    extractor: str
    feature_version: str
    dimension: int | None
    summary: dict[str, Any]
    preprocessing: dict[str, Any]
    created_at: datetime


class ReferenceImageOut(ORMModel):
    id: int
    url: str
    original_url: str
    original_filename: str | None
    width: int
    height: int
    capture_metadata: dict[str, Any]
    quality: dict[str, Any]
    uploaded_at: datetime
    fingerprints: list[FingerprintOut] = []


class ObjectSummary(BaseModel):
    public_id: str
    name: str
    category: str
    description: str | None
    created_at: datetime
    fingerprint_status: str
    image_count: int
    observation_count: int
    thumbnail_url: str | None


class ObjectDetail(ObjectSummary):
    images: list[ReferenceImageOut]


class ObjectList(BaseModel):
    items: list[ObjectSummary]
    total: int
    page: int
    page_size: int
    categories: list[str]


class RegistrationWarning(BaseModel):
    filename: str | None
    message: str


class RegistrationResult(BaseModel):
    object: ObjectDetail
    warnings: list[RegistrationWarning] = []


class OrbMatch(BaseModel):
    qx: float
    qy: float
    rx: float
    ry: float
    inlier: bool


class CandidateOut(BaseModel):
    rank: int
    object_public_id: str
    name: str
    category: str
    score: float = Field(description="Fused similarity score in [0, 1]; NOT a calibrated probability.")
    best_reference_image_id: int
    best_reference_url: str
    supporting_references: int
    references_compared: int
    fused: dict[str, Any]
    evidence: dict[str, Any]
    reference_scores: list[dict[str, Any]]
    matches: list[OrbMatch] = []


class IdentifyResponse(BaseModel):
    identification_id: int
    status: Status
    reason: str
    predicted_object_id: str | None
    predicted_name: str | None
    predicted_category: str | None
    best_score: float | None
    margin: float | None
    candidates: list[CandidateOut]
    thresholds: dict[str, float]
    fusion: dict[str, Any]
    query_image_url: str
    query_processed_size: tuple[int, int]
    query_quality: dict[str, Any]
    inference_ms: float
    timings: dict[str, float]
    references_compared: int
    embedding: dict[str, Any]
    note: str


class CompareResponse(BaseModel):
    decision: Literal["SAME_INSTANCE", "DIFFERENT_INSTANCE", "INCONCLUSIVE"]
    reason: str
    score: float | None
    thresholds: dict[str, float]
    fused: dict[str, Any]
    evidence: dict[str, Any]
    matches: list[OrbMatch]
    image_a: str  # data URL of the processed image (coordinates refer to it)
    image_b: str
    size_a: tuple[int, int]
    size_b: tuple[int, int]
    quality_a: dict[str, Any]
    quality_b: dict[str, Any]
    raw_pixel_similarity: float
    inference_ms: float
    embedding: dict[str, Any]
    note: str


class HistoryItem(BaseModel):
    id: int
    created_at: datetime
    status: str
    selected_object_public_id: str | None
    best_score: float | None
    inference_ms: float
    query_image_url: str
    top_candidates: list[dict[str, Any]]
    source: str


class HistoryList(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int


class ObservationOut(BaseModel):
    id: int
    observed_at: datetime
    similarity: float
    status: str
    image_url: str
    evidence: dict[str, Any]
    surface_changes: dict[str, Any]
    note: str | None


class ObservationResult(BaseModel):
    accepted: bool
    status: Status
    reason: str
    observation: ObservationOut | None
    score: float | None


class ObjectHistory(BaseModel):
    object: ObjectSummary
    first_registered: datetime
    references: list[ReferenceImageOut]
    observations: list[ObservationOut]
    note: str


class DashboardOut(BaseModel):
    usage: dict[str, Any]
    recent: list[HistoryItem]
    categories: list[dict[str, Any]]
    model_status: dict[str, Any]
    fusion: dict[str, Any]
    experiments: list[dict[str, Any]]
    note: str


class ExperimentRequest(BaseModel):
    experiment: Literal["full", "verification", "identification", "robustness"] = "full"
    dataset: str = Field("synthetic_demo", description="Dataset directory name under data/datasets or an "
                                                       "absolute path containing manifest.csv")
    methods: list[str] | None = None
    seed: int = 42
    val_fraction: float = Field(0.4, ge=0.1, le=0.8)
    max_pairs: int = Field(3000, ge=50, le=50_000)
    perturbations: bool = Field(True, description="Also evaluate synthetic blur/occlusion perturbations of test queries")


class ExperimentOut(BaseModel):
    run_uid: str
    experiment: str
    dataset_id: str
    split_id: str | None
    config: dict[str, Any]
    status: str
    progress: float
    message: str | None
    metrics: dict[str, Any]
    artifact_paths: dict[str, Any]
    is_synthetic: bool
    created_at: datetime
    finished_at: datetime | None


class HealthOut(BaseModel):
    status: str
    database: bool
    storage: bool
    embedding: dict[str, Any]
    fusion_source: str
    registered_objects: int
    version: str

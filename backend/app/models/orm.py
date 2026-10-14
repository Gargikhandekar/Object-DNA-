"""ORM models.

Large numerical arrays (ORB descriptors, embeddings, histograms) are stored as
``.npz`` files on disk; the ``fingerprints`` table only keeps references to
those artifacts plus the extractor/version/preprocessing metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PhysicalObject(Base):
    __tablename__ = "objects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(100), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    fingerprint_status: Mapped[str] = mapped_column(String(20), default="pending")

    images: Mapped[list[ReferenceImage]] = relationship(
        back_populates="obj", cascade="all, delete-orphan", passive_deletes=True,
        order_by="ReferenceImage.id",
    )
    fingerprints: Mapped[list[Fingerprint]] = relationship(
        back_populates="obj", cascade="all, delete-orphan", passive_deletes=True
    )
    observations: Mapped[list[Observation]] = relationship(
        back_populates="obj", cascade="all, delete-orphan", passive_deletes=True,
        order_by="Observation.observed_at",
    )


class ReferenceImage(Base):
    __tablename__ = "reference_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    object_id: Mapped[int] = mapped_column(ForeignKey("objects.id", ondelete="CASCADE"), index=True)
    original_path: Mapped[str] = mapped_column(String(500))
    processed_path: Mapped[str] = mapped_column(String(500))
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_sha256: Mapped[str] = mapped_column(String(64), index=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    capture_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    quality: Mapped[dict] = mapped_column(JSON, default=dict)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    obj: Mapped[PhysicalObject] = relationship(back_populates="images")
    fingerprints: Mapped[list[Fingerprint]] = relationship(
        back_populates="image", cascade="all, delete-orphan", passive_deletes=True
    )


class Fingerprint(Base):
    __tablename__ = "fingerprints"
    __table_args__ = (Index("ix_fp_image_extractor", "reference_image_id", "extractor"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    object_id: Mapped[int] = mapped_column(ForeignKey("objects.id", ondelete="CASCADE"), index=True)
    reference_image_id: Mapped[int] = mapped_column(
        ForeignKey("reference_images.id", ondelete="CASCADE"), index=True
    )
    extractor: Mapped[str] = mapped_column(String(50))
    feature_version: Mapped[str] = mapped_column(String(50))
    artifact_path: Mapped[str] = mapped_column(String(500))
    dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    preprocessing: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    obj: Mapped[PhysicalObject] = relationship(back_populates="fingerprints")
    image: Mapped[ReferenceImage] = relationship(back_populates="fingerprints")


class Identification(Base):
    __tablename__ = "identifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query_image_path: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    # Public id is stored (not a FK) so history survives object deletion.
    selected_object_public_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    best_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    candidates: Mapped[list] = mapped_column(JSON, default=list)
    decision_config: Mapped[dict] = mapped_column(JSON, default=dict)
    inference_ms: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(20), default="identify")


class Observation(Base):
    """A later image of a registered object, accepted into its history."""

    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    object_id: Mapped[int] = mapped_column(ForeignKey("objects.id", ondelete="CASCADE"), index=True)
    image_path: Mapped[str] = mapped_column(String(500))
    processed_path: Mapped[str] = mapped_column(String(500))
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    similarity: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30))
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    surface_changes: Mapped[dict] = mapped_column(JSON, default=dict)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    obj: Mapped[PhysicalObject] = relationship(back_populates="observations")


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_uid: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    experiment: Mapped[str] = mapped_column(String(50))
    dataset_id: Mapped[str] = mapped_column(String(200))
    split_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_paths: Mapped[dict] = mapped_column(JSON, default=dict)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

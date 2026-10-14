"""Persistence of reference images and fingerprints + an in-memory reference index.

Layout under ``<storage_dir>``::

    objects/<public_id>/images/<uuid>.<ext>          original upload (server-named)
    objects/<public_id>/images/<uuid>_processed.jpg  normalised image used for features
    objects/<public_id>/fingerprints/<image_id>/     orb.npz, global.npz, embedding_<backbone>.npz
    objects/<public_id>/observations/<uuid>...       later observations (history)
    queries/<yyyy-mm>/<uuid>.<ext>                   identification query images
"""

from __future__ import annotations

import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.orm import Fingerprint, PhysicalObject, ReferenceImage
from app.services.embedding_extractor import EmbeddingExtractor
from app.services.feature_extraction import FeatureConfig, FeatureSet, process_image
from app.services.identification import ReferenceEntry
from app.services.image_validation import ValidatedImage
from app.services.preprocessing import PreprocessConfig, PreprocessedImage

EXT_BY_FORMAT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp"}


def write_rgb_jpeg(path: Path, rgb: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode image.")
    path.write_bytes(buf.tobytes())


class FingerprintStore:
    def __init__(self, storage_dir: Path, prep_config: PreprocessConfig, feat_config: FeatureConfig,
                 embedder: EmbeddingExtractor | None):
        self.storage_dir = Path(storage_dir)
        self.prep_config = prep_config
        self.feat_config = feat_config
        self.embedder = embedder
        self._cache: list[ReferenceEntry] | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------ paths
    def object_dir(self, public_id: str) -> Path:
        return self.storage_dir / "objects" / public_id

    def rel(self, path: Path) -> str:
        return path.relative_to(self.storage_dir).as_posix()

    def abs(self, rel_path: str) -> Path:
        p = (self.storage_dir / rel_path).resolve()
        if not str(p).startswith(str(self.storage_dir.resolve())):
            raise ValueError("Path escapes storage directory.")
        return p

    @property
    def embedding_backbone(self) -> str | None:
        if self.embedder is None or not self.embedder.available:
            return None
        return self.embedder.backbone

    # ------------------------------------------------------------ processing
    def process(self, rgb: np.ndarray) -> tuple[PreprocessedImage, FeatureSet]:
        return process_image(rgb, self.prep_config, self.feat_config, self.embedder)

    def save_upload(self, folder: Path, img: ValidatedImage, prep: PreprocessedImage) -> tuple[Path, Path]:
        folder.mkdir(parents=True, exist_ok=True)
        stem = uuid.uuid4().hex
        original = folder / f"{stem}.{EXT_BY_FORMAT[img.format]}"
        original.write_bytes(img.raw_bytes)
        processed = folder / f"{stem}_processed.jpg"
        write_rgb_jpeg(processed, prep.rgb)
        return original, processed

    def add_reference(self, db: Session, obj: PhysicalObject, img: ValidatedImage,
                      capture_metadata: dict | None = None) -> ReferenceImage:
        prep, feats = self.process(img.rgb)
        odir = self.object_dir(obj.public_id)
        original, processed = self.save_upload(odir / "images", img, prep)
        ref = ReferenceImage(
            object_id=obj.id,
            original_path=self.rel(original),
            processed_path=self.rel(processed),
            original_filename=(img.original_filename or "")[:255] or None,
            content_sha256=img.sha256,
            width=img.width,
            height=img.height,
            capture_metadata=capture_metadata or {},
            quality=prep.quality,
        )
        db.add(ref)
        db.flush()
        paths = feats.save(odir / "fingerprints" / str(ref.id))
        versions = feats.extractor_versions()
        dims = {"orb": feats.orb.count, "lbp": len(feats.lbp), "gradient": len(feats.gradient),
                "color": len(feats.color), "shape": None if feats.shape is None else len(feats.shape)}
        for extractor, path in paths.items():
            summary: dict = {}
            if extractor == "orb":
                summary = {"keypoints": feats.orb.count, "descriptor": "ORB 256-bit (32 bytes)"}
            elif extractor.startswith("embedding:"):
                summary = {"norm": 1.0, "dimension": int(feats.embedding.shape[0])}
                dims[extractor] = int(feats.embedding.shape[0])
            elif extractor == "lbp":
                summary = {"scales": "P,R = (8,1), (16,2), (24,3)", "mapping": "riu2"}
            elif extractor == "color":
                summary = {"space": "HSV", "bins": "18 hue x 8 saturation", "value_channel": "excluded"}
            elif extractor == "gradient":
                summary = {"edge_density": round(float(feats.gradient[0]), 4)}
            elif extractor == "shape":
                summary = {"descriptor": "log Hu moments + aspect/extent/solidity/circularity"}
            db.add(Fingerprint(
                object_id=obj.id,
                reference_image_id=ref.id,
                extractor=extractor,
                feature_version=versions.get(extractor, ""),
                artifact_path=self.rel(path),
                dimension=dims.get(extractor),
                summary=summary,
                preprocessing={**prep.config, "feature_config": self.feat_config.as_dict(),
                               "mask_reliable": prep.mask_reliable},
            ))
        if feats.meta.get("embedding_message"):
            ref.quality = {**ref.quality, "embedding_message": feats.meta["embedding_message"]}
        self.invalidate()
        return ref

    def fingerprint_dir(self, public_id: str, image_id: int) -> Path:
        return self.object_dir(public_id) / "fingerprints" / str(image_id)

    # ------------------------------------------------------------ index
    def invalidate(self) -> None:
        with self._lock:
            self._cache = None

    def references(self, db: Session) -> list[ReferenceEntry]:
        with self._lock:
            if self._cache is not None:
                return self._cache
            backbone = self.embedding_backbone
            entries: list[ReferenceEntry] = []
            objs = db.scalars(select(PhysicalObject).options(selectinload(PhysicalObject.images))).all()
            for obj in objs:
                for img in obj.images:
                    fdir = self.fingerprint_dir(obj.public_id, img.id)
                    if not (fdir / "orb.npz").exists():
                        continue
                    feats = FeatureSet.load(fdir, backbone)
                    entries.append(ReferenceEntry(obj.id, obj.public_id, img.id, feats))
            self._cache = entries
            return entries

    def backfill_embeddings(self, db: Session) -> int:
        """Compute embeddings for references fingerprinted before weights were installed."""
        if self.embedding_backbone is None:
            return 0
        count = 0
        for img in db.scalars(select(ReferenceImage).options(selectinload(ReferenceImage.obj))).all():
            fdir = self.fingerprint_dir(img.obj.public_id, img.id)
            target = fdir / f"embedding_{self.embedding_backbone}.npz"
            if target.exists() or not (fdir / "global.npz").exists():
                continue
            rgb = cv2.cvtColor(cv2.imread(str(self.abs(img.processed_path))), cv2.COLOR_BGR2RGB)
            from app.services.feature_extraction import extract_features
            from app.services.preprocessing import preprocess
            # The processed image is already resized; reprocessing is idempotent w.r.t. size.
            feats = extract_features(preprocess(rgb, self.prep_config), self.feat_config, self.embedder)
            if feats.embedding is None:
                continue
            np.savez_compressed(target, embedding=feats.embedding, version=feats.embedding_version or "",
                                schema="odna-fingerprint-v1")
            db.add(Fingerprint(object_id=img.object_id, reference_image_id=img.id,
                               extractor=f"embedding:{self.embedding_backbone}",
                               feature_version=feats.embedding_version or "", artifact_path=self.rel(target),
                               dimension=int(feats.embedding.shape[0]), summary={"backfilled": True},
                               preprocessing=self.prep_config.as_dict(),
                               created_at=datetime.now(timezone.utc)))
            count += 1
        db.commit()
        self.invalidate()
        return count

    # ------------------------------------------------------------ deletion
    def delete_object_files(self, public_id: str) -> None:
        d = self.object_dir(public_id)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        self.invalidate()

    def delete_image_files(self, public_id: str, image: ReferenceImage) -> None:
        for rel in (image.original_path, image.processed_path):
            try:
                self.abs(rel).unlink(missing_ok=True)
            except ValueError:
                pass
        shutil.rmtree(self.fingerprint_dir(public_id, image.id), ignore_errors=True)
        self.invalidate()

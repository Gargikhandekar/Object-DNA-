# Architecture

```
┌──────────────────────────── frontend (React + Vite + TS + Tailwind) ───────────────────────────┐
│ pages/  Dashboard · Register · Identify · Compare · Library · ObjectDetail · History · Research │
│ components/  MatchViewer (SVG keypoint lines) · EvidenceBreakdown · ImageDropzone · Webcam …   │
│ services/api.ts  typed fetch client  ──  /api/* (Vite proxy → http://127.0.0.1:8000)           │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                             │ HTTP / multipart
┌──────────────────────────────── backend (FastAPI, backend/app) ─────────────────────────────────┐
│ api/        objects.py · identify.py · misc.py (images, history, dashboard, health, settings)    │
│             experiments.py (background worker)                                                   │
│ schemas/    Pydantic request/response models                                                     │
│ core/       config.py (ODNA_* env settings) · runtime.py (shared pipeline objects)               │
│ services/   image_validation → preprocessing → feature_extraction                                │
│               ├─ orb_matcher (ORB, ratio+mutual, RANSAC)                                         │
│               ├─ texture_features (LBP, gradients, HSV, Hu shape + similarities)                 │
│               └─ embedding_extractor (ResNet-50 / DINOv2, local weights only)                    │
│             similarity (pair evidence, normalisation, weighted fusion)                           │
│             identification (ranking, aggregation, decision rule)                                 │
│             fingerprint_store (files + DB rows + in-memory reference index)                      │
│             evaluation (verification / open-set identification metrics)                          │
│ research/   synthetic (procedural instances) · dataset (manifest, splits, pairs) · experiments  │
│ models/     SQLAlchemy ORM          database/  engine, sessions, init                            │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
          │                                  │                                     │
   data/object_dna.db (SQLite)     data/storage/ (images, .npz fingerprints)   experiments/results/<run>/
```

## Request flows

**Registration** (`POST /api/objects`): all files are validated first, so one bad file rejects the request
before anything is written. Duplicate content (SHA-256) within the request or against existing references
returns 409. Then, per image: preprocess → extract → save the original (server-generated name), the processed
image and the `.npz` artifacts → insert `reference_images` and `fingerprints` rows → commit. On failure the
transaction is rolled back and the object folder removed. The in-memory reference index is invalidated.

**Identification** (`POST /api/identify`): validate → preprocess/extract → load the cached reference index →
compare against every reference in a thread pool (OpenCV releases the GIL) → fuse → aggregate per object →
decide → store the processed query image and an `identifications` row → return candidates, evidence, ORB
correspondences for the top 3 and timings.

**Experiments** (`POST /api/experiments/run`): creates an `experiment_runs` row and submits the job to a
single-worker background executor, so long runs never block API requests and only one runs at a time.
Progress is written to the row and polled by the UI. Features are cached on disk in
`experiments/cache/<config-hash>/`. Runs left `running` by a server restart are marked failed at startup.

## Database

| Table | Key columns | Notes |
|---|---|---|
| `objects` | id, **public_id** (`ODNA-XXXXXXXX`, unique), name, category, description, created_at, fingerprint_status | Identity is the public ID, never the name |
| `reference_images` | id, object_id → objects (CASCADE), original_path, processed_path, original_filename, content_sha256 (indexed), width, height, capture_metadata (JSON), quality (JSON), uploaded_at | |
| `fingerprints` | id, object_id (CASCADE), reference_image_id (CASCADE), extractor, feature_version, artifact_path, dimension, summary, preprocessing, created_at | Index on (image, extractor). Arrays live in `.npz` files |
| `identifications` | id, query_image_path, created_at, status, selected_object_public_id, best_score, candidates (JSON), decision_config, inference_ms | Stores the public ID (not an FK), so the audit log survives object deletion |
| `observations` | id, object_id (CASCADE), image_path, processed_path, observed_at, similarity, status, evidence, surface_changes, note | Object history |
| `experiment_runs` | id, run_uid, experiment, dataset_id, split_id, config, status, progress, metrics (JSON), artifact_paths, is_synthetic, created_at, finished_at | |

SQLite foreign keys are enabled (`PRAGMA foreign_keys=ON`) and WAL journaling is used. Deleting an object
cascades in the database first, and files are removed only after the commit succeeds.

## Security measures

* Uploaded files are never executed. Extension, declared MIME type, magic bytes, full decode and pixel count
  are all checked, and size is limited (`ODNA_MAX_UPLOAD_MB`).
* Stored filenames are server-generated UUIDs. Served files are resolved by database ID, and paths are checked
  to stay inside the storage directory.
* Dataset manifests are checked for path traversal.
* There are no external or paid APIs, and model weights are only fetched by an explicit script.

## Known engineering limitations

* Registration and identification are synchronous within their requests (run in FastAPI's threadpool).
  Registration time grows linearly with the number of images (about 0.2–0.4 s per image on CPU without the
  deep model).
* Identification compares against every reference image (O(N)). ORB shortlisting
  (`ODNA_ORB_SHORTLIST`) is available for larger galleries. An ANN index (e.g. FAISS) for embeddings would be
  the next step.
* There is no authentication: the system is designed for local, single-user use.
* The experiment worker runs in-process. A server restart interrupts running experiments.

# 🧬 Object DNA — Visual Fingerprinting for Instance-Level Object Recognition

Object DNA identifies **individual physical objects**, not object categories. Given a new photo, it decides
whether it shows a *specific* object that was registered before (*this* bottle, not *a* bottle), or reports
that the evidence is insufficient.

It builds a multi-part visual fingerprint for every reference image: ORB keypoints with RANSAC geometric
verification, LBP texture, HSV colour, contour shape, and optional deep embeddings (ResNet-50 / DINOv2). It
compares query images against the registered gallery, ranks candidate instances, and makes an open-set
decision: **MATCH FOUND**, **UNCERTAIN** or **NO RELIABLE MATCH**. A research module measures how classical,
deep and hybrid approaches compare on labelled data with leakage-safe splits.

> **Honest scope.** Not every physical object has a recoverable unique fingerprint. Identification depends on
> visible distinguishing detail (wear, scratches, print offsets, stickers), image quality, viewpoint and
> lighting. Scores are similarity scores, **not calibrated probabilities**. Any accuracy figure is valid only for
> the dataset it was measured on.

---

## Contents
1. [Problem statement](#1-problem-statement)
2. [Architecture](#2-architecture)
3. [Installation (Windows)](#3-installation-windows) · [macOS/Linux](#macoslinux)
4. [Running the app](#4-running-the-app)
5. [Model weights](#5-model-weights)
6. [Using the app](#6-using-the-app)
7. [Collecting a dataset](#7-collecting-a-dataset)
8. [Running experiments](#8-running-experiments)
9. [Interpreting the metrics](#9-interpreting-the-metrics)
10. [Tests](#10-tests)
11. [Known limitations](#11-known-limitations)
12. [Future improvements](#12-future-improvements)

Further documentation: [docs/architecture.md](docs/architecture.md) · [docs/methodology.md](docs/methodology.md) ·
[docs/evaluation.md](docs/evaluation.md) · [data/README.md](data/README.md)

---

## 1. Problem statement

Image classification answers *"what kind of object is this?"*. Two bottles with the same brand, colour, shape
and model get the same answer. Object DNA asks a different question: ***is this the same physical object I
registered before?*** That requires distinguishing instances of the **same** model, using fine evidence that
differs between instances, such as surface texture, scratches, dents, wear, print misregistration and small
manufacturing variations, under changes in viewpoint, distance, lighting and background.

**Research question:** Can a computer vision system identify an exact physical object from images taken at
different times, angles, distances and lighting, and distinguish it from other objects of the same category?

## 2. Architecture

```
React/Vite/TS UI ──/api──▶ FastAPI ──▶ services: validation → preprocessing → feature extraction
                                          (ORB · LBP · HSV · Hu shape · ResNet-50/DINOv2)
                                       → pair evidence → normalised weighted fusion
                                       → per-object aggregation → MATCH / UNCERTAIN / NO MATCH
                           SQLite (SQLAlchemy) + data/storage/*.npz fingerprints
                           research/: synthetic data · manifest · identity-disjoint splits · experiments
```

```
object-dna/
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI app (OpenAPI at /docs)
│   │   ├── api/                 routes: objects, identify/compare, history/dashboard/health, experiments
│   │   ├── core/                settings (ODNA_* env vars), runtime singletons
│   │   ├── models/              SQLAlchemy ORM
│   │   ├── schemas/             Pydantic models
│   │   ├── database/            engine/session/init
│   │   ├── services/            preprocessing.py, feature_extraction.py, orb_matcher.py,
│   │   │                        texture_features.py, embedding_extractor.py, similarity.py,
│   │   │                        identification.py, fingerprint_store.py, evaluation.py, image_validation.py
│   │   └── research/            synthetic.py, dataset.py, experiments.py
│   ├── tests/                   pytest suite (58 tests, no network needed)
│   └── requirements.txt
├── frontend/                    React + Vite + TypeScript + Tailwind + Recharts + Lucide
├── scripts/                     init_db, download_weights, import_dataset, generate_synthetic_dataset,
│                                run_experiment, seed_demo
├── data/                        database, storage, datasets, model weights (git-ignored)
├── experiments/                 configs/ (active fusion), results/<run>/ (metrics, CSV, charts)
└── docs/                        architecture.md, methodology.md, evaluation.md
```

Details: [docs/architecture.md](docs/architecture.md).

## 3. Installation (Windows)

Prerequisites: **Python 3.11+** ([python.org](https://www.python.org/downloads/), tick *"Add python.exe to
PATH"*), **Node.js 18+** ([nodejs.org](https://nodejs.org/)), and Git.

Open **PowerShell** in the repository folder:

```powershell
# 1. Python virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 2. Backend dependencies (CPU-only PyTorch wheels are smaller and sufficient)
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend\requirements.txt

# 3. Database + folders
python scripts\init_db.py

# 4. (Recommended) pretrained ResNet-50 weights, ~100 MB, see section 5
python scripts\download_weights.py --backbone resnet50

# 5. Frontend dependencies
cd frontend
npm install
cd ..
```

Optional: copy `.env.example` to `.env` to change settings (thresholds, backbone, upload limits).

### macOS/Linux
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
python scripts/init_db.py
python scripts/download_weights.py --backbone resnet50
(cd frontend && npm install)
```

## 4. Running the app

Two terminals, both from the repository root with the virtual environment activated:

```powershell
# Terminal 1 - backend API on http://127.0.0.1:8000  (OpenAPI docs: http://127.0.0.1:8000/docs)
cd backend
uvicorn app.main:app --port 8000

# Terminal 2 - frontend on http://localhost:5173 (proxies /api to the backend)
cd frontend
npm run dev
```

Open **http://localhost:5173**. The sidebar shows API connectivity and whether the deep encoder is available.

Quick demo without a camera: `python scripts\seed_demo.py` generates the synthetic dataset and registers its
reference images, so you can try *Identify* using images from
`data\datasets\synthetic_demo\objects\<id>\query\`. Production build: `cd frontend && npm run build && npm run preview`.

## 5. Model weights

Weights are **never downloaded silently**. Without them, the app runs with classical features only. The UI
shows *"Deep encoder unavailable"*, and the deep component is excluded from fusion rather than treated as zero.

| Backbone | Command | Size | Licence |
|---|---|---|---|
| ResNet-50 (ImageNet-1k, torchvision V2) — default | `python scripts/download_weights.py --backbone resnet50` | ~100 MB | BSD-3 |
| DINOv2 ViT-S/14 (self-supervised) | `python scripts/download_weights.py --backbone dinov2_vits14` then set `ODNA_EMBEDDING_BACKBONE=dinov2_vits14` | ~90 MB + repo source | Apache-2.0 |

Files go to `data/models/`. Restart the backend afterwards. Objects registered **before** the weights were
installed have no embeddings: press *Backfill missing embeddings* on the Overview page (or
`POST /api/settings/backfill-embeddings`). Set `ODNA_EMBEDDING_BACKBONE=none` to disable deep features
entirely.

## 6. Using the app

| Page | What it does |
|---|---|
| **Overview** | Usage counts (unlabelled), recent attempts, decision outcomes, model status, and summaries of completed experiments. Usage counts are **not** accuracy. |
| **Register Object** | Name, category, description and multiple reference images (drag-and-drop or webcam) with previews. Generates the fingerprint and assigns a unique `ODNA-XXXXXXXX` ID. Identical files are rejected (409); identical *names* are allowed because identity is never based on the name. |
| **Identify Object** | Upload or webcam → processing stages → status banner → ranked candidates with similarity scores → per-candidate evidence: interactive ORB correspondence view (inliers vs outliers), additive score breakdown, per-reference scores. Accepted matches can be added to the object's history. |
| **Compare Images** | Same-instance verification of two images: decision, deep cosine, ORB inliers/matches, texture and colour similarity, raw pixel similarity (to show why pixels are not enough), and the keypoint visualisation. |
| **Object Library** | Search and category filter. Each object page shows all references with quality data and fingerprint records (extractor, version, dimension, preprocessing), lets you add or remove reference images, and deletes the object after confirmation (cascading to images, fingerprints, observations and files). |
| **Object History** | Per-object timeline: registration, accepted observations, similarity over time and measured surface statistics. You can submit a later image; it is accepted only if this object ranks first with MATCH FOUND. Also has the global identification-attempt log. |
| **Research Lab** | Run experiments, inspect the method comparison, ROC curves, confusion matrix, robustness by condition, split details and tuned fusion weights; export CSV/JSON/PNG; apply the validation-tuned fusion to the app. |

**Registering well:** use 3–8 images of the same object from different angles and distances, include close-ups
of distinctive marks, vary lighting, and keep the object large and in focus.

## 7. Collecting a dataset

Each folder is **one physical instance**, not a product category. Put several instances of the same model in
the dataset, otherwise the task degenerates into category recognition.

```
data/datasets/my_pilot/objects/
  object_001/
    object.json                     {"name": "Bottle A", "category": "bottle", "model": "brandX-500ml", "registered": true}
    reference/s1__reference__01.jpg
    query/s2__lighting__01.jpg      <session>__<condition>__<n>.jpg
    query/s3__viewpoint_large__01.jpg
  object_002/ ...
  object_031/ object.json -> "registered": false   (never enrolled: tests unknown-object rejection)
```

* **Pilot target:** 15–30 physical objects with ≥ 2 instances per model, 3–5 reference images (session `s1`),
  and query images from **later** sessions (`s2`, `s3`, …) on different days if possible. This is a starting
  point, not evidence of reliability.
* **Conditions** (used for robustness charts): `same`, `viewpoint_moderate`, `viewpoint_large`, `lighting`,
  `background`, `distance`, `occlusion`, `blur_lowres`. Unregistered objects are evaluated automatically.
* Include distinctive objects, objects with repeated logos and printed patterns, near-identical instances,
  objects with subtle differences, and objects with no obvious marks.
* Build the manifest (and check for near-duplicates):
  `python scripts/import_dataset.py data/datasets/my_pilot`. Add `--register` to also enrol the references in
  the app.

See [data/README.md](data/README.md) for the full capture protocol and leakage rules.

## 8. Running experiments

From the **Research Lab** (choose a dataset, then *Run*) or from the command line:

```powershell
python scripts\generate_synthetic_dataset.py           # optional: procedural sanity-check data
python scripts\run_experiment.py --dataset synthetic_demo
python scripts\run_experiment.py --dataset my_pilot --experiment identification --seed 7
```

Each run evaluates `orb`, `texture`, `color`, `deep` (if weights are installed), `hybrid_default` and
`hybrid_tuned`, and writes `experiments/results/<run>/` containing `metrics.json`, `summary.csv`,
`split.json`, `verification_pairs.csv`, `identification_queries.csv`, `tuned_fusion.json`,
`roc_verification.png`, `robustness_top1.png` and `confusion.png`. Runs appear in the Research Lab.

## 9. Interpreting the metrics

* **Verification (pairs):** *ROC-AUC* (threshold-free separability), *EER*, and at the validation-selected
  threshold: precision, recall, F1, **FMR** (false match rate: different objects accepted as the same) and
  **FNMR** (false non-match rate: same object rejected). For ORB the score is an inlier count, so its threshold
  is a count.
* **Identification (1:N, open set):** *Top-1/Top-5* is closed-set rank accuracy for enrolled objects (Top-5 only
  when the gallery has ≥ 5 objects). *Accepted precision* = correct accepts / all accepts. *Accepted recall* =
  correct accepts / enrolled queries. *Unknown rejection* = share of unregistered-object queries that were not
  accepted. *Open-set accuracy* = (correct accepts + rejected unknowns) / all queries.
* **Latency:** per-query CPU matching time against the gallery, plus per-image extraction time.
* Small test sets produce large uncertainty. With 25 test queries, one error moves accuracy by 4 points.
* Results on the **synthetic** dataset only show that the pipeline works. They are always flagged in the UI.
* App usage statistics (Overview) are never used to compute accuracy.

## 10. Tests

```powershell
cd backend
python -m pytest -q          # 58 tests: validation, features, persistence, fusion, rejection,
                             # API (register/duplicates/identify/compare/delete cascade/history), metrics,
                             # split integrity, end-to-end experiment run
cd ..\frontend
npm test                     # API client tests (vitest)
npm run build                # type-check + production build
```

No test downloads models or needs the internet. Embedding tests use an explicitly labelled
*random-init* network to check shapes and normalisation only.

Versions used for the verified run: Python 3.13.16 and 3.11.17, numpy 2.2.6, opencv-python-headless 5.0.0,
torch 2.14.1, torchvision 0.29.1, scikit-learn 1.9.1, FastAPI 0.143.1, SQLAlchemy 2.1.4, Node 22, Vite 5.4,
React 18.3.

## 11. Known limitations

* **Weights not bundled.** The deep component needs a one-off download. In the development environment used to
  build this project, the weight hosts were unreachable, so **no deep-embedding results have been measured yet**.
  The Research Lab reports the deep method as *unavailable* rather than inventing numbers.
* **Synthetic data is not evidence.** The bundled generator renders printed patterns with procedural wear marks.
  It is far simpler than real objects (no specular reflections, no 3-D self-occlusion, no camera noise model).
* **Same-model siblings are hard.** In the synthetic run, shared print designs produce many valid ORB inliers
  between *different* instances of the same model, so unregistered siblings are often marked UNCERTAIN or even
  accepted. This is the core difficulty of instance recognition and needs real data to quantify.
* **Featureless, transparent, reflective or deformable objects** give few stable keypoints. Identification may be
  impossible from images alone.
* GrabCut foreground masks fail on cluttered or low-contrast scenes; the system then falls back to the whole
  image, and background keypoints can contribute evidence.
* Linear search over all references, single-user and no authentication, and in-process background experiments
  (see [docs/architecture.md](docs/architecture.md)).
* Scores are uncalibrated. Thresholds must be re-selected (Research Lab → *Use in application*) when the data
  or features change.

## 12. Future improvements

* Measure ResNet-50 vs DINOv2 embeddings on a real pilot dataset; add local-feature learners (SuperPoint/LightGlue,
  DISK) for stronger correspondences.
* Supervised metric learning (triplet/ArcFace) on instance identities, with identity-disjoint train/test splits.
* Score calibration (Platt/isotonic on validation) to report calibrated match probabilities.
* An approximate nearest-neighbour index (FAISS) for large galleries; GPU inference; a job queue for experiments.
* Better segmentation (e.g. SAM), multi-view enrolment summaries, change detection that controls for pose and
  lighting.
* Optional Docker Compose packaging.

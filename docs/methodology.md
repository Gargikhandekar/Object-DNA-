# Methodology

Object DNA addresses **instance-level** recognition: deciding whether a photo shows a *specific physical
object* that was enrolled before, not merely an object of the same category or product model. Two bottles of
the same brand and model are two different identities.

The system does **not** assume that every object has a recoverable unique fingerprint. Whether two instances
can be told apart depends on visible distinguishing detail (scratches, wear, print offsets, dents, stickers),
image quality, viewpoint and lighting. The pipeline therefore always allows an *uncertain* or *no reliable match*
outcome, and performance is treated as an experimental question.

## 1. Preprocessing (`backend/app/services/preprocessing.py`)

Every image (reference, query, experiment) is processed identically:

1. Decode with Pillow, apply EXIF orientation, convert to RGB (after strict validation of type, magic bytes,
   decodability, size and pixel count).
2. Resize so the longest side is 640 px (area interpolation).
3. **Foreground mask** with GrabCut initialised from a centred rectangle (6 % border) on a 256 px copy, followed
   by morphological cleaning and largest-component selection. The mask is accepted as *reliable* only if the
   component covers 5–92 % of the frame, holds ≥ 70 % of the foreground area, and does not run into the
   initialisation rectangle on two or more sides (which indicates a frame-filling object); otherwise the whole
   image is used. Optional cropping to the mask's bounding box is configurable (off by default).
4. Grayscale + **CLAHE** (clip 2.0, 8×8 tiles) as input to ORB and LBP, reducing the influence of global
   illumination.
5. Quality statistics (Laplacian variance, brightness, contrast) are stored and surfaced as warnings.

## 2. Fingerprint representation (`feature_extraction.py`, `orb_matcher.py`, `texture_features.py`, `embedding_extractor.py`)

One fingerprint is computed **per reference image** (not one per object), so evidence from different views is
kept separate and aggregated at matching time.

| Component | Descriptor | Size | Purpose |
|---|---|---|---|
| Local keypoints | ORB (FAST + rotated BRIEF), 1 500 keypoints, 8 pyramid levels, restricted to the reliable mask widened by 25 % of the image size (GrabCut tends to trim frame-filling objects) | N × 32 bytes + N × 6 floats | Distinctive local patches: scratches, print details, wear |
| Texture | Rotation-invariant uniform LBP (riu2) at (P,R) = (8,1), (16,2), (24,3) with bilinear sampling; per-scale L1-normalised histograms | 10 + 18 + 26 = 54 | Surface micro-texture |
| Gradients | Canny edge density, mean gradient magnitude, 16-bin unsigned orientation histogram | 18 | Auxiliary evidence, used to describe surface changes |
| Colour | H × S histogram (18 × 8 bins) in HSV; V excluded for lighting robustness | 144 | Colour distribution |
| Shape | log-Hu moments + aspect ratio, extent, solidity, circularity of the largest contour (only with a reliable mask) | 11 | Silhouette |
| Deep embedding | ResNet-50 (ImageNet, GAP 2048-d) or DINOv2 ViT-S/14 (CLS, 384-d), on the mask bounding-box crop, L2-normalised | 2048 / 384 | Global appearance |

Artifacts are stored as versioned, compressed `.npz` files (`orb.npz`, `global.npz`,
`embedding_<backbone>.npz`). The `fingerprints` table records the extractor, its version string
(`orb-v1`, `lbp-riu2-v1`, `torchvision-resnet50-imagenet1k-v2`, …), the dimension, a summary and the full
preprocessing configuration. Changing an extractor therefore never silently mixes incompatible vectors.

Pretrained weights are never downloaded at runtime. Without them the deep component is reported as
*unavailable* and excluded (not zero-filled). Generic ImageNet or self-supervised embeddings are **not
assumed** to separate visually identical instances; whether they help is measured in Experiment 3.

## 3. Matching (`similarity.py`)

For a query image *q* and a reference image *r* the following raw evidence is computed:

| Component | Raw value | Range |
|---|---|---|
| deep | cosine similarity of L2-normalised embeddings | [−1, 1] |
| geometric | ORB matches surviving Lowe's ratio test (0.75) **in both directions** (mutual), then RANSAC: homography (reprojection threshold 6 px, plausibility checks on determinant and perspective terms), falling back to a 4-DoF partial affine model; a model supported by < 4 inliers counts as 0 | inlier count |
| texture | mean per-scale LBP histogram intersection | [0, 1] |
| color | 1 − Bhattacharyya distance of H-S histograms; *unavailable* if either image is near-greyscale (≥ 90 % of mass at saturation < 32) | [0, 1] |
| shape | exp(−(mean \|Δlog-Hu\|/3 + 2·mean \|Δgeometry\|)) | (0, 1] |

### Normalisation and fusion

Each component is mapped to [0, 1] with a clipped linear ramp

    s_i = clip((raw_i − lo_i) / (hi_i − lo_i), 0, 1)

and fused with a **weighted mean over the components that are available for this pair**:

    score = Σ_i w_i s_i / Σ_i w_i      (i ∈ available, w_i > 0)

Missing evidence (no embedding weights, no reliable mask) is excluded and the remaining weights are
renormalised. The contribution `w_i s_i / Σ w` of every component is reported, and the contributions sum to
the score exactly, which makes the evidence breakdown in the UI additive and auditable.

Two configurations exist:

* **default-heuristic**: weights deep 0.40, geometric 0.50, texture 0.05, colour 0.05, shape 0; ranges
  geometric 8–50 inliers (real photographs give tens of inliers) and deep cosine 0.25–0.90 (informed by a
  validation fit on the Objectron subset with ResNet-50: lo 0.086, hi 0.942, fitted weights deep 0.4 / geometric 0.6).
  It is documented as untuned; see `experiments/configs/default_fusion.json`.
* **validation-tuned**: produced by the Research Lab. `lo_i` is the 5th percentile of *different-instance*
  validation pairs and `hi_i` the 95th percentile of *same-instance* validation pairs; the weights are a
  non-negative simplex grid search (step 0.1) that maximises validation ROC-AUC. It can be applied to the app
  with one click (it writes `experiments/configs/active_fusion.json`).

The fused value is a **similarity score**, not a probability. It has not been calibrated, and the UI and API
never label it as one.

## 4. Identification and decision (`identification.py`)

1. Compute the fused score of the query against every reference image (optionally ORB is run only on the N
   best objects by global evidence: `ODNA_ORB_SHORTLIST`).
2. **Aggregate per object**: object score = maximum over its reference images (the best-matching view). The
   number of references above the uncertain threshold is reported as supporting evidence.
3. Rank objects and apply the decision rule shared by the app and the experiments:

| Status | Rule |
|---|---|
| MATCH FOUND | best ≥ accept **and** (best − second best object) ≥ margin |
| UNCERTAIN | best ≥ uncertain, but the accept or margin condition fails |
| NO RELIABLE MATCH | best < uncertain (or nothing to compare) |

The margin rule prevents forced decisions between near-identical candidates, such as two instances of the
same model. Thresholds are configurable from the API or the environment, and the Research Lab can replace them
with validation-selected values (accept = validation open-set optimum, uncertain = accept − 0.10).

## 5. Object history

A later image is accepted into an object's history only if, in a full identification against **all**
registered objects, that object ranks first **and** the status is MATCH FOUND. A user can explicitly confirm an
UNCERTAIN case; it is then stored as `USER_CONFIRMED_UNCERTAIN`. For each observation the app records
descriptive measurements against the references: texture and colour similarity, edge-density change,
sharpness, brightness and keypoint count. These are *not* interpreted as damage or ageing, because
perspective, illumination, camera and preprocessing differences produce the same effects.

## 6. Threshold selection and evaluation design

See [evaluation.md](evaluation.md). In short, the splits are identity-disjoint between validation and test. All
tuning (fusion weights, normalisation ranges, verification and identification thresholds) uses validation data
only. Reported numbers come from the held-out test identities and are computed by the code; none are entered
by hand.

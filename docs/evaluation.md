# Evaluation design and results

## Experiments implemented (`backend/app/research/experiments.py`)

| # | Experiment | Method key | Score used |
|---|---|---|---|
| 1 | Classical local features | `orb` | RANSAC inlier count after ratio test + mutual check |
| 2 | Texture | `texture` | multi-scale LBP histogram intersection |
| – | Colour baseline | `color` | 1 − Bhattacharyya distance (H-S histogram) |
| 3 | Deep embeddings | `deep` | cosine similarity (only when weights are installed) |
| 4 | Hybrid Object DNA | `hybrid_default`, `hybrid_tuned` | normalised weighted fusion (default vs validation-tuned) |
| 5 | Same-instance verification | all | pair scores → ROC-AUC, EER, P/R/F1, FMR, FNMR |
| 6 | Instance identification (open set) | all | per-object max score → Top-1/5, accepted precision/recall, unknown rejection, latency |
| + | Robustness | all | Experiment 6 metrics per acquisition condition, plus synthetic perturbations |

All methods use the **same preprocessing, the same cached features and the same splits**.

## Protocol

1. **Identity-disjoint split** (seeded, stratified by product model): about 40 % of physical objects go to
   validation and 60 % to test. Every image of an object is in exactly one split, so sessions and
   near-duplicates cannot leak between tuning and testing.
2. **Gallery** = reference images of registered objects. Unregistered objects (`registered=false`) are never
   enrolled; their images act as unknown queries.
3. **Leakage checks**: queries from the same capture session as their own gallery images are dropped when
   cross-session images exist; near-duplicates (dHash Hamming ≤ 4 bits) of gallery images are dropped from
   queries and from positive pairs. The counts are written to `split.json`.
4. **Verification pairs** (within each split): positives are different images of the same object from
   different sessions. Negatives are balanced, half *hard* (same model, different instance) and half random.
5. **Tuning on validation only**: hybrid normalisation ranges (5th percentile of negatives, 95th percentile of
   positives), hybrid weights (simplex grid, step 0.1, maximising ROC-AUC), the verification threshold (max F1)
   and the identification threshold (max open-set accuracy).
6. **Testing**: the frozen thresholds and weights are applied to the test identities. Every number shown in the
   app comes from `metrics.json` produced by the run. Conditions that are not present are reported as
   **pending**, and methods whose dependencies are missing (deep weights) are reported as **unavailable**.

Generalisation measured: validation and test contain **different physical identities**, so the results describe
performance on identities that were not used for tuning. Nothing is trained on test identities. No metric
learning is used yet; if it is added, its training identities must be disjoint from the test identities.

## Results on the synthetic sanity-check dataset

> ⚠️ **Synthetic, procedurally generated data**, produced with `scripts/generate_synthetic_dataset.py` (seed 7):
> 5 product models × 4 registered instances + 1 unregistered instance each = 25 objects and 360 images.
> Instances of a model share the printed design and differ only in procedural wear marks. **These numbers show
> that the pipeline and evaluation code behave sensibly. They are not evidence of real-world performance.**
> Deep embeddings: **not evaluated**, because the pretrained weights could not be downloaded in the build
> environment.

Run `20261014-161551-be53e1` (CLI, seed 42, validation fraction 0.4, pipeline `prep-v2`). Split: 10
validation / 15 test identities; gallery 60 images; 119 validation and 177 test queries (153 enrolled + 24
unknown); 4 queries excluded as near-duplicates of gallery images. Test pairs: 940 positive / 940 negative
(470 same-model hard negatives).

| Method | Top-1 | Top-5 | Acc. precision | Acc. recall | Unknown rejection | Open-set acc. | Ver. AUC | EER | FMR | FNMR |
|---|---|---|---|---|---|---|---|---|---|---|
| ORB + RANSAC | 0.993 | 1.000 | 0.910 | 0.856 | 0.458 | 0.802 | 0.906 | 0.182 | 0.224 | 0.146 |
| LBP texture | 0.255 | 0.556 | 1.000 | 0.007 | 1.000 | 0.141 | 0.605 | 0.437 | 1.000 | 0.000 |
| HSV colour | 0.503 | 0.987 | 0.507 | 0.464 | 0.167 | 0.424 | 0.742 | 0.343 | 0.545 | 0.028 |
| Hybrid (default weights) | 0.523 | 1.000 | 0.452 | 0.523 | 0.000 | 0.452 | 0.817 | 0.258 | 0.449 | 0.017 |
| Hybrid (validation-tuned) | 0.993 | 1.000 | 0.910 | 0.856 | 0.458 | 0.802 | 0.906 | 0.182 | 0.224 | 0.146 |
| Deep embedding | unavailable | | | | | | | | | |

Validation-tuned fusion: geometric weight 1.0 (all other components 0), geometric range 0–460 inliers
(validation ROC-AUC 0.916). On this data the tuned hybrid is therefore identical to ORB alone.

### Observations (synthetic data only)

* **Local geometric evidence dominates.** ORB + RANSAC ranks the true instance first for 99 % of enrolled test
  queries, across all synthetic conditions. The generator renders flat printed surfaces, which suit a homography
  model better than real 3-D objects do.
* **Unknown siblings are the failure mode.** Unregistered instances of an enrolled model share its print and
  produce many correct but non-identifying inliers, so only 46 % are rejected. This is the instance-vs-category
  problem the project targets.
* **The default fusion is calibrated for real photos, not for this data.** Its geometric range (8–50 inliers,
  see below) saturates on synthetic printed patterns, where even *different* same-model instances reach 50+
  inliers. Siblings then tie, giving Top-1 0.52 and 0 % unknown rejection. Validation tuning stretches the
  range to 0–460 and recovers ORB-level performance. **Always tune on data that resembles your deployment, then
  press *Use in application*.**
* **Global texture and colour descriptors do not identify instances.** LBP histograms are almost identical across
  siblings, and colour separates models, not instances.
* **Latency.** Matching cost is dominated by ORB, at about 1.8 s CPU time per query against 60 references (summed
  per-thread CPU time). Global descriptors cost about 1–3 ms. The app parallelises reference matching.

History: an earlier run with pipeline `prep-v1` (default geometric range 8–150, tighter keypoint mask) gave
hybrid-default Top-1 0.869 on this data. The defaults were changed after testing on real photographs (next
section), where the old range under-scored real matches.

## Real-photo demo (OpenCV sample images)

`python scripts/download_real_samples.py --register` downloads 5 real objects/scenes from OpenCV's
`samples/data`, each photographed twice (cookie box alone vs. in a cluttered pile, aloe plant stereo pair,
rubber-whale and basketball video frames, graffiti panel at a large viewpoint change), plus 2 unregistered
images (apple, orange). With the default fusion and no deep model:

| Query | Status | Top candidate | Score | RANSAC inliers |
|---|---|---|---|---|
| Cookie box in cluttered scene | MATCH FOUND | cookie box ✔ | 0.604 | 39 |
| Aloe (other stereo view) | MATCH FOUND | aloe ✔ | 1.000 | 216 |
| Rubber whale (next frame) | MATCH FOUND | rubber whale ✔ | 1.000 | 902 |
| Graffiti (large viewpoint change) | MATCH FOUND | graffiti ✔ | 0.802 | 39 |
| Basketball (next frame) | MATCH FOUND | basketball ✔ | 1.000 | 661 |
| Apple (never registered) | NO RELIABLE MATCH | orange | 0.159 | 0 |

This is a **5-object smoke test, not a benchmark**. Two queries are consecutive video frames (flagged as
near-duplicates by the import script), and none of the objects has a same-model sibling. It does show three
issues found and fixed on real photos:

1. GrabCut trimmed part of a frame-filling object, which cut the inliers from 40 to 10. The keypoint mask is now
   widened by 25 % of the image size.
2. Greyscale images produced identical colour histograms (a spurious "perfect" colour match). Colour evidence is
   now marked unavailable when either image is near-greyscale.
3. Real matches give tens of inliers rather than hundreds, so the default geometric range is now 8–50.

## Results on real data

### Objectron subset (real phone videos, 9 categories)

`python scripts/download_objectron.py` (seed 0, ≤ 200 MB): 16 physical objects from Google Objectron's test
index, covering bike ×2, book, bottle ×3, camera, cereal box ×2, chair ×2, cup ×3, laptop and shoe. The 14
registered objects have 3 reference frames each (0 %, 35 %, 70 % of the walk-around). Query frames are +5 %,
+12 % and +22 % of the clip after each reference. One unregistered bottle and one unregistered cup act as
unknown objects. Negative pairs include different physical objects of the same category.

Run `20261014-181236-f7449e` (default pipeline, no deep model): 7 validation / 9 test identities; gallery 42
frames; 80 test queries (56 enrolled + 24 unknown); 13 queries excluded as near-duplicates of gallery frames.
Test pairs: 182 positive / 182 negative (91 same-category).

| Method | Top-1 | Top-5 | Acc. precision | Acc. recall | Unknown rejection | Open-set acc. | Ver. AUC | FMR | FNMR |
|---|---|---|---|---|---|---|---|---|---|
| ORB + RANSAC | 1.000 | 1.000 | 0.982 | 1.000 | 0.958 | 0.988 | 0.874 | 0.000 | 0.253 |
| LBP texture | 0.607 | 0.911 | 0.436 | 0.607 | 0.083 | 0.450 | 0.746 | 0.467 | 0.236 |
| HSV colour | 0.929 | 0.946 | 0.729 | 0.911 | 0.375 | 0.750 | 0.916 | 0.220 | 0.181 |
| Hybrid (default) | 1.000 | 1.000 | 0.933 | 1.000 | 0.833 | 0.950 | 0.900 | 0.066 | 0.154 |
| Hybrid (validation-tuned: geometric 0.9, colour 0.1) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.960 | 0.011 | 0.137 |
| Deep embedding | unavailable | | | | | | | | |

**Read with caution:** there are only 9 test
identities and 80 queries, so a single error changes accuracy by 1–2 points. Each object comes from **one
video**, so queries share lighting and background with their references and differ mainly in viewpoint.
Verification is harder than identification here: FNMR 0.14–0.25 means many same-object pairs far apart in
the walk-around fall below the threshold. In the app with **default** thresholds, a chair seen from a large
viewpoint change ranked first but scored only 0.28 (NO RELIABLE MATCH). Applying the validation-tuned
fusion fixes such cases on this data.

#### Same Objectron subset with ResNet-50 embeddings (run on a user machine)

Run `20261014-184931-fcba36` was produced by the project author's local install with ResNet-50 weights, on the
same 16-object subset and split. These are the values read from the Research Lab:

| Method | Top-1 | Top-5 | Acc. precision | Acc. recall | Unknown rejection | Open-set acc. | Ver. AUC | EER | FMR | FNMR |
|---|---|---|---|---|---|---|---|---|---|---|
| ORB + RANSAC | 1.000 | 1.000 | 0.982 | 1.000 | 0.958 | 0.988 | 0.871 | 0.129 | 0.000 | 0.258 |
| LBP texture | 0.607 | 0.911 | 0.436 | 0.607 | 0.083 | 0.450 | 0.746 | 0.302 | 0.467 | 0.236 |
| HSV colour | 0.929 | 0.946 | 0.729 | 0.911 | 0.375 | 0.750 | 0.916 | 0.179 | 0.220 | 0.181 |
| **Deep embedding (ResNet-50)** | 1.000 | 1.000 | 0.718 | 1.000 | **0.083** | 0.725 | **1.000** | 0.000 | 0.225 | 0.000 |
| **Hybrid (default)** | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | **1.000** | 0.008 | 0.011 | 0.000 |
| Hybrid (validation-tuned: deep 0.4, geometric 0.6) | 1.000 | — | — | — | 0.708 (17/24) | — | 0.999 (val) | — | — | — |

* **Deep embeddings rank perfectly but cannot reject unknowns alone.** Verification AUC is 1.000, but only
  8 % of unregistered-object queries are rejected. Generic ImageNet features make every real photo look
  somewhat similar, so a single threshold cannot separate a *different* bottle from a registered one.
* **The hybrid works best on this subset.** Deep similarity finds the right candidate, and RANSAC-verified
  local matches confirm it is the same physical object, giving 100 % accuracy and 100 % unknown rejection on
  the test split.
* **Validation tuning overfit the tiny validation split.** With only 7 validation identities, the tuned
  thresholds were more permissive. In its open-set confusion matrix, 7 of 24 unknown queries were accepted:
  5 as a registered *bottle*, 1 as the book and 1 as a cup. The unknown objects are a bottle and a cup, so most
  errors are *same-category, different-instance* confusions, the central difficulty of this project.
* The measured ResNet-50 cosine range (5th percentile of different-instance pairs 0.086, 95th percentile of
  same-instance pairs 0.942) was used to set the default deep range to 0.25–0.90. The previous guess of
  0.5–0.95 discarded much real deep evidence.
* Caveats: 9 test identities and 80 test queries, one video per object.

### COIL-100 (real objects, viewpoint changes)

`python scripts/download_coil100.py` converts COIL-100 into the manifest format: 4 reference views per
registered object, query views 10°/25°/45° from the nearest reference (conditions `viewpoint_small`,
`viewpoint_moderate`, `viewpoint_large`), and the last N objects unregistered. Because every COIL object is
its own product, single-instance models are pooled before the identity-disjoint split. Run
`python scripts/run_experiment.py --dataset coil100` and paste the resulting table here. **Not yet measured:**
the archive could not be downloaded from the build environment, so the converter was only tested on a mock
archive with the same layout.

Limitations to report with COIL-100 results: one capture session, fixed lighting, a black background, 128 px
images, and almost no same-model siblings. It measures viewpoint robustness rather than distinguishing
near-identical instances.

### Your own pilot dataset

**Pending** (beyond the 5-object demo above). Collect the pilot dataset described in [data/README.md](../data/README.md), install the deep
weights, and run `python scripts/run_experiment.py --dataset <name>`. Report public-benchmark results (if any
are added) separately from custom-dataset results.

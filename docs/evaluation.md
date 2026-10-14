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

Run `20261014-145104-ec3d44` (Research Lab, seed 42, validation fraction 0.4). Split: 10 validation / 15 test
identities; gallery 60 images; 119 validation and 177 test queries (153 enrolled + 24 unknown); 4 queries
excluded as near-duplicates of gallery images. Test pairs: 940 positive / 940 negative (470 same-model hard
negatives).

| Method | Top-1 | Top-5 | Acc. precision | Acc. recall | Unknown rejection | Open-set acc. | Ver. AUC | EER | FMR | FNMR |
|---|---|---|---|---|---|---|---|---|---|---|
| ORB + RANSAC | 1.000 | 1.000 | 0.908 | 0.843 | 0.458 | 0.791 | 0.909 | 0.173 | 0.145 | 0.210 |
| LBP texture | 0.327 | 0.660 | 0.588 | 0.065 | 0.833 | 0.169 | 0.662 | 0.389 | 1.000 | 0.000 |
| HSV colour | 0.542 | 0.987 | 0.503 | 0.497 | 0.167 | 0.452 | 0.761 | 0.327 | 0.535 | 0.023 |
| Hybrid (default weights) | 0.869 | 1.000 | 0.756 | 0.869 | 0.042 | 0.757 | 0.884 | 0.200 | 0.372 | 0.045 |
| Hybrid (validation-tuned) | 1.000 | 1.000 | 0.925 | 0.810 | 0.583 | 0.780 | 0.910 | 0.173 | 0.273 | 0.078 |
| Deep embedding | unavailable | | | | | | | | | |

Validation-tuned fusion weights: geometric 0.9, colour 0.1, texture 0.0 (validation ROC-AUC 0.921).

### Observations (synthetic data only)

* **Local geometric evidence dominates.** ORB + RANSAC ranks the true instance first for every enrolled test
  query, across all synthetic conditions, including large viewpoint changes, occlusion and blur. The generator
  renders flat printed surfaces, which suit a homography model better than real 3-D objects do.
* **Unknown siblings are the failure mode.** Unregistered instances of an enrolled model share its print, so
  they produce many correct but non-identifying inliers. ORB rejects only 46 % of them, and the tuned hybrid
  58 %. This is the instance-vs-category problem the project targets. Separating siblings requires the
  instance-specific marks to dominate the evidence, for example by down-weighting matches on shared design
  regions.
* **Global texture and colour descriptors do not identify instances.** LBP histograms are almost identical
  across siblings (Top-1 0.33), and the validation-selected threshold collapses to "accept everything" in
  verification (FMR 1.0). Colour separates models, not instances, and is lighting-sensitive.
* **Validation tuning matters.** The hand-set default fusion accepted 96 % of unknown objects. Fitting the
  ranges and weights on validation identities raised Top-1 from 0.87 to 1.00 and unknown rejection from
  0.04 to 0.58.
* **Latency.** Matching cost is dominated by ORB, at about 1.8 s CPU time per query against 60 references in
  the parallel experiment harness. Global descriptors cost about 1–2 ms. The app parallelises reference
  matching across threads.

Charts (`roc_verification.png`, `robustness_top1.png`, `confusion.png`), CSV exports and the full
`metrics.json` are produced in `experiments/results/<run>/` and in the Research Lab.

## Results on real data

**Pending.** Collect the pilot dataset described in [data/README.md](../data/README.md), install the deep
weights, and run `python scripts/run_experiment.py --dataset <name>`. Report public-benchmark results (if any
are added) separately from custom-dataset results.

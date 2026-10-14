# Data directory

| Path | Contents | Committed? |
|---|---|---|
| `object_dna.db` | SQLite database | no |
| `storage/` | uploaded/processed images, fingerprint `.npz` artifacts, query images | no |
| `models/` | pretrained weights fetched by `scripts/download_weights.py` | no |
| `datasets/<name>/` | evaluation datasets with `manifest.csv` | no |
| `objects/` | optional staging area for your own captures | no |

## Capture protocol for a custom dataset

One folder = **one physical instance**. Two identical mugs are two folders.

```
data/datasets/<dataset>/objects/<object_id>/
    object.json                 {"name": "...", "category": "mug", "model": "ikea-faergrik-white", "registered": true}
    reference/<session>__<condition>__<n>.jpg
    query/<session>__<condition>__<n>.jpg
```

(Alternatively `query/<session>/<condition>/<file>.jpg`.)

1. **Choose objects.** Start with 15–30 instances. Use at least 2–4 instances of the same model per category
   (same-model negatives make the task meaningful). Mix distinctive objects, repeated logos and printed patterns,
   near-identical items, items with subtle wear, and items with no obvious marks. Mark a few objects
   `"registered": false`; they are never enrolled and measure unknown-object rejection.
2. **Reference session (`s1`).** 3–5 photos per object: front, two oblique views (30–45°), one closer and one
   farther, all in good diffuse light.
3. **Query sessions (`s2`, `s3`, …).** Capture on a **different day or place**. Take one or more photos per
   condition and name the files by condition: `same`, `viewpoint_moderate` (15–30°), `viewpoint_large`
   (45–90°), `lighting` (lamp, window, dim), `background`, `distance` (½× and 2×), `occlusion` (hand or object
   covering 10–30 %), `blur_lowres` (slight motion blur or a low-resolution camera).
4. **Avoid leakage.** Never copy or crop the same photo into both `reference/` and `query/`. Do not use
   burst/video frames from one moment as separate sessions.
5. Run `python scripts/import_dataset.py data/datasets/<dataset>`. It writes `manifest.csv` and lists
   near-duplicate image pairs (dHash ≤ 4 bits) for you to review.

## Splits used by the experiments

* Validation/test split is **identity-disjoint** and stratified by `model`. Every image of a physical object
  belongs to exactly one split, so no session or near-duplicate can leak between tuning and testing.
* The gallery consists of the `reference` images of registered objects.
* Queries that share a session with their own gallery images are excluded (when cross-session data exists),
  as are near-duplicates of gallery images. The counts are reported in `split.json` and the Research Lab.
* Fusion weights, normalisation ranges and thresholds are fitted on validation identities only. Test metrics
  therefore measure generalisation to identities that were not used for tuning.

# Experiments

* `configs/default_fusion.json` — the documented, hand-set default fusion (weights, normalisation ranges,
  thresholds). It is used when no tuned configuration is active.
* `configs/active_fusion.json` — written when you press **Use in application** in the Research Lab (or call
  `POST /api/experiments/{run}/apply-fusion`) or change thresholds via `PUT /api/settings/decision`. It is
  git-ignored because it is local state. Delete it (or use `POST /api/settings/fusion/reset`) to return to the
  defaults.
* `results/<run_uid>/` — outputs of each run: `metrics.json`, `summary.csv`, `split.json`,
  `verification_pairs.csv`, `identification_queries.csv`, `tuned_fusion.json` and PNG charts (git-ignored).
* `cache/` — cached per-image features keyed by preprocessing/extractor configuration (safe to delete).

Run from the CLI with `python scripts/run_experiment.py --dataset <name>`, or from the Research Lab page.

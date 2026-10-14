"""Run the evaluation suite from the command line (same code as the Research Lab).

    python scripts/run_experiment.py --dataset synthetic_demo
    python scripts/run_experiment.py --dataset my_pilot --experiment identification

Results are written to experiments/results/<run_uid>/ and recorded in the
database so they appear in the Research Lab.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import uuid
from datetime import datetime, timezone

from app.core.runtime import get_runtime
from app.database import get_sessionmaker, init_db
from app.models.orm import ExperimentRun
from app.research import experiments as exp
from app.research.dataset import load_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic_demo")
    ap.add_argument("--experiment", default="full", choices=["full", "verification", "identification", "robustness"])
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.4)
    ap.add_argument("--max-pairs", type=int, default=3000)
    ap.add_argument("--no-perturbations", action="store_true")
    a = ap.parse_args()

    init_db()
    rt = get_runtime()
    root = rt.settings.data_dir / "datasets" / a.dataset
    ds = load_manifest(root)
    cfg = exp.ExperimentConfig(experiment=a.experiment, methods=a.methods, seed=a.seed,
                               val_fraction=a.val_fraction, max_pairs=a.max_pairs,
                               perturbations=not a.no_perturbations)
    uid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    with get_sessionmaker()() as db:
        run = ExperimentRun(run_uid=uid, experiment=a.experiment, dataset_id=ds.dataset_id,
                            config={**vars(a), "source": "cli"}, status="running", is_synthetic=ds.is_synthetic)
        db.add(run)
        db.commit()
        try:
            m = exp.run(root, cfg, rt.prep_config, rt.feat_config, rt.embedder, rt.settings.experiments_dir / "results",
                        rt.settings.experiments_dir / "cache",
                        lambda p, msg: print(f"[{p * 100:5.1f}%] {msg}", flush=True), uid)
        except Exception as e:
            run.status, run.message = "failed", str(e)
            db.commit()
            raise
        run.metrics = json.loads(json.dumps(m, default=exp._json_default))
        run.artifact_paths, run.split_id = m["artifacts"], m["split"]["split_id"]
        run.status, run.progress, run.finished_at = "completed", 1.0, datetime.now(timezone.utc)
        run.message = f"Completed in {m['runtime_s']} s (CLI)"
        db.commit()

    print(f"\nRun {uid}  dataset={ds.dataset_id}  synthetic={ds.is_synthetic}")
    print(f"Unavailable methods: {m['methods_unavailable'] or 'none'}")
    ident = m.get("identification", {}).get("methods", {})
    ver = m.get("verification", {}).get("methods", {})
    print(f"\n{'method':18s} {'top1':>6s} {'top5':>6s} {'acc.prec':>8s} {'unk.rej':>8s} {'open-set':>8s} "
          f"{'ver.AUC':>8s} {'FMR':>6s} {'FNMR':>6s} {'ms/query':>9s}")
    for meth in m["methods_evaluated"]:
        i, v = ident.get(meth, {}), ver.get(meth, {}).get("test", {})
        f = lambda x: "  n/a" if x is None else f"{x:.3f}"  # noqa: E731
        print(f"{meth:18s} {f(i.get('top1_accuracy')):>6s} {f(i.get('top5_accuracy')):>6s} "
              f"{f(i.get('accepted_precision')):>8s} {f(i.get('unknown_rejection_rate')):>8s} "
              f"{f(i.get('open_set_accuracy')):>8s} {f(v.get('roc_auc')):>8s} {f(v.get('fmr')):>6s} "
              f"{f(v.get('fnmr')):>6s} {f(i.get('matching_ms')):>9s}")
    print(f"\nArtifacts: {rt.settings.experiments_dir / 'results' / uid}")


if __name__ == "__main__":
    main()

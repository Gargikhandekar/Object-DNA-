"""Generate the procedural synthetic dataset (pipeline sanity checks only).

    python scripts/generate_synthetic_dataset.py --name synthetic_demo
"""
import _bootstrap  # noqa: F401

import argparse

from app.core.config import get_settings
from app.research.dataset import load_manifest
from app.research.synthetic import generate_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="synthetic_demo")
    ap.add_argument("--models", type=int, default=5)
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--unregistered", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = get_settings().data_dir / "datasets" / a.name
    generate_dataset(out, a.models, a.instances, a.unregistered, seed=a.seed)
    ds = load_manifest(out)
    print(f"Wrote {len(ds.samples)} images of {len(ds.objects())} synthetic instances to {out}")

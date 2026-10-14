"""Download pretrained encoder weights ONCE, explicitly (never at runtime).

    python scripts/download_weights.py --backbone resnet50
    python scripts/download_weights.py --backbone dinov2_vits14

Files are stored in data/models/ (or ODNA_WEIGHTS_DIR). Afterwards restart the
backend; to add embeddings to objects registered earlier, call
POST /api/settings/backfill-embeddings (button on the Dashboard).
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from app.core.config import get_settings
from app.services.embedding_extractor import BACKBONES

SOURCES = {
    "resnet50": {
        "url": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        "sha256_prefix": "11ad3fa6",
        "license": "BSD-3-Clause (torchvision)",
    },
    "dinov2_vits14": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "repo_zip": "https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip",
        "license": "Apache-2.0 (DINOv2)",
    },
}


def fetch(url: str, dest: Path) -> None:
    print(f"Downloading {url}\n  -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=sorted(SOURCES), default="resnet50")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    wdir = get_settings().resolved_weights_dir
    wdir.mkdir(parents=True, exist_ok=True)
    spec, src = BACKBONES[args.backbone], SOURCES[args.backbone]
    dest = wdir / spec.weights_file
    if dest.exists() and not args.force:
        print(f"Already present: {dest}")
    else:
        fetch(src["url"], dest)
    if "sha256_prefix" in src:
        h = hashlib.sha256(dest.read_bytes()).hexdigest()
        if not h.startswith(src["sha256_prefix"]):
            print(f"WARNING: checksum prefix mismatch ({h[:8]} != {src['sha256_prefix']}).", file=sys.stderr)
            return 1
    if args.backbone == "dinov2_vits14":
        repo = wdir / "dinov2_repo"
        if not repo.exists() or args.force:
            z = wdir / "dinov2_main.zip"
            fetch(src["repo_zip"], z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(wdir)
            shutil.rmtree(repo, ignore_errors=True)
            (wdir / "dinov2-main").rename(repo)
            z.unlink()
    print(f"Done. License: {src['license']}. Set ODNA_EMBEDDING_BACKBONE={args.backbone} and restart the backend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

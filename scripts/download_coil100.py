"""Download COIL-100 (real photographs of 100 physical objects) and convert it into an Object DNA dataset.

COIL-100 — Columbia Object Image Library (S. A. Nene, S. K. Nayar, H. Murase, Technical Report
CUCS-006-96, Columbia University, 1996). 100 real objects photographed on a motorised turntable
against a black background, one image every 5 degrees (72 views per object, 128x128 px).
Licence: free for non-commercial research and education; cite the report above when you publish
results. Nothing is redistributed by this repository: the archive is downloaded to your machine.

    python scripts/download_coil100.py                       # 30 objects, dataset only
    python scripts/download_coil100.py --register            # also enrol the reference views in the app
    python scripts/download_coil100.py --objects 60 --unregistered 10 --register
    python scripts/download_coil100.py --zip C:/Downloads/coil-100.zip   # use a manually downloaded archive

Conversion (one folder per physical object, see data/README.md):

* reference views (session ``s1``): 0, 90, 180 and 270 degrees
* query views (session ``s2``), labelled by angular distance to the nearest reference view:
  - ``viewpoint_small``    10 degrees away   (e.g. 10, 100, ...)
  - ``viewpoint_moderate`` 25 degrees away   (e.g. 25, 115, ...)
  - ``viewpoint_large``    45 degrees away   (e.g. 45, 135, ... the hardest possible case)
* the last ``--unregistered`` objects are never enrolled; all their views act as unknown queries.

Limitations to state in any report: a single capture session, fixed lighting and black background,
small images, and mostly *different* products (few same-model siblings), so COIL-100 measures
viewpoint robustness more than instance-vs-sibling discrimination.
"""
import _bootstrap  # noqa: F401

import argparse
import io
import json
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from app.core.config import get_settings

URLS = [
    "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip",
    "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip",
]
NAME_RE = re.compile(r"obj(\d+)__(\d+)\.png$", re.IGNORECASE)
REF_ANGLES = (0, 90, 180, 270)
QUERY_OFFSETS = {"viewpoint_small": 10, "viewpoint_moderate": 25, "viewpoint_large": 45}


def download(dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Using cached archive {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for url in URLS:
        try:
            print(f"Downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as r:
                total = int(r.headers.get("Content-Length") or 0)
                buf, done = io.BytesIO(), 0
                while chunk := r.read(1 << 20):
                    buf.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
            print()
            dest.write_bytes(buf.getvalue())
            return dest
        except Exception as exc:  # try the next mirror
            last = exc
            print(f"  failed: {exc}")
    raise SystemExit(
        f"Could not download COIL-100 ({last}).\n"
        "Download coil-100.zip manually from https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php "
        "and run:  python scripts/download_coil100.py --zip <path to coil-100.zip>"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", type=int, default=30, help="number of COIL objects to use (1-100)")
    ap.add_argument("--unregistered", type=int, default=5, help="how many of them are never enrolled")
    ap.add_argument("--zip", type=Path, help="path to an already downloaded coil-100.zip")
    ap.add_argument("--name", default="coil100", help="dataset folder name under data/datasets/")
    ap.add_argument("--register", action="store_true", help="register reference views in the application")
    a = ap.parse_args()
    if not 1 <= a.objects <= 100 or not 0 <= a.unregistered < a.objects:
        raise SystemExit("--objects must be 1-100 and --unregistered smaller than --objects")

    s = get_settings()
    archive = a.zip or download(s.data_dir / "downloads" / "coil-100.zip")
    root = s.data_dir / "datasets" / a.name

    with zipfile.ZipFile(archive) as zf:
        views: dict[int, dict[int, str]] = {}
        for member in zf.namelist():
            m = NAME_RE.search(member)
            if m:
                views.setdefault(int(m.group(1)), {})[int(m.group(2))] = member
        if not views:
            raise SystemExit(f"No objN__ANGLE.png files found in {archive}; is this the COIL-100 archive?")
        chosen = sorted(views)[: a.objects]
        n_reg = len(chosen) - a.unregistered
        written = 0
        for k, obj in enumerate(chosen):
            registered = k < n_reg
            oid = f"coil_{obj:03d}"
            odir = root / "objects" / oid
            angles = views[obj]

            def put(angle: int, rel: str) -> None:
                nonlocal written
                if angle not in angles:
                    return
                target = odir / rel
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(angles[angle]))
                written += 1

            if registered:
                for ang in REF_ANGLES:
                    put(ang, f"reference/s1__reference__{ang:03d}.png")
            for cond, off in QUERY_OFFSETS.items():
                for ref in REF_ANGLES:
                    ang = (ref + off) % 360
                    put(ang, f"query/s2__{cond}__{ang:03d}.png")
            (odir / "object.json").write_text(json.dumps({
                "name": f"COIL object {obj}", "category": "coil-100", "model": oid,
                "registered": registered, "source": "COIL-100 (Columbia University)",
            }, indent=2))
    (root / "DATASET_INFO.json").write_text(json.dumps({
        "name": a.name, "synthetic": False, "source": "COIL-100, Columbia University (CUCS-006-96)",
        "licence": "Free for non-commercial research and education; cite Nene, Nayar & Murase (1996).",
        "objects": len(chosen), "unregistered": a.unregistered,
        "note": "Real photographs, single session, turntable rotation only, fixed lighting, black background.",
    }, indent=2))
    print(f"Wrote {written} images of {len(chosen)} real objects to {root}")

    cmd = [sys.executable, str(Path(__file__).parent / "import_dataset.py"), str(root)]
    if a.register:
        cmd.append("--register")
    subprocess.check_call(cmd)
    print(f"\nResearch Lab: choose dataset '{a.name}' and press Run.")
    print(f"Identify: upload a file from {root / 'objects' / '<id>' / 'query'}")


if __name__ == "__main__":
    main()

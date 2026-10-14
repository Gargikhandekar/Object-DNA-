"""Download a small set of REAL photographs to try Object DNA without a camera.

Source: sample images shipped with OpenCV (https://github.com/opencv/opencv/tree/4.x/samples/data),
used for computer-vision teaching and benchmarking. Each "object" below has two real photos of the
same physical object or scene taken from different viewpoints, plus a few unrelated images that are
never registered (they test unknown-object rejection).

    python scripts/download_real_samples.py              # dataset only (for the Research Lab)
    python scripts/download_real_samples.py --register   # also register the reference photos in the app

Images are downloaded into data/datasets/opencv_real/ (git-ignored); nothing is redistributed by
this repository. Note: this is a tiny demo set (5 objects), not a benchmark; some images are
greyscale, so colour evidence is unavailable for them.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from app.core.config import get_settings

BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"

# object_id: (name, category, reference file, query file, condition of the query)
OBJECTS = {
    "box": ("Bastoncini cookie box", "box", "box.png", "box_in_scene.png", "background"),
    "aloe": ("Aloe plant", "plant", "aloeL.jpg", "aloeR.jpg", "viewpoint_moderate"),
    "rubberwhale": ("Rubber whale toy scene", "toy", "rubberwhale1.png", "rubberwhale2.png", "same"),
    "graffiti": ("Graffiti wall panel", "artwork", "graf1.png", "graf3.png", "viewpoint_large"),
    "basketball": ("Basketball scene", "ball", "basketball1.png", "basketball2.png", "same"),
}
UNREGISTERED = {"apple": ("Apple", "fruit", "apple.jpg"), "orange": ("Orange", "fruit", "orange.jpg")}


def fetch(name: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE + name, timeout=60) as r:
        dest.write_bytes(r.read())
    print(f"  downloaded {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", action="store_true", help="also register reference photos in the app")
    a = ap.parse_args()
    root = get_settings().data_dir / "datasets" / "opencv_real"
    print(f"Downloading real sample photos to {root}")
    for oid, (name, cat, ref, query, cond) in OBJECTS.items():
        odir = root / "objects" / oid
        ext_r, ext_q = Path(ref).suffix, Path(query).suffix
        fetch(ref, odir / "reference" / f"s1__reference__01{ext_r}")
        fetch(query, odir / "query" / f"s2__{cond}__01{ext_q}")
        (odir / "object.json").write_text(json.dumps(
            {"name": name, "category": cat, "model": oid, "registered": True, "source": BASE + ref}, indent=2))
    for oid, (name, cat, f) in UNREGISTERED.items():
        odir = root / "objects" / oid
        fetch(f, odir / "query" / f"s2__unregistered__01{Path(f).suffix}")
        (odir / "object.json").write_text(json.dumps(
            {"name": name, "category": cat, "model": oid, "registered": False, "source": BASE + f}, indent=2))
    (root / "DATASET_INFO.json").write_text(json.dumps(
        {"name": "opencv_real", "synthetic": False, "source": "OpenCV samples/data",
         "note": "Tiny real-photo demo set (5 registered objects, 2 unregistered). Not a benchmark."}, indent=2))
    cmd = [sys.executable, str(Path(__file__).parent / "import_dataset.py"), str(root)]
    if a.register:
        cmd.append("--register")
    subprocess.check_call(cmd)
    print("\nTry it: Identify Object -> upload a file from", root / "objects" / "<id>" / "query")


if __name__ == "__main__":
    main()

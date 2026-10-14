"""Download real phone videos from Google's Objectron dataset and turn them into an Object DNA dataset.

Objectron (Ahmadyan et al., CVPR 2021, https://github.com/google-research-datasets/Objectron) contains
short videos, each walking around ONE physical object (a specific bottle, cup, shoe, book, ...), recorded
with phones in real homes. Because every category has hundreds of *different physical instances*, it is a
good testbed for instance-level identification: "which of these cups is it?", not "is it a cup?".
Licence: Computational Use of Data Agreement (C-UDA 1.0); cite the paper when you publish results.
Nothing is redistributed by this repository: videos are downloaded to your machine and deleted after
frame extraction (unless --keep-videos).

    python scripts/download_objectron.py                  # all 9 categories, <= 200 MB download
    python scripts/download_objectron.py --register       # also enrol reference frames in the app
    python scripts/download_objectron.py --categories bottle cup shoe --per-category 4 --max-mb 300 --register

Conversion (one folder per physical object):
* frames are sampled along the walk-around; reference frames (session ``s1``) at 0 %, 35 % and 70 % of the clip
* query frames (session ``s2``) at fixed offsets after each reference, labelled by camera motion:
  ``viewpoint_small`` (+5 % of the clip), ``viewpoint_moderate`` (+12 %), ``viewpoint_large`` (+22 %)
* ``--per-category`` objects of every category are registered; ``--unregistered`` extra objects (from the
  first categories) are never enrolled and act as unknown-object queries
* the smallest videos are chosen (sizes are checked with HEAD requests first) so the total download stays
  under ``--max-mb``; videos are deleted after frame extraction
* ``model`` = category, so negative pairs are *different physical objects of the same category*

Limitations: each object is a single capture session (one video), so lighting and background change only
as much as the camera moves; frames are video stills (some motion blur).
"""
import _bootstrap  # noqa: F401

import argparse
import json
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2

from app.core.config import get_settings

BASE = "https://storage.googleapis.com/objectron"
CATEGORIES = ("bottle", "cup", "shoe", "book", "camera", "cereal_box", "laptop", "chair", "bike")
REF_POS = (0.0, 0.35, 0.70)
QUERY_OFFSETS = {"viewpoint_small": 0.05, "viewpoint_moderate": 0.12, "viewpoint_large": 0.22}
MAX_SIDE = 960


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode()


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length") or 0)


def fetch_file(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def extract(video: Path, positions: dict[str, float]) -> dict[str, "cv2.typing.MatLike"]:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = {}
    if n <= 0:
        return out
    for key, pos in positions.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, int(pos * (n - 1))))
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        s = MAX_SIDE / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        out[key] = img
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    ap.add_argument("--per-category", type=int, default=2, help="registered physical objects per category")
    ap.add_argument("--unregistered", type=int, default=2, help="extra never-registered objects")
    ap.add_argument("--max-mb", type=float, default=200, help="total download budget in MB")
    ap.add_argument("--name", default="objectron")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-videos", action="store_true")
    ap.add_argument("--register", action="store_true", help="register reference frames in the application")
    a = ap.parse_args()
    if a.per_category < 1:
        raise SystemExit("--per-category must be at least 1")

    s = get_settings()
    root = s.data_dir / "datasets" / a.name
    vids = s.data_dir / "downloads" / "objectron"
    vids.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    positions = {f"ref{k}": p for k, p in enumerate(REF_POS)}
    for k, p in enumerate(REF_POS):
        for cond, off in QUERY_OFFSETS.items():
            positions[f"{cond}{k}"] = min(0.999, p + off)

    # 1) Plan: list candidate videos per category with their sizes, smallest first.
    print("Checking video sizes ...")
    cands: dict[str, list[tuple[int, str]]] = {}
    for cat in a.categories:
        index = [x.strip() for x in fetch_text(f"{BASE}/v1/index/{cat}_annotations_test").splitlines() if x.strip()]
        rng.shuffle(index)
        sized = []
        for vid_id in index[:40]:
            try:
                sized.append((head_size(f"{BASE}/videos/{vid_id}/video.MOV"), vid_id))
            except Exception:
                continue
        cands[cat] = sorted(sized)
    plan: list[tuple[str, str, bool, int]] = []  # (category, video id, registered, bytes)
    for cat in a.categories:
        plan += [(cat, v, True, size) for size, v in cands[cat][: a.per_category]]
    for k in range(a.unregistered):
        cat = a.categories[k % len(a.categories)]
        used = {v for c, v, *_ in plan if c == cat}
        extra = [(size, v) for size, v in cands[cat] if v not in used]
        if extra:
            plan.append((cat, extra[0][1], False, extra[0][0]))
    budget = a.max_mb * 1e6
    while sum(p[3] for p in plan) > budget and plan:
        # Keep one registered object per category and the unregistered ones as long as possible:
        # first drop the largest *additional* registered instance of a category, then unregistered objects.
        counts: dict[str, int] = {}
        for c, _, reg, _ in plan:
            counts[c] = counts.get(c, 0) + int(reg)
        extras = [p for p in plan if p[2] and counts[p[0]] > 1]
        pool = extras or [p for p in plan if not p[2]] or plan
        drop = max(pool, key=lambda p: p[3])
        plan.remove(drop)
        print(f"  over budget: leaving out {drop[1]} ({drop[3] / 1e6:.0f} MB)")
    print(f"Plan: {len(plan)} videos, {sum(p[3] for p in plan) / 1e6:.0f} MB "
          f"({sum(p[2] for p in plan)} registered, {sum(not p[2] for p in plan)} unregistered)")

    # 2) Download, extract frames, delete the video.
    total = 0
    for cat, vid_id, registered, size in plan:
        oid = f"{cat}_{vid_id.split('/', 1)[1].replace('/', '_')}"
        odir = root / "objects" / oid
        if (odir / "object.json").exists():
            continue
        video = vids / f"{oid}.MOV"
        try:
            if not video.exists():
                print(f"  {vid_id} ({size / 1e6:.0f} MB)")
                fetch_file(f"{BASE}/videos/{vid_id}/video.MOV", video)
            frames = extract(video, positions)
        except Exception as exc:
            print(f"  skipped {vid_id}: {exc}")
            continue
        if len(frames) < len(positions) - 2:
            print(f"  skipped {vid_id}: could not decode frames")
            continue
        for key, img in frames.items():
            if key.startswith("ref"):
                rel = (f"reference/s1__reference__{key[3:]}.jpg" if registered
                       else f"query/s2__unregistered__{key}.jpg")
            else:
                rel = f"query/s2__{key[:-1]}__{key[-1]}.jpg"
            target = odir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            total += 1
        (odir / "object.json").write_text(json.dumps({
            "name": f"{cat.replace('_', ' ').title()} {vid_id.split('/', 1)[1]}", "category": cat,
            "model": cat, "registered": registered, "source": f"Objectron {vid_id}",
        }, indent=2))
        if not a.keep_videos:
            video.unlink(missing_ok=True)

    (root / "DATASET_INFO.json").write_text(json.dumps({
        "name": a.name, "synthetic": False,
        "source": "Google Objectron (Ahmadyan et al., CVPR 2021)", "licence": "C-UDA 1.0",
        "categories": a.categories, "per_category": a.per_category, "unregistered": a.unregistered,
        "seed": a.seed,
        "note": "Real phone videos; one capture session per object; frames sampled along the walk-around.",
    }, indent=2))
    print(f"Wrote {total} new frames to {root}")
    cmd = [sys.executable, str(Path(__file__).parent / "import_dataset.py"), str(root)]
    if a.register:
        cmd.append("--register")
    subprocess.check_call(cmd)
    print(f"\nResearch Lab: choose dataset '{a.name}' and press Run.")
    print(f"Identify: upload a file from {root / 'objects' / '<id>' / 'query'}")


if __name__ == "__main__":
    main()

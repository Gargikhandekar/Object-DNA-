"""Build manifest.csv for a dataset collected with a phone/webcam.

Expected layout (one folder per PHYSICAL INSTANCE, not per product):

    <dataset>/objects/<object_id>/reference/<session>__<condition>__<n>.jpg
    <dataset>/objects/<object_id>/query/<session>__<condition>__<n>.jpg
    <dataset>/objects/<object_id>/object.json   (optional)
        {"name": "...", "category": "bottle", "model": "brandX-500ml", "registered": true}

    python scripts/import_dataset.py data/datasets/my_pilot
    python scripts/import_dataset.py data/datasets/my_pilot --register   # also enrol references into the app
"""
import _bootstrap  # noqa: F401

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from app.research.dataset import import_folder_dataset, load_manifest
from app.services.preprocessing import dhash, hamming


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--register", action="store_true",
                    help="Register every registered=true object's reference images in the application database")
    a = ap.parse_args()
    manifest = import_folder_dataset(a.dataset)
    ds = load_manifest(a.dataset)
    print(f"Manifest: {manifest}")
    print(f"  images: {len(ds.samples)}  physical objects: {len(ds.objects())}")
    print(f"  roles: {dict(Counter(s.role for s in ds.samples))}")
    print(f"  sessions: {dict(Counter(s.session for s in ds.samples))}")
    print(f"  conditions: {dict(Counter(s.condition for s in ds.samples))}")
    print(f"  unregistered objects: {sorted({s.object_id for s in ds.samples if not s.registered})}")

    hashes = {s.idx: dhash(cv2.cvtColor(cv2.imread(str(s.path)), cv2.COLOR_BGR2RGB)) for s in ds.samples}
    dups = [(ds.samples[i].rel_path, ds.samples[j].rel_path) for i in hashes for j in hashes
            if i < j and hamming(hashes[i], hashes[j]) <= 4]
    print(f"  near-duplicate image pairs (dHash <= 4 bits): {len(dups)}")
    for x, y in dups[:20]:
        print(f"    {x}  ~  {y}")

    if a.register:
        from app.api.objects import _new_public_id
        from app.core.runtime import get_runtime
        from app.database import get_sessionmaker, init_db
        from app.models.orm import PhysicalObject
        from app.services.image_validation import validate_image_bytes

        init_db()
        rt = get_runtime()
        by_obj: dict[str, list] = {}
        for s in ds.samples:
            if s.role == "reference" and s.registered:
                by_obj.setdefault(s.object_id, []).append(s)
        with get_sessionmaker()() as db:
            for oid, samples in by_obj.items():
                meta_p = a.dataset / "objects" / oid / "object.json"
                meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
                obj = PhysicalObject(public_id=_new_public_id(db), name=meta.get("name", oid),
                                     category=str(meta.get("category", samples[0].category)).lower(),
                                     description=f"Imported from {a.dataset.name}/{oid}", fingerprint_status="processing")
                db.add(obj)
                db.flush()
                for s in samples:
                    img = validate_image_bytes(s.path.read_bytes(), s.path.name, None,
                                               max_bytes=rt.settings.max_upload_bytes)
                    rt.store.add_reference(db, obj, img, {"session": s.session, "condition": s.condition,
                                                          "dataset_object_id": oid})
                obj.fingerprint_status = "complete"
                db.commit()
                print(f"  registered {oid} -> {obj.public_id} ({len(samples)} references)")


if __name__ == "__main__":
    main()

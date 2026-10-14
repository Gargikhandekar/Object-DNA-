"""Dataset manifest, import and leakage-aware splitting.

Manifest columns (``manifest.csv`` at the dataset root)::

    object_id, image_path, role, session, condition, category, model, registered

* ``object_id`` identifies ONE PHYSICAL INSTANCE (not a product/category).
* ``role`` is ``reference`` (gallery/enrolment) or ``query``.
* ``session`` is the capture session (time/place/camera setup). Images from
  one session are never split across validation and test.
* ``condition`` is the acquisition condition used for robustness analysis
  (``same``, ``viewpoint_moderate``, ``lighting`` ...).
* ``model`` groups different physical instances of the same product model;
  used to build hard negative pairs.
* ``registered=false`` marks identities that are never enrolled; their
  queries are used to measure unknown-object rejection.

Split design
------------
Nothing is *trained* on images except (a) the hybrid fusion weights and
normalisation ranges and (b) the decision thresholds. Both are fitted on a
validation split and evaluated on a held-out test split. Splits are
IDENTITY-DISJOINT: every image (and therefore every session) of a physical
object belongs to exactly one of {validation, test}. This makes validation
and test independent and prevents near-duplicate leakage between them. The
reported metrics therefore measure generalisation to identities that were
not used for tuning.

Within a split, query/reference pairs from the *same capture session* are
excluded from positive pairs where cross-session pairs exist, and
near-duplicates (dHash Hamming distance <= ``NEAR_DUP_BITS``) are removed and
reported.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.research.synthetic import MANIFEST_FIELDS, write_manifest

NEAR_DUP_BITS = 4
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Sample:
    idx: int
    object_id: str
    path: Path
    rel_path: str
    role: str
    session: str
    condition: str
    category: str
    model: str
    registered: bool


@dataclass
class Dataset:
    root: Path
    samples: list[Sample]
    info: dict = field(default_factory=dict)

    @property
    def dataset_id(self) -> str:
        h = hashlib.sha1("\n".join(sorted(s.rel_path for s in self.samples)).encode()).hexdigest()[:10]
        return f"{self.root.name}@{h}"

    @property
    def is_synthetic(self) -> bool:
        return bool(self.info.get("synthetic"))

    def objects(self) -> list[str]:
        return sorted({s.object_id for s in self.samples})


def _truthy(v: str | None) -> bool:
    return str(v if v is not None else "true").strip().lower() not in {"false", "0", "no", "n"}


def load_manifest(root: Path) -> Dataset:
    root = Path(root)
    manifest = root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"No manifest.csv in {root}. Run scripts/import_dataset.py first.")
    samples: list[Sample] = []
    with manifest.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            rel = row["image_path"]
            p = (root / rel).resolve()
            if not str(p).startswith(str(root.resolve())):
                raise ValueError(f"Manifest path escapes dataset root: {rel}")
            role = (row.get("role") or "query").strip().lower()
            if role not in {"reference", "query"}:
                raise ValueError(f"Invalid role '{role}' in row {i + 2}")
            samples.append(Sample(
                idx=len(samples), object_id=row["object_id"].strip(), path=p, rel_path=rel, role=role,
                session=(row.get("session") or "unknown").strip() or "unknown",
                condition=(row.get("condition") or "unspecified").strip() or "unspecified",
                category=(row.get("category") or "unknown").strip() or "unknown",
                model=(row.get("model") or row.get("category") or "unknown").strip() or "unknown",
                registered=_truthy(row.get("registered")),
            ))
    info = {}
    info_path = root / "DATASET_INFO.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
    return Dataset(root=root, samples=samples, info=info)


def import_folder_dataset(root: Path) -> Path:
    """Build ``manifest.csv`` from the folder convention::

        <root>/objects/<object_id>/reference/<file>
        <root>/objects/<object_id>/query/<file>
        <root>/objects/<object_id>/object.json   (optional: name, category, model, registered)

    Session and condition are parsed from filenames of the form
    ``<session>__<condition>__<anything>.jpg``; otherwise from sub-folders
    ``query/<session>/<condition>/file.jpg``; otherwise defaults are used
    (``ref`` / ``q`` sessions, ``unspecified`` condition).
    """
    root = Path(root)
    obj_root = root / "objects"
    if not obj_root.is_dir():
        raise FileNotFoundError(f"Expected {obj_root} with one sub-folder per physical object.")
    rows = []
    for odir in sorted(p for p in obj_root.iterdir() if p.is_dir()):
        meta = {}
        if (odir / "object.json").exists():
            meta = json.loads((odir / "object.json").read_text())
        for role in ("reference", "query"):
            rdir = odir / role
            if not rdir.is_dir():
                continue
            for f in sorted(rdir.rglob("*")):
                if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                    continue
                parts = f.stem.split("__")
                sub = f.relative_to(rdir).parts[:-1]
                if len(parts) >= 3:
                    session, condition = parts[0], parts[1]
                elif len(sub) >= 2:
                    session, condition = sub[0], sub[1]
                elif len(sub) == 1:
                    session, condition = sub[0], "unspecified"
                else:
                    session = "ref" if role == "reference" else "q"
                    condition = "reference" if role == "reference" else "unspecified"
                rows.append({
                    "object_id": odir.name,
                    "image_path": f.relative_to(root).as_posix(),
                    "role": role,
                    "session": session,
                    "condition": condition,
                    "category": meta.get("category", "unknown"),
                    "model": meta.get("model", meta.get("category", "unknown")),
                    "registered": str(bool(meta.get("registered", True))).lower(),
                })
    if not rows:
        raise ValueError(f"No images found under {obj_root}.")
    manifest = root / "manifest.csv"
    write_manifest(manifest, rows)
    return manifest


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@dataclass
class Split:
    split_id: str
    val_objects: list[str]
    test_objects: list[str]
    gallery: list[int]  # sample indices (reference images of registered identities)
    known_objects: list[str]  # identities enrolled in the gallery
    unknown_objects: list[str]
    queries: dict[str, list[int]]  # {"val": [...], "test": [...]}
    excluded: dict[str, list[dict]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def describe(self, ds: Dataset) -> dict:
        by = {s.idx: s for s in ds.samples}

        def summary(idxs):
            sess = sorted({by[i].session for i in idxs})
            conds: dict[str, int] = {}
            for i in idxs:
                conds[by[i].condition] = conds.get(by[i].condition, 0) + 1
            return {"images": len(idxs), "sessions": sess, "conditions": conds,
                    "objects": len({by[i].object_id for i in idxs})}

        return {
            "split_id": self.split_id,
            "strategy": "identity-disjoint validation/test; gallery = reference-role images of registered "
                        "identities; same-session query/reference pairs excluded where cross-session data exists",
            "val_objects": self.val_objects,
            "test_objects": self.test_objects,
            "known_objects": len(self.known_objects),
            "unknown_objects": self.unknown_objects,
            "gallery": summary(self.gallery),
            "val_queries": summary(self.queries["val"]),
            "test_queries": summary(self.queries["test"]),
            "excluded": {k: len(v) for k, v in self.excluded.items()},
            "excluded_examples": {k: v[:10] for k, v in self.excluded.items()},
            "notes": self.notes,
        }


def make_split(ds: Dataset, hashes: dict[int, int], val_fraction: float = 0.4, seed: int = 42) -> Split:
    from app.services.preprocessing import hamming

    rng = np.random.default_rng(seed)
    objs = ds.objects()
    by_obj: dict[str, list[Sample]] = {o: [] for o in objs}
    for s in ds.samples:
        by_obj[s.object_id].append(s)
    known = [o for o in objs if all(s.registered for s in by_obj[o]) and any(s.role == "reference" for s in by_obj[o])]
    unknown = [o for o in objs if o not in known]

    # Identity-disjoint val/test, stratified by model so both splits contain siblings.
    val, test = [], []
    by_model: dict[str, list[str]] = {}
    for o in objs:
        by_model.setdefault(by_obj[o][0].model, []).append(o)
    # Models with several instances are split individually (siblings land in both splits); models with a
    # single instance (e.g. COIL-100, where every object is a different product) are pooled and split together.
    singletons: list[str] = []
    for model in sorted(by_model):
        group = sorted(by_model[model])
        if len(group) == 1:
            singletons += group
            continue
        rng.shuffle(group)
        k = min(max(int(round(len(group) * val_fraction)), 1), len(group) - 1)
        val += group[:k]
        test += group[k:]
    if singletons:
        rng.shuffle(singletons)
        k = int(round(len(singletons) * val_fraction))
        if len(singletons) >= 2:
            k = min(max(k, 1), len(singletons) - 1)
        val += singletons[:k]
        test += singletons[k:]
    notes = []
    if not val or not test:
        notes.append("Too few identities for an identity-disjoint split; validation and test share identities.")
        val = test = objs

    gallery = [s.idx for o in known for s in by_obj[o] if s.role == "reference"]
    gallery_sessions: dict[str, set[str]] = {o: {s.session for s in by_obj[o] if s.role == "reference"} for o in known}
    gallery_hashes: dict[str, list[int]] = {o: [hashes[s.idx] for s in by_obj[o] if s.role == "reference"]
                                            for o in known}

    excluded: dict[str, list[dict]] = {"same_session_as_gallery": [], "near_duplicate_of_gallery": []}
    queries: dict[str, list[int]] = {"val": [], "test": []}
    for name, group in (("val", val), ("test", test)):
        for o in group:
            qs = [s for s in by_obj[o] if s.role == "query"]
            if o in unknown:
                # Unenrolled identities: all their images act as unknown queries.
                qs = by_obj[o]
            for s in qs:
                if o in known:
                    has_cross = any(q.session not in gallery_sessions[o] for q in qs)
                    if has_cross and s.session in gallery_sessions[o]:
                        excluded["same_session_as_gallery"].append({"image": s.rel_path, "object": o})
                        continue
                    if any(hamming(hashes[s.idx], h) <= NEAR_DUP_BITS for h in gallery_hashes[o]):
                        excluded["near_duplicate_of_gallery"].append({"image": s.rel_path, "object": o})
                        continue
                queries[name].append(s.idx)
    sid = hashlib.sha1(json.dumps([sorted(val), sorted(test), seed]).encode()).hexdigest()[:8]
    return Split(split_id=f"idsplit-{sid}", val_objects=sorted(val), test_objects=sorted(test), gallery=gallery,
                 known_objects=known, unknown_objects=unknown, queries=queries, excluded=excluded, notes=notes)


def make_pairs(ds: Dataset, object_ids: list[str], hashes: dict[int, int], rng: np.random.Generator,
               max_pairs: int) -> tuple[list[tuple[int, int, int]], dict]:
    """Positive/negative image pairs among ``object_ids`` -> [(i, j, label)].

    Positives: different images of the same object, preferring different
    capture sessions; near-duplicates removed. Negatives: half "hard" (same
    model, different instance), half random different objects. Balanced.
    """
    from app.services.preprocessing import hamming

    allowed = set(object_ids)
    samples = [s for s in ds.samples if s.object_id in allowed]
    by_obj: dict[str, list[Sample]] = {}
    for s in samples:
        by_obj.setdefault(s.object_id, []).append(s)

    pos, same_session_pos, near_dups = [], [], 0
    for o, ss in by_obj.items():
        for a in range(len(ss)):
            for b in range(a + 1, len(ss)):
                if hamming(hashes[ss[a].idx], hashes[ss[b].idx]) <= NEAR_DUP_BITS:
                    near_dups += 1
                    continue
                (pos if ss[a].session != ss[b].session else same_session_pos).append((ss[a].idx, ss[b].idx, 1))
    used_same_session = False
    if not pos:
        pos, used_same_session = same_session_pos, True
    half = max_pairs // 2
    if len(pos) > half:
        pos = [pos[i] for i in rng.choice(len(pos), half, replace=False)]

    hard, easy = [], []
    objs = list(by_obj)
    for ia in range(len(objs)):
        for ib in range(ia + 1, len(objs)):
            a_s, b_s = by_obj[objs[ia]], by_obj[objs[ib]]
            target = hard if a_s[0].model == b_s[0].model else easy
            for sa in a_s:
                for sb in b_s:
                    target.append((sa.idx, sb.idx, 0))
    n_neg = len(pos)
    n_hard = min(len(hard), n_neg // 2)
    neg = [hard[i] for i in rng.choice(len(hard), n_hard, replace=False)] if n_hard else []
    n_easy = min(len(easy), n_neg - n_hard)
    neg += [easy[i] for i in rng.choice(len(easy), n_easy, replace=False)] if n_easy else []
    stats = {"positives": len(pos), "negatives": len(neg), "hard_negatives_same_model": n_hard,
             "near_duplicates_removed": near_dups, "positives_same_session_only": used_same_session}
    return pos + neg, stats


def manifest_fields() -> list[str]:
    return list(MANIFEST_FIELDS)

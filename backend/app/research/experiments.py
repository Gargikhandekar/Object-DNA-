"""Experiment runner: classical vs. deep vs. hybrid Object DNA.

Experiments (see docs/evaluation.md):

1. Classical local features  -> method ``orb``      (RANSAC inlier count)
2. Texture                   -> method ``texture``  (multi-scale LBP intersection)
   (+ colour baseline        -> method ``color``    (H-S Bhattacharyya))
3. Deep embeddings           -> method ``deep``     (cosine; only if weights installed)
4. Hybrid Object DNA         -> ``hybrid_default`` (hand-set weights) and
                                ``hybrid_tuned`` (weights + normalisation fitted on VALIDATION pairs)
5. Same-instance verification on identity-disjoint val/test pairs.
6. Open-set instance identification against a gallery, with unknown objects.
+  Robustness per acquisition condition (and optional synthetic perturbations).

All thresholds and fused weights are selected on validation data and then
frozen for test. Nothing reported here is hand-entered: every number is
computed from the run.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from app.research.dataset import Dataset, Sample, load_manifest, make_pairs, make_split
from app.services import texture_features as tf
from app.services.embedding_extractor import EmbeddingExtractor
from app.services.evaluation import (
    best_f1_threshold,
    best_open_set_threshold,
    binary_metrics,
    confusion,
    identification_metrics,
    open_set_outcomes,
    roc_summary,
)
from app.services.feature_extraction import FeatureConfig, FeatureSet, extract_features
from app.services.orb_matcher import match_orb
from app.services.preprocessing import PreprocessConfig, dhash, preprocess
from app.services.similarity import COMPONENTS, DEFAULT_RANGES, FusionConfig, fuse

ROBUSTNESS_CONDITIONS = [
    ("same", "Same angle, similar lighting"),
    ("viewpoint_moderate", "Moderate viewpoint change"),
    ("viewpoint_large", "Large viewpoint change"),
    ("lighting", "Different lighting"),
    ("background", "Different background"),
    ("distance", "Different camera distance"),
    ("occlusion", "Partial occlusion"),
    ("blur_lowres", "Mild blur / reduced resolution"),
    ("unregistered", "New, unregistered objects"),
]
PERTURBATIONS = ("occlusion_synthetic", "blur_synthetic", "lowres_synthetic")
BASE_METHODS = ("orb", "texture", "color", "deep")
ALL_METHODS = BASE_METHODS + ("hybrid_default", "hybrid_tuned")
METHOD_COMPONENTS = {"orb": ("geometric",), "texture": ("texture",), "color": ("color",), "deep": ("deep",),
                     "hybrid_default": COMPONENTS, "hybrid_tuned": COMPONENTS}
METHOD_LABELS = {"orb": "ORB + RANSAC (classical)", "texture": "LBP texture", "color": "HSV colour",
                 "deep": "Deep embedding", "hybrid_default": "Hybrid (default weights)",
                 "hybrid_tuned": "Hybrid (validation-tuned)"}


@dataclass
class ExperimentConfig:
    experiment: str = "full"  # full | verification | identification | robustness
    methods: list[str] | None = None
    seed: int = 42
    val_fraction: float = 0.4
    max_pairs: int = 3000
    perturbations: bool = True
    max_perturbed_queries: int = 30
    weight_grid_step: float = 0.1
    workers: int = field(default_factory=lambda: max(1, min(8, os.cpu_count() or 2)))


ProgressFn = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Feature computation with on-disk cache
# ---------------------------------------------------------------------------

def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def perturb(rgb: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = rgb.copy()
    if kind == "occlusion_synthetic":
        ow, oh = int(w * 0.35), int(h * 0.3)
        x0 = int(rng.integers(int(w * 0.2), max(int(w * 0.2) + 1, int(w * 0.8) - ow)))
        y0 = int(rng.integers(int(h * 0.2), max(int(h * 0.2) + 1, int(h * 0.8) - oh)))
        out[y0:y0 + oh, x0:x0 + ow] = rng.integers(40, 200, 3)
    elif kind == "blur_synthetic":
        out = cv2.GaussianBlur(out, (0, 0), 2.5)
    elif kind == "lowres_synthetic":
        small = cv2.resize(out, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


class FeatureBank:
    def __init__(self, cache_dir: Path, prep: PreprocessConfig, feat: FeatureConfig,
                 embedder: EmbeddingExtractor | None):
        self.prep, self.feat, self.embedder = prep, feat, embedder
        self.backbone = embedder.backbone if embedder is not None and embedder.available else None
        cfg_key = json.dumps([prep.as_dict(), feat.as_dict(), self.backbone, tf.LBP_VERSION], sort_keys=True)
        import hashlib
        self.cache_dir = cache_dir / hashlib.sha1(cfg_key.encode()).hexdigest()[:10]
        self.features: dict[str, FeatureSet] = {}
        self.timings: list[dict] = []

    def _key(self, rel: str, variant: str) -> str:
        import hashlib
        return hashlib.sha1(f"{rel}|{variant}".encode()).hexdigest()[:16]

    def get(self, rel: str, rgb_loader: Callable[[], np.ndarray], variant: str = "orig") -> FeatureSet:
        key = self._key(rel, variant)
        if key in self.features:
            return self.features[key]
        d = self.cache_dir / key
        if (d / "orb.npz").exists() and (self.backbone is None or (d / f"embedding_{self.backbone}.npz").exists()):
            fs = FeatureSet.load(d, self.backbone)
        else:
            timings: dict = {}
            t0 = time.perf_counter()
            prep = preprocess(rgb_loader(), self.prep)
            timings["preprocess"] = (time.perf_counter() - t0) * 1000
            fs = extract_features(prep, self.feat, self.embedder, timings)
            fs.save(d)
            self.timings.append(timings)
        self.features[key] = fs
        return fs


# ---------------------------------------------------------------------------
# Pair evidence with timings
# ---------------------------------------------------------------------------

def timed_evidence(q: FeatureSet, r: FeatureSet, ratio: float = 0.75, reproj: float = 6.0) -> tuple[dict, dict]:
    t0 = time.thread_time()
    deep = float(np.dot(q.embedding, r.embedding)) if (q.embedding is not None and r.embedding is not None) else None
    t1 = time.thread_time()
    texture = tf.lbp_similarity(q.lbp, r.lbp)
    t2 = time.thread_time()
    color = tf.bhattacharyya_similarity(q.color, r.color)
    t3 = time.thread_time()
    shape = tf.shape_similarity(q.shape, r.shape)
    t4 = time.thread_time()
    geometric = float(match_orb(q.orb, r.orb, ratio=ratio, reproj_threshold=reproj).inliers)
    t5 = time.thread_time()
    raw = {"deep": deep, "texture": texture, "color": color, "shape": shape, "geometric": geometric}
    times = {"deep": (t1 - t0) * 1000, "texture": (t2 - t1) * 1000, "color": (t3 - t2) * 1000,
             "shape": (t4 - t3) * 1000, "geometric": (t5 - t4) * 1000}
    return raw, times


def method_score(method: str, raw: dict, default_cfg: FusionConfig, tuned_cfg: FusionConfig | None) -> float | None:
    if method == "orb":
        return raw.get("geometric")
    if method in ("texture", "color", "deep"):
        return raw.get(method)
    if method == "hybrid_default":
        return fuse(raw, default_cfg).score
    if method == "hybrid_tuned":
        return fuse(raw, tuned_cfg).score if tuned_cfg is not None else None
    raise ValueError(method)


# ---------------------------------------------------------------------------
# Hybrid tuning on validation pairs
# ---------------------------------------------------------------------------

def _simplex_grid(n: int, step: float) -> np.ndarray:
    k = int(round(1 / step))
    out = [np.array(c, float) / k for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]
    return np.array(out)


def tune_hybrid(raws: list[dict], labels: np.ndarray, available: list[str], step: float,
                accept_hint: float | None = None) -> tuple[FusionConfig, dict]:
    """Fit normalisation ranges + non-negative weights maximising validation ROC-AUC."""
    labels = np.asarray(labels).astype(int)
    ranges = dict(DEFAULT_RANGES)
    range_report = {}
    for c in available:
        vals = np.array([r.get(c) if r.get(c) is not None else np.nan for r in raws], float)
        pos, neg = vals[(labels == 1) & ~np.isnan(vals)], vals[(labels == 0) & ~np.isnan(vals)]
        if len(pos) >= 3 and len(neg) >= 3:
            lo, hi = float(np.percentile(neg, 5)), float(np.percentile(pos, 95))
            if hi - lo > 1e-6:
                ranges[c] = (lo, hi)
        range_report[c] = {"lo": round(ranges[c][0], 5), "hi": round(ranges[c][1], 5)}
    cfg = FusionConfig(ranges=ranges, weights={c: 0.0 for c in COMPONENTS})
    N = np.array([[cfg.normalize(c, r.get(c)) if r.get(c) is not None else np.nan for c in available] for r in raws],
                 float)
    present = ~np.isnan(N)
    Nz = np.nan_to_num(N)
    best_auc, best_w = -1.0, None
    for w in _simplex_grid(len(available), step):
        denom = present @ w
        score = np.where(denom > 0, (Nz @ w) / np.maximum(denom, 1e-12), 0.0)
        if len(np.unique(labels)) < 2:
            break
        auc = roc_auc_score(labels, score)
        if auc > best_auc + 1e-9:
            best_auc, best_w = auc, w
    if best_w is None:
        best_w = np.ones(len(available)) / len(available)
    cfg.weights = {c: 0.0 for c in COMPONENTS}
    cfg.weights.update({c: float(round(w, 4)) for c, w in zip(available, best_w)})
    cfg.source = "validation-tuned"
    return cfg, {"ranges": range_report, "weights": cfg.weights, "validation_roc_auc": round(float(best_auc), 5),
                 "grid_step": step, "components_considered": available}


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run(dataset_root: Path, cfg: ExperimentConfig, prep: PreprocessConfig, feat: FeatureConfig,
        embedder: EmbeddingExtractor | None, results_dir: Path, cache_dir: Path,
        progress: ProgressFn = _noop, run_uid: str | None = None) -> dict:
    t_start = time.perf_counter()
    run_uid = run_uid or uuid.uuid4().hex[:12]
    out_dir = results_dir / run_uid
    out_dir.mkdir(parents=True, exist_ok=True)
    ds: Dataset = load_manifest(dataset_root)
    rng = np.random.default_rng(cfg.seed)
    by_idx: dict[int, Sample] = {s.idx: s for s in ds.samples}

    # ---- features -------------------------------------------------------
    bank = FeatureBank(cache_dir, prep, feat, embedder)
    hashes: dict[int, int] = {}
    feats: dict[int, FeatureSet] = {}
    for n, s in enumerate(ds.samples):
        rgb_cache: dict = {}

        def loader(s=s, rgb_cache=rgb_cache):
            if "rgb" not in rgb_cache:
                rgb_cache["rgb"] = _read_rgb(s.path)
            return rgb_cache["rgb"]
        hashes[s.idx] = dhash(loader())
        feats[s.idx] = bank.get(s.rel_path, loader)
        if n % 10 == 0:
            progress(0.30 * n / len(ds.samples), f"Extracting features ({n}/{len(ds.samples)})")

    deep_available = bank.backbone is not None and all(f.embedding is not None for f in feats.values())
    requested = cfg.methods or list(ALL_METHODS)
    methods = [m for m in requested if m in ALL_METHODS and (m != "deep" or deep_available)]
    unavailable = {m: ("Deep embedding weights not installed; run scripts/download_weights.py"
                       if m == "deep" else "unknown method") for m in requested if m not in methods}
    available_components = ["geometric", "texture", "color"] + (["deep"] if deep_available else [])

    split = make_split(ds, hashes, cfg.val_fraction, cfg.seed)
    split_info = split.describe(ds)
    default_cfg = FusionConfig()
    pool = ThreadPoolExecutor(max_workers=cfg.workers)

    def evid(pairs: list[tuple[int, int]]) -> list[tuple[dict, dict]]:
        return list(pool.map(lambda p: timed_evidence(feats[p[0]], feats[p[1]]), pairs, chunksize=16))

    metrics: dict = {
        "run_uid": run_uid, "dataset_id": ds.dataset_id, "dataset_root": str(ds.root),
        "is_synthetic": ds.is_synthetic, "split": split_info, "methods_evaluated": methods,
        "methods_unavailable": unavailable, "method_labels": METHOD_LABELS,
        "embedding_backbone": bank.backbone, "config": asdict(cfg),
        "preprocessing": prep.as_dict(), "features": feat.as_dict(),
    }

    # ---- validation pairs (always: needed to tune the hybrid) ------------
    progress(0.32, "Building validation pairs")
    val_pairs, val_pair_stats = make_pairs(ds, split.val_objects, hashes, rng, cfg.max_pairs)
    val_ev = evid([(i, j) for i, j, _ in val_pairs])
    val_raw = [r for r, _ in val_ev]
    val_labels = np.array([lab for *_, lab in val_pairs])
    tuned_cfg, tuning = (None, {"message": "Not enough validation pairs"})
    if len(np.unique(val_labels)) == 2:
        tuned_cfg, tuning = tune_hybrid(val_raw, val_labels, available_components, cfg.weight_grid_step)
    metrics["hybrid_tuning"] = tuning

    def scores_for(method: str, raws: list[dict]) -> np.ndarray:
        return np.array([np.nan if (v := method_score(method, r, default_cfg, tuned_cfg)) is None else v
                         for r in raws], float)

    # ---- Experiment 5: verification -------------------------------------
    if cfg.experiment in ("full", "verification"):
        progress(0.40, "Verification: computing test pair evidence")
        test_pairs, test_pair_stats = make_pairs(ds, split.test_objects, hashes, rng, cfg.max_pairs)
        test_ev = evid([(i, j) for i, j, _ in test_pairs])
        test_raw = [r for r, _ in test_ev]
        test_labels = np.array([lab for *_, lab in test_pairs])
        ver: dict = {"val_pairs": val_pair_stats, "test_pairs": test_pair_stats, "methods": {}}
        pair_rows = []
        for m in methods:
            vs, ts = scores_for(m, val_raw), scores_for(m, test_raw)
            vm, tm = ~np.isnan(vs), ~np.isnan(ts)
            if vm.sum() < 2 or tm.sum() < 2:
                ver["methods"][m] = {"available": False}
                continue
            thr = best_f1_threshold(val_labels[vm], vs[vm])
            ver["methods"][m] = {
                "available": True,
                "threshold_selected_on": "validation (max F1)",
                "validation": {**binary_metrics(val_labels[vm], vs[vm], thr), **_roc_lite(val_labels[vm], vs[vm])},
                "test": {**binary_metrics(test_labels[tm], ts[tm], thr), **roc_summary(test_labels[tm], ts[tm])},
                "pairs_with_evidence": int(tm.sum()),
            }
        for (i, j, lab), raw in zip(test_pairs, test_raw):
            pair_rows.append({"split": "test", "image_a": by_idx[i].rel_path, "image_b": by_idx[j].rel_path,
                              "label_same_instance": lab, "same_model": by_idx[i].model == by_idx[j].model,
                              **{f"raw_{k}": v for k, v in raw.items()},
                              **{f"score_{m}": method_score(m, raw, default_cfg, tuned_cfg) for m in methods}})
        _write_csv(out_dir / "verification_pairs.csv", pair_rows)
        metrics["verification"] = ver

    # ---- Experiment 6 + robustness: identification -----------------------
    ident_needed = cfg.experiment in ("full", "identification", "robustness")
    if ident_needed and split.gallery:
        progress(0.55, "Identification: matching queries against the gallery")
        gallery = split.gallery
        all_q = [("val", q) for q in split.queries["val"]] + [("test", q) for q in split.queries["test"]]

        def run_queries(qfeats: list[tuple[str, int, FeatureSet]], p0: float = 0.0, p1: float = 0.0,
                        label: str = "") -> list[dict]:
            out = []
            res: list = []
            step = max(1, len(qfeats) // 20)
            for start in range(0, len(qfeats), step):
                jobs = [(qi, g) for qi in range(start, min(start + step, len(qfeats))) for g in gallery]
                res += list(pool.map(lambda p: timed_evidence(qfeats[p[0]][2], feats[p[1]]), jobs, chunksize=16))
                if p1 > p0:
                    done = min(start + step, len(qfeats))
                    progress(p0 + (p1 - p0) * done / len(qfeats), f"{label} ({done}/{len(qfeats)} queries)")
            for qi, (name, sidx, _) in enumerate(qfeats):
                block = res[qi * len(gallery):(qi + 1) * len(gallery)]
                s = by_idx[sidx]
                rec = {"split": name, "sample": sidx, "true": s.object_id if s.object_id in split.known_objects
                       else None, "object_id": s.object_id, "condition": s.condition, "session": s.session,
                       "ranked": {}, "time_ms": {}}
                comp_time = {c: sum(t[c] for _, t in block) for c in COMPONENTS}
                for m in methods:
                    per_obj: dict[str, float] = {}
                    for g, (raw, _) in zip(gallery, block):
                        v = method_score(m, raw, default_cfg, tuned_cfg)
                        if v is None:
                            continue
                        o = by_idx[g].object_id
                        per_obj[o] = max(per_obj.get(o, -np.inf), v)
                    rec["ranked"][m] = sorted(per_obj.items(), key=lambda kv: kv[1], reverse=True)
                    rec["time_ms"][m] = sum(comp_time[c] for c in METHOD_COMPONENTS[m] if c in comp_time)
                out.append(rec)
            return out

        qrecs = run_queries([(name, q, feats[q]) for name, q in all_q], 0.55, 0.80,
                            "Identification: matching queries against the gallery")
        progress(0.80, "Identification: computing metrics")
        ident: dict = {"gallery_images": len(gallery), "known_objects": len(split.known_objects),
                       "unknown_objects": len(split.unknown_objects), "methods": {},
                       "latency_note": "matching_ms = mean per-thread CPU time to compare one query with the whole "
                                       "gallery (approximates single-threaded latency; excludes I/O); "
                                       "extraction_ms = mean wall-clock per-image preprocessing + extraction time "
                                       "for the extractors the method needs (only for images not served from the "
                                       "feature cache)."}
        ext_ms = _extraction_ms(bank.timings)
        thresholds: dict[str, float] = {}
        for m in methods:
            v = [{"true": r["true"], "ranked": r["ranked"][m]} for r in qrecs if r["split"] == "val" and r["ranked"][m]]
            t = [{"true": r["true"], "ranked": r["ranked"][m]} for r in qrecs if r["split"] == "test" and r["ranked"][m]]
            if not v or not t:
                ident["methods"][m] = {"available": False}
                continue
            thr = best_open_set_threshold(v)
            thresholds[m] = thr
            tm = identification_metrics(t, thr)
            tm["validation"] = identification_metrics(v, thr)
            tm["threshold_selected_on"] = "validation (max open-set accuracy)"
            tm["matching_ms"] = round(float(np.mean([r["time_ms"][m] for r in qrecs if r["split"] == "test"])), 3)
            tm["extraction_ms"] = round(sum(ext_ms.get(k, 0.0) for k in
                                            ["preprocess"] + _extractors_for(m)), 3) if ext_ms else None
            tm["confusion"] = confusion(t, thr, sorted(split.known_objects))
            tm["available"] = True
            ident["methods"][m] = tm
        metrics["identification"] = ident

        # Robustness by condition (test split).
        rob: dict = {"conditions": [], "source": "test-split queries grouped by manifest 'condition'"}
        test_recs = [r for r in qrecs if r["split"] == "test"]
        for cond, label in ROBUSTNESS_CONDITIONS:
            sel = ([r for r in test_recs if r["true"] is None] if cond == "unregistered"
                   else [r for r in test_recs if r["condition"] == cond and r["true"] is not None])
            rob["conditions"].append(_condition_entry(cond, label, sel, methods, thresholds, "dataset"))
        other = sorted({r["condition"] for r in test_recs if r["true"] is not None}
                       - {c for c, _ in ROBUSTNESS_CONDITIONS})
        for cond in other:
            sel = [r for r in test_recs if r["condition"] == cond and r["true"] is not None]
            rob["conditions"].append(_condition_entry(cond, cond, sel, methods, thresholds, "dataset"))

        if cfg.perturbations and cfg.experiment in ("full", "robustness"):
            progress(0.86, "Robustness: synthetic perturbations")
            base = [r for r in test_recs if r["true"] is not None]
            pick = sorted(rng.choice(len(base), min(len(base), cfg.max_perturbed_queries), replace=False)) if base else []
            for kind in PERTURBATIONS:
                qf = []
                for k in pick:
                    s = by_idx[base[k]["sample"]]
                    prng = np.random.default_rng(cfg.seed + s.idx)
                    qf.append(("test", s.idx, bank.get(s.rel_path, lambda s=s, prng=prng, kind=kind:
                                                       perturb(_read_rgb(s.path), kind, prng),
                                                       variant=f"{kind}-seed{cfg.seed}")))
                precs = run_queries(qf)
                label = {"occlusion_synthetic": "Partial occlusion (synthetic perturbation)",
                         "blur_synthetic": "Gaussian blur σ=2.5 (synthetic perturbation)",
                         "lowres_synthetic": "Downscaled to 30% (synthetic perturbation)"}[kind]
                rob["conditions"].append(_condition_entry(kind, label, precs, methods, thresholds, "perturbation"))
        metrics["robustness"] = rob

        q_rows = []
        for r in qrecs:
            row = {"split": r["split"], "image": by_idx[r["sample"]].rel_path, "object_id": r["object_id"],
                   "registered": r["true"] is not None, "condition": r["condition"], "session": r["session"]}
            for m in methods:
                ranked = r["ranked"][m]
                row[f"{m}_top1"] = ranked[0][0] if ranked else None
                row[f"{m}_score"] = round(ranked[0][1], 5) if ranked else None
                if m in thresholds:
                    row[f"{m}_decision"] = "accept" if open_set_outcomes(ranked, r["true"], thresholds[m])["accepted"] else "reject"
            q_rows.append(row)
        _write_csv(out_dir / "identification_queries.csv", q_rows)

        if tuned_cfg is not None and "hybrid_tuned" in thresholds:
            tuned_cfg.accept_threshold = round(thresholds["hybrid_tuned"], 4)
            tuned_cfg.uncertain_threshold = round(max(0.0, thresholds["hybrid_tuned"] - 0.1), 4)
            tuned_cfg.margin_threshold = 0.05
    pool.shutdown()

    if tuned_cfg is not None:
        tuned_cfg.notes = (f"Weights and normalisation ranges fitted on validation pairs of run {run_uid} "
                           f"(dataset {ds.dataset_id}); accept threshold = validation open-set optimum; "
                           f"uncertain = accept - 0.10.")
        tuned_cfg.source = f"validation-tuned:{run_uid}"
        tuned_cfg.save(out_dir / "tuned_fusion.json")

    (out_dir / "split.json").write_text(json.dumps(split_info, indent=2))
    progress(0.93, "Rendering charts")
    artifacts = {"metrics": "metrics.json", "split": "split.json"}
    for name in ("verification_pairs.csv", "identification_queries.csv", "tuned_fusion.json"):
        if (out_dir / name).exists():
            artifacts[name.rsplit(".", 1)[0]] = name
    summary = _summary_table(metrics)
    if not summary.empty:
        summary.to_csv(out_dir / "summary.csv", index=False)
        artifacts["summary"] = "summary.csv"
    try:
        artifacts.update(_charts(metrics, out_dir))
    except Exception as exc:  # pragma: no cover - plotting must never fail a run
        metrics["chart_error"] = str(exc)
    metrics["runtime_s"] = round(time.perf_counter() - t_start, 2)
    metrics["artifacts"] = artifacts
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    progress(1.0, "Completed")
    return metrics


def _summary_table(metrics: dict):
    """One row per method with the headline test metrics (pandas DataFrame)."""
    import pandas as pd

    rows = []
    for m in metrics.get("methods_evaluated", []):
        i = metrics.get("identification", {}).get("methods", {}).get(m, {})
        v = metrics.get("verification", {}).get("methods", {}).get(m, {}).get("test", {})
        rows.append({
            "method": m, "label": METHOD_LABELS.get(m, m),
            "top1": i.get("top1_accuracy"), "top5": i.get("top5_accuracy"),
            "accepted_precision": i.get("accepted_precision"), "accepted_recall": i.get("accepted_recall"),
            "unknown_rejection": i.get("unknown_rejection_rate"), "open_set_accuracy": i.get("open_set_accuracy"),
            "verification_auc": v.get("roc_auc"), "eer": v.get("eer"), "precision": v.get("precision"),
            "recall": v.get("recall"), "f1": v.get("f1"), "fmr": v.get("fmr"), "fnmr": v.get("fnmr"),
            "matching_ms": i.get("matching_ms"), "extraction_ms": i.get("extraction_ms"),
        })
    return pd.DataFrame(rows)


def _roc_lite(labels, scores) -> dict:
    s = roc_summary(labels, scores)
    return {"roc_auc": s["roc_auc"], "eer": s["eer"]}


def _extractors_for(method: str) -> list[str]:
    return {"orb": ["orb"], "texture": ["texture"], "color": ["color"], "deep": ["deep"]}.get(
        method, ["orb", "texture", "color", "deep"])


def _extraction_ms(timings: list[dict]) -> dict[str, float]:
    if not timings:
        return {}
    keys = set().union(*timings)
    return {k: float(np.mean([t.get(k, 0.0) for t in timings])) for k in keys}


def _condition_entry(cond: str, label: str, recs: list[dict], methods: list[str], thresholds: dict,
                     source: str) -> dict:
    entry = {"condition": cond, "label": label, "source": source, "n_queries": len(recs),
             "status": "evaluated" if recs else "pending", "methods": {}}
    if not recs:
        return entry
    for m in methods:
        if m not in thresholds:
            continue
        items = [{"true": r["true"], "ranked": r["ranked"][m]} for r in recs if r["ranked"][m]]
        if not items:
            continue
        im = identification_metrics(items, thresholds[m])
        n = len(items)
        false_matches = 0
        for it in items:
            o = open_set_outcomes(it["ranked"], it["true"], thresholds[m])
            false_matches += int(o["accepted"] and not o["correct"])
        entry["methods"][m] = {
            "top1_accuracy": im["top1_accuracy"],
            "open_set_accuracy": im["open_set_accuracy"],
            "accepted_recall": im["accepted_recall"],
            "unknown_rejection_rate": im["unknown_rejection_rate"],
            "false_match_rate": round(false_matches / n, 5),
            "mean_matching_ms": round(float(np.mean([r["time_ms"][m] for r in recs])), 3),
        }
    return entry


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _charts(metrics: dict, out_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = {}
    ver = metrics.get("verification", {}).get("methods", {})
    if ver:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for m, v in ver.items():
            if v.get("available") and v["test"].get("curve"):
                c = v["test"]["curve"]
                ax.plot([p["fpr"] for p in c], [p["tpr"] for p in c],
                        label=f"{METHOD_LABELS.get(m, m)} (AUC {v['test']['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False match rate (FPR)")
        ax.set_ylabel("True match rate (TPR)")
        ax.set_title("Same-instance verification ROC (test)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "roc_verification.png", dpi=130)
        plt.close(fig)
        files["roc_verification"] = "roc_verification.png"
    rob = metrics.get("robustness", {}).get("conditions", [])
    evaluated = [c for c in rob if c["status"] == "evaluated" and c["condition"] != "unregistered"]
    if evaluated:
        methods = metrics["methods_evaluated"]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        width = 0.8 / max(len(methods), 1)
        x = np.arange(len(evaluated))
        for k, m in enumerate(methods):
            vals = [c["methods"].get(m, {}).get("top1_accuracy") or 0 for c in evaluated]
            ax.bar(x + k * width, vals, width, label=METHOD_LABELS.get(m, m))
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([c["condition"] for c in evaluated], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Top-1 accuracy (test)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, ncol=3)
        ax.set_title("Robustness by acquisition condition")
        fig.tight_layout()
        fig.savefig(out_dir / "robustness_top1.png", dpi=130)
        plt.close(fig)
        files["robustness_top1"] = "robustness_top1.png"
    ident = metrics.get("identification", {}).get("methods", {})
    best = "hybrid_tuned" if ident.get("hybrid_tuned", {}).get("available") else next(
        (m for m, v in ident.items() if v.get("available")), None)
    if best:
        cm = ident[best]["confusion"]
        mat = np.array(cm["matrix"])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(len(cm["cols"])))
        ax.set_yticks(range(len(cm["rows"])))
        ax.set_xticklabels(cm["cols"], rotation=90, fontsize=5)
        ax.set_yticklabels(cm["rows"], fontsize=5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Open-set confusion (test) - {METHOD_LABELS.get(best, best)}")
        fig.tight_layout()
        fig.savefig(out_dir / "confusion.png", dpi=130)
        plt.close(fig)
        files["confusion"] = "confusion.png"
    return files

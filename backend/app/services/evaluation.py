"""Evaluation metrics for verification (1:1) and open-set identification (1:N)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from app.services.identification import MATCH_FOUND, decide


def _nan_to_none(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 5)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Pair verification metrics at a fixed threshold (predict same if score >= t)."""
    labels = np.asarray(labels).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    return {
        "threshold": _nan_to_none(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": _nan_to_none(precision),
        "recall": _nan_to_none(recall),
        "f1": _nan_to_none(f1),
        "fmr": _nan_to_none(fp / (fp + tn)) if fp + tn else None,  # false match rate
        "fnmr": _nan_to_none(fn / (fn + tp)) if fn + tp else None,  # false non-match rate
        "accuracy": _nan_to_none((tp + tn) / max(len(labels), 1)),
    }


def roc_summary(labels: np.ndarray, scores: np.ndarray, max_points: int = 120) -> dict:
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return {"roc_auc": None, "eer": None, "curve": []}
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    k = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[k] + fnr[k]) / 2)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int))
    curve = [{"fpr": round(float(fpr[i]), 5), "tpr": round(float(tpr[i]), 5)} for i in idx]
    return {"roc_auc": round(float(auc), 5), "eer": round(eer, 5), "curve": curve}


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 (used on VALIDATION data only)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, float)
    cands = np.unique(scores)
    if len(cands) > 400:
        cands = np.quantile(scores, np.linspace(0, 1, 400))
    best_t, best_f1 = float(np.median(scores)), -1.0
    for t in cands:
        m = binary_metrics(labels, scores, t)
        f1 = m["f1"] or 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def rank_of_true(object_scores: dict[str, float], true_object: str) -> int | None:
    ranked = sorted(object_scores, key=object_scores.get, reverse=True)
    return ranked.index(true_object) + 1 if true_object in ranked else None


def open_set_outcomes(ranked: list[tuple[str, float]], true_object: str | None, threshold: float,
                      margin: float = 0.0) -> dict:
    d = decide([s for _, s in ranked], threshold, threshold, margin)
    accepted = d.status == MATCH_FOUND
    pred = ranked[0][0] if (accepted and ranked) else None
    return {"accepted": accepted, "predicted": pred, "correct": accepted and pred == true_object}


def identification_metrics(results: list[dict], threshold: float, margin: float = 0.0) -> dict:
    """``results``: [{"true": object_id | None (unknown), "ranked": [(obj, score), ...]}]."""
    known = [r for r in results if r["true"] is not None]
    unknown = [r for r in results if r["true"] is None]
    top1 = top5 = 0
    for r in known:
        ids = [o for o, _ in r["ranked"]]
        top1 += int(ids[:1] == [r["true"]])
        top5 += int(r["true"] in ids[:5])
    acc_correct = acc_wrong = known_rejected = unk_rejected = unk_accepted = 0
    for r in known:
        o = open_set_outcomes(r["ranked"], r["true"], threshold, margin)
        if not o["accepted"]:
            known_rejected += 1
        elif o["correct"]:
            acc_correct += 1
        else:
            acc_wrong += 1
    for r in unknown:
        o = open_set_outcomes(r["ranked"], None, threshold, margin)
        if o["accepted"]:
            unk_accepted += 1
        else:
            unk_rejected += 1
    accepted = acc_correct + acc_wrong + unk_accepted
    n_known, n_unknown = len(known), len(unknown)
    gallery_size = len(results[0]["ranked"]) if results else 0
    return {
        "threshold": round(float(threshold), 5),
        "margin": margin,
        "known_queries": n_known,
        "unknown_queries": n_unknown,
        "top1_accuracy": round(top1 / n_known, 5) if n_known else None,
        "top5_accuracy": round(top5 / n_known, 5) if n_known and gallery_size >= 5 else None,
        "accepted": accepted,
        "accepted_precision": round(acc_correct / accepted, 5) if accepted else None,
        "accepted_recall": round(acc_correct / n_known, 5) if n_known else None,  # detection & identification rate
        "known_rejected_rate": round(known_rejected / n_known, 5) if n_known else None,
        "misidentification_rate": round(acc_wrong / n_known, 5) if n_known else None,
        "unknown_rejection_rate": round(unk_rejected / n_unknown, 5) if n_unknown else None,
        "unknown_false_accept_rate": round(unk_accepted / n_unknown, 5) if n_unknown else None,
        "open_set_accuracy": round((acc_correct + unk_rejected) / max(n_known + n_unknown, 1), 5)
        if (n_known + n_unknown) else None,
    }


def best_open_set_threshold(results: list[dict], margin: float = 0.0) -> float:
    """Threshold maximising open-set accuracy on VALIDATION queries."""
    scores = sorted({s for r in results for _, s in r["ranked"][:1]})
    if not scores:
        return 0.5
    cands = np.unique(np.concatenate([scores, np.array(scores) + 1e-9]))
    if len(cands) > 300:
        cands = np.quantile(cands, np.linspace(0, 1, 300))
    best_t, best = float(cands[0]), -1.0
    for t in cands:
        acc = identification_metrics(results, float(t), margin)["open_set_accuracy"] or 0.0
        if acc > best:
            best, best_t = acc, float(t)
    return best_t


def confusion(results: list[dict], threshold: float, labels: list[str], margin: float = 0.0) -> dict:
    """Rows: true identity (+ "unknown"); columns: predicted identity (+ "rejected")."""
    rows = labels + ["unknown"]
    cols = labels + ["rejected"]
    ri = {k: i for i, k in enumerate(rows)}
    ci = {k: i for i, k in enumerate(cols)}
    m = np.zeros((len(rows), len(cols)), int)
    for r in results:
        o = open_set_outcomes(r["ranked"], r["true"], threshold, margin)
        row = ri.get(r["true"] if r["true"] is not None else "unknown")
        col = ci.get(o["predicted"]) if o["accepted"] and o["predicted"] in ci else ci["rejected"]
        if row is not None:
            m[row, col] += 1
    return {"rows": rows, "cols": cols, "matrix": m.tolist()}

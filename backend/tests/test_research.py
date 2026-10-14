import time

import numpy as np
import pytest

from app.research.dataset import import_folder_dataset, load_manifest, make_pairs, make_split
from app.research.synthetic import generate_dataset
from app.services.evaluation import (
    best_f1_threshold,
    binary_metrics,
    confusion,
    identification_metrics,
    roc_summary,
)
from app.services.preprocessing import dhash


def test_binary_metrics_known_values():
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.4, 0.6])
    m = binary_metrics(labels, scores, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 1, 3, 1)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["fmr"] == pytest.approx(0.25) and m["fnmr"] == pytest.approx(0.25)
    assert m["f1"] == pytest.approx(0.75)


def test_roc_perfect_and_degenerate():
    r = roc_summary(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert r["roc_auc"] == 1.0 and r["eer"] == 0.0
    assert roc_summary(np.array([1, 1]), np.array([0.1, 0.2]))["roc_auc"] is None


def test_best_f1_threshold_separates():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = best_f1_threshold(labels, scores)
    assert 0.3 < t <= 0.7


def test_identification_metrics_open_set():
    res = [
        {"true": "A", "ranked": [("A", 0.9), ("B", 0.2)]},  # correct accept
        {"true": "B", "ranked": [("A", 0.8), ("B", 0.7)]},  # wrong accept, top5 hit
        {"true": "B", "ranked": [("B", 0.3), ("A", 0.1)]},  # correct top1 but rejected
        {"true": None, "ranked": [("A", 0.2), ("B", 0.1)]},  # unknown rejected
        {"true": None, "ranked": [("B", 0.6), ("A", 0.1)]},  # unknown falsely accepted
    ]
    m = identification_metrics(res, 0.5)
    assert m["top1_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["accepted"] == 3 and m["accepted_precision"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["accepted_recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["unknown_rejection_rate"] == 0.5 and m["unknown_false_accept_rate"] == 0.5
    assert m["open_set_accuracy"] == pytest.approx(2 / 5, abs=1e-4)
    assert m["top5_accuracy"] is None  # gallery too small for top-5
    cm = confusion(res, 0.5, ["A", "B"])
    assert cm["matrix"][cm["rows"].index("unknown")][cm["cols"].index("rejected")] == 1
    assert sum(map(sum, cm["matrix"])) == 5


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    generate_dataset(root, n_models=2, instances_per_model=3, unregistered_per_model=1, n_references=2,
                     query_sessions=(("s2", ("same", "lighting")), ("s3", ("viewpoint_moderate",))), seed=3)
    return root


def test_manifest_and_folder_import_agree(small_dataset):
    ds = load_manifest(small_dataset)
    assert len(ds.objects()) == 8 and ds.is_synthetic
    (small_dataset / "manifest.csv").rename(small_dataset / "manifest.orig.csv")
    try:
        import_folder_dataset(small_dataset)
        ds2 = load_manifest(small_dataset)
        assert sorted((s.rel_path, s.session, s.condition, s.registered) for s in ds.samples) == \
            sorted((s.rel_path, s.session, s.condition, s.registered) for s in ds2.samples)
    finally:
        (small_dataset / "manifest.orig.csv").replace(small_dataset / "manifest.csv")


def _hashes(ds):
    import cv2
    return {s.idx: dhash(cv2.imread(str(s.path))[..., ::-1]) for s in ds.samples}


def test_split_is_identity_disjoint_and_leakage_free(small_dataset):
    ds = load_manifest(small_dataset)
    h = _hashes(ds)
    sp = make_split(ds, h, val_fraction=0.4, seed=1)
    assert set(sp.val_objects).isdisjoint(sp.test_objects)
    assert set(sp.val_objects) | set(sp.test_objects) == set(ds.objects())
    by = {s.idx: s for s in ds.samples}
    val_q = {by[i].object_id for i in sp.queries["val"]}
    test_q = {by[i].object_id for i in sp.queries["test"]}
    assert val_q.isdisjoint(test_q)
    # No query is from the same session as its object's gallery images; no gallery image is a query.
    gal = set(sp.gallery)
    for i in sp.queries["val"] + sp.queries["test"]:
        assert i not in gal
        s = by[i]
        if s.object_id in sp.known_objects:
            assert s.session != "s1"
    assert all(by[i].registered for i in sp.gallery)
    assert set(sp.unknown_objects) == {s.object_id for s in ds.samples if not s.registered}
    # Deterministic under the same seed.
    assert make_split(ds, h, 0.4, 1).split_id == sp.split_id


def test_pairs_are_balanced_and_include_hard_negatives(small_dataset):
    ds = load_manifest(small_dataset)
    pairs, stats = make_pairs(ds, ds.objects(), _hashes(ds), np.random.default_rng(0), max_pairs=200)
    labels = [lab for *_, lab in pairs]
    assert sum(labels) == len(labels) - sum(labels)
    assert stats["hard_negatives_same_model"] > 0
    by = {s.idx: s for s in ds.samples}
    for i, j, lab in pairs:
        assert (by[i].object_id == by[j].object_id) == bool(lab)
        if lab:
            assert by[i].session != by[j].session


def test_experiment_runner_end_to_end(small_dataset, tmp_path):
    from app.research import experiments as exp
    from app.services.feature_extraction import FeatureConfig
    from app.services.preprocessing import PreprocessConfig

    progress = []
    t0 = time.time()
    m = exp.run(small_dataset, exp.ExperimentConfig(max_pairs=200, max_perturbed_queries=4, workers=2),
                PreprocessConfig(), FeatureConfig(), None, tmp_path / "results", tmp_path / "cache",
                lambda p, msg: progress.append(p), run_uid="t1")
    assert progress[-1] == 1.0 and time.time() - t0 < 600
    assert "deep" not in m["methods_evaluated"] and "deep" in m["methods_unavailable"]
    ident = m["identification"]["methods"]
    for meth in ("orb", "texture", "color", "hybrid_default", "hybrid_tuned"):
        assert ident[meth]["available"]
        assert 0 <= ident[meth]["top1_accuracy"] <= 1
        assert ident[meth]["threshold_selected_on"].startswith("validation")
    ver = m["verification"]["methods"]["hybrid_tuned"]["test"]
    assert ver["roc_auc"] is not None and ver["fmr"] is not None and ver["fnmr"] is not None
    conds = {c["condition"]: c for c in m["robustness"]["conditions"]}
    assert conds["lighting"]["status"] == "evaluated"
    assert conds["occlusion"]["status"] == "pending"  # not in this dataset -> marked pending, not invented
    assert conds["blur_synthetic"]["status"] == "evaluated"
    out = tmp_path / "results" / "t1"
    for f in ("metrics.json", "split.json", "identification_queries.csv", "verification_pairs.csv",
              "tuned_fusion.json"):
        assert (out / f).exists(), f
    assert abs(sum(m["hybrid_tuning"]["weights"].values()) - 1.0) < 1e-6

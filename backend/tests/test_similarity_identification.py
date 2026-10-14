import numpy as np
import pytest

from app.research.synthetic import synthetic_object_image
from app.services.feature_extraction import FeatureConfig, process_image
from app.services.identification import (
    MATCH_FOUND,
    NO_RELIABLE_MATCH,
    UNCERTAIN,
    ReferenceEntry,
    decide,
    identify,
)
from app.services.preprocessing import PreprocessConfig
from app.services.similarity import FusionConfig, fuse


def test_normalisation_is_clipped_linear():
    cfg = FusionConfig(ranges={**FusionConfig().ranges, "texture": (0.5, 1.0)})
    assert cfg.normalize("texture", 0.4) == 0.0
    assert cfg.normalize("texture", 0.75) == pytest.approx(0.5)
    assert cfg.normalize("texture", 2.0) == 1.0
    assert cfg.normalize("texture", None) is None


def test_fusion_contributions_sum_to_score_and_missing_is_excluded():
    cfg = FusionConfig()
    raw = {"deep": None, "geometric": 79.0, "texture": 0.89, "color": 0.65, "shape": None}
    f = fuse(raw, cfg)
    assert "deep" in f.missing and "deep" not in f.contributions
    assert f.score == pytest.approx(sum(f.contributions.values()))
    assert sum(f.used_weights.values()) == pytest.approx(1.0)
    assert 0.0 <= f.score <= 1.0


def test_fusion_without_any_evidence_returns_none():
    f = fuse({"deep": None, "geometric": None, "texture": None, "color": None, "shape": None}, FusionConfig())
    assert f.score is None


def test_fusion_config_roundtrip(tmp_path):
    cfg = FusionConfig(accept_threshold=0.7, source="x")
    cfg.weights["shape"] = 0.1
    cfg.save(tmp_path / "f.json")
    g = FusionConfig.load(tmp_path / "f.json")
    assert g.accept_threshold == 0.7 and g.weights["shape"] == 0.1 and g.source == "x"


@pytest.mark.parametrize("scores,expected", [
    ([0.9, 0.2], MATCH_FOUND),
    ([0.9, 0.88], UNCERTAIN),  # margin too small
    ([0.45, 0.1], UNCERTAIN),  # between thresholds
    ([0.2, 0.1], NO_RELIABLE_MATCH),
    ([], NO_RELIABLE_MATCH),
    ([0.8], MATCH_FOUND),
])
def test_decision_rule(scores, expected):
    assert decide(scores, 0.6, 0.35, 0.05).status == expected


@pytest.fixture(scope="module")
def gallery():
    pc, fc = PreprocessConfig(), FeatureConfig()
    refs = []
    for k, seed in enumerate((21, 22, 23)):
        for j, cond in enumerate(("same", "viewpoint_moderate")):
            refs.append(ReferenceEntry(k + 1, f"OBJ{seed}", k * 10 + j,
                                       process_image(synthetic_object_image(seed, cond), pc, fc, None)[1]))
    return refs, pc, fc


def test_identify_ranks_true_instance_first(gallery):
    refs, pc, fc = gallery
    q = process_image(synthetic_object_image(22, "lighting"), pc, fc, None)[1]
    res = identify(q, refs, FusionConfig(), top_k=3)
    assert res.candidates[0].object_public_id == "OBJ22"
    assert res.decision.status == MATCH_FOUND
    assert res.candidates[0].matches  # correspondences kept for visualisation
    assert res.candidates[0].references_compared == 2


def test_unknown_object_is_rejected(gallery):
    refs, pc, fc = gallery
    q = process_image(synthetic_object_image(99, "same"), pc, fc, None)[1]
    res = identify(q, refs, FusionConfig())
    assert res.decision.status in (NO_RELIABLE_MATCH, UNCERTAIN)
    assert res.decision.status != MATCH_FOUND


def test_identify_with_empty_gallery():
    pc, fc = PreprocessConfig(), FeatureConfig()
    q = process_image(synthetic_object_image(1), pc, fc, None)[1]
    res = identify(q, [], FusionConfig())
    assert res.decision.status == NO_RELIABLE_MATCH and res.candidates == []


def test_shortlist_limits_orb_matching(gallery):
    refs, pc, fc = gallery
    q = process_image(synthetic_object_image(21, "same"), pc, fc, None)[1]
    res = identify(q, refs, FusionConfig(), orb_shortlist=1, top_k=3)
    with_orb = [c for c in res.candidates if c.evidence.get("orb")]
    assert len(with_orb) == 1
    assert np.isfinite(res.candidates[0].score)

import numpy as np

from app.research.synthetic import synthetic_object_image
from app.services import texture_features as tf
from app.services.feature_extraction import FeatureConfig, FeatureSet, extract_features, process_image
from app.services.orb_matcher import extract_orb, match_orb
from app.services.preprocessing import PreprocessConfig, dhash, hamming, preprocess

PC, FC = PreprocessConfig(), FeatureConfig()


def feats(seed, cond="same", model_seed=None):
    return process_image(synthetic_object_image(seed, cond, model_seed), PC, FC, None)[1]


def test_preprocess_resizes_and_reports_quality():
    p = preprocess(synthetic_object_image(1), PC)
    assert max(p.rgb.shape[:2]) == PC.max_side
    assert p.gray.dtype == np.uint8 and p.gray.shape == p.rgb.shape[:2]
    assert "sharpness_laplacian_var" in p.quality


def test_orb_extraction_and_self_match_is_geometrically_consistent():
    p = preprocess(synthetic_object_image(1), PC)
    f = extract_orb(p.gray, p.feature_mask)
    assert f.count > 100 and f.descriptors.shape[1] == 32
    res = match_orb(f, f)
    assert res.inliers >= 0.9 * res.good_matches > 50


def test_orb_handles_featureless_image():
    blank = np.full((200, 200), 128, np.uint8)
    f = extract_orb(blank)
    assert f.count == 0
    res = match_orb(f, f)
    assert res.inliers == 0 and res.model is None


def test_lbp_histogram_properties():
    p = preprocess(synthetic_object_image(2), PC)
    h = tf.lbp_histogram(p.gray)
    assert h.shape == (sum(P + 2 for P, _ in tf.LBP_SCALES),)
    assert np.isclose(h.sum(), len(tf.LBP_SCALES), atol=1e-4)
    assert tf.lbp_similarity(h, h) == 1.0 or np.isclose(tf.lbp_similarity(h, h), 1.0)


def test_lbp_rotation_invariance_on_90_degree_rotation():
    p = preprocess(synthetic_object_image(3), PC)
    a = tf.lbp_histogram(p.gray)
    b = tf.lbp_histogram(np.ascontiguousarray(np.rot90(p.gray)))
    assert tf.lbp_similarity(a, b) > 0.97


def test_color_similarity_bounds():
    a = tf.color_histogram(synthetic_object_image(1))
    b = tf.color_histogram(synthetic_object_image(4))
    assert np.isclose(tf.bhattacharyya_similarity(a, a), 1.0, atol=1e-5)
    s = tf.bhattacharyya_similarity(a, b)
    assert 0.0 <= s < 1.0


def test_same_instance_gets_more_inliers_than_sibling_and_other_model():
    ref = feats(10, "same")
    same = match_orb(feats(10, "viewpoint_moderate").orb, ref.orb).inliers
    sibling = match_orb(feats(11, "viewpoint_moderate", model_seed=10).orb, ref.orb).inliers
    other = match_orb(feats(12, "viewpoint_moderate").orb, ref.orb).inliers
    assert same > sibling > other


def test_fingerprint_persistence_roundtrip(tmp_path):
    f = feats(5)
    f.embedding = np.ones(8, np.float32) / np.sqrt(8)
    f.embedding_backbone, f.embedding_version = "resnet50", "test"
    paths = f.save(tmp_path / "fp")
    assert {"orb", "lbp", "color", "embedding:resnet50"} <= set(paths)
    g = FeatureSet.load(tmp_path / "fp", "resnet50")
    assert np.array_equal(f.orb.descriptors, g.orb.descriptors)
    assert np.allclose(f.lbp, g.lbp) and np.allclose(f.color, g.color)
    assert np.allclose(f.embedding, g.embedding)
    assert FeatureSet.load(tmp_path / "fp", None).embedding is None


def test_dhash_near_duplicate_detection():
    a = synthetic_object_image(1)
    assert hamming(dhash(a), dhash(a)) == 0
    assert hamming(dhash(a), dhash(np.clip(a.astype(int) + 3, 0, 255).astype(np.uint8))) <= 4
    assert hamming(dhash(a), dhash(synthetic_object_image(7, "viewpoint_large"))) > 4


def test_extract_features_records_versions():
    f = extract_features(preprocess(synthetic_object_image(1), PC), FC, None)
    v = f.extractor_versions()
    assert v["orb"].startswith("orb-") and v["lbp"].startswith("lbp-")

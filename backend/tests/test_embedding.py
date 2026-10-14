"""Embedding extractor tests. They never download weights: a randomly
initialised network is used ONLY to test shapes, normalisation and plumbing."""

import numpy as np
import pytest

from app.services.embedding_extractor import (
    EmbeddingExtractor,
    EmbeddingUnavailable,
    cosine_similarity,
    l2_normalize,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_missing_weights_reported_not_downloaded(tmp_path):
    e = EmbeddingExtractor("resnet50", tmp_path)
    st = e.status()
    assert st["available"] is False
    assert "download_weights.py" in st["message"]
    with pytest.raises(EmbeddingUnavailable):
        e.load()
    assert not any(tmp_path.iterdir())  # nothing was downloaded


def test_unknown_backbone(tmp_path):
    assert EmbeddingExtractor("nope", tmp_path).status()["available"] is False


def test_random_init_embedding_is_l2_normalised(tmp_path):
    e = EmbeddingExtractor("resnet50", tmp_path, allow_random_init=True)
    rng = np.random.default_rng(0)
    imgs = [(rng.random((180, 240, 3)) * 255).astype(np.uint8) for _ in range(2)]
    emb = e.embed(imgs)
    assert emb.shape == (2, 2048)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
    assert "RANDOM-INIT" in e.version  # never mistaken for a pretrained model


def test_cosine_helpers():
    a, b = np.array([3.0, 4.0]), np.array([4.0, 3.0])
    assert np.isclose(np.linalg.norm(l2_normalize(a)), 1.0)
    assert np.isclose(cosine_similarity(a, a), 1.0)
    assert np.isclose(cosine_similarity(a, b), 24 / 25)


def test_backfill_adds_embeddings_to_existing_references(settings, img_bytes):
    """Objects registered before weights were installed get embeddings via backfill."""
    from fastapi.testclient import TestClient

    from app.core.runtime import build_runtime, set_runtime
    from app.main import create_app
    from tests.conftest import register

    with TestClient(create_app()) as c:
        assert register(c, "Bottle", [("a.jpg", img_bytes(1)), ("b.jpg", img_bytes(1, "lighting"))]).status_code == 201
        assert c.post("/api/settings/backfill-embeddings").status_code == 409  # no encoder yet

        embedder = EmbeddingExtractor("resnet50", settings.resolved_weights_dir, allow_random_init=True)
        set_runtime(build_runtime(settings, embedder))
        r = c.post("/api/settings/backfill-embeddings")
        assert r.status_code == 200 and r.json()["computed"] == 2
        assert c.post("/api/settings/backfill-embeddings").json()["computed"] == 0  # idempotent

        res = c.post("/api/identify", files={"file": ("q.jpg", img_bytes(1, "viewpoint_moderate"), "image/jpeg")}).json()
        assert res["candidates"][0]["evidence"]["raw"]["deep"] is not None

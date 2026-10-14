"""Shared fixtures. Every test runs against a fresh temporary data directory
and SQLite database; no network access or model download is required."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.research.synthetic import synthetic_object_image


def encode(rgb: np.ndarray, ext: str = ".jpg") -> bytes:
    ok, buf = cv2.imencode(ext, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


@pytest.fixture
def img_bytes():
    """img_bytes(seed, condition) -> JPEG bytes of a synthetic physical instance."""
    def make(seed: int, condition: str = "same", model_seed: int | None = None) -> bytes:
        return encode(synthetic_object_image(seed, condition, model_seed))
    return make


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("ODNA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ODNA_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("ODNA_EMBEDDING_BACKBONE", "none")
    from app.core import config
    from app.core.runtime import set_runtime
    from app.database import reset_engine

    config.get_settings.cache_clear()
    reset_engine()
    set_runtime(None)
    s = config.get_settings()
    s.ensure_dirs()
    yield s
    reset_engine()
    set_runtime(None)
    config.get_settings.cache_clear()


@pytest.fixture
def client(settings):
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


def register(client, name: str, files: list[tuple[str, bytes]], category: str = "bottle", **extra):
    return client.post("/api/objects", data={"name": name, "category": category, **extra},
                       files=[("files", (fn, data, "image/jpeg")) for fn, data in files])

"""Object DNA FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import experiments, identify, misc, objects
from app.core.config import get_settings
from app.database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_settings().ensure_dirs()
    init_db()
    experiments.mark_interrupted_runs()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Object DNA API",
        version=misc.VERSION,
        description=(
            "Visual fingerprinting for instance-level physical object recognition. Similarity scores returned by "
            "this API are fused similarity scores, not calibrated probabilities."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])
    for r in (objects.router, identify.router, misc.router, experiments.router):
        app.include_router(r)
    return app


app = create_app()

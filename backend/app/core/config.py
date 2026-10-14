"""Application configuration.

All settings can be overridden through environment variables prefixed with
``ODNA_`` (see ``.env.example`` at the repository root).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ODNA_",
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths -----------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    storage_dir: Path | None = None  # defaults to <data_dir>/storage
    weights_dir: Path | None = None  # defaults to <data_dir>/models
    experiments_dir: Path = REPO_ROOT / "experiments"
    database_url: str | None = None  # defaults to sqlite:///<data_dir>/object_dna.db

    # --- Upload validation ----------------------------------------------
    max_upload_mb: float = 15.0
    max_images_per_request: int = 30
    min_image_side: int = 64
    max_image_pixels: int = 40_000_000

    # --- Preprocessing ---------------------------------------------------
    processing_max_side: int = 640
    use_foreground_mask: bool = True
    crop_to_foreground: bool = False

    # --- Feature extraction ---------------------------------------------
    orb_nfeatures: int = 1500
    orb_ratio: float = 0.75
    ransac_reproj_threshold: float = 6.0

    # Deep embedding backbone: "resnet50", "dinov2_vits14" or "none".
    embedding_backbone: str = "resnet50"
    device: str = "cpu"

    # --- Identification decision ----------------------------------------
    accept_threshold: float = 0.55
    uncertain_threshold: float = 0.35
    margin_threshold: float = 0.05
    top_k: int = 5
    # Run ORB matching only against the N best objects according to the
    # cheap global descriptors. 0 means "match against every object".
    orb_shortlist: int = 0

    # --- API -------------------------------------------------------------
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    @property
    def resolved_storage_dir(self) -> Path:
        return Path(self.storage_dir or (self.data_dir / "storage"))

    @property
    def resolved_weights_dir(self) -> Path:
        return Path(self.weights_dir or (self.data_dir / "models"))

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'object_dna.db').as_posix()}"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def active_fusion_path(self) -> Path:
        return self.experiments_dir / "configs" / "active_fusion.json"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.resolved_storage_dir, self.resolved_weights_dir,
                     self.experiments_dir / "results", self.experiments_dir / "configs"):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()

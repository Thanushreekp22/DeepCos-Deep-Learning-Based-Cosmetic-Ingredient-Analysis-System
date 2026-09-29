"""
Backend configuration for the DeepCos API  (Module 5).

Values come from environment variables (see ``.env.example``) with sensible
defaults so ``uvicorn backend.main:app`` works out of the box.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # optional: load a local .env file when python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass

from ml import config as ml_config

PROJECT_ROOT: Path = ml_config.PROJECT_ROOT
ARTIFACT_DIR: Path = ml_config.ARTIFACT_DIR
REPORT_DIR: Path = ml_config.REPORT_DIR
MODELS_DIR: Path = ml_config.MODEL_DIR

API_TITLE = "DeepCos API"
API_DESCRIPTION = (
    "DeepCos - Deep Learning based cosmetic ingredient analysis. "
    "CNN text localisation + OCR, an Embedding->LSTM->MLP ingredient model for "
    "formulation-profile prediction, and a cosmetic knowledge base for "
    "ingredient functions and potential-concern screening."
)
API_VERSION = "1.0.0"

HOST: str = os.getenv("DEEPCOS_HOST", "127.0.0.1")
PORT: int = int(os.getenv("DEEPCOS_PORT", "8000"))

CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv(
        "DEEPCOS_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if origin.strip()
]

# --- persistence -----------------------------------------------------------
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
MONGODB_DB: str = os.getenv("MONGODB_DB", "deepcos")
MONGODB_TIMEOUT_MS: int = int(os.getenv("MONGODB_TIMEOUT_MS", "1500"))

# Fallback store used when MongoDB is unreachable, so the demo never breaks
JSON_STORE_PATH: Path = ARTIFACT_DIR / "reports" / "analyses_store.json"
ANALYSIS_ASSET_DIR: Path = ARTIFACT_DIR / "reports" / "analyses"

MAX_UPLOAD_BYTES: int = int(os.getenv("DEEPCOS_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))

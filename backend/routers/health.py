"""
System endpoints: liveness, pipeline status and model metadata.

* ``GET  /api/health``        - overall status of every component
* ``GET  /api/model/info``    - what is trained / available
* ``GET  /api/model/metrics`` - saved training metrics (JSON)
* ``GET  /api/model/curves``  - training-curves PNG
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend import config
from backend.services.analysis_service import store_status
from ml.inference import model_status

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", summary="Liveness + component status")
def health() -> dict:
    status = model_status()
    return {
        "status": "ok" if status.get("ingredient_model") else "degraded",
        "api": {"version": config.API_VERSION, "title": config.API_TITLE},
        "database": store_status(),
        "knowledge_base": status.get("knowledge_base", {}),
        "image_pipeline": {
            "text_region_cnn": status.get("text_region_cnn", False),
            "label_category_cnn": status.get("label_category_cnn", False),
            "text_detector_loaded": status.get("text_detector_loaded", False),
            "ocr": status.get("ocr", {}),
        },
        "ingredient_model": {
            "trained": bool(status.get("ingredient_model")),
            "path": status.get("ingredient_model_path"),
            "hint": status.get("hint", ""),
        },
    }


@router.get("/model/info", summary="Ingredient-model availability and capabilities")
def model_info() -> dict:
    status = model_status()
    bundle_status: dict = {}
    try:
        from ml.inference import get_ingredient_model

        bundle = get_ingredient_model()
        if bundle is not None:
            bundle_status = bundle.summary() if hasattr(bundle, "summary") else {}
    except Exception:
        bundle_status = {}
    return {**status, "capabilities": bundle_status}


@router.get("/model/metrics", summary="Saved training metrics for all models")
def model_metrics() -> dict:
    payload: dict = {}
    for key, filename in (
        ("ingredient_model", "ingredient_model_metrics.json"),
        ("text_region_cnn", "textregion_cnn_metrics.json"),
        ("label_cnn", "label_cnn_metrics.json"),
    ):
        path = config.MODELS_DIR / filename
        if path.exists():
            payload[key] = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        raise HTTPException(status_code=404, detail="No training metrics found yet.")
    return payload


@router.get("/model/curves", summary="Training-curves image (PNG)")
def model_curves() -> FileResponse:
    path = config.REPORT_DIR / "ingredient_training_curves.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Training curves not found.")
    return FileResponse(path, media_type="image/png", filename=path.name)

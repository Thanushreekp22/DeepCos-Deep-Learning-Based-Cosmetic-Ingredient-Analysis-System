"""Pydantic request/response models for the DeepCos API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalyzeTextRequest(BaseModel):
    """Body for ``POST /api/analyze/text``."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Ingredient list (INCI-style) as typed or pasted by the user.",
        examples=["Aqua, Glycerin, Niacinamide, Sodium Hyaluronate, Panthenol, Parfum"],
    )
    explain: bool = Field(True, description="Run occlusion explainability.")
    persist: bool = Field(True, description="Store the report (MongoDB / JSON fallback).")

    model_config = {"json_schema_extra": {"examples": [{"text": "Aqua, Glycerin, Niacinamide", "explain": True}]}}


class AnalyzeImageResponse(BaseModel):
    """Thin wrapper so OpenAPI documents that an image analysis returns a report."""

    report: dict[str, Any] = Field(..., description="Full DeepCos analysis report.")


class HealthResponse(BaseModel):
    status: str
    api: str
    database: dict[str, Any]
    knowledge_base: dict[str, Any]
    image_pipeline: dict[str, Any]
    ingredient_model: dict[str, Any]


class ErrorResponse(BaseModel):
    detail: str
    hint: Optional[str] = None

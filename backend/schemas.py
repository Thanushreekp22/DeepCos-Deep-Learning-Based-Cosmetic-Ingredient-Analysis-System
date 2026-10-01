"""Pydantic request/response models for the DeepCos API."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


# ---------------------------------------------------------------------------
# AI ingredient enrichment (optional Groq fallback)
# ---------------------------------------------------------------------------
_IDENTIFICATION_ALIASES = {
    "inci": "inci",
    "inci name": "inci",
    "in ci": "inci",
    "trade name": "trade_name",
    "tradename": "trade_name",
    "trade_name": "trade_name",
    "botanical": "botanical",
    "botanical name": "botanical",
    "plant": "botanical",
    "unknown": "unknown",
}

# Hard caps so a verbose model response can never bloat the cache or the report
_MAX_DESCRIPTION = 500
_MAX_NOTE = 400
_MAX_LIST_ITEMS = 12
_MAX_LIST_ITEM_LEN = 80


def _clean_text(value: Any, limit: int) -> str:
    """Conservative string clean-up: text only, trimmed, length-capped."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.split())[:limit]


def _clean_list(value: Any) -> list[str]:
    """List clean-up: accept a bare string, drop empties, de-duplicate, cap."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _clean_text(item, _MAX_LIST_ITEM_LEN)
        if text and text.lower() not in {existing.lower() for existing in out}:
            out.append(text)
        if len(out) >= _MAX_LIST_ITEMS:
            break
    return out


class EnrichmentData(BaseModel):
    """
    Validated payload for one AI-enriched ingredient.

    Every field is deliberately conservative: missing or uncertain information
    is an empty string / empty list, and ``confidence`` defaults to ``low``.
    This data is informational only - it never feeds the model, the concern
    rules, or any regulatory determination.
    """

    model_config = ConfigDict(extra="ignore")

    inci_name: str = ""
    identification: Literal["inci", "trade_name", "botanical", "unknown"] = "unknown"
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    functions: list[str] = Field(default_factory=list)
    possible_concerns: list[str] = Field(default_factory=list)
    regulatory_note: str = ""
    confidence: Literal["low", "medium", "high"] = "low"
    evidence_note: str = ""

    @model_validator(mode="before")
    @classmethod
    def _sanitise(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        identification = _clean_text(payload.get("identification"), 40).lower().replace("_", " ")
        confidence = _clean_text(payload.get("confidence"), 10).lower()
        return {
            "inci_name": _clean_text(payload.get("inci_name"), 200),
            "identification": _IDENTIFICATION_ALIASES.get(identification, "unknown"),
            "aliases": _clean_list(payload.get("aliases")),
            "description": _clean_text(payload.get("description"), _MAX_DESCRIPTION),
            "functions": _clean_list(payload.get("functions")),
            "possible_concerns": _clean_list(payload.get("possible_concerns")),
            "regulatory_note": _clean_text(payload.get("regulatory_note"), _MAX_NOTE),
            "confidence": confidence if confidence in {"low", "medium", "high"} else "low",
            "evidence_note": _clean_text(payload.get("evidence_note"), _MAX_NOTE),
        }

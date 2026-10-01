"""
Service layer between the FastAPI routers and the ML pipeline.

Responsibilities
----------------
* call :mod:`ml.inference` for text / image analyses;
* persist reports through :class:`backend.database.AnalysisStore`
  (MongoDB with JSON fallback) **and** as readable files on disk
  (``artifacts/reports/<id>.json`` / ``.md`` / ``.txt``);
* expose report retrieval / listing / deletion for the history UI.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Optional

from backend import config
from backend.database import AnalysisStore, get_store
from backend.services.ingredient_enrichment import get_enricher
from ml.inference import analyze as run_analysis
from ml.rules.report_builder import to_markdown, to_text

EXPORT_FORMATS = {"markdown": ".md", "md": ".md", "text": ".txt", "txt": ".txt", "json": ".json"}


class AnalysisError(RuntimeError):
    """Raised when the ML pipeline cannot fulfil a request (mapped to HTTP 422/503)."""

    def __init__(self, message: str, status_code: int = 422, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.hint = hint


# -- run analyses -----------------------------------------------------------
def analyse_text(text: str, *, explain: bool = True, persist: bool = True) -> dict:
    """Run the ingredient-analysis pipeline on a typed / pasted list."""
    try:
        report = run_analysis(text=text, explain=explain, enricher=get_enricher())
    except RuntimeError as exc:  # model not trained yet
        raise AnalysisError(str(exc), status_code=503, hint="Train the ingredient model first: python -m ml.train_ingredient_model") from exc
    except ValueError as exc:
        raise AnalysisError(str(exc), status_code=422) from exc
    return save_report(report) if persist else report


def analyse_image(
    image_bytes: bytes,
    *,
    filename: str | None = None,
    text: str | None = None,
    explain: bool = True,
    persist: bool = True,
    max_ocr_variants: int | None = 5,
) -> dict:
    """Run CNN text localisation -> OCR -> ingredient analysis on an uploaded image."""
    if not image_bytes:
        raise AnalysisError("The uploaded file is empty.", status_code=422)
    try:
        report = run_analysis(
            text=text,
            image_bytes=image_bytes,
            explain=explain,
            max_ocr_variants=max_ocr_variants,
            enricher=get_enricher(),
        )
    except RuntimeError as exc:
        raise AnalysisError(
            str(exc),
            status_code=503,
            hint="Train the ingredient model first: python -m ml.train_ingredient_model",
        ) from exc
    except ValueError as exc:
        raise AnalysisError(
            str(exc),
            status_code=422,
            hint="Check the photo (lighting, focus) or paste the ingredient list instead.",
        ) from exc
    except Exception as exc:  # unreadable image etc.
        traceback.print_exc()
        raise AnalysisError(
            f"Could not process the image ({exc.__class__.__name__}: {exc}).",
            status_code=422,
            hint="Upload a PNG/JPEG photo of the ingredient list.",
        ) from exc
    if filename:
        report.setdefault("input", {})["filename"] = filename
    return save_report(report) if persist else report


# -- persistence ------------------------------------------------------------
def save_report(report: dict, store: AnalysisStore | None = None) -> dict:
    """
    Store the report in the database and export JSON / Markdown / text copies
    under ``artifacts/reports/``.
    """
    store = store or get_store()
    analysis_id = report.get("analysis_id") or "analysis"
    folder = config.REPORT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    try:
        (folder / f"{analysis_id}.md").write_text(to_markdown(report), encoding="utf-8")
        (folder / f"{analysis_id}.txt").write_text(to_text(report), encoding="utf-8")
    except Exception:  # export failures must never break an analysis
        traceback.print_exc()
    stored = store.save(report)  # strips base64 previews into PNG assets
    try:
        (folder / f"{analysis_id}.json").write_text(
            json.dumps(stored, indent=1), encoding="utf-8"
        )
    except Exception:
        traceback.print_exc()
    return report  # the caller still receives the in-memory report (with previews)


# -- retrieval --------------------------------------------------------------
def get_report(analysis_id: str) -> dict | None:
    return get_store().get(analysis_id)


def list_reports(limit: int = 50, offset: int = 0) -> list[dict]:
    return get_store().list(limit=limit, offset=offset)


def delete_report(analysis_id: str) -> bool:
    removed = get_store().delete(analysis_id)
    folder = config.REPORT_DIR
    for suffix in (".json", ".md", ".txt"):
        path = folder / f"{analysis_id}{suffix}"
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
    return removed


def report_bytes(analysis_id: str, fmt: str = "markdown") -> tuple[bytes, str, str] | None:
    """
    Render an export of a stored report.

    Returns ``(payload, media_type, filename)`` or ``None`` when the analysis
    does not exist. ``fmt`` is one of ``markdown`` / ``text`` / ``json``.
    """
    report = get_report(analysis_id)
    if report is None:
        return None
    fmt = (fmt or "markdown").lower()
    if fmt not in EXPORT_FORMATS:
        raise AnalysisError(
            f"Unknown export format '{fmt}'. Use markdown, text or json.",
            status_code=400,
        )
    if fmt == "json":
        payload = json.dumps(report, indent=1).encode("utf-8")
        return payload, "application/json", f"{analysis_id}.json"
    rendered = to_text(report) if fmt == "text" else to_markdown(report)
    media = "text/plain; charset=utf-8"
    return rendered.encode("utf-8"), media, f"{analysis_id}{EXPORT_FORMATS[fmt]}"


def store_status() -> dict:
    return get_store().status()


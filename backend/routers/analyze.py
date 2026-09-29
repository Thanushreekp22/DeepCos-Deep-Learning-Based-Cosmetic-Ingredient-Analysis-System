"""
Analysis endpoints - the core of the DeepCos API.

* ``POST /api/analyze/text``            - ingredient text -> full report
* ``POST /api/analyze/image``           - product photo -> CNN + OCR -> report
* ``GET  /api/analyses``                - history listing (MongoDB / JSON)
* ``GET  /api/analyses/{id}``           - one stored report
* ``DELETE /api/analyses/{id}``         - remove a report
* ``GET  /api/analyses/{id}/export``    - Markdown / text / JSON download
* ``GET  /api/analyses/{id}/image/{n}`` - stored preview PNGs
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from backend import config
from backend.database import IMAGE_KEYS, get_store
from backend.schemas import AnalyzeTextRequest
from backend.services import analysis_service
from backend.services.analysis_service import AnalysisError

router = APIRouter(prefix="/api", tags=["analysis"])


def _error(exc: AnalysisError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "hint": exc.hint},
    )


@router.post("/analyze/text", summary="Analyse a typed ingredient list")
def analyze_text(request: AnalyzeTextRequest):
    try:
        report = analysis_service.analyse_text(
            request.text, explain=request.explain, persist=request.persist
        )
    except AnalysisError as exc:
        return _error(exc)
    return report


@router.post("/analyze/image", summary="Analyse a product-label photo")
async def analyze_image(
    file: UploadFile = File(..., description="PNG/JPEG photo of the ingredient list"),
    text: str | None = Form(None, description="Optional corrected ingredient text"),
    explain: bool = Form(True),
    persist: bool = Form(True),
    max_ocr_variants: int = Form(5),
):
    raw = await file.read()
    if len(raw) > config.MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": f"File too large (limit {config.MAX_UPLOAD_BYTES} bytes)."},
        )
    try:
        report = analysis_service.analyse_image(
            raw,
            filename=file.filename,
            text=text,
            explain=explain,
            persist=persist,
            max_ocr_variants=max(1, min(max_ocr_variants, 8)),
        )
    except AnalysisError as exc:
        return _error(exc)
    return report


@router.get("/analyses", summary="List stored analyses (history)")
def list_analyses(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    items = analysis_service.list_reports(limit=limit, offset=offset)
    return {"count": len(items), "offset": offset, "items": items}


@router.get("/analyses/{analysis_id}", summary="Fetch one stored report")
def get_analysis(analysis_id: str):
    report = analysis_service.get_report(analysis_id)
    if report is None:
        return JSONResponse(status_code=404, content={"detail": "Analysis not found."})
    return report


@router.delete("/analyses/{analysis_id}", summary="Delete a stored analysis")
def delete_analysis(analysis_id: str):
    if not analysis_service.delete_report(analysis_id):
        return JSONResponse(status_code=404, content={"detail": "Analysis not found."})
    return {"deleted": analysis_id}


@router.get("/analyses/{analysis_id}/export", summary="Download report as Markdown/text/JSON")
def export_analysis(
    analysis_id: str,
    format: str = Query("markdown", pattern="^(markdown|text|json)$"),
):
    try:
        result = analysis_service.report_bytes(analysis_id, format)
    except AnalysisError as exc:
        return _error(exc)
    if result is None:
        return JSONResponse(status_code=404, content={"detail": "Analysis not found."})
    payload, media_type, filename = result
    return Response(
        content=payload,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/analyses/{analysis_id}/image/{name}", summary="Stored preview PNGs")
def analysis_image(analysis_id: str, name: str):
    path = get_store().asset_path(analysis_id, name)
    if path is None:
        return JSONResponse(status_code=404, content={"detail": "Image not found."})
    return Response(content=path.read_bytes(), media_type="image/png")


@router.post("/analyses/{analysis_id}/preview", summary="Inline base64 previews (small payloads)")
def analysis_preview(analysis_id: str):
    """
    Return the stored preview PNGs as inline data-URLs. The UI can drop the
    returned strings straight into ``<img src>`` without extra requests.
    """
    report = analysis_service.get_report(analysis_id)
    if report is None:
        return JSONResponse(status_code=404, content={"detail": "Analysis not found."})
    images: dict[str, str] = {}
    for key in IMAGE_KEYS:
        path = get_store().asset_path(analysis_id, key)
        if path is not None:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            images[key] = f"data:image/png;base64,{encoded}"
    return {"analysis_id": analysis_id, "images": images}


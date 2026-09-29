"""Service layer: wires the ML pipeline into the FastAPI routers."""

from backend.services.analysis_service import (
    delete_report,
    get_report,
    list_reports,
    report_bytes,
    save_report,
    analyse_image,
    analyse_text,
)

__all__ = [
    "analyse_text",
    "analyse_image",
    "get_report",
    "list_reports",
    "save_report",
    "delete_report",
    "report_bytes",
]

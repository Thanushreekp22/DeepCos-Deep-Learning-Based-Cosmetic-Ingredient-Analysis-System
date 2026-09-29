"""
DeepCos FastAPI application  (Module 5).

Run it with::

    python -m uvicorn backend.main:app --reload --port 8000
    # or
    python -m backend.main

Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import config
from backend.routers import analyze, health, knowledge
from backend.services.analysis_service import AnalysisError


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover - startup bookkeeping
    from backend.database import get_store
    from ml.inference import model_status

    status = model_status()
    store = get_store()
    print(f"[deepcos] API '{config.API_TITLE}' v{config.API_VERSION} starting")
    print(f"[deepcos] persistence backend: {store.backend}")
    print(f"[deepcos] ingredient model trained: {status.get('ingredient_model')}")
    print(f"[deepcos] OCR available: {status.get('ocr', {}).get('available')}")
    print(f"[deepcos] text detector loaded: {status.get('text_detector_loaded')}")
    if not status.get("ingredient_model"):
        print("[deepcos] WARNING - ingredient model missing; run: python -m ml.train_ingredient_model")
    yield


app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(knowledge.router)
app.include_router(analyze.router)


@app.exception_handler(AnalysisError)
async def analysis_error_handler(request: Request, exc: AnalysisError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message, "hint": exc.hint})


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "name": config.API_TITLE,
        "version": config.API_VERSION,
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("backend.main:app", host=config.HOST, port=config.PORT, reload=False)

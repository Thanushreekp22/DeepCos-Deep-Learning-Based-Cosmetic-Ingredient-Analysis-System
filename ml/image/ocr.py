"""
OCR stage of DeepCos  (Module 1, after the CNN text localisation).

Design
------
* The native Tesseract engine is optional. :func:`ocr_status` reports whether it
  is usable so the API and UI can react gracefully (the text path always works).
* OCR is run over **all** preprocessing variants produced by
  :func:`ml.image.preprocess.prepare_variants` and the most *plausible* result is
  kept. Plausibility is scored with the ingredient parser: the variant that
  yields the most ingredient-looking names (and the highest Tesseract confidence)
  wins. This is far more robust than one fixed preprocessing chain.
* Nothing here invents text: if the engine is unavailable or finds nothing, the
  result is reported as empty and ``engine`` says so.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ml import config
from ml.dataset.preprocessing import clean_raw_text, parse_ingredient_text
from ml.image import preprocess

try:  # pytesseract is a pure wrapper; the engine itself may be missing
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore[assignment]


@dataclass
class OcrResult:
    """Outcome of the OCR stage."""

    text: str = ""
    engine: str = "unavailable"
    available: bool = False
    command: str = ""
    confidence: float = 0.0
    variant: str = ""
    variants_tried: list[str] = field(default_factory=list)
    ingredient_count: int = 0
    ingredients: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "engine": self.engine,
            "available": self.available,
            "command": self.command,
            "confidence": round(float(self.confidence), 2),
            "variant": self.variant,
            "variants_tried": self.variants_tried,
            "ingredient_count": self.ingredient_count,
            "ingredients": self.ingredients,
            "notes": self.notes,
        }


def _candidate_commands() -> list[str]:
    candidates = [config.TESSERACT_CMD, "tesseract"]
    found = shutil.which("tesseract")
    if found:
        candidates.append(found)
    return [c for c in candidates if c]


def _engine_available() -> tuple[bool, str]:
    """Locate a usable Tesseract binary and return (available, command)."""
    if pytesseract is None:
        return False, ""
    for candidate in _candidate_commands():
        path = Path(candidate)
        if path.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(path)
            return True, str(path)
        resolved = shutil.which(candidate)
        if resolved:
            pytesseract.pytesseract.tesseract_cmd = resolved
            return True, resolved
    return False, ""


def ocr_status() -> dict:
    """Report OCR availability for the /api/health endpoint."""
    available, command = _engine_available()
    version = ""
    if available:
        try:  # pragma: no cover - depends on the local engine
            version = str(pytesseract.get_tesseract_version())
        except Exception:
            version = "unknown"
    return {
        "available": available,
        "engine": "tesseract" if available else "unavailable",
        "command": command,
        "version": version,
        "hint": (
            "Install Tesseract (https://github.com/UB-Mannheim/tesseract/wiki) and set "
            "TESSERACT_CMD in .env, or paste the ingredient list manually."
        )
        if not available
        else "",
    }


_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _plausibility(text: str, confidence: float) -> float:
    """
    Score an OCR attempt.

    Rewards ingredient-like comma-separated names and average confidence, and
    penalises output that is mostly garbage characters.
    """
    if not text.strip():
        return 0.0
    ingredients = parse_ingredient_text(text)
    letters = len(_WORD_RE.findall(text))
    noise = sum(text.count(ch) for ch in "|~^`[]{}<>") / max(1, len(text))
    return float(len(ingredients)) * 1.0 + letters * 0.02 + confidence / 100.0 - noise * 20.0


def _recognise(image: np.ndarray) -> tuple[str, float]:
    """Run Tesseract on one image and return (text, mean word confidence)."""
    if pytesseract is None:  # pragma: no cover - guarded by ocr_status
        return "", 0.0
    pil_image = preprocess.png_bytes_to_pil(image)
    data = pytesseract.image_to_data(
        pil_image,
        lang=config.OCR_LANGS,
        config=config.OCR_CONFIG,
        output_type=pytesseract.Output.DICT,
    )
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for i in range(len(data["text"])):
        token = (data["text"][i] or "").strip()
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1.0
        if not token:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(token)
        if confidence >= 0:
            confidences.append(confidence)

    ordered_lines = [" ".join(lines[key]) for key in sorted(lines)]
    text = "\n".join(ordered_lines)
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    return text, mean_confidence


def run_ocr_on_variants(variants: dict[str, np.ndarray]) -> OcrResult:
    """
    OCR every preprocessing variant and keep the most plausible reading.

    The result's ``variant`` field records which preprocessing chain won, which
    is useful evidence in the report and when debugging difficult labels.
    """
    available, command = _engine_available()
    if not available:
        return OcrResult(
            engine="unavailable",
            available=False,
            variants_tried=sorted(variants),
            notes=[
                "No Tesseract engine found - the image path cannot extract text.",
                ocr_status()["hint"],
            ],
        )

    best = OcrResult(engine="tesseract", available=True, command=command)  # type: ignore[call-arg]
    best_score = -1.0
    tried: list[str] = []

    for name, image in variants.items():
        tried.append(name)
        try:
            text, confidence = _recognise(image)
        except Exception as exc:  # pragma: no cover - engine runtime failure
            best.notes.append(f"{name}: {exc}")
            continue
        cleaned = clean_raw_text(text)
        score = _plausibility(cleaned, confidence)
        if score > best_score:
            ingredients = parse_ingredient_text(cleaned)
            best_score = score
            best.text = cleaned
            best.confidence = confidence
            best.variant = name
            best.ingredients = ingredients
            best.ingredient_count = len(ingredients)

    best.variants_tried = tried
    if not best.text:
        best.notes.append("OCR ran but produced no readable text.")
    return best


def run_ocr(image: np.ndarray, max_variants: int | None = None) -> OcrResult:
    """
    Full OCR entry point for an (already loaded) label image.

    ``max_variants`` optionally limits how many preprocessing chains are tried
    (each one costs one Tesseract pass).
    """
    variants = preprocess.prepare_variants(image)
    if max_variants:
        priority = ["adaptive", "clahe", "denoised", "raw_gray", "otsu", "deskewed", "sharpened", "upscaled"]
        keep = [name for name in priority if name in variants][: int(max_variants)]
        variants = {name: variants[name] for name in keep}
    return run_ocr_on_variants(variants)


def ocr_text_from_image_bytes(data: bytes, max_variants: int | None = None) -> tuple[OcrResult, np.ndarray]:
    """Convenience wrapper used by the API: bytes in, (result, loaded image) out."""
    image = preprocess.load_image(data)
    return run_ocr(image, max_variants=max_variants), image


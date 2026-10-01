"""
DeepCos inference service  (the orchestration layer).

One place where the whole pipeline runs end to end:

    image bytes ──> OpenCV preprocessing ──> AlexNet CNN text localisation
                                                     │
                                                     ▼
                                            Tesseract OCR (multi-variant)
                                                     │
    ingredient text ──> cleaning ──> normalisation ──> canonical names
                                                     │
                                                     ▼
                                        vocabulary ──> token sequence
                                                     │
                                                     ▼
                                    Embedding -> LSTM -> MLP  (Keras)
                                                     │
                        ┌────────────────────────────┴───────────────────────┐
                        ▼                                                    ▼
          formulation profile + category                 occlusion attribution
                        │                                                    │
                        └───────────────┬────────────────────────────────────┘
                                        ▼
                       knowledge base: ingredient functions + concerns
                                        ▼
                                 user-friendly report
"""

from __future__ import annotations

import json
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from ml import config
from ml.dataset.preprocessing import (
    canonical_names,
    clean_raw_text,
    ingredient_list_from_any,
    text_preprocessing_report,
)
from ml.dataset.vocabulary import Vocabulary
from ml.knowledge_base import IngredientKnowledgeBase, ResolvedIngredient, get_knowledge_base
from ml.models.deepcos_model import load_ingredient_model, model_architecture_summary
from ml.models.explain import occlusion_contributions
from ml.rules import concern_engine
from ml.rules.profile import band, band_fraction, profile_payload, summary_sentence

DISCLAIMER = (
    "DeepCos analyses ingredient composition and predicts formulation characteristics from "
    "patterns learned from training data. It is an informational tool: it does not determine "
    "whether a product is safe, and it does not give medical or dermatological advice."
)

# Shown with every report so AI-derived content is never mistaken for the
# verified local knowledge base.
AI_ENRICHMENT_DISCLAIMER = (
    "AI-generated informational content. It is not a safety, medical, or regulatory determination."
)


class EnrichmentProvider(Protocol):
    """
    Optional AI-enrichment hook supplied by the backend.

    Implementations receive the local-knowledge-base resolution and return the
    ``ai_enrichment`` report block. They must never raise, but inference wraps
    them defensively anyway: enrichment can only add a report field, never
    change model inputs, scores or the deterministic concern screening.
    """

    def enrich_unresolved(self, resolution: Sequence["ResolvedIngredient"]) -> dict: ...


def _empty_enrichment_block() -> dict:
    return {
        "enabled": False,
        "requested_count": 0,
        "resolved_count": 0,
        "cached_count": 0,
        "failed_count": 0,
        "promotion_enabled": False,
        "promoted_count": 0,
        "items": [],
        "disclaimer": AI_ENRICHMENT_DISCLAIMER,
    }


def _run_enrichment(
    enricher: EnrichmentProvider | None,
    resolution: Sequence["ResolvedIngredient"],
) -> dict:
    """
    Build the ``ai_enrichment`` block. A disabled or broken enricher (or any
    failure inside it) degrades to the empty block - the analysis itself always
    completes, and AI data never touches tokens, scores, concerns or attribution.
    """
    if enricher is None:
        block = _empty_enrichment_block()
    else:
        try:
            block = enricher.enrich_unresolved(resolution)
        except Exception as exc:  # noqa: BLE001 - enrichment is best-effort only
            print(f"[deepcos] AI enrichment skipped ({type(exc).__name__}: {exc})")
            block = _empty_enrichment_block()
    if not isinstance(block, dict):
        block = _empty_enrichment_block()
    block.setdefault("enabled", False)
    for count_key in ("requested_count", "resolved_count", "cached_count", "failed_count"):
        block.setdefault(count_key, 0)
    block.setdefault("promotion_enabled", False)
    block.setdefault("promoted_count", 0)
    block.setdefault("items", [])
    block.setdefault("disclaimer", AI_ENRICHMENT_DISCLAIMER)
    return block

# Functions that are structural rather than characteristic: they rarely belong in
# the "key ingredients" list unless the model found them influential.
ANCILLARY_FUNCTIONS: frozenset[str] = frozenset(
    {"solvent", "chelating", "ph-adjuster", "thickener", "colorant", "emulsifier"}
)


# ---------------------------------------------------------------------------
# Model registry (loaded once per process)
# ---------------------------------------------------------------------------
class IngredientModelBundle:
    """Trained ingredient network + vocabulary + training metrics."""

    def __init__(self, model, vocab: Vocabulary, metrics: dict) -> None:
        self.model = model
        self.vocab = vocab
        self.metrics = metrics

    @classmethod
    def load(cls) -> "IngredientModelBundle":
        model = load_ingredient_model()
        vocab = Vocabulary.load()
        metrics: dict = {}
        if config.INGREDIENT_METRICS_PATH.exists():
            metrics = json.loads(config.INGREDIENT_METRICS_PATH.read_text(encoding="utf-8"))
        return cls(model, vocab, metrics)

    @property
    def available(self) -> bool:
        return True

    def encode(self, names: Sequence[str], seq_len: int | None = None) -> np.ndarray:
        return self.vocab.encode(list(names), seq_len or config.SEQ_LEN)

    def predict(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (profile_scores[4], category_probabilities[8])."""
        raw = self.model.predict(
            np.asarray(sequence, dtype="int32").reshape(1, -1), verbose=0
        )
        profiles = np.asarray(raw["formulation_profile"])[0]
        categories = np.asarray(raw["product_category"])[0]
        return profiles, categories

    def profile_reliability(self) -> dict[str, dict]:
        """
        Honest per-target reliability taken from the held-out test metrics.

        Reported in the API so the UI can show "model confidence" instead of
        pretending the raw sigmoid value is a probability of a medical fact.
        """
        out: dict[str, dict] = {}
        for target in config.PROFILE_TARGETS:
            values = self.metrics.get("profile_metrics", {}).get(target, {})
            r2 = float(values.get("r2", 0.0))
            band_accuracy = float(values.get("band_accuracy", 0.0))
            if r2 >= 0.6:
                level = "good"
            elif r2 >= 0.35:
                level = "moderate"
            elif r2 > 0:
                level = "indicative"
            else:
                level = "low"
            out[target] = {
                "r2": round(r2, 4),
                "mae": round(float(values.get("mae", 0.0)), 4),
                "band_accuracy": round(band_accuracy, 4),
                "level": level,
            }
        return out

    def summary(self) -> dict:
        return {
            "available": True,
            "trained_at": self.metrics.get("trained_at"),
            "architecture": model_architecture_summary(self.model),
            "vocabulary_size": len(self.vocab),
            "category_metrics": self.metrics.get("category_metrics", {}),
            "profile_metrics": self.metrics.get("profile_metrics", {}),
            "profile_reliability": self.profile_reliability(),
            "dataset": self.metrics.get("dataset", {}),
        }


@lru_cache(maxsize=1)
def get_ingredient_model() -> IngredientModelBundle | None:
    """Cached ingredient-model bundle; ``None`` when the model is not trained yet."""
    try:
        return IngredientModelBundle.load()
    except FileNotFoundError:
        return None


@lru_cache(maxsize=1)
def get_kb() -> IngredientKnowledgeBase:
    return get_knowledge_base()


def model_status() -> dict:
    """Availability report used by /api/health and the dashboard banner."""
    bundle = get_ingredient_model()
    from ml.image import ocr
    from ml.image.text_detector import get_text_detector

    return {
        "ingredient_model": bool(bundle),
        "ingredient_model_path": str(config.INGREDIENT_MODEL_PATH),
        "vocabulary_path": str(config.INGREDIENT_VOCAB_PATH),
        "text_region_cnn": config.TEXT_CNN_MODEL_PATH.exists(),
        "label_category_cnn": config.CATEGORY_CNN_MODEL_PATH.exists(),
        "text_detector_loaded": get_text_detector() is not None,
        "ocr": ocr.ocr_status(),
        "knowledge_base": get_kb().stats(),
        "hint": (
            ""
            if bundle
            else "Train the models first: python -m ml.train_ingredient_model"
        ),
    }


# ---------------------------------------------------------------------------
# Report building blocks
# ---------------------------------------------------------------------------
def _key_ingredients(
    rows: Sequence[dict],
    explanation: dict,
    top_n: int = 6,
) -> list[dict]:
    """
    Pick the ingredients that matter most for this product.

    Importance combines the model's own attribution (occlusion deltas) with the
    documented function of the ingredient, so both what the network found and
    what the reference data says contribute to the selection.
    """
    deltas: dict[str, float] = {}
    for entry in explanation.get("contributions", []):
        name = entry["ingredient"]
        contribution = entry["contributions"]
        deltas[name] = max(deltas.get(name, 0.0), max(abs(v) for v in contribution.values()))

    scored: list[tuple[float, dict]] = []
    for row in rows:
        functions = set(row["functions"])
        importance = deltas.get(row["ingredient"], 0.0) * 3.0
        if functions & ANCILLARY_FUNCTIONS:
            importance -= 0.4
        if functions & {"brightening", "exfoliating", "retinoid", "barrier-repair", "humectant",
                        "uv-filter", "peptide", "antioxidant", "soothing", "anti-acne", "cleansing"}:
            importance += 0.6
        if row["resolved"]:
            importance += 0.15
        scored.append((importance, row))

    scored.sort(key=lambda pair: -pair[0])
    selected: list[dict] = []
    for importance, row in scored[:top_n]:
        selected.append(
            {
                "ingredient": row["ingredient"],
                "primary_function": row["primary_function"],
                "function_labels": row["function_labels"],
                "note": row["note"],
                "importance": round(float(importance), 4),
                "model_attribution": round(float(deltas.get(row["ingredient"], 0.0)), 4),
            }
        )
    return selected


def _profile_confidence(scores: dict[str, float], reliability: dict[str, dict]) -> dict[str, dict]:
    """
    Per-target confidence for the report.

    ``band_margin`` expresses how far the prediction is from the nearest band
    boundary: a score of 0.66+ is far more decisive than one sitting at 0.39/0.41.
    ``level`` comes from the held-out R2 of that target, so a weak target is
    labelled as weak rather than presented as certain.
    """
    thresholds = [limit for limit, _ in config.PROFILE_BANDS]
    out: dict[str, dict] = {}
    for target, score in scores.items():
        score = float(score)
        distance = min(abs(score - limit) for limit in thresholds)
        if distance >= 0.12:
            margin = "decisive"
        elif distance >= 0.05:
            margin = "clear"
        else:
            margin = "near boundary"
        info = reliability.get(target, {})
        out[target] = {
            "band_margin": margin,
            "band_margin_value": round(float(distance), 4),
            "level": info.get("level", "unknown"),
            "r2": info.get("r2", 0.0),
            "band_accuracy": info.get("band_accuracy", 0.0),
        }
    return out


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def analyze_ingredients(
    ingredient_input: str | Iterable[str],
    *,
    analysis_id: str | None = None,
    mode: str = "text",
    raw_text: str | None = None,
    image_analysis: dict | None = None,
    ocr_result: dict | None = None,
    explain: bool = True,
    enricher: EnrichmentProvider | None = None,
) -> dict:
    """
    Run the complete DeepCos analysis for an ingredient list.

    ``enricher`` is an optional backend-supplied AI fallback for ingredients the
    local knowledge base could not resolve; when ``None`` (the default) the
    report simply carries an ``ai_enrichment`` block with ``enabled: false``.

    Raises
    ------
    RuntimeError  if the trained model is missing.
    ValueError    if no ingredient could be extracted.
    """
    bundle = get_ingredient_model()
    if bundle is None:
        raise RuntimeError(
            "The DeepCos ingredient model is not trained yet. "
            "Run: python -m ml.train_ingredient_model"
        )

    kb = get_kb()
    started = time.perf_counter()
    warnings: list[str] = []

    # 1. preprocessing -----------------------------------------------------
    raw_names = ingredient_list_from_any(ingredient_input)
    if not raw_names:
        raise ValueError("No ingredient could be extracted from the supplied input.")

    resolution = kb.resolve_all(raw_names, allow_fuzzy=True)
    names = canonical_names(resolution)
    unresolved = [item.raw for item in resolution if not item.resolved]
    if unresolved:
        warnings.append(
            f"{len(unresolved)} ingredient name(s) were not found in the knowledge base: "
            + ", ".join(unresolved[:6])
            + ("..." if len(unresolved) > 6 else "")
        )
    # AI-learned entries (promoted by the optional enrichment feature) resolve
    # locally, but they are unverified AI content - say so instead of presenting
    # them like the curated reference data.
    learned_used = sorted(
        {item.inci for item in resolution if item.info is not None and item.info.learned}
    )
    if learned_used:
        warnings.append(
            f"{len(learned_used)} ingredient(s) resolved from the AI-learned knowledge-base "
            "overlay (AI-generated, unreviewed - not verified reference data): "
            + ", ".join(learned_used[:6])
            + ("..." if len(learned_used) > 6 else "")
        )
    if len(raw_names) > config.SEQ_LEN:
        warnings.append(
            f"The list contains {len(raw_names)} ingredients; the model reads the first "
            f"{config.SEQ_LEN} in list order (the rest still appear in the ingredient table)."
        )

    # 2. tokenisation ------------------------------------------------------
    sequence = bundle.encode(names)
    tokens = [
        {"token": int(index), "ingredient": name}
        for index, name in zip(sequence, names)
        if int(index) != config.PAD_INDEX
    ]

    # 3. deep learning prediction -----------------------------------------
    profile_scores_raw, category_probabilities = bundle.predict(sequence)
    profile_scores = {
        target: float(profile_scores_raw[i])
        for i, target in enumerate(config.PROFILE_TARGETS)
    }
    category_index = int(np.argmax(category_probabilities))
    category = config.CATEGORIES[category_index]
    category_probabilities_map = {
        name: round(float(category_probabilities[i]), 4)
        for i, name in enumerate(config.CATEGORIES)
    }

    # 4. explainability (model-derived attribution) ------------------------
    explanation = (
        occlusion_contributions(bundle.model, sequence, names)
        if explain
        else {"method": "disabled", "contributions": [], "top_contributors": {}}
    )

    # 5. knowledge base ----------------------------------------------------
    rows = concern_engine.ingredient_breakdown(resolution, kb)
    concerns = concern_engine.screen(resolution, kb)
    reliability = bundle.profile_reliability()

    # 5b. optional AI enrichment for unresolved names -----------------------
    # Runs strictly after the model/rule outputs above and only appends a
    # separate report block: no effect on tokens, profile scores, category
    # prediction, concern flags, regulatory severity or attribution.
    ai_enrichment = _run_enrichment(enricher, resolution)

    # 6. assemble the report ----------------------------------------------
    raw_text_value = raw_text if raw_text is not None else ", ".join(raw_names)
    report = {
        "analysis_id": analysis_id or uuid.uuid4().hex[:12],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "input": {
            "raw_text": raw_text_value,
            "characters": len(raw_text_value),
            "ingredient_count": len(names),
            "extra_ingredients_beyond_model_window": max(0, len(names) - config.SEQ_LEN),
        },
        "preprocessing": (
            text_preprocessing_report(raw_text)
            if isinstance(raw_text, str) and raw_text
            else {
                "raw_ingredient_count": len(raw_names),
                "ingredients": raw_names,
                "cleaned_text": ", ".join(raw_names),
            }
        ),
        "product": {
            "category": category,
            "display_name": category,
            "category_confidence": round(float(category_probabilities[category_index]), 4),
            "category_probabilities": category_probabilities_map,
            "brand": None,
            "identification_note": (
                "The category is predicted by the trained network from the ingredient "
                "sequence; the brand is not read from the label."
            ),
        },
        "profile": profile_payload(profile_scores),
        "profile_scores": {k: band_fraction(v) for k, v in profile_scores.items()},
        "profile_summary": summary_sentence(profile_scores, category),
        "profile_confidence": _profile_confidence(profile_scores, reliability),
        "key_ingredients": _key_ingredients(rows, explanation),
        "ingredients": rows,
        "unresolved_ingredients": unresolved,
        "function_histogram": concern_engine.function_histogram(resolution),
        "concerns": concerns,
        "explainability": {
            "method": explanation.get("method"),
            "description": (
                "Each ingredient is removed from the sequence in turn and the change in every "
                "predicted profile is measured (occlusion attribution). The listed ingredients "
                "are the ones that actually moved the network's output."
                if explain
                else "Explanation not requested."
            ),
            "top_contributors": explanation.get("top_contributors", {}),
        },
        "sequence": {"tokens": tokens, "seq_len": config.SEQ_LEN},
        "model": {
            "name": "Embedding -> LSTM -> MLP (multi-task)",
            "vocabulary_size": len(bundle.vocab),
            "seq_len": config.SEQ_LEN,
            "profile_targets": list(config.PROFILE_TARGETS),
            "categories": list(config.CATEGORIES),
            "profile_reliability": reliability,
            "category_accuracy": bundle.metrics.get("category_metrics", {}).get("accuracy"),
        },
        "knowledge_base": {
            "version": kb.version,
            "ingredient_count": len(kb),
            "matched": len(names) - len(unresolved),
            "match_rate": round((len(names) - len(unresolved)) / max(1, len(names)), 4),
            "learned_count": kb.learned_count(),
            "learned_used": len(learned_used),
        },
        "ai_enrichment": ai_enrichment,
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
        "processing_ms": round((time.perf_counter() - started) * 1000, 1),
    }

    if ocr_result is not None:
        report["ocr"] = ocr_result
    if image_analysis is not None:
        report["image_analysis"] = image_analysis
    return report


# ---------------------------------------------------------------------------
# Image path
# ---------------------------------------------------------------------------
def analyze_image_bytes(
    image_bytes: bytes,
    *,
    provided_text: str | None = None,
    explain: bool = True,
    max_ocr_variants: int | None = 5,
    enricher: EnrichmentProvider | None = None,
) -> dict:
    """
    Full image path: CNN text localisation -> OCR -> ingredient analysis.

    When the caller also supplies the ingredient text (for example the user
    corrected an OCR mistake) that text is analysed and the OCR output is kept in
    the report for transparency.
    """
    from ml.image import ocr, preprocess
    from ml.image.text_detector import get_text_detector

    started = time.perf_counter()
    image = preprocess.load_image(image_bytes)
    quality = preprocess.assess_quality(image)

    detector = get_text_detector()
    image_analysis: dict[str, Any] = {
        "quality": quality.to_dict(),
        "cnn_text_detector": {
            "available": detector is not None,
            "model_path": str(config.TEXT_CNN_MODEL_PATH),
            "description": (
                "AlexNet-style CNN applied as a sliding window to locate the ingredient text "
                "block before OCR."
            ),
        },
        "original_png_base64": preprocess.encode_png_base64(image, max_width=700),
    }

    if detector is not None:
        visual = detector.visualise(image)
        cropped = visual.pop("cropped_image")
        image_analysis["cnn_text_detector"].update(
            {
                "box": [int(v) for v in visual["box"]],
                "mean_text_probability": visual["mean_text_probability"],
                "coverage": visual["coverage"],
                "grid_shape": visual["grid_shape"],
            }
        )
        image_analysis["detected_region_png_base64"] = visual["annotated_png_base64"]
        image_analysis["heatmap_png_base64"] = visual["heatmap_png_base64"]
        image_analysis["cropped_region_png_base64"] = visual["cropped_png_base64"]
        ocr_target = cropped
    else:
        image_analysis["cnn_text_detector"]["note"] = (
            "Text-region CNN not trained - OCR ran on the full frame. "
            "Train it with: python -m ml.train_label_cnn --task text_region"
        )
        ocr_target = image

    ocr_result = ocr.run_ocr(ocr_target, max_variants=max_ocr_variants)
    ocr_payload = ocr_result.to_dict()

    supplied_text = (provided_text or "").strip()
    ingredient_text = supplied_text or ocr_result.text

    if not ingredient_text:
        notes = " ".join(ocr_payload.get("notes", []))
        raise ValueError(
            "No ingredient text could be extracted from the image. "
            + (notes if notes else "Try a sharper, closer photo of the ingredient list.")
            + " You can also paste the ingredient list manually."
        )

    image_analysis["ocr_variant_used"] = ocr_result.variant
    image_analysis["ocr_confidence"] = round(float(ocr_result.confidence), 2)
    image_analysis["ingredient_text_source"] = (
        "user-supplied text (OCR output kept for transparency)" if supplied_text else "ocr"
    )
    image_analysis["ocr_ms_estimate"] = round((time.perf_counter() - started) * 1000, 1)

    report = analyze_ingredients(
        ingredient_text,
        mode="image+text" if supplied_text else "image",
        raw_text=ingredient_text,
        image_analysis=image_analysis,
        ocr_result=ocr_payload,
        explain=explain,
        enricher=enricher,
    )
    report["preprocessing"]["ocr_text"] = ocr_result.text
    return report


def analyze(
    *,
    text: str | None = None,
    image_bytes: bytes | None = None,
    explain: bool = True,
    max_ocr_variants: int | None = 5,
    enricher: EnrichmentProvider | None = None,
) -> dict:
    """Unified entry point: accepts text, an image, or both."""
    if image_bytes is not None:
        return analyze_image_bytes(
            image_bytes,
            provided_text=text,
            explain=explain,
            max_ocr_variants=max_ocr_variants,
            enricher=enricher,
        )
    if text is None or not str(text).strip():
        raise ValueError("Supply either an ingredient text or an image.")
    return analyze_ingredients(text, mode="text", raw_text=text, explain=explain, enricher=enricher)






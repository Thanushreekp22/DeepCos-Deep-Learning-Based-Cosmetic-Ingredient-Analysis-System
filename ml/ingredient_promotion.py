"""
Promotion of *validated* AI enrichment output into the AI-learned KB overlay.

Why a separate module (and a separate file)
-------------------------------------------
The verified reference data (``data/knowledge_base/ingredients.json`` and
``concern_rules.json``) is curated and never machine-written. Ingredients the
knowledge base does not document (for example ``Zinc PCA``) can, however, be
looked up through the optional Groq enrichment feature. This module turns such a
result into a knowledge-base entry that

* uses the **existing taxonomy**: AI function words are mapped onto the real
  ``function_taxonomy`` tags (``humectant``, ``anti-acne``, ...) and AI concern
  phrases onto the real concern-rule tags (``comedogenic-potential``, ...), so
  the deterministic rule engine can screen them;
* keeps anything unmappable as a plain lower-cased slug tag instead of
  pretending it is a documented rule;
* is always stamped ``origin="ai-learned"`` / ``review_status="unreviewed"``, so
  no consumer can mistake it for verified reference data;
* is stored in its own overlay file (see :data:`ml.config.LEARNED_INGREDIENTS_PATH`),
  where it can be inspected and deleted again.

Everything here is pure: no IO, no network, no globals. The write path lives in
``ml/learned_store.py`` and ``backend/services/kb_promotion.py``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from ml.knowledge_base import titlecase_ingredient

# Labels used by every consumer (API, reports, exports, UI).
ORIGIN = "ai-learned"
REVIEW_STATUS = "unreviewed"
SCHEMA_VERSION = 1

# A promotion needs at least this much confidence from the provider.
CONFIDENCE_RANK: dict[str, int] = {"low": 1, "medium": 2, "high": 3}
DEFAULT_MIN_CONFIDENCE = "medium"

# Guards against junk keys such as OCR fragments ("x", "and", "a b").
MIN_NAME_LENGTH = 4
MAX_NAME_LENGTH = 80
MAX_TAGS = 8

# AI wording -> real taxonomy tags (see ingredients.json -> function_taxonomy).
FUNCTION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "humectant": ("humectant",),
    "moisturizer": ("humectant",),
    "moisturiser": ("humectant",),
    "moisturizing": ("humectant",),
    "moisturising": ("humectant",),
    "moisturizing agent": ("humectant",),
    "moisturising agent": ("humectant",),
    "moisture binding agent": ("humectant",),
    "water binding agent": ("humectant",),
    "hydrating agent": ("humectant",),
    "humidity regulator": ("humectant",),
    "emollient": ("emollient",),
    "skin softener": ("emollient",),
    "softener": ("emollient",),
    "occlusive": ("occlusive",),
    "occlusion agent": ("occlusive",),
    "film former": ("occlusive",),
    "solvent": ("solvent",),
    "carrier": ("solvent",),
    "diluent": ("solvent",),
    "conditioning": ("skin-conditioning",),
    "conditioning agent": ("skin-conditioning",),
    "skin conditioning": ("skin-conditioning",),
    "skin conditioning agent": ("skin-conditioning",),
    "skin conditioner": ("skin-conditioning",),
    "brightening": ("brightening",),
    "skin brightening": ("brightening",),
    "whitening": ("brightening",),
    "lightening": ("brightening",),
    "tyrosinase inhibitor": ("brightening",),
    "anti pigmentation": ("brightening",),
    "depigmenting": ("brightening",),
    "skin tone evening": ("brightening",),
    "exfoliating": ("exfoliating",),
    "exfoliant": ("exfoliating",),
    "keratolytic": ("exfoliating",),
    "skin peeling agent": ("exfoliating",),
    "antioxidant": ("antioxidant",),
    "antioxidative": ("antioxidant",),
    "free radical scavenger": ("antioxidant",),
    "radical scavenger": ("antioxidant",),
    "soothing": ("soothing",),
    "skin calming": ("soothing",),
    "calming": ("soothing",),
    "anti irritant": ("soothing",),
    "anti inflammatory": ("soothing",),
    "anti redness": ("soothing",),
    "barrier repair": ("barrier-repair",),
    "barrier support": ("barrier-repair",),
    "barrier restoring": ("barrier-repair",),
    "skin barrier repair": ("barrier-repair",),
    "barrier strengthening": ("barrier-repair",),
    "anti acne": ("anti-acne",),
    "antiacne": ("anti-acne",),
    "acne treatment": ("anti-acne",),
    "anti blemish": ("anti-acne",),
    "sebum regulating": ("anti-acne",),
    "sebum regulation": ("anti-acne",),
    "sebum control": ("anti-acne",),
    "sebum regulator": ("anti-acne",),
    "sebostatic": ("anti-acne",),
    "anti seborrheic": ("anti-acne",),
    "astringent": ("anti-acne",),
    "pore refining": ("anti-acne",),
    "matte agent": ("anti-acne",),
    "uv filter": ("uv-filter",),
    "uv filters": ("uv-filter",),
    "uv absorber": ("uv-filter",),
    "ultraviolet filter": ("uv-filter",),
    "sunscreen": ("uv-filter",),
    "sunscreen agent": ("uv-filter",),
    "photoprotective": ("uv-filter",),
    "preservative": ("preservative",),
    "antimicrobial preservative": ("preservative",),
    "antimicrobial": ("preservative",),
    "antifungal": ("preservative",),
    "antiseptic": ("preservative",),
    "cleansing": ("cleansing",),
    "cleanser": ("cleansing",),
    "surfactant": ("cleansing",),
    "cleansing surfactant": ("cleansing",),
    "detergent": ("cleansing",),
    "foaming agent": ("cleansing",),
    "emulsifier": ("emulsifier",),
    "emulsifying agent": ("emulsifier",),
    "surfactant emulsifier": ("emulsifier",),
    "co emulsifier": ("emulsifier",),
    "thickener": ("thickener",),
    "thickening agent": ("thickener",),
    "rheology modifier": ("thickener",),
    "viscosity controlling": ("thickener",),
    "viscosity modifier": ("thickener",),
    "gelling agent": ("thickener",),
    "gel former": ("thickener",),
    "fragrance": ("fragrance",),
    "perfume": ("fragrance",),
    "parfum": ("fragrance",),
    "odorant": ("fragrance",),
    "masking agent": ("fragrance",),
    "ph adjuster": ("ph-adjuster",),
    "ph regulator": ("ph-adjuster",),
    "ph buffer": ("ph-adjuster",),
    "buffering agent": ("ph-adjuster",),
    "buffer": ("ph-adjuster",),
    "acidulant": ("ph-adjuster",),
    "chelating": ("chelating",),
    "chelating agent": ("chelating",),
    "chelator": ("chelating",),
    "sequestrant": ("chelating",),
    "silicone": ("silicone",),
    "siloxane": ("silicone",),
    "colorant": ("colorant",),
    "colourant": ("colorant",),
    "pigment": ("colorant",),
    "dye": ("colorant",),
    "opacifier": ("colorant",),
    "botanical": ("botanical",),
    "botanical extract": ("botanical",),
    "plant extract": ("botanical",),
    "plant derived extract": ("botanical",),
    "herbal extract": ("botanical",),
    "peptide": ("peptide",),
    "polypeptide": ("peptide",),
    "peptide derivative": ("peptide",),
    "retinoid": ("retinoid",),
    "retinol": ("retinoid",),
    "retinal": ("retinoid",),
    "retinyl ester": ("retinoid",),
    "vitamin a derivative": ("retinoid",),
    "vitamin": ("vitamin",),
    "vitamin derivative": ("vitamin",),
    "vitamin c": ("vitamin", "antioxidant"),
}

# AI wording -> real concern-rule tags (see concern_rules.json -> rules).
CONCERN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "fragrance allergen": ("fragrance-allergen", "fragrance"),
    "fragrance allergens": ("fragrance-allergen", "fragrance"),
    "allergenic fragrance": ("fragrance-allergen", "fragrance"),
    "fragrance": ("fragrance",),
    "perfume": ("fragrance",),
    "parfum": ("fragrance",),
    "scent": ("fragrance",),
    "aroma": ("fragrance",),
    "essential oil": ("essential-oil", "fragrance"),
    "essential oils": ("essential-oil", "fragrance"),
    "essential oil component": ("essential-oil", "fragrance"),
    "paraben": ("paraben",),
    "parabens": ("paraben",),
    "sulfate": ("sulfate-surfactant",),
    "sulfates": ("sulfate-surfactant",),
    "sulfate surfactant": ("sulfate-surfactant",),
    "sls": ("sulfate-surfactant",),
    "sles": ("sulfate-surfactant",),
    "anionic surfactant": ("sulfate-surfactant",),
    "drying alcohol": ("drying-alcohol",),
    "denatured alcohol": ("drying-alcohol",),
    "alcohol denat": ("drying-alcohol",),
    "volatile alcohol": ("drying-alcohol",),
    "ethanol": ("drying-alcohol",),
    "photosensitizing": ("photosensitising",),
    "photosensitising": ("photosensitising",),
    "photosensitization": ("photosensitising",),
    "photosensitivity": ("photosensitising",),
    "phototoxic": ("photosensitising",),
    "phototoxicity": ("photosensitising",),
    "comedogenic": ("comedogenic-potential",),
    "comedogenic potential": ("comedogenic-potential",),
    "comedogenicity": ("comedogenic-potential",),
    "pore clogging": ("comedogenic-potential",),
    "pore clogging potential": ("comedogenic-potential",),
    "acnegenic": ("comedogenic-potential",),
    "sensitizer": ("potential-sensitizer",),
    "sensitiser": ("potential-sensitizer",),
    "sensitizer potential": ("potential-sensitizer",),
    "contact allergen": ("potential-sensitizer",),
    "allergen": ("potential-sensitizer",),
    "allergenic": ("potential-sensitizer",),
    "irritant": ("potential-sensitizer",),
    "skin irritation": ("potential-sensitizer",),
    "irritation": ("potential-sensitizer",),
    "irritation potential": ("potential-sensitizer",),
    "formaldehyde": ("formaldehyde-releaser",),
    "formaldehyde releaser": ("formaldehyde-releaser",),
    "formaldehyde releasing": ("formaldehyde-releaser",),
    "formaldehyde donor": ("formaldehyde-releaser",),
    "restricted": ("restricted",),
    "restricted ingredient": ("restricted",),
    "banned": ("restricted",),
    "prohibited": ("restricted",),
    "concentration limit": ("restricted",),
    "regulatory limit": ("restricted",),
    "eu restriction": ("restricted",),
    "pregnancy caution": ("pregnancy-caution",),
    "not recommended during pregnancy": ("pregnancy-caution",),
    "pregnancy risk": ("pregnancy-caution",),
    "avoid during pregnancy": ("pregnancy-caution",),
    "environmental concern": ("environmental-concern",),
    "environmental impact": ("environmental-concern",),
    "environmental risk": ("environmental-concern",),
    "ecotoxicity": ("environmental-concern",),
    "aquatic toxicity": ("environmental-concern",),
    "coral reef damage": ("environmental-concern",),
    "bioaccumulation": ("environmental-concern",),
    "not readily biodegradable": ("environmental-concern",),
    "exfoliating acid": ("exfoliating-acid",),
    "aha": ("exfoliating-acid",),
    "bha": ("exfoliating-acid",),
    "pha": ("exfoliating-acid",),
    "glycolic acid": ("exfoliating-acid",),
    "lactic acid": ("exfoliating-acid",),
    "salicylic acid": ("exfoliating-acid",),
}

# Keys that carry no information on their own.
STOPWORD_KEYS: frozenset[str] = frozenset(
    {"and", "other", "others", "aqua water", "ingredients", "extract", "unknown", "n a"}
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_HAS_LETTER_RE = re.compile(r"[a-z]")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def phrase_key(text: object) -> str:
    """Normalise an AI phrase for the synonym tables: ``"Sebum-Regulating"`` -> ``"sebum regulating"``."""
    return _NON_ALNUM_RE.sub(" ", str(text or "").lower()).strip()


def slug(text: object, limit: int = 48) -> str:
    """Tag-style slug for an AI phrase that has no entry in the synonym tables."""
    return _NON_ALNUM_RE.sub("-", str(text or "").lower()).strip("-")[:limit].strip("-")


def _as_phrases(value: object) -> list[str]:
    """Accept a single string or a sequence of strings and keep the usable ones."""
    if value is None:
        return []
    if isinstance(value, str):
        candidates: Sequence[object] = [value]
    elif isinstance(value, Sequence):
        candidates = value
    else:
        return []
    out: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            out.append(candidate.strip())
    return out


def _match_synonyms(key: str, synonyms: Mapping[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    """Exact match first, then a word-boundary match inside the phrase."""
    if not key:
        return None
    exact = synonyms.get(key)
    if exact:
        return exact
    if " " in key:
        for synonym_key, tags in synonyms.items():
            if len(synonym_key) >= 5 and re.search(rf"\b{re.escape(synonym_key)}\b", key):
                return tags
    return None


def _map_phrases(
    phrases: object,
    synonyms: Mapping[str, tuple[str, ...]],
    known: set[str] | None,
) -> tuple[list[str], list[str]]:
    """
    Split AI phrases into (tags usable by the rule engine, unmapped slug tags).

    ``known`` is the set of tags that exist in the reference data; ``None``
    disables that check (the synonym tables are then trusted as-is).
    """
    tags: list[str] = []
    unmapped: list[str] = []
    for phrase in _as_phrases(phrases):
        matched = _match_synonyms(phrase_key(phrase), synonyms)
        for tag in matched or ():
            if known is None or tag in known:
                if tag not in tags:
                    tags.append(tag)
            else:
                candidate = slug(tag)
                if candidate and candidate not in unmapped:
                    unmapped.append(candidate)
        if not matched:
            candidate = slug(phrase)
            if candidate and candidate not in unmapped:
                unmapped.append(candidate)
    return tags[:MAX_TAGS], unmapped[:MAX_TAGS]


def map_functions(phrases: object, kb: object | None = None) -> tuple[list[str], list[str]]:
    """Map AI function words onto ``function_taxonomy`` tags (plus leftovers)."""
    known = set(getattr(kb, "function_taxonomy", {}) or {}) if kb is not None else None
    return _map_phrases(phrases, FUNCTION_SYNONYMS, known)


def map_concerns(phrases: object, kb: object | None = None) -> tuple[list[str], list[str]]:
    """Map AI concern phrases onto concern-rule tags (plus leftovers)."""
    known = set(getattr(kb, "concern_rules", {}) or {}) if kb is not None else None
    return _map_phrases(phrases, CONCERN_SYNONYMS, known)


# ---------------------------------------------------------------------------
# Promotion gates
# ---------------------------------------------------------------------------
def promotion_blockers(
    data: Mapping | None,
    key: str,
    *,
    min_confidence: str = DEFAULT_MIN_CONFIDENCE,
) -> str | None:
    """
    Reason why an enrichment result must **not** be promoted (``None`` = promote).

    Only results that identify the ingredient with enough confidence and that
    carry at least one function or concern are worth persisting; everything else
    stays a per-analysis AI note.
    """
    if not isinstance(data, Mapping):
        return "no enrichment data available"

    normalized = phrase_key(key)
    if len(normalized) < MIN_NAME_LENGTH or len(normalized) > MAX_NAME_LENGTH:
        return "the ingredient name is too short or too long to promote"
    if normalized in STOPWORD_KEYS or not _HAS_LETTER_RE.search(normalized):
        return "the ingredient name carries no usable information"

    identification = str(data.get("identification", "") or "").strip().lower()
    if not identification or identification == "unknown":
        return "the provider could not identify the ingredient"

    confidence = str(data.get("confidence", "") or "").strip().lower()
    threshold = str(min_confidence or DEFAULT_MIN_CONFIDENCE).strip().lower()
    minimum = CONFIDENCE_RANK.get(threshold, CONFIDENCE_RANK[DEFAULT_MIN_CONFIDENCE])
    if CONFIDENCE_RANK.get(confidence, 0) < minimum:
        return (
            f"confidence '{confidence or 'unknown'}' is below the "
            f"promotion threshold '{threshold}'"
        )

    if not _as_phrases(data.get("functions")) and not _as_phrases(data.get("possible_concerns")):
        return "the provider returned neither functions nor concerns"
    return None


def promotable(data: Mapping | None, key: str, *, min_confidence: str = DEFAULT_MIN_CONFIDENCE) -> bool:
    """Convenience inverse of :func:`promotion_blockers`."""
    return promotion_blockers(data, key, min_confidence=min_confidence) is None


# ---------------------------------------------------------------------------
# Learned-entry construction
# ---------------------------------------------------------------------------
def _canonical_name(data: Mapping, key: str) -> str:
    candidate = str(data.get("inci_name", "") or "").strip()
    if candidate and len(phrase_key(candidate)) >= MIN_NAME_LENGTH:
        return candidate
    return titlecase_ingredient(key)


def _aliases(data: Mapping, key: str, requested_name: str, inci: str) -> list[str]:
    aliases: list[str] = []
    for candidate in [requested_name, *_as_phrases(data.get("aliases"))]:
        cleaned = str(candidate or "").strip()
        normalized = phrase_key(cleaned)
        if not cleaned or normalized == phrase_key(inci) or normalized == key:
            continue
        if cleaned not in aliases:
            aliases.append(cleaned)
    return aliases[:8]


def build_learned_entry(
    *,
    key: str,
    requested_name: str,
    data: Mapping,
    record: Mapping | None = None,
    kb: object | None = None,
    now: datetime | None = None,
) -> dict:
    """
    Build the overlay record for one validated AI enrichment result.

    The result always keeps *both* views of the AI output: the mapped tags that
    the deterministic rule engine can screen (``functions`` / ``concerns``) and
    the raw phrases for auditability (``ai_functions`` / ``ai_concerns``).
    """
    record = record or {}
    functions, unmapped_functions = map_functions(data.get("functions"), kb)
    concerns, unmapped_concerns = map_concerns(data.get("possible_concerns"), kb)
    inci = _canonical_name(data, key)
    description = str(data.get("description", "") or "").strip()
    note = (
        f"{description} (AI-learned entry, unreviewed - not verified reference data.)"
        if description
        else "AI-learned entry from a Groq enrichment lookup; unreviewed, not verified reference data."
    )
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    return {
        "normalized_key": key,
        "inci": inci,
        "requested_name": requested_name,
        "aliases": _aliases(data, key, requested_name, inci),
        "functions": functions,
        "unmapped_functions": unmapped_functions,
        "concerns": concerns,
        "unmapped_concerns": unmapped_concerns,
        "note": note,
        # AI regulatory text is informational only: the entry never claims a
        # regulatory status, so it cannot reach the regulatory-notes block.
        "regulatory_status": "unknown",
        "regulatory_note": "",
        "ai_regulatory_note": str(data.get("regulatory_note", "") or "").strip(),
        "identification": str(data.get("identification", "") or ""),
        "confidence": str(data.get("confidence", "") or ""),
        "evidence_note": str(data.get("evidence_note", "") or "").strip(),
        "ai_functions": _as_phrases(data.get("functions")),
        "ai_concerns": _as_phrases(data.get("possible_concerns")),
        "origin": ORIGIN,
        "review_status": REVIEW_STATUS,
        "source": record.get("source"),
        "provider": record.get("provider"),
        "model": record.get("model"),
        "promoted_at": timestamp,
        "promoted_from": str(record.get("normalized_key") or requested_name),
        "enrichment_updated_at": record.get("updated_at"),
        "schema_version": SCHEMA_VERSION,
    }

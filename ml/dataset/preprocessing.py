"""
Text preprocessing for DeepCos  (Module 2 - Ingredient Processing).

Raw OCR output or a pasted ingredient list is not directly usable:

    "Ingredients: Water, Glycerin, Niacinamide (2%), Hyaluronic Acid, ..."

becomes

    ["Water", "Glycerin", "Niacinamide", "Hyaluronic Acid"]

Implemented pipeline
--------------------
text cleaning -> splitting -> normalisation -> canonical ingredient mapping

The module has no ML dependency, so it can be unit tested and reused by both
the training pipeline and the FastAPI service.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from ml.knowledge_base import (
    IngredientKnowledgeBase,
    ResolvedIngredient,
    normalize_ingredient_name,
)

# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------
# Separators found in real INCI lists and in OCR output
LIST_SEPARATOR_RE = re.compile(
    r"[,;•·\u2022\u00b7\u2013\u2014|\u2028\u2029]+|\s+\.\s+|(?:\r?\n)+"
)

PREFIX_RE = re.compile(
    r"^\s*(?:ingredients?|inci|contents?|composition|contains)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)

# Sections that are not ingredient names
TRAILING_SECTION_RE = re.compile(
    r"(?:may contain|free from|allergens?|directions?|warning|caution|"
    r"how to use|storage|best before|net (?:wt|weight|content)|mfg|batch no)"
    r"[\s:].*$",
    flags=re.IGNORECASE | re.DOTALL,
)

# OCR noise characters
OCR_NOISE_RE = re.compile(r"[|~^`_\[\]{}<>*#\"']+")
STANDALONE_CHAR_RE = re.compile(r"(?:^|\s)[a-z](?=\s|$)")

# Hyphenated line breaks produced by OCR: "Niacina-\nmide" -> "Niacinamide"
HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")

# Concentration markers: "(2%)", "0.5 %" - must not damage "1,2-Hexanediol"
CONCENTRATION_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*%")

MIN_NAME_LENGTH = 3
MAX_NAME_LENGTH = 60


def _basic_clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
    )


def clean_raw_text(text: str) -> str:
    """
    Clean an ingredient block coming from OCR or from the UI textarea.

    Removes a leading ``Ingredients:`` label, trailing instruction sections,
    OCR noise characters and stray concentration percentages, and joins
    hyphenated line breaks.
    """
    cleaned = _basic_clean(text)
    cleaned = HYPHEN_BREAK_RE.sub(r"\1\2", cleaned)
    cleaned = cleaned.replace("\r", "\n")
    cleaned = TRAILING_SECTION_RE.sub(" ", cleaned)
    cleaned = PREFIX_RE.sub("", cleaned)
    cleaned = OCR_NOISE_RE.sub(" ", cleaned)
    cleaned = CONCENTRATION_RE.sub(" ", cleaned)
    cleaned = STANDALONE_CHAR_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def split_ingredient_string(text: str) -> list[str]:
    """Split a cleaned ingredient block into individual raw name strings."""
    parts = LIST_SEPARATOR_RE.split(text)
    names: list[str] = []
    for part in parts:
        candidate = part.strip(" .:-*\t\n")
        if not candidate:
            continue
        if len(candidate) < MIN_NAME_LENGTH or len(candidate) > MAX_NAME_LENGTH:
            continue
        if not re.search(r"[A-Za-z]", candidate):
            continue
        names.append(candidate)
    return names


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    """Remove duplicate ingredient names while keeping the original order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = normalize_ingredient_name(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def parse_ingredient_text(text: str) -> list[str]:
    """
    Full raw-text -> list-of-ingredients step.

    ``parse_ingredient_text("Ingredients: Water, Glycerin, Niacinamide (2%)")``
    -> ``["Water", "Glycerin", "Niacinamide"]``
    """
    return dedupe_preserving_order(split_ingredient_string(clean_raw_text(text)))


def resolve_ingredients(
    names: Iterable[str],
    kb: IngredientKnowledgeBase | None = None,
    allow_fuzzy: bool = True,
) -> list[ResolvedIngredient]:
    """Map raw ingredient names onto canonical knowledge-base ingredients."""
    kb = kb or IngredientKnowledgeBase()
    return kb.resolve_all(names, allow_fuzzy=allow_fuzzy)


def canonical_names(resolved: Iterable[ResolvedIngredient]) -> list[str]:
    """Canonical INCI names for a resolved list (unknowns keep their best guess)."""
    return [item.inci for item in resolved]


def ingredient_list_from_any(value: str | Iterable[str]) -> list[str]:
    """
    Normalise whatever the API receives into a list of ingredient names.

    Accepts a raw ingredient string (``"Water, Glycerin"``) or an already
    separated list of ingredient names.
    """
    if isinstance(value, str):
        return parse_ingredient_text(value)
    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        if LIST_SEPARATOR_RE.search(item) and not item.isupper():
            names.extend(split_ingredient_string(clean_raw_text(item)))
        else:
            cleaned = clean_raw_text(item)
            if cleaned:
                names.append(cleaned)
    return dedupe_preserving_order(names)


def text_preprocessing_report(text: str) -> dict:
    """Diagnostic used by the API and the tests to show what preprocessing did."""
    cleaned = clean_raw_text(text)
    raw_parts = split_ingredient_string(cleaned)
    return {
        "raw_characters": len(text or ""),
        "cleaned_text": cleaned,
        "raw_ingredient_count": len(raw_parts),
        "ingredients": dedupe_preserving_order(raw_parts),
    }

"""
Knowledge-base layer of DeepCos.

The neural network learns *patterns* from ingredient sequences. Documented
facts - what an ingredient is normally used for, which characteristics are
associated with it and which regulatory notes exist - are looked up here
instead of being memorised by the network.

Public API
----------
normalize_ingredient_name(name) -> str
IngredientInfo                       dataclass for one knowledge-base entry
IngredientKnowledgeBase              loader + normaliser + fuzzy matcher
get_knowledge_base()                 cached singleton accessor
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from ml import config

# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------
_PAREN_RE = re.compile(r"\([^)]*\)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_MULTISPACE_RE = re.compile(r"\s+")
_PERCENT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*%")
_CONC_SUFFIX_RE = re.compile(r"[;,]\s*\d+(?:[.,]\d+)?\s*%")


def normalize_ingredient_name(raw: str) -> str:
    """
    Normalise one raw ingredient string into a lookup key.

    ``"Niacinamide (2%)"``       -> ``"niacinamide"``
    ``"  GLYCERIN,  "``          -> ``"glycerin"``
    ``"Sodium Hyaluronate(HA)"`` -> ``"sodium hyaluronate ha"``
    """
    if not raw:
        return ""
    text = unicodedata.normalize("NFKD", str(raw))
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
    text = text.lower()
    text = _PERCENT_RE.sub(" ", text)
    text = _CONC_SUFFIX_RE.sub(" ", text)
    text = _PAREN_RE.sub(lambda m: " " + m.group(0)[1:-1] + " ", text)
    text = text.replace(".", " ")
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def titlecase_ingredient(name: str) -> str:
    """Give a normalised key a presentable capitalisation."""
    return " ".join(part.capitalize() for part in name.split())


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IngredientInfo:
    """One documented cosmetic ingredient."""

    inci: str
    aliases: tuple[str, ...] = ()
    functions: tuple[str, ...] = ()
    concerns: tuple[str, ...] = ()
    note: str = ""
    common: bool = False
    regulatory_status: str = "allowed"
    regulatory_note: str = ""

    def to_dict(self) -> dict:
        return {
            "inci": self.inci,
            "aliases": list(self.aliases),
            "functions": list(self.functions),
            "concerns": list(self.concerns),
            "note": self.note,
            "common": self.common,
            "regulatory_status": self.regulatory_status,
            "regulatory_note": self.regulatory_note,
        }


@dataclass
class ResolvedIngredient:
    """A raw OCR/user ingredient string plus how it was resolved."""

    raw: str
    key: str
    info: Optional[IngredientInfo] = None
    match_score: float = 0.0
    match_type: str = "unknown"  # exact | alias | fuzzy | unknown

    @property
    def inci(self) -> str:
        return self.info.inci if self.info else titlecase_ingredient(self.key)

    @property
    def resolved(self) -> bool:
        return self.info is not None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "matched_name": self.inci,
            "resolved": self.resolved,
            "match_type": self.match_type,
            "match_score": round(float(self.match_score), 4),
            "functions": list(self.info.functions) if self.info else [],
            "concerns": list(self.info.concerns) if self.info else [],
            "note": self.info.note if self.info else "",
            "regulatory_status": self.info.regulatory_status if self.info else "unknown",
            "regulatory_note": self.info.regulatory_note if self.info else "",
        }


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
class IngredientKnowledgeBase:
    """Loads the DeepCos reference data and resolves ingredient names."""

    def __init__(
        self,
        kb_path: Path | str | None = None,
        concern_rules_path: Path | str | None = None,
        fuzzy_cutoff: float = 0.86,
    ) -> None:
        self.kb_path = Path(kb_path or config.INGREDIENT_KB_PATH)
        self.concern_rules_path = Path(concern_rules_path or config.CONCERN_RULES_PATH)
        self.fuzzy_cutoff = fuzzy_cutoff

        raw = json.loads(self.kb_path.read_text(encoding="utf-8"))
        self.version: str = raw.get("version", "unknown")
        self.purpose: str = raw.get("purpose", "")
        self.disclaimer: str = raw.get("disclaimer", "")
        self.function_taxonomy: dict[str, dict] = raw.get("function_taxonomy", {})
        self.restricted: list[dict] = raw.get("banned_or_restricted_in_cosmetics", [])

        restricted_by_key = {
            normalize_ingredient_name(item["inci"]): item for item in self.restricted
        }

        self.ingredients: dict[str, IngredientInfo] = {}
        for entry in raw.get("ingredients", []):
            inci = entry["inci"]
            key = normalize_ingredient_name(inci)
            reg = restricted_by_key.get(key, {})
            self.ingredients[key] = IngredientInfo(
                inci=inci,
                aliases=tuple(entry.get("aliases", [])),
                functions=tuple(entry.get("functions", [])),
                concerns=tuple(entry.get("concerns", [])),
                note=entry.get("note", ""),
                common=bool(entry.get("common", False)),
                regulatory_status=reg.get("status", "allowed"),
                regulatory_note=reg.get("note", ""),
            )

        # Restricted substances that are not marketed ingredients (e.g. banned)
        for item in self.restricted:
            key = normalize_ingredient_name(item["inci"])
            self.ingredients.setdefault(
                key,
                IngredientInfo(
                    inci=item["inci"],
                    regulatory_status=item.get("status", "restricted"),
                    regulatory_note=item.get("note", ""),
                ),
            )

        # Lookup index: normalised INCI name + every alias --------------------
        self._index: dict[str, str] = {}
        for key, info in self.ingredients.items():
            self._index[key] = key
            for alias in info.aliases:
                alias_key = normalize_ingredient_name(alias)
                if alias_key:
                    self._index.setdefault(alias_key, key)

        # Concern rules ------------------------------------------------------
        self.concern_rules: dict[str, dict] = {}
        self.concern_severities: dict[str, dict] = {}
        if self.concern_rules_path.exists():
            rules_raw = json.loads(self.concern_rules_path.read_text(encoding="utf-8"))
            self.concern_rules = rules_raw.get("rules", {})
            self.concern_severities = rules_raw.get("severities", {})

    # -- basics ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.ingredients)

    def all_inci(self) -> list[str]:
        """Canonical ingredient names in knowledge-base file order."""
        return [info.inci for info in self.ingredients.values()]

    def get(self, name: str) -> Optional[IngredientInfo]:
        key = self._index.get(normalize_ingredient_name(name))
        return self.ingredients.get(key) if key else None

    def keys(self) -> list[str]:
        return list(self._index.keys())

    def _fuzzy_key(self, key: str) -> tuple[Optional[str], float]:
        """Nearest known ingredient key, for OCR-noise tolerance."""
        if len(key) < 4:
            return None, 0.0
        best_key: Optional[str] = None
        best_score = 0.0
        for candidate in self._index:
            if abs(len(candidate) - len(key)) > max(4, len(key) // 2):
                continue
            score = SequenceMatcher(None, key, candidate).ratio()
            if score > best_score:
                best_key, best_score = candidate, score
        if best_score >= self.fuzzy_cutoff:
            return best_key, best_score
        return None, best_score

    def resolve(self, raw: str, allow_fuzzy: bool = True) -> ResolvedIngredient:
        """Resolve one raw ingredient string to a knowledge-base entry."""
        key = normalize_ingredient_name(raw)
        if not key:
            return ResolvedIngredient(raw=raw, key="", match_type="unknown")

        exact_key = self._index.get(key)
        if exact_key:
            return ResolvedIngredient(
                raw=raw,
                key=key,
                info=self.ingredients[exact_key],
                match_score=1.0,
                match_type="exact" if exact_key == key else "alias",
            )

        if allow_fuzzy:
            fuzzy_key, score = self._fuzzy_key(key)
            if fuzzy_key:
                return ResolvedIngredient(
                    raw=raw,
                    key=key,
                    info=self.ingredients[fuzzy_key],
                    match_score=score,
                    match_type="fuzzy",
                )
        return ResolvedIngredient(raw=raw, key=key, match_type="unknown", match_score=0.0)

    def resolve_all(
        self, names: Iterable[str], allow_fuzzy: bool = True
    ) -> list[ResolvedIngredient]:
        return [self.resolve(name, allow_fuzzy=allow_fuzzy) for name in names]

    # -- functions / concerns ---------------------------------------------
    def function_label(self, tag: str) -> str:
        return self.function_taxonomy.get(tag, {}).get("label", tag.replace("-", " ").title())

    def function_description(self, tag: str) -> str:
        return self.function_taxonomy.get(tag, {}).get("description", "")

    def describe_functions(self, tags: Iterable[str]) -> list[dict]:
        return [
            {
                "tag": tag,
                "label": self.function_label(tag),
                "description": self.function_description(tag),
            }
            for tag in tags
        ]

    def concern_rule(self, tag: str) -> dict:
        return self.concern_rules.get(tag, {})

    def severity_rank(self, severity: str) -> int:
        return int(self.concern_severities.get(severity, {}).get("rank", 0))

    def search(self, query: str, limit: int = 25) -> list[IngredientInfo]:
        """Simple ranked text search over names, aliases and notes."""
        q = normalize_ingredient_name(query)
        if not q:
            return list(self.ingredients.values())[:limit]
        scored: list[tuple[float, IngredientInfo]] = []
        for info in self.ingredients.values():
            haystacks = [normalize_ingredient_name(info.inci)] + [
                normalize_ingredient_name(alias) for alias in info.aliases
            ]
            score = 0.0
            for haystack in haystacks:
                if haystack == q:
                    score = max(score, 1.0)
                elif haystack.startswith(q):
                    score = max(score, 0.9)
                elif q in haystack:
                    score = max(score, 0.75)
            if score == 0.0 and info.note and q in info.note.lower():
                score = 0.5
            if score > 0:
                scored.append((score + (0.05 if info.common else 0.0), info))
        scored.sort(key=lambda pair: (-pair[0], pair[1].inci))
        return [info for _, info in scored[:limit]]

    def stats(self) -> dict:
        from collections import Counter

        function_counts = Counter(
            fn for info in self.ingredients.values() for fn in info.functions
        )
        concern_counts = Counter(c for info in self.ingredients.values() for c in info.concerns)
        return {
            "version": self.version,
            "ingredient_count": len(self.ingredients),
            "alias_count": max(0, len(self._index) - len(self.ingredients)),
            "function_taxonomy_count": len(self.function_taxonomy),
            "concern_rule_count": len(self.concern_rules),
            "restricted_entries": len(self.restricted),
            "top_functions": [list(item) for item in function_counts.most_common(12)],
            "top_concerns": [list(item) for item in concern_counts.most_common(12)],
        }


@lru_cache(maxsize=1)
def get_knowledge_base() -> IngredientKnowledgeBase:
    """Cached knowledge-base accessor (the JSON reference files are static)."""
    return IngredientKnowledgeBase()



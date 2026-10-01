"""
Knowledge-base promotion: persist *validated* AI enrichment into the KB overlay.

This is the write path for the feature requested by ``AI_PROMOTE_TO_KB``: when an
unknown ingredient (for example ``Zinc PCA``) was enriched through Groq and the
result passes the promotion gates in :mod:`ml.ingredient_promotion`, the entry is

1. stored in the AI-learned overlay (``data/knowledge_base/learned_ingredients.json``
   plus the MongoDB collection ``ingredient_knowledge_learned``);
2. added to the running knowledge base, so the **next** analysis in this process
   resolves it locally instead of calling the provider again;
3. labelled ``origin="ai-learned"`` / ``review_status="unreviewed"`` in every
   report, export and API response it appears in.

The verified reference files (``ingredients.json`` / ``concern_rules.json``) are
never modified, verified entries are never overwritten, and every failure here is
swallowed: promotion can only ever add information, never break an analysis.
"""

from __future__ import annotations

import threading
from typing import Mapping, Optional

from backend import config
from ml import ingredient_promotion
from ml.knowledge_base import IngredientKnowledgeBase, get_knowledge_base, normalize_ingredient_name
from ml.learned_store import LearnedIngredientStore

REASON_DISABLED = "promotion to the knowledge base is disabled (AI_PROMOTE_TO_KB)"
REASON_PROMOTED = "saved to the AI-learned knowledge-base overlay (unreviewed)"
REASON_REFRESHED = "AI-learned knowledge-base entry refreshed"


class KnowledgePromoter:
    """Writes validated enrichment results into the AI-learned KB overlay."""

    def __init__(
        self,
        *,
        kb: IngredientKnowledgeBase | None = None,
        store: LearnedIngredientStore | None = None,
        min_confidence: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.kb = kb
        self.store = store if store is not None else LearnedIngredientStore()
        self.min_confidence = (
            config.AI_PROMOTE_MIN_CONFIDENCE if min_confidence is None else min_confidence
        )
        self.enabled = bool(config.AI_PROMOTE_TO_KB) if enabled is None else bool(enabled)

    # -- helpers ------------------------------------------------------------
    def knowledge_base(self) -> IngredientKnowledgeBase:
        """The shared KB instance (created on first use)."""
        if self.kb is None:
            self.kb = get_knowledge_base()
        return self.kb

    def block(self, *, promoted: bool, reason: str, entry: Mapping | None = None) -> dict:
        return {
            "promoted": bool(promoted),
            "reason": reason,
            "origin": ingredient_promotion.ORIGIN if promoted else None,
            "review_status": ingredient_promotion.REVIEW_STATUS if promoted else None,
            "functions": list((entry or {}).get("functions", [])),
            "concerns": list((entry or {}).get("concerns", [])),
        }

    # -- promotion ----------------------------------------------------------
    def promote(
        self,
        raw_name: str,
        key: str | None = None,
        data: Mapping | None = None,
        record: Mapping | None = None,
    ) -> dict:
        """
        Promote one enrichment result. **Never raises.**

        Returns a small status block (``promoted`` / ``reason`` / ``origin`` /
        ``review_status`` / mapped tags) that is attached to the enrichment item
        in the report, so the UI can show what was saved and why.
        """
        if not self.enabled:
            return self.block(promoted=False, reason=REASON_DISABLED)
        try:
            normalized = key or normalize_ingredient_name(raw_name)
            blocker = ingredient_promotion.promotion_blockers(
                data, normalized, min_confidence=self.min_confidence
            )
            if blocker:
                return self.block(promoted=False, reason=blocker)

            kb = self.knowledge_base()
            documented = kb.ingredients.get(normalized)
            if documented is not None and not documented.learned:
                # Never shadow curated reference data with AI output.
                return self.block(
                    promoted=False,
                    reason="the knowledge base already documents this ingredient",
                )
            existing = kb.learned_entries.get(normalized)
            if (
                existing
                and existing.get("enrichment_updated_at")
                and existing.get("enrichment_updated_at") == (record or {}).get("updated_at")
            ):
                return self.block(promoted=True, reason=REASON_PROMOTED, entry=existing)

            entry = ingredient_promotion.build_learned_entry(
                key=normalized,
                requested_name=raw_name,
                data=dict(data or {}),
                record=dict(record or {}),
                kb=kb,
            )
            self.store.save(entry)
            kb.add_learned(entry)
        except Exception as exc:  # noqa: BLE001 - promotion must never break analysis
            print(f"[deepcos] knowledge-base promotion skipped ({type(exc).__name__})")
            return self.block(
                promoted=False,
                reason=f"promotion failed ({type(exc).__name__}); the AI result was still cached",
            )
        reason = REASON_REFRESHED if existing else REASON_PROMOTED
        return self.block(promoted=True, reason=reason, entry=entry)

    # -- inspection / removal ----------------------------------------------
    def get(self, name: str) -> Optional[dict]:
        key = normalize_ingredient_name(name)
        if not key:
            return None
        return self.store.get(key)

    def list(self, limit: int = 25, offset: int = 0) -> list[dict]:
        return self.store.list(limit, offset)

    def delete(self, name: str) -> bool:
        key = normalize_ingredient_name(name)
        if not key:
            return False
        removed = self.store.delete(key)
        self.knowledge_base().remove_learned(key)
        return removed

    def status(self) -> dict:
        try:
            store_status = self.store.status()
        except Exception:  # noqa: BLE001
            store_status = {"backend": "unknown", "count": 0}
        return {
            "enabled": self.enabled,
            "min_confidence": self.min_confidence,
            "store_backend": store_status.get("backend", "unknown"),
            "count": store_status.get("count", 0),
            "origin": ingredient_promotion.ORIGIN,
            "review_status": ingredient_promotion.REVIEW_STATUS,
        }


# ---------------------------------------------------------------------------
# Module-level accessors
# ---------------------------------------------------------------------------
_promoter: Optional[KnowledgePromoter] = None
_promoter_lock = threading.Lock()


def get_promoter() -> KnowledgePromoter:
    """Process-wide singleton promoter (shares the cached knowledge base)."""
    global _promoter
    with _promoter_lock:
        if _promoter is None:
            _promoter = KnowledgePromoter(kb=get_knowledge_base())
        return _promoter


def promoter_status() -> dict:
    """Status for ``/api/health``: booleans and counts only, never secrets."""
    try:
        return get_promoter().status()
    except Exception as exc:  # noqa: BLE001
        return {
            "enabled": bool(config.AI_PROMOTE_TO_KB),
            "min_confidence": config.AI_PROMOTE_MIN_CONFIDENCE,
            "store_backend": "unknown",
            "count": 0,
            "error": type(exc).__name__,
        }


def list_learned_entries(limit: int = 25, offset: int = 0) -> list[dict]:
    return get_promoter().list(limit, offset)


def get_learned_entry(name: str) -> Optional[dict]:
    return get_promoter().get(name)


def delete_learned_entry(name: str) -> bool:
    return get_promoter().delete(name)

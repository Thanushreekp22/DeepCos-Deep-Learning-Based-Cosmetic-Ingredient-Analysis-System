"""
Potential-concern screening engine  (knowledge base, not a neural network).

DeepCos deliberately separates two very different questions:

* "what formulation characteristics does this pattern suggest?"  -> neural
  network (learned from data, probabilistic)
* "which documented characteristics or regulatory notes apply to these
  ingredients?"                                                 -> rule engine
  over curated reference data (deterministic, auditable)

This module implements the second question. It reports *ingredient-level*
observations with a severity and the reference note behind them. It never
outputs a verdict such as "safe" or "unsafe", and it never gives medical advice.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from ml.knowledge_base import IngredientKnowledgeBase, ResolvedIngredient


def _ingredients_by_concern(resolved: Sequence[ResolvedIngredient]) -> dict[str, list[tuple[str, str]]]:
    """Concern tag -> [(ingredient, origin), ...] where origin is local|ai-learned."""
    mapping: dict[str, list[tuple[str, str]]] = {}
    for item in resolved:
        if not item.info:
            continue
        for tag in item.info.concerns:
            mapping.setdefault(tag, []).append((item.inci, item.info.origin))
    return mapping


def screen(resolved: Sequence[ResolvedIngredient], kb: IngredientKnowledgeBase) -> dict:
    """
    Run the ingredient screening and return the concern block of the report.

    Findings are sorted by severity (highest first) and then by how many
    ingredients triggered the rule. Findings that were triggered by an entry
    from the AI-learned overlay carry ``ai_derived: True`` and a tag with no
    local rule keeps severity ``info`` plus an explicit "unverified" message.
    """
    by_concern = _ingredients_by_concern(resolved)
    findings: list[dict] = []

    for tag, hits in by_concern.items():
        rule = kb.concern_rule(tag)
        documented = bool(rule)
        severity = str(rule.get("severity", "info")) if documented else "info"
        ingredients = [inci for inci, _ in hits]
        ai_derived = any(origin == "ai-learned" for _, origin in hits)
        findings.append(
            {
                "tag": tag,
                "label": rule.get("label", tag.replace("-", " ").title()),
                "severity": severity,
                "severity_rank": kb.severity_rank(severity),
                "message": rule.get("message", "")
                if documented
                else (
                    "AI enrichment flagged this characteristic for an ingredient that is not in "
                    "the verified local knowledge base; treat it as unverified information."
                ),
                "advice": rule.get("advice", "")
                if documented
                else (
                    "Check the ingredient against a regulatory or dermatological source before "
                    "drawing conclusions."
                ),
                "ingredients": sorted(set(ingredients)),
                "ingredient_count": len(set(ingredients)),
                "documented_rule": documented,
                "ai_derived": ai_derived,
            }
        )

    # Regulatory notes get their own, clearly separated block ---------------
    regulatory: list[dict] = []
    for item in resolved:
        if item.info and item.info.regulatory_status not in ("allowed", "unknown"):
            regulatory.append(
                {
                    "ingredient": item.inci,
                    "status": item.info.regulatory_status,
                    "note": item.info.regulatory_note,
                }
            )

    findings.sort(key=lambda f: (-f["severity_rank"], -f["ingredient_count"], f["label"]))

    has_fragrance = "fragrance" in by_concern or "fragrance-allergen" in by_concern
    has_exfoliating = "exfoliating-acid" in by_concern
    has_preservative = any(
        item.info and "preservative" in item.info.functions for item in resolved
    )
    has_uv_filter = any(item.info and "uv-filter" in item.info.functions for item in resolved)

    ai_derived_findings = [f for f in findings if f["ai_derived"]]

    summary_bits: list[str] = []
    if has_fragrance:
        summary_bits.append("fragrance material detected")
    if has_exfoliating:
        summary_bits.append("exfoliating acid detected")
    if has_preservative:
        summary_bits.append("preservative system identified")
    if has_uv_filter:
        summary_bits.append("UV filter present")
    if ai_derived_findings:
        summary_bits.append(
            f"{len(ai_derived_findings)} characteristic(s) come from unreviewed AI-learned entries"
        )
    if not summary_bits:
        summary_bits.append(
            "no fragrance, exfoliating acid or restricted ingredient flagged"
        )

    return {
        "findings": findings,
        "highest_severity": findings[0]["severity"] if findings else "info",
        "highest_severity_label": (
            kb.concern_severities.get(findings[0]["severity"], {}).get("label", "Information")
            if findings
            else "Information"
        ),
        "regulatory_notes": regulatory,
        "ai_derived_count": len(ai_derived_findings),
        "flags": {
            "fragrance": has_fragrance,
            "exfoliating_acid": has_exfoliating,
            "preservative": has_preservative,
            "uv_filter": has_uv_filter,
            "restricted": bool(regulatory),
        },
        "summary": (
            "Based on the available ingredient information, " + ", ".join(summary_bits) + "."
        ),
        "ingredient_assessment": (
            "This is a screening of documented ingredient characteristics, not a verdict on "
            "safety or suitability for any individual. Concentration, pH, the rest of the "
            "formulation and your own skin matter, and regulatory limits differ by market."
        ),
    }


def concern_counts(resolved: Sequence[ResolvedIngredient]) -> dict[str, int]:
    """Compact count of concern tags - handy for charts."""
    counter = Counter(tag for item in resolved if item.info for tag in item.info.concerns)
    return dict(counter)


def function_histogram(resolved: Sequence[ResolvedIngredient]) -> dict[str, int]:
    """Count of documented functions across the ingredient list."""
    counter = Counter(
        function for item in resolved if item.info for function in item.info.functions
    )
    return {tag: count for tag, count in counter.most_common()}


def ingredient_breakdown(
    resolved: Sequence[ResolvedIngredient],
    kb: IngredientKnowledgeBase,
) -> list[dict]:
    """
    Per-ingredient analysis rows for the report.

    Each row carries the documented functions (with human labels), the reference
    note, the concern tags and whether the knowledge base recognised the name.
    ``origin``/``review_status`` tell verified reference data (``local`` /
    ``verified``) apart from entries promoted by AI enrichment (``ai-learned`` /
    ``unreviewed``).
    """
    rows: list[dict] = []
    for position, item in enumerate(resolved):
        functions = list(item.info.functions) if item.info else []
        rows.append(
            {
                "position": position + 1,
                "ingredient": item.inci,
                "raw": item.raw,
                "resolved": item.resolved,
                "match_type": item.match_type,
                "match_score": round(float(item.match_score), 3),
                "functions": functions,
                "function_labels": [kb.function_label(fn) for fn in functions],
                "primary_function": kb.function_label(functions[0]) if functions else "",
                "note": item.info.note if item.info else "",
                "concerns": list(item.info.concerns) if item.info else [],
                "regulatory_status": item.info.regulatory_status if item.info else "unknown",
                "origin": item.info.origin if item.info else "unknown",
                "review_status": item.info.review_status if item.info else "unknown",
                "ai_learned": bool(item.info and item.info.learned),
            }
        )
    return rows


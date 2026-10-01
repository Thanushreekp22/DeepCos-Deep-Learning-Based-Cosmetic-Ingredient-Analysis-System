"""
Knowledge-base endpoints - the cosmetic reference data behind the rules.

* ``GET    /api/knowledge/stats``      - ingredient / function / concern counts
* ``GET    /api/knowledge/search``     - ranked ingredient search
* ``GET    /api/knowledge/ingredient`` - resolve one ingredient (exact/alias/fuzzy)
* ``GET    /api/knowledge/functions``  - full function taxonomy
* ``GET    /api/knowledge/concerns``   - concern rules + severity ladder
* ``GET    /api/knowledge/samples``    - ready-made example ingredient lists (UI)
* ``GET    /api/knowledge/enrichment`` - one cached AI-enrichment record
* ``GET    /api/knowledge/enrichments``- list cached AI-enrichment records
* ``DELETE /api/knowledge/enrichment`` - remove one cached AI-enrichment record
* ``GET    /api/knowledge/learned``    - list the AI-learned KB overlay entries
* ``GET    /api/knowledge/learned/entry`` - one AI-learned entry
* ``DELETE /api/knowledge/learned``    - remove one AI-learned entry

The enrichment endpoints only expose cached, validated AI content (source,
model, timestamps, review status). They never expose secrets and never trigger
a live provider call. The learned endpoints expose the AI-learned knowledge-base
overlay written when ``AI_PROMOTE_TO_KB`` is enabled; every record is labelled
``origin="ai-learned"`` / ``review_status="unreviewed"`` and can be deleted here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.config import AI_PROMOTE_MIN_CONFIDENCE, AI_PROMOTE_TO_KB
from backend.services.ingredient_enrichment import (
    delete_cached_enrichment,
    get_cached_enrichment,
    list_cached_enrichments,
)
from backend.services.kb_promotion import (
    delete_learned_entry,
    get_learned_entry,
    list_learned_entries,
)
from ml.inference import get_kb

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

SAMPLE_PRODUCTS: list[dict] = [
    {
        "id": "hydrating-serum",
        "title": "Hydrating serum",
        "text": (
            "Aqua, Glycerin, Sodium Hyaluronate, Panthenol, Niacinamide, "
            "Betaine, Allantoin, Carbomer, Sodium Hydroxide, Phenoxyethanol, "
            "Ethylhexylglycerin, Tocopherol"
        ),
    },
    {
        "id": "brightening-toner",
        "title": "Brightening toner",
        "text": (
            "Aqua, Glycerin, Niacinamide, Alpha-Arbutin, Ascorbyl Glucoside, "
            "Panthenol, Sodium Citrate, Citric Acid, Hydroxyethylcellulose, "
            "Phenoxyethanol, Parfum, Limonene"
        ),
    },
    {
        "id": "exfoliating-cleanser",
        "title": "Exfoliating gel cleanser",
        "text": (
            "Aqua, Cocamidopropyl Betaine, Glycerin, Salicylic Acid, "
            "Sodium Lauroyl Sarcosinate, Carbomer, Sodium Hydroxide, "
            "Menthol, Mentha Piperita Oil, Phenoxyethanol, Chlorphenesin"
        ),
    },
    {
        "id": "night-retinal-cream",
        "title": "Night retinal cream",
        "text": (
            "Aqua, Glycerin, Caprylic/Capric Triglyceride, Retinal, Squalane, "
            "Ceramide NP, Cholesterol, Niacinamide, Panthenol, Cetearyl Alcohol, "
            "Polysorbate 60, Tocopherol, Phenoxyethanol"
        ),
    },
]


@router.get("/stats", summary="Knowledge-base statistics")
def stats() -> dict:
    return get_kb().stats()


@router.get("/search", summary="Search ingredients")
def search(
    q: str = Query("", description="Ingredient name, alias or keyword"),
    limit: int = Query(25, ge=1, le=100),
) -> dict:
    results = get_kb().search(q, limit=limit)
    return {"query": q, "count": len(results), "results": [item.to_dict() for item in results]}


@router.get("/ingredient", summary="Resolve one ingredient against the KB")
def ingredient(name: str = Query(..., description="Raw ingredient string")) -> dict:
    resolved = get_kb().resolve(name)
    payload = {
        "raw": resolved.raw,
        "key": resolved.key,
        "inci": resolved.inci,
        "match_type": resolved.match_type,
        "match_score": resolved.match_score,
        "info": resolved.info.to_dict() if resolved.info else None,
        "functions": get_kb().describe_functions(resolved.info.functions if resolved.info else []),
    }
    if resolved.info:
        payload["concerns"] = [
            {
                "tag": tag,
                "rule": get_kb().concern_rule(tag),
                "severity_rank": get_kb().severity_rank(
                    get_kb().concern_rule(tag).get("severity", "")
                ),
            }
            for tag in resolved.info.concerns
        ]
    return payload


@router.get("/functions", summary="Function taxonomy")
def functions() -> dict:
    kb = get_kb()
    return {
        "count": len(kb.function_taxonomy),
        "taxonomy": [
            {"tag": tag, **({"label": kb.function_label(tag)} if kb.function_label(tag) else {}),
             "description": kb.function_description(tag)}
            for tag in sorted(kb.function_taxonomy)
        ],
    }


@router.get("/concerns", summary="Concern rules and severity ladder")
def concerns() -> dict:
    kb = get_kb()
    return {
        "severities": getattr(kb, "concern_severities", {}),
        "rules": {tag: kb.concern_rule(tag) for tag in sorted(getattr(kb, "concern_rules", {}))},
        "disclaimer": kb.disclaimer,
    }


@router.get("/samples", summary="Example ingredient lists for the demo UI")
def samples() -> dict:
    return {"count": len(SAMPLE_PRODUCTS), "samples": SAMPLE_PRODUCTS}


# -- AI enrichment cache (read-only inspection + removal) --------------------
@router.get("/enrichment", summary="One cached AI-enrichment record")
def enrichment(name: str = Query(..., description="Ingredient name (raw or normalised)")) -> dict:
    record = get_cached_enrichment(name)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached AI enrichment found for '{name}'.",
        )
    return record


@router.get("/enrichments", summary="List cached AI-enrichment records")
def enrichments(
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> dict:
    records = list_cached_enrichments(limit=limit, offset=offset)
    return {"count": len(records), "records": records}


@router.delete("/enrichment", summary="Delete one cached AI-enrichment record")
def delete_enrichment(name: str = Query(..., description="Ingredient name (raw or normalised)")) -> dict:
    deleted = delete_cached_enrichment(name)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No cached AI enrichment found for '{name}'.",
        )
    return {"deleted": True, "name": name}


# -- AI-learned knowledge-base overlay (promoted AI enrichment) --------------
@router.get("/learned", summary="List AI-learned knowledge-base entries")
def learned(
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """
    Entries the optional AI-enrichment feature persisted for ingredients the
    verified knowledge base does not document. They are served to the analysis
    pipeline, and they stay labelled as unreviewed AI content.
    """
    records = list_learned_entries(limit=limit, offset=offset)
    return {
        "count": len(records),
        "promotion_enabled": bool(AI_PROMOTE_TO_KB),
        "min_confidence": AI_PROMOTE_MIN_CONFIDENCE,
        "origin": "ai-learned",
        "review_status": "unreviewed",
        "entries": records,
    }


@router.get("/learned/entry", summary="One AI-learned knowledge-base entry")
def learned_entry(name: str = Query(..., description="Ingredient name (raw or normalised)")) -> dict:
    record = get_learned_entry(name)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No AI-learned knowledge-base entry found for '{name}'.",
        )
    return record


@router.delete("/learned", summary="Delete one AI-learned knowledge-base entry")
def delete_learned(name: str = Query(..., description="Ingredient name (raw or normalised)")) -> dict:
    """Remove an entry again (the ingredient stops resolving locally)."""
    deleted = delete_learned_entry(name)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No AI-learned knowledge-base entry found for '{name}'.",
        )
    return {"deleted": True, "name": name, "learned_count": get_kb().learned_count()}

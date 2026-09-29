"""
Knowledge-base endpoints - the cosmetic reference data behind the rules.

* ``GET /api/knowledge/stats``     - ingredient / function / concern counts
* ``GET /api/knowledge/search``    - ranked ingredient search
* ``GET /api/knowledge/ingredient``- resolve one ingredient (exact/alias/fuzzy)
* ``GET /api/knowledge/functions`` - full function taxonomy
* ``GET /api/knowledge/concerns``  - concern rules + severity ladder
* ``GET /api/knowledge/samples``   - ready-made example ingredient lists (UI)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

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

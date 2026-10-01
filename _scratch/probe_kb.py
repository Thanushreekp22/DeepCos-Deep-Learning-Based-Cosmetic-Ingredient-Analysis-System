"""Scratch probe: inspect knowledge-base behaviour (temporary, delete after use)."""
import json

from ml.knowledge_base import get_knowledge_base

kb = get_knowledge_base()
print("STATS", json.dumps(kb.stats())[:900])
names = [
    "Aqua",
    "Glycerin",
    "Niacinamide",
    "Sodium Hyaluronate",
    "Parfum",
    "Salicylic Acid",
    "Phenoxyethanol",
    "Retinal",
    "Totallymadeup Extract",
]
for n in names:
    r = kb.resolve(n)
    info = r.info
    print(
        "RESOLVE",
        n,
        "| inci=", r.inci,
        "| type=", r.match_type,
        "| score=", round(r.match_score, 3),
        "| resolved=", r.resolved,
        "| functions=", (info.functions if info else None),
        "| concerns=", (info.concerns if info else None),
        "| reg=", (info.regulatory_status if info else None),
        "| origin=", (info.origin if info else None),
    )
print("FRAGRANCE RULE", kb.concern_rule("fragrance"))
print("SEV RANKS", {s: kb.severity_rank(s) for s in ("info", "low", "medium", "high", "bogus")})
print("LEARNED COUNT", kb.learned_count(), "TOTAL", len(kb))
print("SEARCH niaci", [i.inci for i in kb.search("niaci", limit=5)])
print("ALL INCI sample", kb.all_inci()[:8], "n=", len(kb.all_inci()))

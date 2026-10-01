"""Scratch smoke check (temporary): does the trained pipeline actually work?"""
import json
import os

os.environ["AI_ENRICHMENT_ENABLED"] = "false"
os.environ["AI_PROMOTE_TO_KB"] = "false"
os.environ["MONGODB_URI"] = "mongodb://127.0.0.1:27017"

from ml import inference  # noqa: E402

status = inference.model_status()
print("STATUS:", json.dumps(status, indent=1, default=str)[:1500])

report = inference.analyze(
    text="Aqua, Glycerin, Niacinamide, Sodium Hyaluronate, Panthenol, Carbomer, Phenoxyethanol"
)
print("KEYS:", sorted(report.keys()))
print("PRODUCT:", report["product"])
print("PROFILES:", json.dumps(report["profile_scores"]))
print("SUMMARY:", report["profile_summary"])
print("UNRESOLVED:", report["unresolved_ingredients"])
print("CONCERNS:", report["concerns"]["highest_severity"], len(report["concerns"]["findings"]))
print("KEY_ING:", [k["ingredient"] for k in report["key_ingredients"]])
print("EXPLAIN:", list(report["explainability"]["top_contributors"].keys()))
print("KB:", report["knowledge_base"])
print("CONF:", json.dumps(report["profile_confidence"], indent=1)[:800])

"""Scratch probe #2: config + preprocessing + model sanity (temporary)."""
import json
import time

from ml import config

print("TARGETS", config.PROFILE_TARGETS)
print("CATEGORIES", config.CATEGORIES)
print("SEQ_LEN", config.SEQ_LEN, "PAD", config.PAD_INDEX)
print("BANDS", config.PROFILE_BANDS)
print("LABELS", config.PROFILE_LABELS)

from ml.dataset.preprocessing import (
    canonical_names,
    clean_raw_text,
    ingredient_list_from_any,
    parse_ingredient_text,
    text_preprocessing_report,
)

raw = "Ingredient INCI: Aqua (Water), Glycerin*, Sodium Hyaluronate; Panthenol.  PARFUM "
print("CLEAN", repr(clean_raw_text(raw)))
print("PARSE", parse_ingredient_text(raw))
print("FROM_ANY list", ingredient_list_from_any(["Aqua", " Glycerin ", ""]))
print("REPORT", json.dumps(text_preprocessing_report(raw))[:400])

import importlib.util  # noqa: E402

for name in ("httpx", "PIL", "pytesseract", "cv2", "pymongo"):
    print("DEP", name, importlib.util.find_spec(name) is not None)

t0 = time.time()
from ml.inference import analyze  # noqa: E402

samples = {
    "serum": "Aqua, Glycerin, Sodium Hyaluronate, Panthenol, Niacinamide, Betaine, Allantoin, Carbomer, Sodium Hydroxide, Phenoxyethanol, Ethylhexylglycerin, Tocopherol",
    "cleanser": "Aqua, Cocamidopropyl Betaine, Glycerin, Salicylic Acid, Sodium Lauroyl Sarcosinate, Carbomer, Sodium Hydroxide, Menthol, Mentha Piperita Oil, Phenoxyethanol, Chlorphenesin",
    "cream": "Aqua, Glycerin, Caprylic/Capric Triglyceride, Squalane, Ceramide NP, Cholesterol, Niacinamide, Panthenol, Cetearyl Alcohol, Polysorbate 60, Tocopherol, Phenoxyethanol",
}
out = {}
for key, text in samples.items():
    r = analyze(text=text, explain=False)
    out[key] = {
        "category": r["product"]["category"],
        "conf": r["product"]["category_confidence"],
        "scores": r["profile_scores"],
        "match_rate": r["knowledge_base"]["match_rate"],
        "unresolved": r["unresolved"],
        "key_ing": [k["ingredient"] for k in r["key_ingredients"]],
    }
print("MODELS SEC", round(time.time() - t0, 1))
for k, v in out.items():
    print("RESULT", k, json.dumps(v))

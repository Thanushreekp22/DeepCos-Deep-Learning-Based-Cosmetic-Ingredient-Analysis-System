# DeepCos — Deep Learning Based Cosmetic Ingredient Analysis System

Paste an INCI ingredient list (or upload a product-label photo) and get an **explainable
analysis report**: predicted product category, four formulation-profile scores, ingredient
reference information, potential-concern screening, and model-derived ingredient attributions.
Every analysis is persisted so you can browse, re-open, export or delete it later.

> Educational project. Predictions come from a model trained on **weakly supervised synthetic
> labels**, concern flags come from local reference data and deterministic rules. Nothing here
> is medical advice, a safety assessment or a regulatory opinion.

## Features

- **Two input modes** — typed/pasted ingredient list, or a label photo (AlexNet-style CNN text-region
  detector + Tesseract OCR with variant fallback).
- **Multi-task ingredient model** — Embedding → LSTM → MLP predicting 8 product categories and
  4 formulation profiles (hydration, brightening, exfoliation, barrier support).
- **Explainability** — occlusion-based attributions showing which ingredients pushed each profile
  score up or down, plus sequence-window reporting (ingredients past the model window are flagged).
- **Knowledge base** — 226 ingredients, 156 aliases, 25 functions and 14 concern rules loaded from
  local JSON; exact, alias and fuzzy matching; browsable/searchable in the UI.
- **Reports & history** — MongoDB (or JSON fallback) persistence, Markdown/text/JSON export,
  stored PNG previews, printable report page.
- **Optional AI enrichment** — off by default; unresolved ingredients can be looked up through the
  Groq API, cached separately and labelled `ai-learned` / `unreviewed`. The verified reference
  files are never rewritten.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite 5, React Router 6, plain CSS (`frontend/src/styles/global.css`) |
| Backend | FastAPI 0.141 + Uvicorn, Pydantic v2 |
| ML | TensorFlow 2.21 / Keras 3, scikit-learn, NumPy, pandas |
| Image | OpenCV, Pillow, pytesseract + native Tesseract, custom AlexNet-style CNN |
| Storage | MongoDB via pymongo (auto-falls back to a JSON file store) |

## Repository layout

```text
backend/      FastAPI app: main.py, config.py, database.py, routers/, services/
ml/           Inference, model definition, dataset generation, image pipeline, rules, training
frontend/     React dashboard (Vite dev server on 5173, proxies /api to 127.0.0.1:8000)
data/knowledge_base/   ingredients.json, concern_rules.json, profile_weights.json, product_templates.json
artifacts/models/      Checked-in vocabulary + metrics JSON (trained *.keras weights are gitignored)
tests/        Pytest suite for AI enrichment and knowledge-base promotion
projectDetails.md  Full architecture, API reference and metric details
```

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11** | Required — TensorFlow 2.21 / Keras 3 are pinned for it |
| **Node.js 18+** | For the Vite frontend (npm included) |
| **Git** | To clone |
| MongoDB *(optional)* | Skip it and the backend uses a local JSON store automatically |
| Tesseract OCR *(optional)* | Only needed for image/label analysis. `pytesseract` is just a wrapper — the native engine must be installed: Windows [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki), Linux `sudo apt install tesseract-ocr`, macOS `brew install tesseract` |

## Quick start

### 1. Clone

```powershell
git clone https://github.com/Thanushreekp22/DeepCos-Deep-Learning-Based-Cosmetic-Ingredient-Analysis-System.git
cd DeepCos-Deep-Learning-Based-Cosmetic-Ingredient-Analysis-System
```

### 2. Python environment + dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
Copy-Item .env.example .env         # macOS/Linux: cp .env.example .env
```

Edit `.env` only if MongoDB, Tesseract or the ports differ from the defaults.

### 3. Train the ingredient model (required)

Trained weights are **not in the repository** (`.gitignore` excludes `artifacts/models/*.keras`), so
train once. No dataset download is needed — the weakly labelled dataset is generated on demand from
`data/knowledge_base/`, which ships with the repo:

```powershell
python -m ml.train_ingredient_model
```

Quick smoke run instead of the full job: `python -m ml.train_ingredient_model --n 1500 --epochs 5`
(faster, but held-out metrics will be lower than a full run).

Outputs: `artifacts/models/deepcos_ingredient_model.keras`, `ingredient_vocabulary.json`,
`ingredient_model_metrics.json` and `artifacts/reports/ingredient_training_curves.png`.

### 4. Run the backend (terminal 1)

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The first start takes a few seconds while TensorFlow loads. Startup prints the persistence backend,
model status, OCR availability and text-detector status. Interactive API docs:
<http://127.0.0.1:8000/docs>.

### 5. Run the frontend (terminal 2)

```powershell
cd frontend
npm install
npm run dev
```

Open **<http://localhost:5173>**. The Vite dev server proxies `/api` to `http://127.0.0.1:8000`,
so no frontend configuration is needed in development.

### 6. Verify

```powershell
# API health: "status": "ok" once the ingredient model loads
curl.exe http://127.0.0.1:8000/api/health

# End-to-end smoke test (returns an analysis_id)
curl.exe -X POST http://127.0.0.1:8000/api/analyze/text -H "Content-Type: application/json" -d "{\"text\":\"Aqua, Glycerin, Niacinamide, Fragrance, Phenoxyethanol\"}"
```

If `status` is `degraded` with `ingredient_model.trained = false`, step 3 was skipped or failed.

## Optional extras

### Train the image CNNs (only for the photo path)

The ingredient model is enough for text analysis. The label-photo pipeline uses two optional CNNs
and falls back to plain OCR when they are missing:

```powershell
python -m ml.train_label_cnn --task text_region      # text / background patch classifier
python -m ml.train_label_cnn --task label_category   # 8-class product-category CNN
# or both at once
python -m ml.train_label_cnn --task both
```

### MongoDB (optional)

Point `MONGODB_URI` / `MONGODB_DB` in `.env` at a local server or a MongoDB Atlas cluster. Without a
reachable server the backend transparently switches to `artifacts/reports/analyses_store.json`
(the dashboard shows "JSON fallback"), and reports written there are migrated to MongoDB on the next
successful connection. Never commit real credentials — keep them in your local `.env` only.

### Optional AI enrichment (Groq)

Disabled unless **both** `AI_ENRICHMENT_ENABLED=true` and a `GROQ_API_KEY` are set in `.env`.
Unresolved ingredients are then looked up through the Groq API, validated with Pydantic and cached
(`ingredient_enrichments` collection, or the JSON fallback). With `AI_PROMOTE_TO_KB=true`, a
validated result is additionally written to the clearly-labelled AI-learned overlay
(`data/knowledge_base/learned_ingredients.json`, `origin="ai-learned"`, `review_status="unreviewed"`)
so later runs resolve it locally. The verified reference files are never modified, and the key is
never logged or returned by the API.

## Tests

```powershell
python -m pytest tests -q      # run from the repository root
```

The suite covers AI enrichment (with a mocked provider client — no network calls) and
knowledge-base promotion. `pytest` is already installed by `requirements.txt`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `only one usage of each socket address ... port 8000` | A previous backend is still running. `Get-NetTCPConnection -LocalPort 8000 -State Listen` then `Stop-Process -Id <PID> -Force` |
| `WARNING - ingredient model missing; run: python -m ml.train_ingredient_model` | Step 3 not done, or it failed — retrain; analysis endpoints return `503` until a model exists |
| `/api/health` reports `"ocr": {"available": false}` | Tesseract is not installed/on PATH. Set `TESSERACT_CMD` in `.env` to the full `tesseract.exe` path |
| `/api/model/metrics` or `/api/model/curves` return 404 | Metrics JSON or the training-curve PNG does not exist yet — train the model |
| Model metrics look poor | Expected: labels are synthetic/weakly supervised, so profile R² values are modest by design |
| Frontend loads but every panel errors | Backend not running, or not on port 8000 (the dev proxy target is fixed at `127.0.0.1:8000`) |

## Notes

- Retrain the model and rebuild the vocabulary together — they must stay in sync
  (`ingredient_vocabulary.json` is written by the same training run).
- Generated outputs (`artifacts/logs/`, `artifacts/reports/`, `data/processed/`, `data/synthetic/`)
  are gitignored and safe to delete; the checked-in `artifacts/models/*.json` files are metric
  snapshots from the last committed training run.
- Frontend production build: `npm run build` → `frontend/dist`, served with `npm run preview`. Set
  `VITE_API_URL` at build time when the API is not on the frontend origin.
- Full architecture, endpoint-by-endpoint API reference and metric tables: see
  [`projectDetails.md`](projectDetails.md).


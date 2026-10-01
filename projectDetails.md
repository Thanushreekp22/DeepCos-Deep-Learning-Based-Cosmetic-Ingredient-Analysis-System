# DeepCos — Project Details

> Repository guide for the code and artifacts present in this checkout. Values and behavior below were read from the source on 2026-09-29; retrained artifacts or edited configuration may change them.

## 1. Project overview

DeepCos is a local web application for exploring cosmetic ingredient lists. Users can paste an INCI-style list or upload a product-label photo. The application returns a predicted product category, four formulation-profile scores, ingredient-level reference information, potential-concern screening, and (when enabled) model-derived ingredient attributions. Analyses are persisted for history and can be exported.

The repository has three cooperating parts:

1. **React/Vite frontend** (`frontend/`) provides the dashboard and analysis/report screens.
2. **FastAPI backend** (`backend/`) exposes analysis, knowledge-base, health, model, history, image preview, and export endpoints.
3. **ML/reference pipeline** (`ml/`, `data/`, `artifacts/`) handles preprocessing, CNN/OCR image input, ingredient-sequence inference, explainability, rule-based screening, training, and model metadata.

The learned profile and category outputs are predictions from the ingredient-sequence model. Ingredient functions and concern flags come from local JSON reference data and deterministic rules. Neither output is a product-safety decision or medical advice.

## 2. Repository map

```text
backend/
  main.py                    FastAPI app, startup lifecycle, CORS, routes, error mapping
  config.py                  Environment-backed API, paths, persistence and upload settings
  database.py                MongoDB store, JSON fallback, preview-image assets
  schemas.py                 Pydantic request/response schemas
  routers/
    analyze.py               Analyze, history, exports, and image endpoints
    health.py                Health, model metadata, metrics and curves
    knowledge.py             Ingredient knowledge-base API and demo samples
  services/analysis_service.py  ML orchestration and report persistence/export
ml/
  config.py                  Shared paths, categories, targets and model hyperparameters
  inference.py               End-to-end inference and report assembly
  knowledge_base.py          JSON loader, aliases, exact/fuzzy matching and search
  dataset/                    Ingredient parsing, vocabulary, synthetic weak-label dataset
  models/                     Embedding/LSTM/MLP model and occlusion explanations
  image/                      Image preprocessing, AlexNet-style CNN, text detection and OCR
  rules/                      Profile presentation, concern screening and report rendering
  train_ingredient_model.py   Train/evaluate the ingredient model
  train_label_cnn.py          Train text-region or label-category CNN tasks
data/knowledge_base/          Ingredient, concern, profile-weight and product-template JSON
artifacts/models/             Checked-in vocabulary and training metrics; model binaries ignored
frontend/src/
  App.jsx, api.js, main.jsx   App shell, API client, React entry point
  pages/                      Dashboard, Analyze, Report, History, Knowledge, ModelPage
  components/ui.jsx           Shared UI primitives
  styles/global.css           Global styling and responsive layout
scripts/fix_created_at.py    One-off MongoDB timestamp normalization helper
requirements.txt             Pinned Python dependencies
frontend/package.json        React/Vite dependencies and npm scripts
.env.example                 Runtime configuration template
.gitignore                   Excluded environments, generated data and binary artifacts
```

There is no README, tracked test source, `pyproject.toml`, Dockerfile, or compose file in this checkout. The `tests/` and `docs/` directories do not contain files in the tracked-file inventory. Generated files under `artifacts/reports/`, `artifacts/logs/`, `data/processed/`, and `data/synthetic/` are expected at runtime/training but are not currently present in the inventory.

## 3. User interface

The frontend is a React 18 single-page app using React Router 6. Vite serves it on port **5173** and proxies `/api` to `http://127.0.0.1:8000` in development. `frontend/src/api.js` uses `VITE_API_URL` as its optional base URL (empty by default, so same-origin/proxy URLs are used).

Routes and page responsibilities:

| Route | Page | Purpose |
|---|---|---|
| `/` | Dashboard | API/model/storage/knowledge status and six recent analyses |
| `/analyze` | Analyze | Text or image input, sample lists, submits analysis |
| `/report/:id` | Report | Prediction, confidence/reliability, attribution, concern findings, OCR/images, ingredient table; print/delete/export links |
| `/history` | History | Browse and delete saved reports (up to 200 in one request) |
| `/knowledge` | Knowledge | Search ingredient reference entries, inspect details, browse functions and concern rules |
| `/model` | ModelPage | Availability, saved metrics, architecture/training information and curves |

`App.jsx` fetches health once on mount and displays model/database status and degraded/unreachable banners. The shared components implement meters, chips, severity badges, loading, empty and error states. Global styles define the dark dashboard theme and responsive layout.

## 4. Runtime setup

### Prerequisites

- Python **3.11** is the version identified in `requirements.txt` comments.
- Node.js/npm for the frontend.
- Tesseract OCR native executable for photo-to-text analysis. `pytesseract` is only a Python wrapper. Without the native engine, typed-text analysis still works, but OCR cannot extract text from an image.
- MongoDB is optional: the backend falls back to a local JSON store if MongoDB is unavailable.
- A trained ingredient model and matching vocabulary are required for analyses. If absent, the API reports degraded status and analysis endpoints return an unavailable/model-training error.

### Install and run (PowerShell from repository root)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env if MongoDB, Tesseract, ports, or the frontend API URL differ.
python -m uvicorn backend.main:app --reload --port 8000
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`; interactive API docs are at `http://127.0.0.1:8000/docs`.

Production frontend build commands are `npm run build` and `npm run preview` in `frontend/`. Set `VITE_API_URL` at build time when the API is not on the frontend origin. The code does not include deployment/container configuration.

### Environment settings

| Variable | Default / example | Effect |
|---|---|---|
| `DEEPCOS_HOST` | `127.0.0.1` | Host used by `python -m backend.main` |
| `DEEPCOS_PORT` | `8000` | Port used by `python -m backend.main` |
| `DEEPCOS_CORS_ORIGINS` | localhost Vite origins | Comma-separated allowed browser origins; backend default additionally includes port 4173 |
| `MONGODB_URI` | `mongodb://127.0.0.1:27017` | MongoDB connection URI |
| `MONGODB_DB` | `deepcos` | Database; analyses collection is `analyses` |
| `MONGODB_TIMEOUT_MS` | `1500` | MongoDB server-selection timeout |
| `TESSERACT_CMD` | Windows Tesseract path in `.env.example` | Optional native Tesseract binary path; can also be on `PATH` |
| `DEEPCOS_MAX_UPLOAD_BYTES` | 12 MiB | Maximum image request payload accepted by API |
| `VITE_API_URL` | empty | Frontend API base URL; note `.env.example` uses `VITE_API_BASE_URL`, which the client does not read |

The backend loads the root `.env` via `python-dotenv` if installed. The frontend Vite config proxy target is hard-coded to `127.0.0.1:8000` for development.

## 5. Analysis lifecycle

### Typed ingredient list

1. `POST /api/analyze/text` validates a non-empty string up to 20,000 characters.
2. `ml.dataset.preprocessing` applies Unicode normalization, removes an `Ingredients:`/similar prefix, trailing directions/warnings sections, percentage annotations and OCR noise, joins OCR hyphenated line breaks, splits on commas/semicolons/bullets/newlines, filters implausible names, and deduplicates while preserving order.
3. `IngredientKnowledgeBase` resolves canonical INCI names and aliases; fuzzy matching uses `SequenceMatcher` with a default cutoff of 0.86. Unmatched names remain in the report as best-effort title-cased names and generate a warning.
4. Vocabulary encoding maps names to integer ids, pads to 32 positions, and uses id 1 for unknown tokens. The model sees the first 32 ingredients in list order; all parsed ingredients remain in the ingredient table.
5. The ingredient model predicts profile scores and an eight-way product category distribution. Optional occlusion attribution re-predicts after removing each token in turn.
6. Rule-based functions/concerns are looked up independently from reference JSON and assembled into ingredient rows, concern findings and flags.
7. The report is returned and persisted by default. The UI navigates to `/report/{analysis_id}`.

### Image label

1. `POST /api/analyze/image` accepts multipart form data, defaults to `explain=true`, `persist=true`, and limits OCR variants to 1–8 (default 5). The API checks the byte limit; the image loader validates/decodes it and assesses quality.
2. If available, the text-region CNN scans overlapping grayscale 64×64 windows, creates a probability heatmap, chooses a text band and crops it. If unavailable, OCR receives the full image.
3. OpenCV preprocessing creates multiple OCR variants; Tesseract is run on up to the requested number and the most plausible parsed ingredient string is selected.
4. If optional multipart `text` is supplied, it replaces OCR text for analysis while OCR output remains attached for transparency. Otherwise empty OCR results produce a 422 error and a manual-paste hint.
5. The selected text runs through the same ingredient model and reference screening path as typed input. The report includes image quality, detector output, OCR metadata/text and preview PNGs.

The separate label-category CNN can be trained and its availability/metrics are exposed, but the current inference report's product category is generated by the **ingredient sequence model**. The label-category CNN is not invoked from `analyze_image_bytes`.

## 6. Models and training data

### Ingredient sequence model

The multi-task Keras model has a 32-token `int32` input, a 64-dimensional embedding with padding mask, an LSTM with 96 units, mask-aware mean and max pooling, dense layers of 128 and 64 units with dropout, and two output heads:

- `formulation_profile`: four sigmoid values — hydration, brightening, exfoliation and barrier support.
- `product_category`: eight-class softmax — Serum, Moisturizer, Cleanser, Sunscreen, Toner, Mask, Exfoliant and Eye Cream.

Training defaults in `ml/config.py`: Adam learning rate 0.001, batch size 64, up to 18 epochs, dropout 0.3, seed 42; early stopping, learning-rate reduction and best-validation-loss checkpointing are configured. Profile regression uses MSE and category classification uses sparse categorical cross-entropy.

Dataset labels are **weakly supervised synthetic labels**, not dermatologist- or lab-annotated outcomes. `ml.dataset.generate_dataset` samples known ingredients and category templates, applies INCI position ordering, derives profile values from `profile_weights.json` (function weights, ingredient marker boosts, positional decay, category bias and Gaussian noise), and writes a CSV. Default size: 8,000 products, split 70/15/15. The checked-in metric snapshot reports 1,000 products per category and 8–32 ingredients per product.

Train/rebuild commands:

```powershell
python -m ml.dataset.generate_dataset --n 8000
python -m ml.train_ingredient_model
```

`train_ingredient_model` generates the dataset if needed; `--n N` regenerates with N products, `--regenerate` forces generation, and `--epochs`, `--batch-size`, `--learning-rate`, and `--seed` are supported. It writes `artifacts/models/deepcos_ingredient_model.keras`, `ingredient_vocabulary.json`, `ingredient_model_metrics.json`, and `artifacts/reports/ingredient_training_curves.png`.

### CNNs and OCR

`ml.image.cnn_alexnet` defines the shared AlexNet-style CNN. Text-region training classifies grayscale 64×64 patches as text/background and the runtime detector applies a sliding window. Label-category training classifies 128×128 RGB synthetic labels into the eight categories. Training data is rendered synthetically by `ml.image.synthetic_labels`.

```powershell
python -m ml.train_label_cnn --task text_region
python -m ml.train_label_cnn --task label_category
python -m ml.train_label_cnn --task both
```

Defaults are 3,000 text patches / 12 epochs and 2,400 label images / 10 epochs. Each task uses a 70/15/15 split, early stopping and reduce-on-plateau, then writes a `.keras` model and JSON metrics. Text recognition uses the installed Tesseract executable, English language data and `--oem 3 --psm 6`; image preprocessing/OCR variants are implemented under `ml/image/`.

### Snapshot of checked-in metrics

The metrics files in `artifacts/models/` were trained on **2026-09-29**. They describe synthetic/weakly supervised task data and must not be read as real-world product performance guarantees.

| Artifact/task | Held-out result in checked-in metrics |
|---|---|
| Ingredient category head | Accuracy 0.9183; macro F1 0.9183 |
| Hydration profile | MAE 0.0592; R² 0.3151; band accuracy 0.6967 |
| Brightening profile | MAE 0.1312; R² 0.3976; band accuracy 0.4567 |
| Exfoliation profile | MAE 0.1375; R² 0.5594; band accuracy 0.4742 |
| Barrier support profile | MAE 0.0785; R² 0.6764; band accuracy 0.6683 |
| Text-region CNN | Accuracy 1.0; macro F1 1.0 on synthetic test patches |
| Label-category CNN | Accuracy 1.0; macro F1 1.0 on synthetic test labels |

Model `.keras` binaries are ignored by `.gitignore` (`artifacts/models/*.keras` and `*.h5`), so the checked-in vocabulary and metrics alone do not make inference available. A compatible trained model file is needed locally. Vocabulary/model must be regenerated together when retraining.

## 7. Reference data and rule logic

All reference files are JSON version **1.0.0** in this checkout:

| File | Contents |
|---|---|
| `data/knowledge_base/ingredients.json` | 226 ingredient entries, 25 function-taxonomy terms, aliases, notes, concerns and declared metadata |
| `data/knowledge_base/concern_rules.json` | 14 concern rules and four severity levels (`info`, `low`, `medium`, `high`) |
| `data/knowledge_base/profile_weights.json` | Synthetic weak-label generation parameters: base/noise/squash, per-function weights, decay, scaling and ingredient marker boosts |
| `data/knowledge_base/product_templates.json` | Eight category templates plus function-position grouping used to synthesize product lists |

Knowledge lookup handles normalized exact names, aliases, and fuzzy matches. Search ranks exact/prefix/substring matches and note matches, with a small preference for common ingredients. Concern severity is ordered by the JSON ladder; concern-engine findings aggregate the distinct triggering ingredients per concern and regulatory notes are returned separately. These are deterministic reference-data matches, not neural-network predictions.

The four displayed profile bands use score cutoffs **High ≥ 0.66**, **Moderate ≥ 0.40**, **Low ≥ 0.20**, otherwise **Minimal**. These bands are display labels; they do not establish efficacy or safety.

## 8. Backend API

FastAPI title/version: **DeepCos API 1.0.0**. Interactive OpenAPI UI: `/docs`; root `GET /` identifies the service and points to docs/health.

| Method and path | Purpose / notable parameters |
|---|---|
| `GET /api/health` | API, database, knowledge base, OCR/CNN and ingredient-model status; status is `ok` when the ingredient model loads, otherwise `degraded` |
| `GET /api/model/info` | Model availability, architecture summary, metrics and capabilities |
| `GET /api/model/metrics` | Existing ingredient/text-region/label-CNN metrics JSON; 404 when none exist |
| `GET /api/model/curves` | `artifacts/reports/ingredient_training_curves.png`; 404 when missing |
| `POST /api/analyze/text` | JSON `{text, explain=true, persist=true}`; text length 1–20,000 |
| `POST /api/analyze/image` | Multipart `file` required; optional `text`, `explain`, `persist`, `max_ocr_variants`; configured upload-size limit |
| `GET /api/analyses?limit=50&offset=0` | Newest-first history, limit 1–200, excludes heavy report sections |
| `GET /api/analyses/{analysis_id}` | Full persisted report or 404 |
| `DELETE /api/analyses/{analysis_id}` | Deletes report, exports, and stored preview assets; 404 if absent |
| `GET /api/analyses/{analysis_id}/export?format=markdown\|text\|json` | Download report in selected format; default Markdown |
| `GET /api/analyses/{analysis_id}/image/{name}` | Serve one stored PNG preview; accepted names are fixed internally |
| `POST /api/analyses/{analysis_id}/preview` | Return available preview images as inline base64 data URLs |
| `GET /api/knowledge/stats` | Counts and most-common function/concern tags |
| `GET /api/knowledge/search?q=...&limit=25` | Search ingredient names, aliases and notes; limit 1–100 |
| `GET /api/knowledge/ingredient?name=...` | Resolve a name and return match type/score, info, functions, concerns and severity ranks |
| `GET /api/knowledge/functions` | Function taxonomy with labels/descriptions |
| `GET /api/knowledge/concerns` | Severity ladder, rules and disclaimer |
| `GET /api/knowledge/samples` | Four built-in demo lists for hydration, brightening, exfoliating cleanser and retinal cream |

Analysis errors are JSON `{detail, hint?}`. Typical status codes are 422 for invalid/unreadable inputs, 413 for over-limit uploads, 503 when the required ingredient model is unavailable, 404 for missing reports/assets/metrics, and 400 for unsupported export formats.

## 9. Analysis report shape

The analysis endpoints return the report as a top-level JSON object (not under a `report` wrapper). Persisted report fields include:

- Identity and timing: `analysis_id`, `created_at`, `mode`, `processing_ms`, `title` may be derived by UI fallback.
- Input/preprocessing: raw text, character and ingredient counts, items beyond sequence window, cleaned text, deduplicated ingredients, optional OCR text.
- Product: predicted category/display name, top class confidence, full category probability map, explanatory note; brand is currently `null` and not read from the label.
- Model output: `profile` entries (key/label/icon/score/band/bar), `profile_scores`, `profile_summary`, per-target `profile_confidence`, sequence tokens and model metadata/reliability.
- Evidence: `key_ingredients`, full `ingredients` breakdown, `unresolved_ingredients`, `function_histogram`, `explainability` positive/negative top contributors.
- Reference screening: `concerns.findings`, severity, ingredient lists/counts, `regulatory_notes`, flags, summary and assessment caveat.
- Image-only additions: OCR object and `image_analysis` detector/quality/source metadata and `assets` URLs after persistence.
- `warnings` and `disclaimer`.

Occlusion attribution is the per-token difference between the baseline profile and prediction with that ingredient replaced by the padding token. Positive/negative deltas indicate a direction of change in the model output, not causal biological effects. If `explain=false`, attribution is omitted/disabled.

## 10. Persistence, exports and generated files

MongoDB is primary (`<MONGODB_DB>.analyses`). At startup it creates indexes, backfills missing legacy `analysis_id` values, and migrates JSON fallback reports to MongoDB with insert-only upserts when MongoDB is available. If connection fails, the app stores reports in `artifacts/reports/analyses_store.json`.

Image base64 fields are removed from the persisted document and written as PNGs under `artifacts/reports/analyses/{analysis_id}/`; the report contains API asset URLs. The immediate analysis response still carries base64 previews. Listing responses omit heavy fields (`image_analysis`, `explainability`, `sequence`, `ingredients`, `preprocessing`). The analysis service also writes `.md`, `.txt`, and `.json` report files to `artifacts/reports/`. Export endpoint renders the saved object. The timestamp repair utility is a one-off Mongo script hard-coded for local MongoDB/database `deepcos`.

## 11. Dependencies and architecture boundaries

Python packages are pinned in `requirements.txt`: TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.6, scikit-learn 1.9.0, pandas 2.3.3, OpenCV 5.0.0.93, Pillow 12.3.0, pytesseract 0.3.13, FastAPI 0.141.1, Uvicorn 0.52.4, python-multipart 0.0.32, Pydantic 2.13.5, python-dotenv 1.2.3, PyMongo 4.17.0, and matplotlib 3.11.1. Frontend uses React/React DOM 18.3, React Router 6.26, Vite 5.4 and the React Vite plugin 4.3.

The ML boundary is `ml.inference.analyze`; the backend service maps domain errors to HTTP and handles persistence; the frontend API client centralizes HTTP parsing and routes. `ml.config` is the shared source of truth for model dimensions, target names, category order, model paths and training defaults.

## 12. Known implementation notes

- The example environment file defines `VITE_API_BASE_URL`, while `frontend/src/api.js` reads `VITE_API_URL`. For a deployed frontend, use `VITE_API_URL` or adjust the code/config naming.
- `ModelPage.jsx` describes `python -m ml.train_text_region_detector`, but no such module exists in this repository. The actual text-region training command is `python -m ml.train_label_cnn --task text_region`.
- `ModelPage.jsx` notes `artifacts/models/curves.png`; the API and ingredient training script use `artifacts/reports/ingredient_training_curves.png`.
- The label-category CNN has a model path and metrics route/UI, but the image inference path currently predicts category from ingredients and does not load/use that CNN for report predictions.
- `requirements.txt` comments claim Python 3.11 tested; several dependencies are pinned to recent versions, so verify installation compatibility for the target OS/Python runtime.
- Image endpoint accepts PNG/JPEG in its API description, while the frontend file chooser also accepts WebP. Actual decoding depends on OpenCV support.
- Training dataset is synthetic/weakly labeled, and image CNN test sets are synthetically rendered. Their metrics should be interpreted only as measurements on those generated distributions.
- `scripts/fix_created_at.py` connects to local MongoDB directly and targets the `deepcos.analyses` collection; review its connection/database constants before running against a non-demo environment.

## 13. Scope and limitations

DeepCos does not read product brand, determine concentrations from ordering, validate finished-product formulation, or establish that an ingredient/product is safe or suitable for an individual. Ingredient order is retained, but the learned model truncates at 32 positions. OCR can be affected by photo quality and depends on native Tesseract. Knowledge-base coverage is finite, and fuzzy matching can make imperfect matches; reports expose match type and unresolved names to make that visible. Regulatory metadata is local reference content and may vary by jurisdiction, concentration, formulation and date.

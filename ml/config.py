"""
Central configuration for the DeepCos Deep-Learning pipeline.

Everything that the training scripts, the inference service and the FastAPI
backend need to agree on lives here: filesystem locations, the sequence length,
the embedding / LSTM / MLP hyper-parameters, the formulation-profile targets and
the product categories.

Import from this module instead of hard-coding numbers so that a retrained
model and the running API can never drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
KB_DIR: Path = DATA_DIR / "knowledge_base"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"

ARTIFACT_DIR: Path = PROJECT_ROOT / "artifacts"
MODEL_DIR: Path = ARTIFACT_DIR / "models"
REPORT_DIR: Path = ARTIFACT_DIR / "reports"
LOG_DIR: Path = ARTIFACT_DIR / "logs"

# Knowledge base (documented cosmetic ingredient facts - not learned)
INGREDIENT_KB_PATH: Path = KB_DIR / "ingredients.json"
CONCERN_RULES_PATH: Path = KB_DIR / "concern_rules.json"
PROFILE_WEIGHTS_PATH: Path = KB_DIR / "profile_weights.json"
CATEGORY_TEMPLATES_PATH: Path = KB_DIR / "product_templates.json"

# AI-learned overlay: entries promoted from *validated* AI enrichment output.
# Kept in a separate file so the verified reference data (ingredients.json /
# concern_rules.json) is never rewritten; every surfaced entry is labelled
# ``origin="ai-learned"`` + ``review_status="unreviewed"``.
LEARNED_INGREDIENTS_PATH: Path = KB_DIR / "learned_ingredients.json"

# Generated datasets
INGREDIENT_DATASET_PATH: Path = PROCESSED_DIR / "deepcos_products.csv"
IMAGE_DATASET_DIR: Path = SYNTHETIC_DIR / "label_images"

# Model artifacts
INGREDIENT_MODEL_PATH: Path = MODEL_DIR / "deepcos_ingredient_model.keras"
INGREDIENT_VOCAB_PATH: Path = MODEL_DIR / "ingredient_vocabulary.json"
INGREDIENT_METRICS_PATH: Path = MODEL_DIR / "ingredient_model_metrics.json"

TEXT_CNN_MODEL_PATH: Path = MODEL_DIR / "deepcos_textregion_cnn.keras"
TEXT_CNN_METRICS_PATH: Path = MODEL_DIR / "textregion_cnn_metrics.json"
CATEGORY_CNN_MODEL_PATH: Path = MODEL_DIR / "deepcos_label_cnn.keras"
CATEGORY_CNN_METRICS_PATH: Path = MODEL_DIR / "label_cnn_metrics.json"

# ---------------------------------------------------------------------------
# Formulation targets predicted by the LSTM + MLP network
# ---------------------------------------------------------------------------
PROFILE_TARGETS: tuple[str, ...] = (
    "hydration",
    "brightening",
    "exfoliation",
    "barrier_support",
)

PROFILE_LABELS: dict[str, str] = {
    "hydration": "Hydration",
    "brightening": "Brightening",
    "exfoliation": "Exfoliation",
    "barrier_support": "Barrier support",
}

PROFILE_ICONS: dict[str, str] = {
    "hydration": "\U0001F4A7",        # droplet
    "brightening": "\u2728",          # sparkles
    "exfoliation": "\U0001F9EA",      # test tube
    "barrier_support": "\U0001F6E1",  # shield
}

# Score -> human readable band
PROFILE_BANDS: tuple[tuple[float, str], ...] = (
    (0.66, "High"),
    (0.40, "Moderate"),
    (0.20, "Low"),
    (0.00, "Minimal"),
)

# ---------------------------------------------------------------------------
# Product categories  (must stay in sync with product_templates.json)
# ---------------------------------------------------------------------------
CATEGORIES: tuple[str, ...] = (
    "Serum",
    "Moisturizer",
    "Cleanser",
    "Sunscreen",
    "Toner",
    "Mask",
    "Exfoliant",
    "Eye Cream",
)

NUM_CATEGORIES: int = len(CATEGORIES)

# ---------------------------------------------------------------------------
# Text (ingredient sequence) model hyper-parameters
# ---------------------------------------------------------------------------
SEQ_LEN: int = 32            # padded ingredient-sequence length
EMBEDDING_DIM: int = 64      # ingredient embedding size
LSTM_UNITS: int = 96         # LSTM hidden units (learned ingredient representation)
MLP_HIDDEN: tuple[int, int] = (128, 64)
DROPOUT: float = 0.3
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 64
EPOCHS: int = 18
PROFILE_LOSS_WEIGHT: float = 1.0
CATEGORY_LOSS_WEIGHT: float = 1.0

# Vocabulary special tokens
PAD_TOKEN: str = "<pad>"
UNK_TOKEN: str = "<unk>"
PAD_INDEX: int = 0
UNK_INDEX: int = 1

# ---------------------------------------------------------------------------
# CNN (AlexNet style) hyper-parameters
# ---------------------------------------------------------------------------
TEXT_PATCH_SIZE: int = 64        # grayscale patch for text-region detection
TEXT_CNN_EPOCHS: int = 12
LABEL_IMAGE_SIZE: int = 128      # RGB label image for category CNN
LABEL_CNN_EPOCHS: int = 10
CNN_BATCH_SIZE: int = 32
CNN_LEARNING_RATE: float = 5e-4

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
TESSERACT_CMD: str = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
OCR_LANGS: str = "eng"
OCR_CONFIG: str = "--oem 3 --psm 6"   # assume a single uniform block of text

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
DATASET_SIZE: int = 8000          # synthetic products (weak supervision)
TEXT_PATCH_SAMPLES: int = 3000
LABEL_IMAGE_SAMPLES: int = 2400


def ensure_directories() -> None:
    """Create every output directory used by the pipeline."""
    for path in (
        DATA_DIR,
        KB_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        SYNTHETIC_DIR,
        ARTIFACT_DIR,
        MODEL_DIR,
        REPORT_DIR,
        LOG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def band_for_score(score: float) -> str:
    """Convert a 0-1 formulation score into a 'Minimal/Low/Moderate/High' band."""
    for threshold, label in PROFILE_BANDS:
        if score >= threshold:
            return label
    return PROFILE_BANDS[-1][1]

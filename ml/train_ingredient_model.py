"""
Train the DeepCos ingredient-sequence model (Embedding -> LSTM -> MLP).

Pipeline
--------
    1. load (or generate) the weakly labelled product dataset
    2. build the ingredient vocabulary from the knowledge base
    3. encode ingredient sequences into padded token ids
    4. train the multi-task network (4 formulation scores + category)
    5. evaluate on the held-out test split (MAE / R2 / accuracy / confusion)
    6. save model, vocabulary, metrics and training-curve plot

Usage
-----
    python -m ml.train_ingredient_model                        # full run
    python -m ml.train_ingredient_model --n 1500 --epochs 5     # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from ml import config
from ml.dataset.generate_dataset import (
    ReferenceData,
    build_arrays,
    dataset_summary,
    generate_dataset,
    read_csv,
    split_dataset,
    write_csv,
)
from ml.dataset.vocabulary import Vocabulary
from ml.models.deepcos_model import build_deepcos_model, model_architecture_summary

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
tf.keras.utils.set_random_seed(config.RANDOM_SEED)


def build_vocabulary(reference: ReferenceData | None = None) -> Vocabulary:
    """Vocabulary over every knowledge-base ingredient, plus alias shortcuts."""
    reference = reference or ReferenceData.load()
    aliases: dict[str, str] = {}
    for info in reference.kb.ingredients.values():
        for alias in info.aliases:
            aliases[alias] = info.inci
    return Vocabulary.build(reference.kb.all_inci(), extra_aliases=aliases)


def load_dataset(
    n: int | None = None,
    regenerate: bool = False,
    seed: int = config.RANDOM_SEED,
) -> list:
    """Load the CSV dataset, generating it on demand."""
    path = config.INGREDIENT_DATASET_PATH
    if regenerate or n is not None or not path.exists():
        count = int(n or config.DATASET_SIZE)
        print(f"[dataset] generating {count} weakly labelled products ...")
        products = generate_dataset(count, seed=seed)
        write_csv(products, path)
        print(f"[dataset] written -> {path}")
        return products
    products = read_csv(path)
    print(f"[dataset] loaded {len(products)} products from {path}")
    return products


def metrics_for_targets(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, dict[str, float]]:
    """MAE / RMSE / R2 / band accuracy for every formulation target."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    out: dict[str, dict[str, float]] = {}
    for i, target in enumerate(config.PROFILE_TARGETS):
        true = y_true[:, i]
        pred = y_pred[:, i]
        band_accuracy = float(
            np.mean(
                [
                    config.band_for_score(float(a)) == config.band_for_score(float(b))
                    for a, b in zip(true, pred)
                ]
            )
        )
        out[target] = {
            "mae": round(float(mean_absolute_error(true, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(true, pred))), 4),
            "r2": round(float(r2_score(true, pred)), 4),
            "band_accuracy": round(band_accuracy, 4),
        }
    return out


def category_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro-F1 and confusion matrix for the category head."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    labels = list(range(config.NUM_CATEGORIES))
    predicted = y_pred.argmax(axis=1)
    report = classification_report(
        y_true,
        predicted,
        labels=labels,
        target_names=list(config.CATEGORIES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
        "macro_f1": round(float(f1_score(y_true, predicted, average="macro")), 4),
        "per_category": {
            name: {
                "precision": round(float(report[name]["precision"]), 4),
                "recall": round(float(report[name]["recall"]), 4),
                "f1": round(float(report[name]["f1-score"]), 4),
                "support": int(report[name]["support"]),
            }
            for name in config.CATEGORIES
        },
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=labels).tolist(),
    }


def save_training_curves(history: dict, path: Path) -> Path | None:
    """Render the training curves to a PNG for the dashboard."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        axes[0].plot(history.get("loss", []), label="train total loss", color="#6366f1")
        axes[0].plot(history.get("val_loss", []), label="val total loss", color="#f97316")
        axes[0].set_title("DeepCos ingredient model - total loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        for key in sorted(k for k in history if "formulation_profile" in k and "loss" in k):
            color = "#0ea5e9" if key.startswith("val") else "#22c55e"
            axes[1].plot(history[key], label=key, color=color)
        axes[1].set_title("Profile head loss (MSE)")
        axes[1].set_xlabel("epoch")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.25)

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception as exc:  # pragma: no cover - plotting is best effort
        print(f"[warn] could not render training curves: {exc}")
        return None


def sample_predictions(products: Sequence, model, vocab: Vocabulary, limit: int = 5) -> list[dict]:
    """A few qualitative examples for the metrics report."""
    subset = list(products[:limit])
    x, y_profile, y_category = build_arrays(subset, vocab)
    raw = model.predict(x, verbose=0)
    profiles = np.asarray(raw["formulation_profile"])
    categories = np.asarray(raw["product_category"])
    out: list[dict] = []
    for row, product in enumerate(subset):
        out.append(
            {
                "product_id": product.product_id,
                "category_true": product.category,
                "category_predicted": config.CATEGORIES[int(categories[row].argmax())],
                "ingredients": product.ingredients[:10],
                "profile_true": {
                    t: round(float(y_profile[row][i]), 3)
                    for i, t in enumerate(config.PROFILE_TARGETS)
                },
                "profile_predicted": {
                    t: round(float(profiles[row][i]), 3)
                    for i, t in enumerate(config.PROFILE_TARGETS)
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    n: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    regenerate: bool = False,
    learning_rate: float | None = None,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Run the full training + evaluation + artifact-saving pipeline."""
    config.ensure_directories()
    tf.keras.utils.set_random_seed(seed)

    reference = ReferenceData.load()
    print(
        f"[kb] {len(reference.kb)} ingredients, "
        f"{len(reference.kb.function_taxonomy)} function tags"
    )

    products = load_dataset(n=n, regenerate=regenerate, seed=seed)
    splits = split_dataset(products, seed=seed)
    print("[dataset] split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    vocab = build_vocabulary(reference)
    vocab_path = vocab.save()
    print(f"[vocab] {len(vocab)} tokens -> {vocab_path}")

    arrays = {name: build_arrays(subset, vocab) for name, subset in splits.items()}
    (x_train, y_train_profile, y_train_category) = arrays["train"]
    (x_val, y_val_profile, y_val_category) = arrays["val"]
    (x_test, y_test_profile, y_test_category) = arrays["test"]

    y_train = {
        "formulation_profile": y_train_profile,
        "product_category": y_train_category,
    }
    y_val = {"formulation_profile": y_val_profile, "product_category": y_val_category}

    model = build_deepcos_model(
        vocab_size=len(vocab),
        seq_len=config.SEQ_LEN,
        num_categories=config.NUM_CATEGORIES,
        learning_rate=learning_rate or config.LEARNING_RATE,
    )
    model.summary(expand_nested=False)

    epochs = int(epochs or config.EPOCHS)
    batch_size = int(batch_size or config.BATCH_SIZE)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            min_delta=1e-5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.INGREDIENT_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.perf_counter() - started

    evaluation = model.evaluate(
        x_test,
        {"formulation_profile": y_test_profile, "product_category": y_test_category},
        verbose=0,
    )
    raw = model.predict(x_test, batch_size=256, verbose=0)
    profile_metrics = metrics_for_targets(
        y_test_profile, np.asarray(raw["formulation_profile"])
    )
    category_metrics = category_report(y_test_category, np.asarray(raw["product_category"]))

    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_seconds": round(train_seconds, 2),
        "dataset": {
            "total_products": len(products),
            "splits": {name: len(subset) for name, subset in splits.items()},
            "dataset_summary": dataset_summary(products),
            "weak_supervision": (
                "Labels derived from documented ingredient functions "
                "(profile_weights.json) plus category bias and Gaussian noise."
            ),
        },
        "vocabulary_size": len(vocab),
        "architecture": model_architecture_summary(model),
        "profile_metrics": profile_metrics,
        "category_metrics": category_metrics,
        "test_loss": [round(float(v), 5) for v in np.atleast_1d(evaluation)],
        "history": {
            key: [round(float(v), 5) for v in values]
            for key, values in history.history.items()
        },
        "sample_predictions": sample_predictions(splits["test"], model, vocab, limit=5),
    }

    metrics_path = config.INGREDIENT_METRICS_PATH
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.save(config.INGREDIENT_MODEL_PATH)
    curves = save_training_curves(
        history.history, config.REPORT_DIR / "ingredient_training_curves.png"
    )

    print("\n=== DeepCos ingredient model ===")
    print(f"training time      : {train_seconds:.1f} s")
    print(f"vocabulary         : {len(vocab)} tokens")
    print(f"model parameters   : {model.count_params():,}")
    print(
        f"category accuracy  : {category_metrics['accuracy']:.3f} "
        f"(macro-F1 {category_metrics['macro_f1']:.3f})"
    )
    for target, values in profile_metrics.items():
        print(
            f"{target:<16}: MAE {values['mae']:.4f}  R2 {values['r2']:+.3f}  "
            f"band-accuracy {values['band_accuracy']:.3f}"
        )
    print(f"model saved        : {config.INGREDIENT_MODEL_PATH}")
    print(f"metrics saved      : {metrics_path}")
    if curves:
        print(f"curves saved       : {curves}")
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the DeepCos ingredient-sequence model (Embedding -> LSTM -> MLP)."
    )
    parser.add_argument("--n", type=int, default=None, help="regenerate the dataset with N products")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--regenerate", action="store_true", help="force dataset regeneration")
    args = parser.parse_args(argv)

    train(
        n=args.n,
        epochs=args.epochs,
        batch_size=args.batch_size,
        regenerate=args.regenerate,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())




"""
Train the AlexNet-style CNN used by the DeepCos image path  (Module 1).

Two tasks
---------
``text_region``     binary text / background patch classifier on 64x64 grayscale
                    patches. Used at inference time to localise the ingredient
                    block before OCR.
``label_category``  multi-class product-category classifier on 128x128 label
                    images. Demonstrates the CNN as a visual feature extractor
                    for product identification.

Usage
-----
    python -m ml.train_label_cnn --task text_region
    python -m ml.train_label_cnn --task label_category
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Sequence

import numpy as np
import tensorflow as tf

from ml import config
from ml.dataset.generate_dataset import ReferenceData
from ml.image.cnn_alexnet import alexnet_architecture_summary, build_alexnet
from ml.image.synthetic_labels import (
    build_label_category_dataset,
    build_text_region_dataset,
    save_preview_images,
)

tf.keras.utils.set_random_seed(config.RANDOM_SEED)


def _split(x: np.ndarray, y: np.ndarray, seed: int = config.RANDOM_SEED):
    """70 / 15 / 15 train-validation-test split."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    n_train = int(len(x) * 0.7)
    n_val = int(len(x) * 0.15)
    return (
        x[order[:n_train]], y[order[:n_train]],
        x[order[n_train : n_train + n_val]], y[order[n_train : n_train + n_val]],
        x[order[n_train + n_val :]], y[order[n_train + n_val :]],
    )


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "precision": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "per_class": {
            name: {
                "precision": round(float(report[name]["precision"]), 4),
                "recall": round(float(report[name]["recall"]), 4),
                "f1": round(float(report[name]["f1-score"]), 4),
                "support": int(report[name]["support"]),
            }
            for name in labels
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(labels)))
        ).tolist(),
    }


def train_text_region(n: int = 3000, epochs: int = 12, batch_size: int = 32) -> dict:
    """Train the text / background patch classifier used before OCR."""
    print(f"[text_region] building {n} labelled patches ...")
    x, y = build_text_region_dataset(
        n, seed=config.RANDOM_SEED, patch_size=config.TEXT_PATCH_SIZE
    )
    x_train, y_train, x_val, y_val, x_test, y_test = _split(x, y)

    model = build_alexnet(
        input_shape=(config.TEXT_PATCH_SIZE, config.TEXT_PATCH_SIZE, 1),
        num_classes=2,
        base_filters=32,
        model_name="deepcos_textregion_cnn",
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
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
    seconds = time.perf_counter() - started

    probabilities = model.predict(x_test, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype("int32")
    metrics = _classification_metrics(
        y_test.astype("int32"), predictions, ["background", "text"]
    )

    model.save(config.TEXT_CNN_MODEL_PATH)
    payload = {
        "task": "text_region",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_seconds": round(seconds, 2),
        "dataset": {
            "samples": int(len(x)),
            "patch_size": config.TEXT_PATCH_SIZE,
            "splits": {
                "train": int(len(x_train)),
                "val": int(len(x_val)),
                "test": int(len(x_test)),
            },
        },
        "architecture": alexnet_architecture_summary(model, "text_region"),
        "metrics": metrics,
        "history": {
            key: [round(float(v), 5) for v in values]
            for key, values in history.history.items()
        },
        "notes": [
            "Trained on synthetically rendered label text vs. background patches.",
            "Applied as a sliding window to crop the ingredient block before OCR.",
        ],
    }
    config.TEXT_CNN_METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[text_region] accuracy {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}  "
        f"({seconds:.1f}s) -> {config.TEXT_CNN_MODEL_PATH}"
    )
    return payload


def train_label_category(n: int = 2400, epochs: int = 10, batch_size: int = 32) -> dict:
    """Train the product-category label-image classifier."""
    reference = ReferenceData.load()
    ingredient_names = reference.kb.all_inci()
    print(f"[label_category] rendering {n} synthetic label images ...")
    x, y = build_label_category_dataset(
        n, ingredient_names, seed=config.RANDOM_SEED, size=config.LABEL_IMAGE_SIZE
    )
    x_train, y_train, x_val, y_val, x_test, y_test = _split(x, y)

    model = build_alexnet(
        input_shape=(config.LABEL_IMAGE_SIZE, config.LABEL_IMAGE_SIZE, 3),
        num_classes=config.NUM_CATEGORIES,
        base_filters=32,
        model_name="deepcos_label_cnn",
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
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
    seconds = time.perf_counter() - started

    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    metrics = _classification_metrics(y_test, predictions, list(config.CATEGORIES))

    model.save(config.CATEGORY_CNN_MODEL_PATH)
    payload = {
        "task": "label_category",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_seconds": round(seconds, 2),
        "dataset": {
            "samples": int(len(x)),
            "image_size": config.LABEL_IMAGE_SIZE,
            "splits": {
                "train": int(len(x_train)),
                "val": int(len(x_val)),
                "test": int(len(x_test)),
            },
        },
        "architecture": alexnet_architecture_summary(model, "label_category"),
        "metrics": metrics,
        "history": {
            key: [round(float(v), 5) for v in values]
            for key, values in history.history.items()
        },
        "sample_predictions": [
            {
                "true": config.CATEGORIES[int(y_test[i])],
                "predicted": config.CATEGORIES[int(predictions[i])],
                "confidence": round(float(probabilities[i].max()), 4),
            }
            for i in range(min(8, len(y_test)))
        ],
        "notes": [
            "Trained on synthetically rendered labels with category-specific shapes, colours and copy.",
            "Provides a visual category prior that complements the text-based prediction.",
        ],
    }
    config.CATEGORY_CNN_METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[label_category] accuracy {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}  "
        f"({seconds:.1f}s) -> {config.CATEGORY_CNN_MODEL_PATH}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the DeepCos AlexNet-style CNN.")
    parser.add_argument(
        "--task", choices=["text_region", "label_category", "both"], default="text_region"
    )
    parser.add_argument("--n", type=int, default=None, help="number of training images")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)

    config.ensure_directories()
    if args.task in ("text_region", "both"):
        train_text_region(
            n=args.n or config.TEXT_PATCH_SAMPLES,
            epochs=args.epochs or config.TEXT_CNN_EPOCHS,
            batch_size=config.CNN_BATCH_SIZE,
        )
    if args.task in ("label_category", "both"):
        train_label_category(
            n=args.n or config.LABEL_IMAGE_SAMPLES,
            epochs=args.epochs or config.LABEL_CNN_EPOCHS,
            batch_size=config.CNN_BATCH_SIZE,
        )
    save_preview_images(config.REPORT_DIR / "label_previews")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



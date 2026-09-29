"""
AlexNet-style CNN used for the visual part of DeepCos  (Module 1).

Why a CNN here
--------------
A cosmetic label photo contains a lot of structure that is *not* the ingredient
list (logos, marketing copy, barcodes, textures, shadows). Before OCR it helps
enormously to localise the printed-text region, and CNNs are the standard tool
for that because they learn spatial features - edges, strokes, text-like
textures - directly from pixels.

Two tasks share this backbone
-----------------------------
1. ``text_region``   : binary text / non-text patch classification on 64x64
                       grayscale patches -> gives the bounding box that is fed
                       to OCR (default task, used by the API).
2. ``label_category``: multi-class product-category classification of a whole
                       label image (128x128 RGB) - the CNN as a visual feature
                       extractor for product identification.

The layer stack mirrors the classic AlexNet design (Krizhevsky et al., 2012):
five convolutional layers in a 1-2-2 configuration with max-pooling, followed by
fully-connected layers and a softmax output. Kernel and pooling sizes are scaled
down to suit small grayscale patches instead of 227x227 RGB photos.
"""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
from keras import layers

from ml import config

TEXT_REGION_CLASSES: tuple[str, ...] = ("background", "text")


def build_alexnet(
    input_shape: tuple[int, int, int] = (config.TEXT_PATCH_SIZE, config.TEXT_PATCH_SIZE, 1),
    num_classes: int = 2,
    base_filters: int = 32,
    dropout: float = 0.4,
    learning_rate: float | None = None,
    compile_model: bool = True,
    model_name: str = "deepcos_alexnet",
) -> keras.Model:
    """
    AlexNet-style convolutional network.

    ``base_filters`` scales the width of the network (32 keeps the model small
    enough to train on CPU in a couple of minutes); the filter progression
    (1, 2, 2, 4, 4) follows the original architecture.
    """
    inputs = keras.Input(shape=input_shape, name="label_image")
    f = int(base_filters)

    # Block 1: large kernel + strided convolution (AlexNet's hallmarks)
    x = layers.Conv2D(
        f, (5, 5), strides=(2, 2), padding="same", activation="relu", name="conv1"
    )(inputs)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(f * 2, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(f * 2, (3, 3), padding="same", activation="relu", name="conv3")(x)

    # Block 4 (in AlexNet the 4th conv keeps the resolution of the 3rd)
    x = layers.Conv2D(f * 4, (3, 3), padding="same", activation="relu", name="conv4")(x)

    # Block 5
    x = layers.Conv2D(f * 4, (3, 3), padding="same", activation="relu", name="conv5")(x)
    x = layers.MaxPooling2D((2, 2), name="pool5")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="fc6")(x)
    x = layers.Dropout(dropout, name="dropout6")(x)
    x = layers.Dense(128, activation="relu", name="fc7")(x)
    x = layers.Dropout(dropout, name="dropout7")(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
        loss, metric = "binary_crossentropy", "accuracy"
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
        loss, metric = "sparse_categorical_crossentropy", "accuracy"

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate or config.CNN_LEARNING_RATE
            ),
            loss=loss,
            metrics=[metric],
        )
    return model


def alexnet_architecture_summary(model: keras.Model, task: str) -> dict:
    """Architecture description for the dashboard (mirrors the ingredient model)."""
    return {
        "model_name": model.name,
        "task": task,
        "input_shape": list(model.input_shape[1:]),
        "stages": [
            {
                "stage": "conv1 + pool1",
                "description": "5x5 strided convolution - learns edges and stroke fragments.",
            },
            {
                "stage": "conv2 + pool2",
                "description": "Builds text-like texture and character-part features.",
            },
            {
                "stage": "conv3 / conv4 / conv5",
                "description": "3x3 stack - higher-level text and layout patterns.",
            },
            {
                "stage": "fc6 / fc7",
                "description": "Fully-connected visual feature extractor.",
            },
            {
                "stage": "predictions",
                "description": "Task head (text/background or product category).",
            },
        ],
        "layers": [
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(getattr(layer, "output", None).shape)
                if getattr(layer, "output", None) is not None
                else "n/a",
                "params": int(layer.count_params()),
            }
            for layer in model.layers
        ],
        "total_params": int(model.count_params()),
    }


def load_text_region_cnn(path: Path | str | None = None) -> keras.Model:
    """Load the trained text-region (text vs background) CNN."""
    path = Path(path or config.TEXT_CNN_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "Text-region CNN not found at "
            f"{path}. Run: python -m ml.train_label_cnn --task text_region"
        )
    return keras.models.load_model(path)


def load_label_category_cnn(path: Path | str | None = None) -> keras.Model:
    """Load the trained label-category CNN."""
    path = Path(path or config.CATEGORY_CNN_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "Label-category CNN not found at "
            f"{path}. Run: python -m ml.train_label_cnn --task label_category"
        )
    return keras.models.load_model(path)


def predict_text_probabilities(model: keras.Model, batch: np.ndarray) -> np.ndarray:
    """Sigmoid text probability for a batch of normalised patches."""
    raw = model.predict(np.asarray(batch, dtype="float32"), verbose=0)
    return np.asarray(raw).reshape(-1)


def predict_category_probabilities(model: keras.Model, batch: np.ndarray) -> np.ndarray:
    """Softmax class probabilities for a batch of label images."""
    raw = model.predict(np.asarray(batch, dtype="float32"), verbose=0)
    return np.asarray(raw)


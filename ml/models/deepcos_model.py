"""
DeepCos ingredient-sequence network  (Modules 3 and 4 of the design).

Architecture
------------
    ingredient tokens (int32, length 32)
        -> Embedding                 (learned ingredient representation)
        -> LSTM                      (sequence / order modelling)
        -> masked mean + max pooling (fixed-size learned representation)
        -> Dense + Dropout           (MLP)
        -> Dense 64                  ("formulation_features")
        -> two heads
               formulation_profile  : 4 sigmoid outputs
                                      (hydration, brightening, exfoliation,
                                       barrier support)
               product_category     : softmax over the product categories

This single multi-task network is what the report calls "LSTM + MLP": the LSTM
learns an ingredient-sequence representation and the MLP maps it to formulation
characteristics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
from keras import layers, ops

from ml import config


@keras.saving.register_keras_serializable(package="deepcos")
class MaskedSequencePooling(layers.Layer):
    """
    Pool an LSTM output sequence into a fixed-size vector.

    Padding tokens carry no information, so both the mean and the maximum are
    computed over the real (unmasked) timesteps only. Mask-aware pooling matters
    here because product labels differ in length (10-32 ingredients).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):  # noqa: D102 - Keras layer API
        x = ops.cast(inputs, "float32")
        if mask is None:
            mask = ops.ones(ops.shape(x)[:2], dtype="bool")
        mask_f = ops.expand_dims(ops.cast(mask, "float32"), axis=-1)
        denom = ops.maximum(ops.sum(mask_f, axis=1), 1.0)
        mean = ops.sum(x * mask_f, axis=1) / denom
        neg_inf = ops.cast(-1e9, dtype="float32")
        masked = ops.where(ops.cast(mask_f, "bool"), x, neg_inf)
        maximum = ops.max(masked, axis=1)
        return ops.concatenate([mean, maximum], axis=-1)

    def compute_mask(self, inputs, mask=None):  # noqa: D102
        return None

    def compute_output_shape(self, input_shape):  # noqa: D102
        return (input_shape[0], int(input_shape[2]) * 2)

    def get_config(self) -> dict:  # noqa: D102
        return super().get_config()


def build_deepcos_model(
    vocab_size: int,
    seq_len: int | None = None,
    num_categories: int | None = None,
    embedding_dim: int | None = None,
    lstm_units: int | None = None,
    mlp_hidden: tuple[int, int] | None = None,
    dropout: float | None = None,
    learning_rate: float | None = None,
    compile_model: bool = True,
) -> keras.Model:
    """Construct the DeepCos multi-task ingredient model."""
    seq_len = int(seq_len or config.SEQ_LEN)
    num_categories = int(num_categories or config.NUM_CATEGORIES)
    embedding_dim = int(embedding_dim or config.EMBEDDING_DIM)
    lstm_units = int(lstm_units or config.LSTM_UNITS)
    mlp_hidden = tuple(mlp_hidden or config.MLP_HIDDEN)
    dropout = float(config.DROPOUT if dropout is None else dropout)
    learning_rate = float(learning_rate or config.LEARNING_RATE)

    sequence_input = keras.Input(
        shape=(seq_len,), dtype="int32", name="ingredient_sequence"
    )

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="ingredient_embedding",
    )(sequence_input)

    x = layers.LSTM(
        lstm_units,
        return_sequences=True,
        dropout=0.2,
        name="lstm_encoder",
    )(x)

    x = MaskedSequencePooling(name="masked_sequence_pooling")(x)

    x = layers.Dense(mlp_hidden[0], activation="relu", name="mlp_dense_1")(x)
    x = layers.Dropout(dropout, name="mlp_dropout_1")(x)

    features = layers.Dense(
        mlp_hidden[1], activation="relu", name="formulation_features"
    )(x)
    x = layers.Dropout(dropout, name="mlp_dropout_2")(features)

    profile_output = layers.Dense(
        len(config.PROFILE_TARGETS), activation="sigmoid", name="formulation_profile"
    )(x)
    category_output = layers.Dense(
        num_categories, activation="softmax", name="product_category"
    )(x)

    model = keras.Model(
        inputs=sequence_input,
        outputs={
            "formulation_profile": profile_output,
            "product_category": category_output,
        },
        name="deepcos_ingredient_model",
    )

    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                "formulation_profile": "mse",
                "product_category": "sparse_categorical_crossentropy",
            },
            loss_weights={
                "formulation_profile": config.PROFILE_LOSS_WEIGHT,
                "product_category": config.CATEGORY_LOSS_WEIGHT,
            },
            metrics={
                "formulation_profile": [
                    keras.metrics.MeanAbsoluteError(name="mae"),
                ],
                "product_category": [
                    keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                ],
            },
        )
    return model


def build_feature_extractor(model: keras.Model) -> keras.Model:
    """
    Re-use the trained network as a feature extractor.

    Returns a model that maps an ingredient token sequence to the 64-dimensional
    "formulation_features" vector learned by the LSTM + MLP stack.
    """
    return keras.Model(
        inputs=model.input,
        outputs=model.get_layer("formulation_features").output,
        name="deepcos_feature_extractor",
    )


def load_ingredient_model(path: Path | str | None = None) -> keras.Model:
    """Load the trained DeepCos ingredient model from disk."""
    path = Path(path or config.INGREDIENT_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run: python -m ml.train_ingredient_model"
        )
    return keras.models.load_model(path)


def model_architecture_summary(model: keras.Model) -> dict:
    """
    Structured architecture description (used by the /api/model/info endpoint).

    Returns the module-level stages described in the DeepCos design plus the
    per-layer table so the dashboard can render the network.
    """
    layers_table: list[dict] = []
    for layer in model.layers:
        try:
            output_shape = str(layer.output.shape)
        except Exception:  # pragma: no cover - shape inference edge cases
            output_shape = "n/a"
        layers_table.append(
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": output_shape,
                "params": int(layer.count_params()),
                "trainable": bool(layer.trainable),
            }
        )

    stages = [
        {
            "stage": "Embedding",
            "layer": "ingredient_embedding",
            "description": "Each INCI ingredient name is mapped to a learned dense vector.",
        },
        {
            "stage": "LSTM",
            "layer": "lstm_encoder",
            "description": "Reads the ingredient list as a sequence and models order and combinations.",
        },
        {
            "stage": "Masked pooling",
            "layer": "masked_sequence_pooling",
            "description": "Mean + max over real ingredients only, ignoring padding.",
        },
        {
            "stage": "MLP",
            "layer": "mlp_dense_1 -> formulation_features",
            "description": "Maps the sequence representation to learned formulation features.",
        },
        {
            "stage": "Prediction heads",
            "layer": "formulation_profile, product_category",
            "description": "Multi-task output: 4 formulation scores + product category.",
        },
    ]

    return {
        "model_name": model.name,
        "input": {
            "name": "ingredient_sequence",
            "seq_len": config.SEQ_LEN,
            "dtype": "int32",
            "note": "Padded ingredient token ids (0 = padding, 1 = unknown).",
        },
        "outputs": [
            {
                "name": "formulation_profile",
                "shape": len(config.PROFILE_TARGETS),
                "activation": "sigmoid",
                "targets": list(config.PROFILE_TARGETS),
            },
            {
                "name": "product_category",
                "shape": config.NUM_CATEGORIES,
                "activation": "softmax",
                "targets": list(config.CATEGORIES),
            },
        ],
        "hyperparameters": {
            "vocab_size": int(model.get_layer("ingredient_embedding").input_dim),
            "embedding_dim": int(model.get_layer("ingredient_embedding").output_dim),
            "lstm_units": int(model.get_layer("lstm_encoder").units),
            "mlp_hidden": list(config.MLP_HIDDEN),
            "dropout": config.DROPOUT,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "learning_rate": config.LEARNING_RATE,
        },
        "total_params": int(model.count_params()),
        "trainable_params": int(
            sum(int(layer.count_params()) for layer in model.layers if layer.trainable)
        ),
        "stages": stages,
        "layers": layers_table,
    }


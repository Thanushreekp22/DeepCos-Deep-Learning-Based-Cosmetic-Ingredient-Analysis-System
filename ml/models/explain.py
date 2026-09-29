"""
Explainability for the DeepCos ingredient model.

DeepCos answers "why" by measuring how much each ingredient actually changes the
network's output. The method is **occlusion attribution**:

    1. run the model on the full ingredient sequence -> baseline prediction
    2. for each position, re-run the model with that single ingredient removed
       (its token is replaced by the padding token, and because the network uses
       mask-aware pooling the ingredient really disappears from the sequence)
    3. the change in each formulation score is that ingredient's contribution

This gives a genuine model-derived explanation rather than a hard-coded lookup,
and the sign of the contribution shows whether the ingredient pushes a
characteristic up or down.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ml import config


def _profile_output(model_output) -> np.ndarray:
    """Extract the formulation_profile head from a possibly dict-valued prediction."""
    if isinstance(model_output, dict):
        return np.asarray(model_output["formulation_profile"], dtype="float32")
    if isinstance(model_output, (list, tuple)):
        return np.asarray(model_output[0], dtype="float32")
    return np.asarray(model_output, dtype="float32")


def occlusion_contributions(
    model,
    sequence: np.ndarray,
    ingredient_names: Sequence[str],
    batch_size: int | None = None,
) -> dict:
    """
    Compute per-ingredient contributions to every formulation target.

    Parameters
    ----------
    model : keras.Model                trained DeepCos ingredient model
    sequence : np.ndarray              (1, seq_len) or (seq_len,) encoded tokens
    ingredient_names : Sequence[str]   canonical names aligned with the sequence

    Returns
    -------
    dict with ``baseline`` scores, a per-ingredient ``contributions`` matrix and
    ``top_contributors`` per target (both positive and negative directions).
    """
    x = np.asarray(sequence, dtype="int32").reshape(1, -1)
    seq_len = x.shape[1]

    baseline = _profile_output(model.predict(x, verbose=0))[0]

    positions = [i for i, token in enumerate(x[0]) if int(token) != config.PAD_INDEX]
    if not positions:
        return {
            "method": "occlusion",
            "baseline": {t: float(v) for t, v in zip(config.PROFILE_TARGETS, baseline)},
            "contributions": [],
            "top_contributors": {},
        }

    perturbed = np.repeat(x, len(positions), axis=0).astype("int32")
    for row, position in enumerate(positions):
        perturbed[row, position] = config.PAD_INDEX

    perturbed_output = _profile_output(
        model.predict(perturbed, batch_size=batch_size or 64, verbose=0)
    )

    contributions: list[dict] = []
    for row, position in enumerate(positions):
        delta = baseline - perturbed_output[row]
        name = ingredient_names[row] if row < len(ingredient_names) else f"#{row}"
        contributions.append(
            {
                "position": int(position),
                "ingredient": name,
                "contributions": {
                    target: round(float(delta[i]), 4)
                    for i, target in enumerate(config.PROFILE_TARGETS)
                },
            }
        )

    top_contributors: dict[str, dict] = {}
    for i, target in enumerate(config.PROFILE_TARGETS):
        ranked = sorted(
            contributions, key=lambda item: item["contributions"][target], reverse=True
        )
        top_contributors[target] = {
            "positive": [
                {"ingredient": item["ingredient"], "delta": item["contributions"][target]}
                for item in ranked
                if item["contributions"][target] > 0.002
            ][:5],
            "negative": [
                {"ingredient": item["ingredient"], "delta": item["contributions"][target]}
                for item in reversed(ranked)
                if item["contributions"][target] < -0.002
            ][:3],
        }

    return {
        "method": "occlusion",
        "seq_len": int(seq_len),
        "baseline": {
            target: round(float(baseline[i]), 4)
            for i, target in enumerate(config.PROFILE_TARGETS)
        },
        "contributions": contributions,
        "top_contributors": top_contributors,
    }


def describe_drivers(
    explanation: dict,
    profile_flags: Iterable[str] | None = None,
    max_items: int = 3,
) -> dict[str, list[str]]:
    """
    Turn occlusion results into short "why" lists for the report, e.g.

        Hydration: Glycerin, Sodium Hyaluronate, Panthenol
    """
    drivers: dict[str, list[str]] = {}
    for target, payload in explanation.get("top_contributors", {}).items():
        if profile_flags is not None and target not in set(profile_flags):
            continue
        names = [item["ingredient"] for item in payload.get("positive", [])][:max_items]
        drivers[target] = names
    return drivers

"""
Formulation-profile helpers  (shared by the report builder and the API).

The scores themselves come from the trained network; this module only turns a
0-1 score into the human-readable band shown in the report and produces the
short natural-language summary of the whole profile.
"""

from __future__ import annotations

from typing import Mapping

from ml import config


def band(score: float) -> str:
    """Map a 0-1 formulation score to Minimal / Low / Moderate / High."""
    return config.band_for_score(float(score))


def band_fraction(score: float) -> float:
    """Score rounded for display (kept explicit so the UI and API agree)."""
    return round(max(0.0, min(1.0, float(score))), 4)


def profile_payload(scores: Mapping[str, float]) -> list[dict]:
    """
    Build the ordered, display-ready profile block used by the API and the UI.

    Includes label, icon, raw score, band and bar width (0-100).
    """
    payload: list[dict] = []
    for target in config.PROFILE_TARGETS:
        score = float(scores.get(target, 0.0))
        payload.append(
            {
                "key": target,
                "label": config.PROFILE_LABELS.get(target, target.title()),
                "icon": config.PROFILE_ICONS.get(target, ""),
                "score": band_fraction(score),
                "band": band(score),
                "bar": int(round(band_fraction(score) * 100)),
            }
        )
    return payload


def dominant_targets(scores: Mapping[str, float], threshold: float = 0.55) -> list[str]:
    """Targets scored at or above ``threshold``, strongest first."""
    ranked = sorted(
        config.PROFILE_TARGETS,
        key=lambda target: float(scores.get(target, 0.0)),
        reverse=True,
    )
    return [target for target in ranked if float(scores.get(target, 0.0)) >= threshold]


def summary_sentence(scores: Mapping[str, float], category: str | None = None) -> str:
    """
    One-sentence description of the formulation, e.g.

    "Mainly hydration-oriented with a secondary brightening profile."
    """
    strong = dominant_targets(scores, 0.55)
    moderate = [
        target
        for target in config.PROFILE_TARGETS
        if target not in strong and float(scores.get(target, 0.0)) >= 0.4
    ]
    label_of = lambda key: config.PROFILE_LABELS.get(key, key).lower()  # noqa: E731

    prefix = f"Likely {category.lower()}: " if category else ""
    if not strong and not moderate:
        return prefix + "no characteristic formulation pattern stands out in the predicted profile."
    if strong:
        primary = label_of(strong[0])
        others = [label_of(target) for target in strong[1:]]
        if others:
            return (
                prefix
                + f"mainly {primary}-oriented, with additional "
                + " and ".join(others)
                + " characteristics."
            )
        if moderate:
            return (
                prefix
                + f"mainly {primary}-oriented with a secondary "
                + label_of(moderate[0])
                + " profile."
            )
        return prefix + f"mainly {primary}-oriented."
    return (
        prefix
        + "a balanced profile: moderate "
        + " and ".join(label_of(target) for target in moderate)
        + " characteristics."
    )

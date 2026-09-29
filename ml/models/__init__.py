"""
DeepCos model package: Keras architectures and explainability utilities.
"""

from ml.models.deepcos_model import (
    MaskedSequencePooling,
    build_deepcos_model,
    build_feature_extractor,
    load_ingredient_model,
    model_architecture_summary,
)
from ml.models.explain import describe_drivers, occlusion_contributions

__all__ = [
    "MaskedSequencePooling",
    "build_deepcos_model",
    "build_feature_extractor",
    "load_ingredient_model",
    "model_architecture_summary",
    "describe_drivers",
    "occlusion_contributions",
]

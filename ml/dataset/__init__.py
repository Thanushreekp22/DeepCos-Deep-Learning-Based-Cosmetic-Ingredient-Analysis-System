"""
DeepCos dataset package: text preprocessing, vocabulary and (weakly) labelled
dataset generation for the ingredient-sequence model.
"""

from ml.dataset.preprocessing import (
    canonical_names,
    clean_raw_text,
    ingredient_list_from_any,
    parse_ingredient_text,
    resolve_ingredients,
    split_ingredient_string,
    text_preprocessing_report,
)

__all__ = [
    "canonical_names",
    "clean_raw_text",
    "ingredient_list_from_any",
    "parse_ingredient_text",
    "resolve_ingredients",
    "split_ingredient_string",
    "text_preprocessing_report",
]

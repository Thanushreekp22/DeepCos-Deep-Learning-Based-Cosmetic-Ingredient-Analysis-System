"""
Weak-supervision dataset generation for the DeepCos ingredient model.

Real cosmetic ingredient lists are used as *inputs*; the formulation-profile
labels are derived from the ingredient composition with the documented weights
in ``profile_weights.json`` (functions such as humectant/brightening/exfoliating
plus ingredient markers), together with a category bias and Gaussian noise.

Why weak supervision
--------------------
Public cosmetic datasets rarely ship with curated hydration / brightening /
exfoliation labels. Generating labels from documented ingredient functions
gives the LSTM+MLP a genuine supervised task today, and the generated CSV is
schema-compatible with a real labelled dataset, so the same training code can
be re-run on real labels without modification.

Usage
-----
    python -m ml.dataset.generate_dataset --n 8000
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ml import config
from ml.dataset.vocabulary import Vocabulary
from ml.knowledge_base import IngredientKnowledgeBase, normalize_ingredient_name


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------
@dataclass
class Product:
    """One (weakly) labelled cosmetic product."""

    product_id: str
    category: str
    display_name: str
    ingredients: list[str]
    profile: dict[str, float]
    bands: dict[str, str]
    ingredient_count: int = 0
    resolved_ratio: float = 1.0
    split: str = "train"

    def __post_init__(self) -> None:
        if not self.ingredient_count:
            self.ingredient_count = len(self.ingredients)

    # -- CSV helpers -------------------------------------------------------
    def to_row(self) -> dict:
        row = {
            "product_id": self.product_id,
            "category": self.category,
            "display_name": self.display_name,
            "ingredient_count": self.ingredient_count,
            "resolved_ratio": round(self.resolved_ratio, 4),
            "split": self.split,
            "ingredients": "|".join(self.ingredients),
        }
        for target in config.PROFILE_TARGETS:
            row[f"profile_{target}"] = round(float(self.profile.get(target, 0.0)), 5)
            row[f"band_{target}"] = self.bands.get(target, "")
        return row

    @classmethod
    def from_row(cls, row: dict) -> "Product":
        ingredients = [name for name in (row.get("ingredients") or "").split("|") if name]
        return cls(
            product_id=row["product_id"],
            category=row["category"],
            display_name=row.get("display_name", row["category"]),
            ingredients=ingredients,
            profile={
                target: float(row.get(f"profile_{target}", 0.0))
                for target in config.PROFILE_TARGETS
            },
            bands={
                target: row.get(f"band_{target}", config.band_for_score(
                    float(row.get(f"profile_{target}", 0.0))
                ))
                for target in config.PROFILE_TARGETS
            },
            ingredient_count=int(row.get("ingredient_count") or len(ingredients)),
            resolved_ratio=float(row.get("resolved_ratio") or 1.0),
            split=row.get("split", "train"),
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["ingredient_count"] = self.ingredient_count
        return payload


CSV_COLUMNS: tuple[str, ...] = (
    "product_id",
    "category",
    "display_name",
    "ingredient_count",
    "resolved_ratio",
    "split",
    "ingredients",
    *[f"profile_{target}" for target in config.PROFILE_TARGETS],
    *[f"band_{target}" for target in config.PROFILE_TARGETS],
)


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
@dataclass
class ReferenceData:
    kb: IngredientKnowledgeBase
    templates: dict
    weights: dict
    position_groups: dict[str, int]

    @classmethod
    def load(cls) -> "ReferenceData":
        kb = IngredientKnowledgeBase()
        templates_raw = json.loads(
            config.CATEGORY_TEMPLATES_PATH.read_text(encoding="utf-8")
        )
        weights = json.loads(config.PROFILE_WEIGHTS_PATH.read_text(encoding="utf-8"))
        return cls(
            kb=kb,
            templates=templates_raw["categories"],
            weights=weights,
            position_groups=templates_raw["position_groups"],
        )

    def ingredients_by_function(self) -> dict[str, list[str]]:
        """Map each function tag to the canonical ingredients providing it."""
        out: dict[str, list[str]] = {}
        for info in self.kb.ingredients.values():
            for fn in info.functions:
                out.setdefault(fn, []).append(info.inci)
        return out

    def sampling_weights(self) -> dict[str, float]:
        """Common ingredients are sampled more often than exotic ones."""
        return {
            info.inci: (3.0 if info.common else 1.0) for info in self.kb.ingredients.values()
        }


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def _weighted_choice(
    candidates: Sequence[str],
    weights: dict[str, float],
    exclude: set[str],
    rng: np.random.Generator,
) -> str | None:
    pool = [c for c in candidates if normalize_ingredient_name(c) not in exclude]
    if not pool:
        return None
    w = np.asarray([weights.get(c, 1.0) for c in pool], dtype="float64")
    w = w / w.sum()
    return str(rng.choice(np.asarray(pool, dtype=object), p=w))


def position_group_of(
    name: str, kb: IngredientKnowledgeBase, position_groups: dict[str, int]
) -> int:
    """Realistic INCI ordering group: aqua first, colourants last."""
    if normalize_ingredient_name(name) == "aqua":
        return 0
    info = kb.get(name)
    if not info or not info.functions:
        return 6
    return min(position_groups.get(fn, 6) for fn in info.functions)


def order_ingredients(
    names: Sequence[str],
    kb: IngredientKnowledgeBase,
    position_groups: dict[str, int],
    rng: np.random.Generator,
    shuffle_within_group: float = 0.3,
) -> list[str]:
    """
    Order a sampled ingredient set the way a real INCI list is written:
    solvent first, then humectants, emollients, actives, emulsifiers,
    preservatives, pH adjusters, with fragrance and colourants last.
    """
    groups: dict[int, list[str]] = {}
    for name in names:
        groups.setdefault(position_group_of(name, kb, position_groups), []).append(name)
    ordered: list[str] = []
    for group in sorted(groups):
        items = groups[group]
        if len(items) > 1 and rng.random() < shuffle_within_group:
            rng.shuffle(items)
        ordered.extend(items)
    return ordered


# ---------------------------------------------------------------------------
# Label derivation (weak supervision)
# ---------------------------------------------------------------------------
def compute_profile(
    ingredients: Sequence[str],
    category: str,
    reference: "ReferenceData",
    rng: np.random.Generator,
    with_noise: bool = True,
) -> dict[str, float]:
    """
    Derive the formulation-profile vector of one product from its ingredients.

        composition = sum_i decay^i * (function_weight(fn_i, dim)
                                       + marker_boost(ingredient_i, dim))
        score       = squash( base + composition / dimension_scale[dim]
                              + category_bias + noise )

    ``decay`` gives earlier (higher-concentration) ingredients slightly more
    influence; ``dimension_scale`` keeps all four dimensions on a comparable
    0-1 range so that no single profile saturates.
    """
    weights = reference.weights
    function_weights: dict[str, dict[str, float]] = weights["function_weights"]
    marker_boosts: dict[str, dict[str, float]] = weights.get("marker_boosts", {})
    decay = float(weights.get("decay", 0.82))
    base_level = float(weights.get("base_level", 0.04))
    squash_k = float(weights.get("squash_k", 1.25))
    noise_std = float(weights.get("noise_std", 0.045))
    dimension_scale: dict[str, float] = weights.get("dimension_scale", {})

    composition = {target: 0.0 for target in config.PROFILE_TARGETS}
    for position, name in enumerate(ingredients):
        position_decay = decay**position
        info = reference.kb.get(name)
        if info:
            for fn in info.functions:
                for target in config.PROFILE_TARGETS:
                    contribution = function_weights.get(target, {}).get(fn)
                    if contribution:
                        composition[target] += contribution * position_decay
        for target, boost in marker_boosts.get(name, {}).items():
            if target in composition:
                composition[target] += float(boost) * position_decay

    category_bias = reference.templates[category]["profile_bias"]
    profile: dict[str, float] = {}
    for target in config.PROFILE_TARGETS:
        scale = float(dimension_scale.get(target, 1.0) or 1.0)
        total = base_level + composition[target] / scale + float(category_bias.get(target, 0.0))
        if with_noise and noise_std > 0:
            total += float(rng.normal(0.0, noise_std))
        profile[target] = float(np.clip(1.0 - np.exp(-squash_k * max(total, 0.0)), 0.0, 1.0))
    return profile


def bands_for_profile(profile: dict[str, float]) -> dict[str, str]:
    return {target: config.band_for_score(score) for target, score in profile.items()}


# ---------------------------------------------------------------------------
# Product generation
# ---------------------------------------------------------------------------
def generate_product(
    category: str,
    index: int,
    reference: ReferenceData,
    rng: np.random.Generator,
    by_function: dict[str, list[str]],
    sampling_weights: dict[str, float],
) -> Product:
    """Sample one coherent, realistically ordered product of a given category."""
    template = reference.templates[category]
    slots: dict[str, list[int]] = template["slots"]

    sampled: list[str] = []
    exclude: set[str] = set()

    def add(name: str | None) -> None:
        if not name:
            return
        key = normalize_ingredient_name(name)
        if key in exclude:
            return
        exclude.add(key)
        sampled.append(name)

    # 1. base ingredients of the category (always / probabilistic)
    for name, probability in template["base_ingredients"]:
        if rng.random() <= float(probability):
            add(name)

    # 2. one sampling pass per functional slot
    for function, (low, high) in slots.items():
        count = int(rng.integers(int(low), int(high) + 1))
        candidates = by_function.get(function, [])
        for _ in range(count):
            add(_weighted_choice(candidates, sampling_weights, exclude, rng))

    # 3. fragrance and colourant behaviour
    if rng.random() <= float(template.get("fragrance_probability", 0.4)):
        fragrance_pool = by_function.get("fragrance", [])
        if fragrance_pool and rng.random() < 0.55:
            add(_weighted_choice(fragrance_pool, sampling_weights, exclude, rng))
        for _ in range(int(rng.integers(0, 3))):
            add(_weighted_choice(fragrance_pool, sampling_weights, exclude, rng))
    if rng.random() <= float(template.get("colorant_probability", 0.05)):
        add(_weighted_choice(by_function.get("colorant", []), sampling_weights, exclude, rng))

    # 4. grow / trim to the category size range
    low, high = template["size_range"]
    target_size = int(rng.integers(int(low), int(high) + 1))
    available_functions = [fn for fn in slots if by_function.get(fn)] or ["humectant"]
    guard = 0
    while len(sampled) < target_size and guard < 4 * target_size:
        guard += 1
        function = available_functions[int(rng.integers(0, len(available_functions)))]
        add(_weighted_choice(by_function[function], sampling_weights, exclude, rng))
    if len(sampled) > config.SEQ_LEN:
        sampled = sampled[: config.SEQ_LEN]

    # 5. realistic INCI ordering
    ordered = order_ingredients(
        sampled,
        reference.kb,
        reference.position_groups,
        rng,
        shuffle_within_group=float(template.get("shuffle_within_group", 0.3)),
    )

    profile = compute_profile(ordered, category, reference, rng)
    resolved = reference.kb.resolve_all(ordered, allow_fuzzy=False)
    resolved_ratio = sum(1 for item in resolved if item.resolved) / max(1, len(resolved))

    return Product(
        product_id=f"P{index:06d}",
        category=category,
        display_name=template.get("display_name", category),
        ingredients=ordered,
        profile=profile,
        bands=bands_for_profile(profile),
        ingredient_count=len(ordered),
        resolved_ratio=resolved_ratio,
    )


def generate_dataset(
    n: int = config.DATASET_SIZE,
    seed: int = config.RANDOM_SEED,
    reference: ReferenceData | None = None,
    categories: Sequence[str] | None = None,
) -> list[Product]:
    """Generate ``n`` weakly labelled products, balanced across categories."""
    reference = reference or ReferenceData.load()
    rng = np.random.default_rng(seed)
    by_function = reference.ingredients_by_function()
    sampling_weights = reference.sampling_weights()
    categories = list(categories or config.CATEGORIES)

    return [
        generate_product(
            categories[i % len(categories)],
            i,
            reference,
            rng,
            by_function,
            sampling_weights,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Persistence / dataset splitting
# ---------------------------------------------------------------------------
def write_csv(products: Sequence[Product], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for product in products:
            writer.writerow(product.to_row())
    return path


def read_csv(path: Path | str) -> list[Product]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [Product.from_row(row) for row in csv.DictReader(handle)]


def split_dataset(
    products: Sequence[Product],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = config.RANDOM_SEED,
) -> dict[str, list[Product]]:
    """Random split into train / validation / test."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(products))
    n_train = int(len(products) * ratios[0])
    n_val = int(len(products) * ratios[1])
    index = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    splits: dict[str, list[Product]] = {}
    for name, indices in index.items():
        subset: list[Product] = []
        for i in indices:
            product = products[int(i)]
            product.split = name
            subset.append(product)
        splits[name] = subset
    return splits


def build_arrays(
    products: Sequence[Product],
    vocab: Vocabulary,
    seq_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode a product list into (X_sequence, y_profile, y_category) arrays."""
    seq_len = int(seq_len or config.SEQ_LEN)
    category_index = {category: i for i, category in enumerate(config.CATEGORIES)}
    x = np.zeros((len(products), seq_len), dtype="int32")
    y_profile = np.zeros((len(products), len(config.PROFILE_TARGETS)), dtype="float32")
    y_category = np.zeros((len(products),), dtype="int32")
    for row, product in enumerate(products):
        x[row] = vocab.encode(product.ingredients, seq_len)
        y_profile[row] = [product.profile[target] for target in config.PROFILE_TARGETS]
        y_category[row] = category_index.get(product.category, 0)
    return x, y_profile, y_category


def dataset_summary(products: Sequence[Product]) -> dict:
    """Descriptive statistics printed after generation (and stored as JSON)."""
    import collections

    categories = collections.Counter(p.category for p in products)
    sizes = np.asarray([p.ingredient_count for p in products], dtype="float32")
    profile_stats: dict[str, dict] = {}
    for target in config.PROFILE_TARGETS:
        values = np.asarray([p.profile[target] for p in products], dtype="float32")
        band_counts = collections.Counter(p.bands[target] for p in products)
        profile_stats[target] = {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "bands": dict(band_counts),
        }
    return {
        "products": len(products),
        "categories": dict(categories),
        "ingredients_per_product": {
            "mean": round(float(sizes.mean()), 2),
            "min": int(sizes.min()),
            "max": int(sizes.max()),
        },
        "resolved_ratio_mean": round(
            float(np.mean([p.resolved_ratio for p in products])), 4
        ),
        "profile_statistics": profile_stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the DeepCos weak-supervision dataset.")
    parser.add_argument("--n", type=int, default=config.DATASET_SIZE, help="number of products")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--out", type=str, default=str(config.INGREDIENT_DATASET_PATH))
    parser.add_argument("--summary", type=str, default=str(config.PROCESSED_DIR / "dataset_summary.json"))
    args = parser.parse_args(argv)

    config.ensure_directories()
    products = generate_dataset(args.n, args.seed)
    splits = split_dataset(products, seed=args.seed)

    write_csv(products, args.out)
    for name, subset in splits.items():
        write_csv(subset, Path(args.out).with_name(f"{Path(args.out).stem}_{name}.csv"))

    summary = dataset_summary(products)
    summary["split_sizes"] = {name: len(subset) for name, subset in splits.items()}
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Generated {len(products)} products -> {args.out}")
    print(f"Splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print(f"Summary -> {args.summary}")
    print(json.dumps(summary["profile_statistics"], indent=2)[:1200])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())




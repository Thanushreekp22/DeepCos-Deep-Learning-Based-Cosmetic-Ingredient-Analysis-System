"""
Synthetic cosmetic-label images for training the visual CNN.

Real photographs of cosmetic labels are not redistributable, so the CNN is
trained on rendered labels produced from the ingredient knowledge base used by
the sequence model. Each rendered label contains:

* a product "shape" silhouette (dropper / jar / tube / sachet / bottle)
* a colour theme, a product name and a block of small ingredient text
* photo-like degradations: rotation, blur, noise, uneven lighting

Two datasets are produced:

``text_region``    64x64 grayscale patches labelled text / background - teaches
                   the CNN to localise the printed ingredient block for OCR.
``label_category`` 128x128 RGB label images labelled with the product category -
                   teaches the CNN to recognise product type from pixels.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:  # pragma: no cover - corrupt font file
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Per-category visual themes (this is what makes the category task learnable)
# ---------------------------------------------------------------------------
CATEGORY_VISUALS: dict[str, dict] = {
    "Serum": {
        "background": (247, 248, 252),
        "accent": (99, 102, 241),
        "shape": "dropper",
        "titles": ["Hydrating Serum", "Glow Serum", "Barrier Serum"],
    },
    "Moisturizer": {
        "background": (252, 248, 242),
        "accent": (217, 119, 6),
        "shape": "jar",
        "titles": ["Daily Moisturizer", "Rich Cream", "Barrier Cream"],
    },
    "Cleanser": {
        "background": (243, 250, 247),
        "accent": (16, 148, 128),
        "shape": "tube",
        "titles": ["Gentle Cleanser", "Foaming Wash", "Gel Cleanser"],
    },
    "Sunscreen": {
        "background": (250, 250, 240),
        "accent": (234, 179, 8),
        "shape": "tube",
        "titles": ["Sunscreen SPF 50", "Daily UV Fluid", "Mineral SPF 30"],
    },
    "Toner": {
        "background": (244, 249, 253),
        "accent": (14, 165, 233),
        "shape": "bottle",
        "titles": ["Balancing Toner", "Hydrating Toner", "PHA Toner"],
    },
    "Mask": {
        "background": (250, 244, 250),
        "accent": (168, 85, 247),
        "shape": "sachet",
        "titles": ["Clay Mask", "Sheet Mask", "Sleeping Mask"],
    },
    "Exfoliant": {
        "background": (253, 246, 246),
        "accent": (239, 68, 68),
        "shape": "bottle",
        "titles": ["AHA Peel", "BHA Treatment", "Resurfacing Solution"],
    },
    "Eye Cream": {
        "background": (247, 247, 250),
        "accent": (100, 116, 139),
        "shape": "jar",
        "titles": ["Eye Cream", "Eye Contour Cream", "Brightening Eye Cream"],
    },
}


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, accent: tuple[int, int, int], size: int) -> None:
    """Draw a simple product silhouette in the upper part of the label."""
    w = size
    if shape == "dropper":
        draw.rounded_rectangle([w * 0.42, w * 0.10, w * 0.58, w * 0.40], radius=6, outline=accent, width=3)
        draw.line([w * 0.50, w * 0.05, w * 0.50, w * 0.12], fill=accent, width=3)
    elif shape == "jar":
        draw.rounded_rectangle([w * 0.34, w * 0.16, w * 0.66, w * 0.40], radius=10, outline=accent, width=3)
        draw.line([w * 0.34, w * 0.22, w * 0.66, w * 0.22], fill=accent, width=3)
    elif shape == "tube":
        draw.polygon(
            [(w * 0.38, w * 0.10), (w * 0.62, w * 0.10), (w * 0.66, w * 0.42), (w * 0.34, w * 0.42)],
            outline=accent,
        )
    elif shape == "bottle":
        draw.rounded_rectangle([w * 0.40, w * 0.14, w * 0.60, w * 0.42], radius=8, outline=accent, width=3)
        draw.rectangle([w * 0.46, w * 0.07, w * 0.54, w * 0.15], outline=accent, width=3)
    else:  # sachet
        draw.rectangle([w * 0.32, w * 0.10, w * 0.68, w * 0.42], outline=accent, width=3)
        draw.line([w * 0.32, w * 0.16, w * 0.68, w * 0.16], fill=accent, width=2)


def _wrap(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    """Greedy word wrapping so the rendered text stays inside the label."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _degrade(
    image: Image.Image,
    rng: random.Random,
    np_rng: np.random.Generator,
    strength: float = 1.0,
) -> Image.Image:
    """Add photo-like degradations so the CNN sees realistic variation."""
    fill = 255 if image.mode == "L" else (255, 255, 255)
    if rng.random() < 0.7 * strength:
        image = image.rotate(
            rng.uniform(-8, 8) * strength, resample=Image.BICUBIC, fillcolor=fill
        )
    if rng.random() < 0.5 * strength:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.8) * strength))

    array = np.asarray(image).astype("float32")
    array += np_rng.normal(0.0, 6.0 * strength, size=array.shape)
    if rng.random() < 0.4 * strength:  # soft uneven lighting
        width = array.shape[1]
        ramp = np.linspace(0.80, 1.08, width, dtype="float32").reshape(
            1, width, *([1] * (array.ndim - 2))
        )
        array *= ramp
    return Image.fromarray(np.clip(array, 0, 255).astype("uint8"), mode=image.mode)


# ---------------------------------------------------------------------------
# Label rendering
# ---------------------------------------------------------------------------
def render_label_image(
    category: str,
    ingredients: list[str],
    rng: random.Random,
    np_rng: np.random.Generator,
    size: int = 128,
    degrade: bool = True,
) -> Image.Image:
    """Render one synthetic cosmetic label of the given category."""
    theme = CATEGORY_VISUALS.get(category, CATEGORY_VISUALS["Serum"])
    image = Image.new("RGB", (size, size), theme["background"])
    draw = ImageDraw.Draw(image)

    _draw_shape(draw, theme["shape"], theme["accent"], size)

    title = rng.choice(theme["titles"])
    title_font = load_font(max(9, size // 12))
    draw.text((size * 0.05, size * 0.45), title, fill=(28, 28, 34), font=title_font)

    body_font = load_font(max(6, size // 20))
    ingredient_text = "Ingredients: " + ", ".join(ingredients)
    lines = _wrap(ingredient_text, body_font, int(size * 0.90), draw)[:6]
    for index, line in enumerate(lines):
        draw.text((size * 0.05, size * 0.58 + index * (size // 18)), line, fill=(70, 70, 78), font=body_font)

    if degrade:
        image = _degrade(image, rng, np_rng, strength=1.0)
    return image


def _text_patch_source(rng: random.Random, size: int = 64) -> Image.Image:
    """A card of small ingredient text, cropped later into patches."""
    image = Image.new("L", (size * 4, size * 3), 255)
    draw = ImageDraw.Draw(image)
    font = load_font(rng.randint(9, 15))
    for line_index in range(size * 3 // (size // 8)):
        y = 4 + line_index * (size // 8)
        text = " ".join(rng.choice(WORD_POOL) for _ in range(rng.randint(4, 9)))
        draw.text((6, y), text, fill=rng.randint(0, 60), font=font)
    return image


WORD_POOL: tuple[str, ...] = (
    "aqua", "glycerin", "niacinamide", "panthenol", "sodium", "hyaluronate",
    "butylene", "glycol", "cetearyl", "alcohol", "dimethicone", "tocopherol",
    "salicylic", "acid", "phenoxyethanol", "xanthan", "gum", "parfum",
    "linalool", "limonene", "citric", "squalane", "ceramide", "allantoin",
)


def _background_patch_source(rng: random.Random, np_rng: np.random.Generator, size: int = 64) -> Image.Image:
    """Plain label area: gradient, speckle, sometimes a logo blob or a barcode."""
    width, height = size * 3, size * 2
    base = np_rng.integers(200, 255)
    array = np.full((height, width), base, dtype="float32")
    array += np_rng.normal(0, 4, size=array.shape)
    ramp = np.linspace(0.92, 1.05, width, dtype="float32")[None, :]
    array *= ramp
    image = Image.fromarray(np.clip(array, 0, 255).astype("uint8"))
    draw = ImageDraw.Draw(image)
    if rng.random() < 0.5:  # logo blob
        cx, cy = rng.randint(size, width - size), rng.randint(size // 2, height - size // 2)
        radius = rng.randint(size // 6, size // 3)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     outline=rng.randint(120, 200), width=rng.randint(2, 4))
    if rng.random() < 0.3:  # barcode-like stripes
        x0 = rng.randint(0, width - size)
        for x in range(x0, min(width, x0 + size + rng.randint(0, size)), rng.randint(3, 6)):
            draw.line([x, height - size // 2, x, height - 4], fill=rng.randint(20, 90), width=2)
    return image


def build_text_region_dataset(
    n: int,
    seed: int = 42,
    patch_size: int = 64,
    text_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the text/background patch dataset for the AlexNet-style CNN.

    Returns ``(X, y)`` with ``X`` shaped ``(n, patch, patch, 1)`` float32 in [0,1]
    and ``y`` float32 labels (1.0 = text, 0.0 = background).
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    x = np.zeros((n, patch_size, patch_size, 1), dtype="float32")
    y = np.zeros((n,), dtype="float32")

    for i in range(n):
        is_text = rng.random() < text_ratio
        if is_text:
            source = _text_patch_source(rng, patch_size)
        else:
            source = _background_patch_source(rng, np_rng, patch_size)
        array = np.asarray(source, dtype="float32")
        max_y = max(1, array.shape[0] - patch_size)
        max_x = max(1, array.shape[1] - patch_size)
        top = rng.randint(0, max_y)
        left = rng.randint(0, max_x)
        patch = array[top : top + patch_size, left : left + patch_size]
        if patch.shape != (patch_size, patch_size):
            patch = np.resize(patch, (patch_size, patch_size))
        if rng.random() < 0.35:  # patch-level photo degradation
            patch = np.asarray(
                _degrade(
                    Image.fromarray(patch.astype("uint8")),
                    rng,
                    np_rng,
                    strength=0.5,
                ),
                dtype="float32",
            )
        x[i, :, :, 0] = patch / 255.0
        y[i] = 1.0 if is_text else 0.0
    return x, y


def build_label_category_dataset(
    n: int,
    ingredient_names: list[str],
    seed: int = 42,
    size: int = 128,
    categories: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the label-image classifier dataset (all 8 DeepCos categories).

    Returns ``(X, y)`` with ``X`` shaped ``(n, size, size, 3)`` float32 in [0,1]
    and integer class labels following the ``categories`` order.
    """
    from ml import config as cfg

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    categories = list(categories or cfg.CATEGORIES)

    x = np.zeros((n, size, size, 3), dtype="float32")
    y = np.zeros((n,), dtype="int32")
    for i in range(n):
        label = i % len(categories)
        category = categories[label]
        count = rng.randint(8, 18)
        ingredients = rng.sample(ingredient_names, min(count, len(ingredient_names)))
        image = render_label_image(category, ingredients, rng, np_rng, size=size)
        x[i] = np.asarray(image, dtype="float32") / 255.0
        y[i] = label
    return x, y


def save_preview_images(out_dir: str | Path, per_category: int = 2, size: int = 256) -> list[Path]:
    """Write a few rendered labels to disk (docs / manual inspection)."""
    from ml import config as cfg

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    np_rng = np.random.default_rng(7)
    written: list[Path] = []
    for category in cfg.CATEGORIES:
        for index in range(per_category):
            ingredients = [
                "Aqua", "Glycerin", "Niacinamide", "Panthenol", "Sodium Hyaluronate",
                "Cetearyl Alcohol", "Dimethicone", "Tocopherol", "Phenoxyethanol",
            ]
            image = render_label_image(category, ingredients, rng, np_rng, size=size)
            path = out_dir / f"{category.replace(' ', '_').lower()}_{index + 1}.png"
            image.save(path)
            written.append(path)
    return written



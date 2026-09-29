"""
CNN-driven text-region localisation  (Module 1).

The AlexNet-style CNN is applied as a sliding-window classifier over the label
photo. Each 64x64 patch receives a "contains printed text" probability, the
probabilities form a heatmap, and the heatmap is reduced to the ingredient-text
block that is then cropped and handed to OCR.

Effect of this stage: OCR only sees the region that actually matters, which
removes logos, marketing copy and background texture - the usual causes of
garbage ingredient names.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ml import config
from ml.image import cnn_alexnet, preprocess


class TextRegionDetector:
    """Sliding-window text-block detector built on the trained CNN."""

    def __init__(
        self,
        model,
        patch_size: int | None = None,
        stride: int | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.patch_size = int(patch_size or config.TEXT_PATCH_SIZE)
        self.stride = int(stride or max(8, self.patch_size // 2))
        self.threshold = float(threshold)

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str | None = None, **kwargs) -> "TextRegionDetector | None":
        """Load the trained CNN; ``None`` when it has not been trained yet."""
        try:
            model = cnn_alexnet.load_text_region_cnn(path)
        except FileNotFoundError:
            return None
        return cls(model, **kwargs)

    # -- inference ---------------------------------------------------------
    def heatmap(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        """
        Per-patch text probability reshaped to the sliding-window grid.

        Returns ``(grid, rows, cols)`` with ``grid`` shaped ``(rows, cols)``.
        """
        patches: list[np.ndarray] = []
        positions: list[tuple[int, int, int, int]] = []
        for patch, box in preprocess.sliding_window_patches(
            image, patch_size=self.patch_size, stride=self.stride
        ):
            patches.append(patch)
            positions.append(box)
        if not patches:
            return np.zeros((1, 1), dtype="float32"), 1, 1

        batch = preprocess.patches_to_batch(patches, self.patch_size)
        probabilities = cnn_alexnet.predict_text_probabilities(self.model, batch)

        unique_x = sorted({box[0] for box in positions})
        unique_y = sorted({box[1] for box in positions})
        x_index = {value: i for i, value in enumerate(unique_x)}
        y_index = {value: i for i, value in enumerate(unique_y)}

        grid = np.zeros((len(unique_y), len(unique_x)), dtype="float32")
        for probability, (x, y, _, _) in zip(probabilities, positions):
            grid[y_index[y], x_index[x]] = float(probability)
        return grid, len(unique_y), len(unique_x)

    def detect(self, image: np.ndarray, threshold: float | None = None) -> dict:
        """
        Locate the densest text band on the label.

        Returns the bounding box of the ingredient block, the mean text
        probability inside it, the coverage fraction and the full-size heatmap.
        """
        threshold = float(self.threshold if threshold is None else threshold)
        grid, rows, cols = self.heatmap(image)
        height, width = image.shape[:2]

        # Row profile: which horizontal bands look like printed text?
        if grid.size:
            row_density = (grid >= threshold).mean(axis=1)
        else:
            row_density = np.zeros(rows, dtype="float32")
        band_rows = np.where(row_density >= max(0.25, float(row_density.max(initial=0.0)) * 0.5))[0]

        if band_rows.size == 0:  # nothing convincing -> keep the whole frame
            box = (0, 0, width, height)
            mean_probability = float(grid.mean()) if grid.size else 0.0
            coverage = 1.0
        else:
            # keep the largest contiguous run of text rows
            runs = np.split(band_rows, np.where(np.diff(band_rows) > 1)[0] + 1)
            run = max(runs, key=len)
            row_start, row_end = int(run[0]), int(run[-1])
            band = grid[row_start : row_end + 1, :]
            columns = np.where((band >= threshold).mean(axis=0) >= 0.2)[0]
            column_start = int(columns.min()) if columns.size else 0
            column_end = int(columns.max()) if columns.size else cols - 1

            scale_x = width / max(1, cols)
            scale_y = height / max(1, rows)
            box = (
                int(column_start * scale_x),
                int(row_start * scale_y),
                int(min(width, (column_end + 1) * scale_x)),
                int(min(height, (row_end + 1) * scale_y)),
            )
            mean_probability = float(band.mean())
            coverage = float((box[3] - box[1]) * (box[2] - box[0]) / max(1, height * width))

        if coverage < 0.02:  # degenerate box -> keep the full frame
            box = (0, 0, width, height)
            coverage = 1.0

        heatmap_full = cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)
        return {
            "box": box,
            "box_normalised": [
                round(box[0] / width, 4),
                round(box[1] / height, 4),
                round(box[2] / width, 4),
                round(box[3] / height, 4),
            ],
            "mean_text_probability": round(mean_probability, 4),
            "coverage": round(coverage, 4),
            "grid": grid,
            "grid_shape": [rows, cols],
            "heatmap": heatmap_full,
            "threshold": threshold,
        }

    def extract(self, image: np.ndarray, padding: int = 10) -> tuple[np.ndarray, dict]:
        """Crop the detected text block (with a small padding)."""
        detection = self.detect(image)
        cropped = preprocess.crop_region(image, detection["box"], padding=padding)
        if cropped.size == 0:
            cropped = image
            detection["box"] = (0, 0, image.shape[1], image.shape[0])
            detection["coverage"] = 1.0
        return cropped, detection

    def visualise(self, image: np.ndarray, padding: int = 10) -> dict:
        """
        Full visual output for the dashboard.

        Returns base64 PNGs of the label with the detected box, of the
        text-probability heatmap, and of the crop that was sent to OCR.
        """
        cropped, detection = self.extract(image, padding=padding)
        annotated = preprocess.draw_regions(image, [detection["box"]])
        overlay = preprocess.heatmap_overlay(image, detection["heatmap"])
        return {
            "box": detection["box"],
            "mean_text_probability": detection["mean_text_probability"],
            "coverage": detection["coverage"],
            "grid_shape": detection["grid_shape"],
            "annotated_png_base64": preprocess.encode_png_base64(annotated),
            "heatmap_png_base64": preprocess.encode_png_base64(overlay),
            "cropped_png_base64": preprocess.encode_png_base64(cropped),
            "cropped_image": cropped,
        }


@lru_cache(maxsize=1)
def get_text_detector() -> TextRegionDetector | None:
    """
    Cached detector instance.

    Returns ``None`` when the CNN has not been trained yet; the API then falls
    back to whole-image OCR instead of failing.
    """
    return TextRegionDetector.load()


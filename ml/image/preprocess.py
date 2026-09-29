"""
Image preprocessing for DeepCos  (Module 1 - Image Processing, OpenCV part).

Responsibilities
----------------
* robust loading of an uploaded label photo (bytes / path / ndarray)
* quality assessment (blur, brightness, resolution) so the API can warn the user
* the classical enhancement chain used before OCR
* sliding-window patch extraction that feeds the AlexNet-style text detector
* visual debugging helpers (region overlay, heatmap overlay) returned as
  base64 PNGs so the React dashboard can show what the pipeline "saw"
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Iterator, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Loading / encoding
# ---------------------------------------------------------------------------
def load_image(source) -> np.ndarray:
    """Load an image from a path, raw bytes, a file-like object or an ndarray."""
    if isinstance(source, np.ndarray):
        image = source
    elif isinstance(source, (bytes, bytearray)):
        buffer = np.frombuffer(bytes(source), dtype="uint8")
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    elif hasattr(source, "read"):
        data = source.read()
        buffer = np.frombuffer(bytes(data), dtype="uint8")
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the supplied image.")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def encode_png_base64(image: np.ndarray, max_width: int = 900) -> str:
    """Encode an image as a base64 PNG string for the API response."""
    preview = resize_max_width(image, max_width)
    ok, buffer = cv2.imencode(".png", preview)
    if not ok:  # pragma: no cover - encoding failure
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def resize_max_width(image: np.ndarray, max_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / float(width)
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def resize_max_side(image: np.ndarray, max_side: int = 1600) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------
@dataclass
class ImageQuality:
    width: int
    height: int
    blur_score: float
    brightness: float
    contrast: float
    is_sharp: bool
    is_bright_enough: bool
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "blur_score": round(self.blur_score, 2),
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "is_sharp": self.is_sharp,
            "is_bright_enough": self.is_bright_enough,
            "warnings": self.warnings,
        }


def assess_quality(image: np.ndarray) -> ImageQuality:
    """
    Basic capture-quality check.

    ``blur_score`` is the variance of the Laplacian: low values mean a blurry
    photo, which is the main cause of OCR failure on cosmetic labels.
    """
    gray = to_grayscale(image)
    height, width = gray.shape[:2]
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    warnings: list[str] = []
    is_sharp = blur_score >= 60.0
    is_bright = 45.0 <= brightness <= 225.0
    if not is_sharp:
        warnings.append("The photo looks blurry - OCR accuracy may drop.")
    if brightness < 45.0:
        warnings.append("The photo looks dark - try to capture it in better light.")
    if brightness > 225.0:
        warnings.append("The photo looks over-exposed or washed out.")
    if min(height, width) < 480:
        warnings.append("Low resolution - the ingredient text may be too small for OCR.")
    if contrast < 30.0:
        warnings.append("Low contrast between text and background.")

    return ImageQuality(
        width=width,
        height=height,
        blur_score=blur_score,
        brightness=brightness,
        contrast=contrast,
        is_sharp=is_sharp,
        is_bright_enough=is_bright,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Enhancement / binarisation chain
# ---------------------------------------------------------------------------
def denoise(gray: np.ndarray, strength: int = 7) -> np.ndarray:
    """Edge-preserving denoising - keeps letters crisp, removes sensor noise."""
    return cv2.bilateralFilter(gray, d=strength, sigmaColor=55, sigmaSpace=55)


def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE: local contrast equalisation, effective on glossy printed labels."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Rotate a photographed label so its text lines are approximately horizontal.

    The skew angle is estimated from the dominant orientation of the ink pixels
    with ``minAreaRect`` over the binarised image.
    """
    inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(inverted)
    if coords is None or len(coords) < 50:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.3 or abs(angle) > 20:
        return gray
    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def binarise(gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """Binarise an enhanced grayscale image for OCR."""
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=12,
    )


def sharpen(gray: np.ndarray) -> np.ndarray:
    kernel = np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype="float32")
    return cv2.filter2D(gray, -1, kernel)


def upscale(gray: np.ndarray, factor: float = 1.8, min_height: int = 900) -> np.ndarray:
    """Small text is the most common OCR killer - upscale before recognition."""
    if gray.shape[0] >= min_height or factor <= 1.0:
        return gray
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def prepare_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Produce several OCR-ready versions of the same label.

    The OCR stage runs on each variant and keeps the most plausible result, which
    is far more robust than one fixed preprocessing chain across different label
    designs, lighting conditions and paper types.
    """
    gray = resize_max_side(to_grayscale(image), 1800)
    denoised = denoise(gray)
    enhanced = enhance_contrast(denoised)
    straightened = deskew(enhanced)
    return {
        "raw_gray": gray,
        "denoised": denoised,
        "clahe": enhanced,
        "deskewed": straightened,
        "adaptive": binarise(straightened, "adaptive"),
        "otsu": binarise(straightened, "otsu"),
        "sharpened": sharpen(enhanced),
        "upscaled": upscale(enhanced),
    }


# ---------------------------------------------------------------------------
# Patch extraction (feeds the AlexNet-style text detector)
# ---------------------------------------------------------------------------
def sliding_window_patches(
    image: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
) -> Iterator[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Yield ``(patch, (x1, y1, x2, y2))`` windows over the grayscale image."""
    gray = to_grayscale(image)
    height, width = gray.shape[:2]
    for y in range(0, max(1, height - patch_size + 1), stride):
        for x in range(0, max(1, width - patch_size + 1), stride):
            x2 = min(x + patch_size, width)
            y2 = min(y + patch_size, height)
            patch = gray[y:y2, x:x2]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.copyMakeBorder(
                    patch,
                    0,
                    patch_size - patch.shape[0],
                    0,
                    patch_size - patch.shape[1],
                    cv2.BORDER_REPLICATE,
                )
            yield patch, (x, y, x2, y2)


def normalise_patch(patch: np.ndarray, patch_size: int = 64) -> np.ndarray:
    """Resize + normalise one patch exactly the way the CNN expects it."""
    resized = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0


def patches_to_batch(patches: Sequence[np.ndarray], patch_size: int = 64) -> np.ndarray:
    """Stack patches into an (N, patch, patch, 1) float32 batch."""
    return np.stack([normalise_patch(patch, patch_size) for patch in patches])[..., None]


def crop_region(image: np.ndarray, box: tuple[int, int, int, int], padding: int = 8) -> np.ndarray:
    """Crop a bounding box with a small padding, clipped to the image bounds."""
    x1, y1, x2, y2 = box
    height, width = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return image[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Visual debugging for the UI
# ---------------------------------------------------------------------------
def draw_regions(
    image: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (34, 197, 94),
    thickness: int = 3,
) -> np.ndarray:
    """Draw detected text regions on a copy of the label image."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return canvas


def heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a text-probability heatmap on the label image.

    ``heatmap`` is a float array in [0, 1]; it is resized to the image size and
    blended with the JET colour map.
    """
    base = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = base.shape[:2]
    resized = cv2.resize(
        np.asarray(heatmap, dtype="float32"), (width, height), interpolation=cv2.INTER_CUBIC
    )
    coloured = cv2.applyColorMap(
        (np.clip(resized, 0.0, 1.0) * 255).astype("uint8"), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(base, 1.0 - alpha, coloured, alpha, 0.0)


def image_to_png_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:  # pragma: no cover - encoding failure
        raise ValueError("Could not encode image to PNG.")
    return buffer.tobytes()


def png_bytes_to_pil(image: np.ndarray):
    """Convert an OpenCV image into a PIL image (used by the OCR path)."""
    from PIL import Image

    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_numpy(pil_image) -> np.ndarray:
    """Convert a PIL image into an OpenCV BGR ndarray."""
    array = np.asarray(pil_image.convert("RGB"))
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)



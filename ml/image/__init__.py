"""
DeepCos image package (Module 1).

preprocess      : OpenCV loading, quality checks, enhancement, patch extraction
cnn_alexnet     : AlexNet-style CNN (text-region detection / label category)
text_detector   : sliding-window CNN inference that localises the text block
ocr             : multi-variant Tesseract OCR with graceful degradation
synthetic_labels: rendered label images used to train the visual CNN
"""

from ml.image import preprocess
from ml.image.cnn_alexnet import build_alexnet, load_text_region_cnn

__all__ = ["preprocess", "build_alexnet", "load_text_region_cnn"]

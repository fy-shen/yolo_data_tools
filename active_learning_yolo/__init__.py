"""面向 Ultralytics YOLO 的轻量 PPAL 主动学习工具包。"""

from .ppal import (
    ClassQualityEMA,
    Detection,
    ImagePrediction,
    OBJECT_FEATURES,
    PPALSelector,
    SelectionResult,
    binary_entropy,
    compute_class_weights,
)

__all__ = [
    "ClassQualityEMA", "Detection", "ImagePrediction", "OBJECT_FEATURES",
    "PPALSelector", "SelectionResult", "binary_entropy", "compute_class_weights",
]

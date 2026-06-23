"""检测框架适配器。"""

from .ultralytics import (
    class_qualities_from_metrics,
    predict_with_object_features,
    results_to_predictions,
)

__all__ = [
    "class_qualities_from_metrics",
    "predict_with_object_features",
    "results_to_predictions",
]

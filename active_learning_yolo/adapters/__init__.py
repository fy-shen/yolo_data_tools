"""检测框架适配器。"""

from .ultralytics import (
    class_qualities_from_metrics,
    extract_feature_maps,
    predict_with_object_features,
    results_to_predictions,
    sample_detection_features,
)

__all__ = [
    "class_qualities_from_metrics", "extract_feature_maps",
    "predict_with_object_features", "results_to_predictions",
    "sample_detection_features",
]

"""PPAL 多样性阶段使用的目标级图片距离。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .schemas import Detection, ImagePrediction

OBJECT_FEATURES = "object_features"
EPS = 1e-12


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector) if norm <= EPS else vector / norm


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个特征向量的余弦距离。"""
    return float(1.0 - np.dot(_normalize(left), _normalize(right)))


def _valid_detections(
    prediction: ImagePrediction, score_threshold: float
) -> list[Detection]:
    return [
        det for det in prediction.detections
        if det.feature is not None and det.confidence >= score_threshold
    ]


def _directed_object_distance(
    source: Sequence[Detection], target: Sequence[Detection]
) -> float:
    """计算 source 到 target 的有向目标级距离，只匹配相同预测类别。"""
    weighted_distance = 0.0
    confidence_sum = 0.0
    for source_det in source:
        if not target:
            continue
        candidates = [
            target_det for target_det in target
            if target_det.class_id == source_det.class_id
        ]
        nearest = (
            min(
                cosine_distance(source_det.feature, target_det.feature)
                for target_det in candidates
            )
            if candidates else 2.0
        )
        weighted_distance += source_det.confidence * nearest
        confidence_sum += source_det.confidence
    return 0.0 if confidence_sum <= EPS else weighted_distance / confidence_sum


def object_feature_distance_matrix(
    predictions: Sequence[ImagePrediction], score_threshold: float = 0.05
) -> np.ndarray:
    """根据目标级局部特征构造论文 PPAL 风格的图片距离矩阵。"""
    detections = [_valid_detections(item, score_threshold) for item in predictions]
    size = len(predictions)
    distances = np.zeros((size, size), dtype=np.float32)
    for left in range(size):
        for right in range(left + 1, size):
            distance = 0.5 * (
                _directed_object_distance(detections[left], detections[right])
                + _directed_object_distance(detections[right], detections[left])
            )
            distances[left, right] = distances[right, left] = distance
    return distances


def validate_object_features(predictions: Sequence[ImagePrediction]) -> None:
    """确认每张图至少有一个检测框带目标级局部特征。"""
    has_objects = all(
        any(det.feature is not None for det in item.detections)
        for item in predictions
    )
    if not has_objects:
        raise ValueError("严格 PPAL 要求每张图至少有一个框级 feature")


def build_distance_matrix(
    predictions: Sequence[ImagePrediction],
    mode: str = OBJECT_FEATURES,
    score_threshold: float = 0.05,
) -> np.ndarray:
    """根据目标级局部特征构造 PPAL 图片距离矩阵。"""
    if mode not in (OBJECT_FEATURES, "object"):
        raise ValueError("严格 PPAL 只支持 object_features 模式")
    validate_object_features(predictions)
    return object_feature_distance_matrix(predictions, score_threshold)

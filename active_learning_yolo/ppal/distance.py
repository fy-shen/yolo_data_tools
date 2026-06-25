"""PPAL 多样性阶段使用的目标级图片距离。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .schemas import Detection, ImagePrediction

OBJECT_FEATURES = "object_features"
EPS = 1e-12
ProgressCallback = Callable[[str], None]


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector) if norm <= EPS else vector / norm


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= EPS, 1.0, norms)
    return matrix / norms


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个特征向量的余弦距离。"""
    return float(1.0 - np.dot(_normalize(left), _normalize(right)))


def _valid_detections(
    prediction: ImagePrediction,
    score_threshold: float,
    max_detections_per_image: int | None = None,
) -> list[Detection]:
    detections = [
        det for det in prediction.detections
        if det.feature is not None and det.confidence >= score_threshold
    ]
    if max_detections_per_image is not None and max_detections_per_image > 0:
        detections = sorted(
            detections, key=lambda item: item.confidence, reverse=True
        )[:max_detections_per_image]
    return detections


def _group_detections(detections: Sequence[Detection]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[int, list[Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.class_id, []).append(detection)

    packed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for class_id, items in grouped.items():
        features = _normalize_rows(np.stack([item.feature for item in items]))
        confidences = np.asarray([item.confidence for item in items], dtype=np.float32)
        packed[class_id] = (features, confidences)
    return packed


def _directed_object_distance(
    source: Sequence[Detection], target: Sequence[Detection]
) -> float:
    """计算 source 到 target 的有向目标级距离，只匹配相同预测类别。"""
    return _directed_group_distance(_group_detections(source), _group_detections(target))


def _directed_group_distance(
    source: dict[int, tuple[np.ndarray, np.ndarray]],
    target: dict[int, tuple[np.ndarray, np.ndarray]],
) -> float:
    if not target:
        return 0.0

    weighted_distance = 0.0
    confidence_sum = 0.0
    for class_id, (source_features, source_confidences) in source.items():
        target_item = target.get(class_id)
        if target_item is None:
            nearest = np.full(len(source_confidences), 2.0, dtype=np.float32)
        else:
            target_features, _target_confidences = target_item
            similarities = source_features @ target_features.T
            nearest = 1.0 - np.max(similarities, axis=1)
            nearest = np.clip(nearest, 0.0, 2.0)
        weighted_distance += float(source_confidences @ nearest)
        confidence_sum += float(source_confidences.sum())
    return 0.0 if confidence_sum <= EPS else weighted_distance / confidence_sum


def object_feature_distance_matrix(
    predictions: Sequence[ImagePrediction],
    score_threshold: float = 0.05,
    progress_interval: int = 0,
    progress_callback: ProgressCallback | None = None,
    max_detections_per_image: int | None = None,
) -> np.ndarray:
    """根据目标级局部特征构造论文 PPAL 风格的图片距离矩阵。"""
    detections = [
        _valid_detections(item, score_threshold, max_detections_per_image)
        for item in predictions
    ]
    grouped = [_group_detections(item) for item in detections]
    size = len(predictions)
    total_pairs = size * (size - 1) // 2
    processed_pairs = 0
    last_report = 0
    distances = np.zeros((size, size), dtype=np.float32)

    if progress_callback is not None:
        progress_callback(
            f"diversity distance: candidates={size} pairs={total_pairs}"
        )

    for left in range(size):
        for right in range(left + 1, size):
            distance = 0.5 * (
                _directed_group_distance(grouped[left], grouped[right])
                + _directed_group_distance(grouped[right], grouped[left])
            )
            distances[left, right] = distances[right, left] = distance
        processed_pairs += size - left - 1
        if (
            progress_callback is not None
            and progress_interval > 0
            and processed_pairs - last_report >= progress_interval
        ):
            progress_callback(
                f"diversity distance: processed_pairs={processed_pairs}/{total_pairs} "
                f"rows={left + 1}/{size}"
            )
            last_report = processed_pairs

    if progress_callback is not None:
        progress_callback(
            f"diversity distance: processed_pairs={processed_pairs}/{total_pairs} done"
        )
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
    progress_interval: int = 0,
    progress_callback: ProgressCallback | None = None,
    max_detections_per_image: int | None = None,
) -> np.ndarray:
    """根据目标级局部特征构造 PPAL 图片距离矩阵。"""
    if mode not in (OBJECT_FEATURES, "object"):
        raise ValueError("严格 PPAL 只支持 object_features 模式")
    validate_object_features(predictions)
    return object_feature_distance_matrix(
        predictions,
        score_threshold,
        progress_interval=progress_interval,
        progress_callback=progress_callback,
        max_detections_per_image=max_detections_per_image,
    )

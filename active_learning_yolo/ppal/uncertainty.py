"""难度校准的不确定性计算，对应 PPAL 的第一阶段。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .schemas import ImagePrediction

EPS = 1e-10


def binary_entropy(confidence: float) -> float:
    """计算二元熵；置信度越接近 0.5，不确定性越高。"""

    p = float(np.clip(confidence, EPS, 1.0 - EPS))
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def compute_class_weights(
    class_qualities: Mapping[int, float],
    upper_bound: float = 0.2,
    alpha: float = 0.3,
) -> dict[int, float]:
    """把类别学习质量转换成 PPAL 使用的难度权重。"""

    if alpha <= 0:
        raise ValueError("alpha 必须大于 0")
    b = np.exp(1.0 / alpha) - 1.0
    weights = {}
    for class_id, quality in class_qualities.items():
        reverse_quality = 1.0 - float(np.clip(quality, 0.0, 1.0))
        value = 1.0 + alpha * np.log(b * reverse_quality + 1.0) * upper_bound
        weights[int(class_id)] = float(value)
    return weights


def score_image_uncertainty(
    prediction: ImagePrediction,
    class_weights: Mapping[int, float] | None = None,
    score_threshold: float = 0.05,
    min_box_side: float = 1.0,
    max_aspect_ratio: float = 5.0,
) -> float:
    """累加图片中有效检测框的难度加权分类熵。"""

    weights = class_weights or {}
    total = 0.0
    for det in prediction.detections:
        if det.confidence < score_threshold:
            continue
        x1, y1, x2, y2 = map(float, det.bbox_xyxy)
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if min(width, height) < min_box_side:
            continue
        ratio = max(width / max(height, EPS), height / max(width, EPS))
        if ratio > max_aspect_ratio:
            continue
        total += binary_entropy(det.confidence) * float(
            weights.get(det.class_id, 1.0)
        )
    return float(total)


def rank_by_uncertainty(
    predictions: Sequence[ImagePrediction],
    class_weights: Mapping[int, float] | None = None,
    score_threshold: float = 0.05,
) -> list[tuple[ImagePrediction, float]]:
    """按图片不确定性从高到低排序。"""

    ranked = [
        (item, score_image_uncertainty(item, class_weights, score_threshold))
        for item in predictions
    ]
    return sorted(ranked, key=lambda item: item[1], reverse=True)

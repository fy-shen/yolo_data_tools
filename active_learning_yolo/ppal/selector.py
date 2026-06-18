"""PPAL 的两阶段图片选择流程。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .distance import OBJECT_FEATURES, build_distance_matrix, validate_object_features
from .diversity import kmedoids
from .schemas import ImageId, ImagePrediction
from .uncertainty import rank_by_uncertainty


@dataclass(frozen=True)
class SelectionResult:
    """一次主动学习选择的完整结果。"""
    selected_ids: list[ImageId]
    candidate_ids: list[ImageId]
    uncertainty_scores: dict[ImageId, float]
    diversity_mode: str = OBJECT_FEATURES


class PPALSelector:
    """难度校准不确定性 + 多样性的两阶段选择器。"""

    def __init__(
        self,
        budget: int,
        candidate_multiplier: int = 4,
        score_threshold: float = 0.05,
        diversity_mode: str = OBJECT_FEATURES,
        seed: int = 0,
    ) -> None:
        if budget <= 0 or candidate_multiplier <= 0:
            raise ValueError("budget 和 candidate_multiplier 必须大于 0")
        if diversity_mode not in (OBJECT_FEATURES, "object"):
            raise ValueError("严格 PPAL 只支持 object_features 模式")
        self.budget = budget
        self.candidate_multiplier = candidate_multiplier
        self.score_threshold = score_threshold
        self.diversity_mode = OBJECT_FEATURES
        self.seed = seed

    def select(
        self,
        predictions: Sequence[ImagePrediction],
        class_weights: Mapping[int, float] | None = None,
    ) -> SelectionResult:
        """从未标注预测中选出本轮需要标注的图片。"""
        if len(predictions) < self.budget:
            raise ValueError("未标注图片数量小于 budget")
        ids = [item.image_id for item in predictions]
        if len(set(ids)) != len(ids):
            raise ValueError("image_id 必须唯一")
        ranked = rank_by_uncertainty(
            predictions, class_weights, self.score_threshold
        )
        candidate_count = min(
            len(ranked), self.budget * self.candidate_multiplier
        )
        candidates = [item[0] for item in ranked[:candidate_count]]
        scores = {item.image_id: score for item, score in ranked}
        validate_object_features(candidates)
        if candidate_count == self.budget:
            selected = candidates
        else:
            matrix = build_distance_matrix(
                candidates, self.diversity_mode, self.score_threshold
            )
            indices = kmedoids(matrix, self.budget, seed=self.seed)
            selected = [candidates[index] for index in indices]
        return SelectionResult(
            selected_ids=[item.image_id for item in selected],
            candidate_ids=[item.image_id for item in candidates],
            uncertainty_scores=scores,
            diversity_mode=self.diversity_mode,
        )

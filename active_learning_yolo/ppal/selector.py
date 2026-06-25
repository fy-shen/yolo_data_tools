"""PPAL 的两阶段图片选择流程。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .distance import OBJECT_FEATURES, build_distance_matrix, validate_object_features
from .diversity import kmedoids
from .schemas import ImageId, ImagePrediction
from .uncertainty import rank_by_uncertainty

ProgressCallback = Callable[[str], None]


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
        diversity_progress_interval: int = 500_000,
        kmedoids_max_iter: int = 100,
        max_detections_per_image: int | None = None,
        progress_callback: ProgressCallback | None = None,
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
        self.diversity_progress_interval = diversity_progress_interval
        self.kmedoids_max_iter = kmedoids_max_iter
        self.max_detections_per_image = max_detections_per_image
        self.progress_callback = progress_callback

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
        if self.progress_callback is not None:
            self.progress_callback(f"ppal select: ranking uncertainty for {len(predictions)} candidates")
        ranked = rank_by_uncertainty(
            predictions, class_weights, self.score_threshold
        )
        candidate_count = min(
            len(ranked), self.budget * self.candidate_multiplier
        )
        candidates = [item[0] for item in ranked[:candidate_count]]
        scores = {item.image_id: score for item, score in ranked}
        validate_object_features(candidates)
        if self.progress_callback is not None:
            self.progress_callback(
                f"ppal select: candidates={candidate_count} budget={self.budget}"
            )
        if candidate_count == self.budget:
            selected = candidates
        else:
            matrix = build_distance_matrix(
                candidates,
                self.diversity_mode,
                self.score_threshold,
                progress_interval=self.diversity_progress_interval,
                progress_callback=self.progress_callback,
                max_detections_per_image=self.max_detections_per_image,
            )
            indices = kmedoids(
                matrix,
                self.budget,
                max_iter=self.kmedoids_max_iter,
                seed=self.seed,
                progress_callback=self.progress_callback,
            )
            selected = [candidates[index] for index in indices]
        return SelectionResult(
            selected_ids=[item.image_id for item in selected],
            candidate_ids=[item.image_id for item in candidates],
            uncertainty_scores=scores,
            diversity_mode=self.diversity_mode,
        )

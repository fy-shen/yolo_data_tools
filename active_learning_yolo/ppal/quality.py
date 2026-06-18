"""类别学习质量的指数滑动平均（EMA）。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClassQualityEMA:
    """维护各类别质量：confidence^xi * IoU^(1-xi)。"""

    num_classes: int
    momentum: float = 0.99
    quality_xi: float = 0.6
    values: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes 必须大于 0")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum 必须位于 [0, 1)")
        if not 0.0 <= self.quality_xi <= 1.0:
            raise ValueError("quality_xi 必须位于 [0, 1]")
        self.values = np.zeros(self.num_classes, dtype=np.float32)

    def update(self, matches: Iterable[tuple[int, float, float]]) -> np.ndarray:
        """使用 (class_id, confidence, iou) 匹配结果更新质量。"""

        grouped = [[] for _ in range(self.num_classes)]
        for class_id, confidence, iou in matches:
            class_id = int(class_id)
            if not 0 <= class_id < self.num_classes:
                raise ValueError(f"类别编号越界: {class_id}")
            p = float(np.clip(confidence, 0.0, 1.0))
            overlap = float(np.clip(iou, 0.0, 1.0))
            grouped[class_id].append(
                p**self.quality_xi * overlap ** (1.0 - self.quality_xi)
            )
        for class_id, qualities in enumerate(grouped):
            if qualities:
                current = float(np.mean(qualities))
                self.values[class_id] = (
                    self.momentum * self.values[class_id]
                    + (1.0 - self.momentum) * current
                )
        return self.values.copy()

    def load(self, values: Mapping[int, float] | np.ndarray) -> None:
        """从检查点或验证结果恢复类别质量。"""

        if isinstance(values, Mapping):
            for class_id, quality in values.items():
                self.values[int(class_id)] = float(quality)
            return
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(array) != self.num_classes:
            raise ValueError("类别质量数量与 num_classes 不一致")
        self.values[:] = array

    def as_dict(self) -> dict[int, float]:
        return {index: float(value) for index, value in enumerate(self.values)}

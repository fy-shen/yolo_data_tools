"""PPAL 核心算法使用的通用数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Optional, Sequence

import numpy as np

ImageId = Hashable


@dataclass
class Detection:
    """单个目标检测结果。

    feature 是可选的目标级局部特征。严格 PPAL 需要为每个检测框提供该特征。
    """

    class_id: int
    confidence: float
    bbox_xyxy: Sequence[float]
    feature: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy 必须包含 4 个坐标")
        self.class_id = int(self.class_id)
        self.confidence = float(self.confidence)
        if self.feature is not None:
            self.feature = np.asarray(self.feature, dtype=np.float32).reshape(-1)


@dataclass
class ImagePrediction:
    """一张未标注图片的预测结果。"""

    image_id: ImageId
    detections: list[Detection] = field(default_factory=list)
    source: Optional[str] = None

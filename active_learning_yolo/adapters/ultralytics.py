"""Ultralytics YOLO 的 PPAL 目标级特征适配器。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..ppal.schemas import Detection, ImagePrediction


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def results_to_predictions(
    results: Sequence[Any], image_ids: Sequence[Any] | None = None
) -> list[ImagePrediction]:
    """把普通 YOLO.predict() Results 转为 PPAL 通用结构。"""
    if image_ids is not None and len(image_ids) != len(results):
        raise ValueError("image_ids 数量必须与 results 一致")
    predictions = []
    for index, result in enumerate(results):
        source = str(result.path)
        image_id = image_ids[index] if image_ids is not None else Path(source).stem
        detections = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for xyxy, confidence, class_id in zip(
                _to_numpy(boxes.xyxy), _to_numpy(boxes.conf), _to_numpy(boxes.cls)
            ):
                detections.append(Detection(
                    class_id=int(class_id), confidence=float(confidence),
                    bbox_xyxy=xyxy.tolist(),
                ))
        predictions.append(ImagePrediction(
            image_id=image_id, source=source, detections=detections
        ))
    return predictions


def _to_torch(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return torch.as_tensor(value)


def _input_hw(imgsz: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(imgsz, int):
        return imgsz, imgsz
    values = list(imgsz)
    if len(values) == 1:
        return int(values[0]), int(values[0])
    if len(values) != 2:
        raise ValueError("imgsz 必须是 int 或长度为 2 的序列")
    return int(values[0]), int(values[1])


def _result_orig_hw(result: Any) -> tuple[int, int]:
    shape = getattr(result, "orig_shape", None)
    if shape is None and getattr(result, "boxes", None) is not None:
        shape = getattr(result.boxes, "orig_shape", None)
    if shape is None:
        raise ValueError("Ultralytics Result 缺少 orig_shape，无法映射到 letterbox 坐标")
    return int(shape[0]), int(shape[1])


def _letterbox_boxes(
    boxes_xyxy: torch.Tensor,
    orig_hw: tuple[int, int],
    input_hw: tuple[int, int],
) -> torch.Tensor:
    """把原图坐标的 bbox 映射回 YOLO letterbox 输入坐标。"""
    orig_h, orig_w = orig_hw
    input_h, input_w = input_hw
    gain = min(input_h / orig_h, input_w / orig_w)
    pad_x = (input_w - orig_w * gain) / 2
    pad_y = (input_h - orig_h * gain) / 2
    mapped = boxes_xyxy.clone().to(dtype=torch.float32)
    mapped[:, [0, 2]] = mapped[:, [0, 2]] * gain + pad_x
    mapped[:, [1, 3]] = mapped[:, [1, 3]] * gain + pad_y
    return mapped


def _extract_level_indices(boxes: Any, n_detections: int) -> np.ndarray | None:
    """读取可选的检测来源层索引。

    严格复现 PPAL 的多层采样需要 NMS 后每个检测框保留其来源特征层。
    """
    for name in ("level", "levels", "lvl", "lvls", "lvl_ind", "lvl_inds"):
        value = getattr(boxes, name, None)
        if value is not None:
            levels = _to_numpy(value).reshape(-1).astype(np.int64)
            if levels.size != n_detections:
                raise ValueError(f"boxes.{name} 数量与检测框数量不一致")
            return levels
    return None


def _collect_feature_maps(outputs: Sequence[Any], n_layers: int) -> list[torch.Tensor]:
    """把 Ultralytics 按 batch 产出的特征图整理为每层一个 [N,C,H,W] Tensor。"""
    if n_layers <= 0:
        raise ValueError("embed_layers 不能为空")
    tensors = [_to_torch(item) for item in outputs]
    if not tensors:
        raise RuntimeError("Ultralytics 未返回特征图")
    if any(tensor.ndim != 4 for tensor in tensors):
        shapes = [tuple(tensor.shape) for tensor in tensors]
        raise ValueError(f"特征图必须是 [B,D,H,W]，实际为 {shapes}")
    if len(tensors) % n_layers != 0:
        raise RuntimeError("特征图数量不能按 embed_layers 分组")
    grouped = [[] for _ in range(n_layers)]
    for offset, tensor in enumerate(tensors):
        grouped[offset % n_layers].append(tensor)
    return [torch.cat(items, dim=0) for items in grouped]


def sample_detection_features(
    predictions: Sequence[ImagePrediction],
    results: Sequence[Any],
    feature_maps: Sequence[Any],
    imgsz: int | Sequence[int],
    require_level_indices: bool = True,
) -> None:
    """按 PPAL 原版中心点双线性采样，为每个 Detection 写入 feature。"""
    if len(predictions) != len(results):
        raise ValueError("predictions 与 results 数量不一致")
    maps = [_to_torch(item) for item in feature_maps]
    if not maps:
        raise ValueError("feature_maps 不能为空")
    if any(item.ndim != 4 for item in maps):
        raise ValueError("feature_maps 中每个张量都必须是 [B,D,H,W]")
    if any(item.shape[0] < len(predictions) for item in maps):
        raise ValueError("feature_maps 的 batch 维度小于图片数量")
    channel_dims = {int(item.shape[1]) for item in maps}
    if len(channel_dims) != 1:
        raise ValueError("严格 PPAL 要求所有采样层的通道数一致")

    input_hw = _input_hw(imgsz)
    for image_index, (prediction, result) in enumerate(zip(predictions, results)):
        if not prediction.detections:
            continue
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        levels = _extract_level_indices(boxes, len(prediction.detections))
        if levels is None:
            if len(maps) > 1 and require_level_indices:
                raise ValueError(
                    "多层特征采样需要 Ultralytics 在 boxes 中提供 level/lvl_inds"
                )
            levels = np.zeros(len(prediction.detections), dtype=np.int64)
        if np.any(levels < 0) or np.any(levels >= len(maps)):
            raise ValueError("检测框 level 索引超出 feature_maps 范围")

        xyxy = torch.as_tensor(
            [det.bbox_xyxy for det in prediction.detections], dtype=torch.float32
        )
        xyxy = _letterbox_boxes(xyxy, _result_orig_hw(result), input_hw)
        cx = ((0.5 * (xyxy[:, 0] + xyxy[:, 2]) / input_hw[1]) - 0.5) * 2
        cy = ((0.5 * (xyxy[:, 1] + xyxy[:, 3]) / input_hw[0]) - 0.5) * 2
        coords = torch.stack((cx, cy), dim=-1)

        for level_index, feature_map in enumerate(maps):
            mask = torch.as_tensor(levels == level_index, dtype=torch.bool)
            if not bool(mask.any()):
                continue
            image_feature = feature_map[image_index:image_index + 1]
            grid = coords[mask].to(image_feature.device, image_feature.dtype)
            sampled = F.grid_sample(
                image_feature, grid[None, None, :, :],
                mode="bilinear", align_corners=False,
            )
            sampled = sampled.squeeze(0).squeeze(1).transpose(0, 1)
            sampled = sampled.detach().cpu().numpy().astype(np.float32)
            det_indices = np.flatnonzero(levels == level_index)
            for det_index, feature in zip(det_indices, sampled):
                prediction.detections[int(det_index)].feature = feature


def extract_feature_maps(
    model: Any,
    sources: Sequence[str],
    embed_layers: Sequence[int],
    **predict_kwargs: Any,
) -> list[torch.Tensor]:
    """提取修改版 Ultralytics 返回的 `[B,D,H,W]` 特征图。"""
    source_list = list(sources)
    if "embed" in predict_kwargs:
        raise ValueError("请通过 embed_layers 参数指定层号")
    layers = [int(index) for index in embed_layers]
    if not layers:
        raise ValueError("embed_layers 不能为空")
    outputs = list(model.predict(
        source=source_list, stream=False, embed=layers, **predict_kwargs
    ))
    feature_maps = _collect_feature_maps(outputs, len(layers))
    if any(item.shape[0] != len(source_list) for item in feature_maps):
        raise RuntimeError("特征图 batch 数量与输入图片数量不一致")
    return feature_maps


def predict_with_object_features(
    model: Any,
    sources: Sequence[str],
    image_ids: Sequence[Any] | None = None,
    embed_layers: Sequence[int] = (0,),
    require_level_indices: bool = True,
    **predict_kwargs: Any,
) -> list[ImagePrediction]:
    """执行检测和特征图采样，返回严格 PPAL 所需的框级特征。"""
    source_list = list(sources)
    imgsz = predict_kwargs.get("imgsz", 640)
    results = list(model.predict(
        source=source_list, stream=False, embed=None, **predict_kwargs
    ))
    predictions = results_to_predictions(results, image_ids)
    feature_maps = extract_feature_maps(
        model, source_list, embed_layers=embed_layers, **predict_kwargs
    )
    sample_detection_features(
        predictions, results, feature_maps, imgsz,
        require_level_indices=require_level_indices,
    )
    return predictions


def class_qualities_from_metrics(metrics: Any) -> dict[int, float]:
    """使用 model.val() 的逐类别 mAP50-95 作为类别质量工程近似。"""
    maps = getattr(getattr(metrics, "box", None), "maps", None)
    if maps is None:
        raise ValueError("metrics.box.maps 不存在")
    values = _to_numpy(maps).reshape(-1)
    return {index: float(value) for index, value in enumerate(values)}

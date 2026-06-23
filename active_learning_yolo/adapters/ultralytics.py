"""Ultralytics YOLO 的 PPAL 目标级特征适配器。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
import contextlib

import numpy as np
import torch

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

    predictions: list[ImagePrediction] = []
    for index, result in enumerate(results):
        source = str(result.path)
        image_id = image_ids[index] if image_ids is not None else Path(source).stem
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for xyxy, confidence, class_id in zip(
                _to_numpy(boxes.xyxy), _to_numpy(boxes.conf), _to_numpy(boxes.cls)
            ):
                detections.append(Detection(
                    class_id=int(class_id),
                    confidence=float(confidence),
                    bbox_xyxy=xyxy.tolist(),
                ))
        predictions.append(ImagePrediction(
            image_id=image_id,
            source=source,
            detections=detections,
        ))
    return predictions


def _remove_callback(model: Any, event: str, callback: Callable[[Any], None]) -> None:
    callbacks = getattr(model, "callbacks", None)
    if callbacks is None:
        return
    items = callbacks.get(event)
    if not items:
        return
    with contextlib.suppress(ValueError):
        items.remove(callback)


def _make_detect_feature_callback(hooks: list[Any]) -> Callable[[Any], None]:
    """创建临时 callback，复用 Ultralytics ReID 的 Detect 输入特征回溯机制。"""

    def on_predict_start(predictor: Any) -> None:
        if hooks:
            hooks.pop().remove()
        predictor._feats = None

        try:
            from ultralytics.nn.modules.head import Detect
        except ImportError as exc:
            raise RuntimeError("无法导入 Ultralytics Detect head") from exc

        backend_model = getattr(getattr(predictor, "model", None), "model", None)
        layers = getattr(backend_model, "model", None)
        detect_head = layers[-1] if layers is not None and len(layers) else None
        if not isinstance(detect_head, Detect):
            raise ValueError("当前模型最后一层不是 Ultralytics Detect，无法提取目标特征")
        if getattr(detect_head, "end2end", False):
            raise ValueError("当前仅支持非 end2end 的 YOLOv8 Detect 模型")

        # PPAL 需要目标级局部特征。这里捕获进入 Detect head 前的多尺度特征图，
        # Ultralytics NMS 会返回保留框在展平候选中的索引，get_obj_feats 再回溯到对应位置。
        def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            predictor._feats = list(inputs[0])

        hooks.append(detect_head.register_forward_pre_hook(pre_hook))

    return on_predict_start


def _predict_results_with_object_features(
    model: Any,
    sources: Sequence[str],
    **predict_kwargs: Any,
) -> list[Any]:
    """执行一次 YOLO predict，并让 Results 携带 NMS 后目标对应的特征。"""
    if "embed" in predict_kwargs and predict_kwargs["embed"] is not None:
        raise ValueError("目标级特征提取使用 Detect hook，不应传入 embed")
    predict_kwargs["embed"] = None

    if not hasattr(model, "add_callback"):
        raise ValueError("model 必须是 ultralytics.YOLO 实例或兼容对象")

    hooks: list[Any] = []
    callback = _make_detect_feature_callback(hooks)
    model.add_callback("on_predict_start", callback)
    try:
        return list(model.predict(source=list(sources), stream=False, **predict_kwargs))
    finally:
        for hook in hooks:
            hook.remove()
        _remove_callback(model, "on_predict_start", callback)


def _attach_object_features(
    predictions: Sequence[ImagePrediction],
    results: Sequence[Any],
) -> None:
    """把 Ultralytics Results.feats 写回 PPAL 的 Detection.feature。"""
    if len(predictions) != len(results):
        raise ValueError("predictions 与 results 数量不一致")

    for prediction, result in zip(predictions, results):
        if not prediction.detections:
            continue
        feats = getattr(result, "feats", None)
        if feats is None:
            raise RuntimeError("Ultralytics Result 缺少 feats，请确认 Detect hook 已启用")
        feats_np = _to_numpy(feats).astype(np.float32)
        if feats_np.shape[0] != len(prediction.detections):
            raise RuntimeError("目标特征数量与检测框数量不一致")
        for detection, feature in zip(prediction.detections, feats_np):
            detection.feature = feature


def predict_with_object_features(
    model: Any,
    sources: Sequence[str],
    image_ids: Sequence[Any] | None = None,
    **predict_kwargs: Any,
) -> list[ImagePrediction]:
    """执行一次 YOLOv8 检测，并返回带目标级特征的 PPAL 预测。"""
    source_list = list(sources)
    results = _predict_results_with_object_features(model, source_list, **predict_kwargs)
    predictions = results_to_predictions(results, image_ids)
    _attach_object_features(predictions, results)
    return predictions


def class_qualities_from_metrics(metrics: Any) -> dict[int, float]:
    """使用 model.val() 的逐类别 mAP50-95 作为类别质量工程近似。"""
    maps = getattr(getattr(metrics, "box", None), "maps", None)
    if maps is None:
        raise ValueError("metrics.box.maps 不存在")
    values = _to_numpy(maps).reshape(-1)
    return {index: float(value) for index, value in enumerate(values)}

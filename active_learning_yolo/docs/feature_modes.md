# PPAL 目标级特征提取

当前主流程面向普通 YOLOv8 Detect 模型，不修改 Ultralytics 源码。适配器在
`predict_with_object_features()` 的一次预测期间临时注册 Detect head `forward_pre_hook`，
捕获进入 Detect head 前的多尺度特征图。Ultralytics 的 NMS 在 `return_idxs=True`
时会返回最终保留检测框在展平候选中的索引，随后官方 `get_obj_feats()` 会用这些索引
把检测框回溯到对应的 anchor point / grid cell 特征。

## PPAL 需要的数据

```text
图片 A -> [(类别, 分数, 目标特征), ...]
图片 B -> [(类别, 分数, 目标特征), ...]
```

计算两张图片距离时，它会：

1. 只匹配相同预测类别的目标；
2. 为每个目标寻找另一张图中最近的目标特征；
3. 用检测置信度加权目标距离；
4. 对两个方向的距离取平均。

这些目标特征会写入 `Detection.feature`，之后 `PPALSelector` 固定使用
`object_features`。

## 与原版 PPAL 的关系

原版 PPAL 在 RetinaNet 中保留 NMS 后的 `keep_idxs` 和来源 FPN level，然后从分类塔
特征中为每个目标取局部特征。YOLOv8 的检测候选天然对应多尺度特征图上的网格点，
因此当前实现使用 NMS 返回的候选索引直接回溯对应位置特征，而不是再按 bbox 中心
做 `grid_sample`。这仍然是目标级局部特征，不是整图 embedding。

## 使用方式

```python
from active_learning_yolo.adapters import predict_with_object_features
from active_learning_yolo.ppal import PPALSelector

predictions = predict_with_object_features(
    model,
    image_paths,
    imgsz=640,
    conf=0.05,
    verbose=False,
)
result = PPALSelector(budget=100).select(predictions)
```

## 当前限制

默认 `end2end=True` 的 YOLO26 不会在 NMS 中返回候选索引，暂不纳入主流程。若后续
需要支持 YOLO26，可以显式切到 `end2end=False` 后复用同一机制，或为 end2end top-k
输出额外保留索引。

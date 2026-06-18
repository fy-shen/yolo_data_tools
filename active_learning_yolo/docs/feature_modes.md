# 严格 PPAL 目标级特征采样

当前代码只保留 PPAL 原版的目标级局部特征路径。修改版 Ultralytics 需要在：

```python
model.predict(source, embed=[...], imgsz=640, conf=0.25)
```

返回一个或多个 `[B,D,H,W]` 特征图，而不是全局平均池化后的一维向量。

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

## 坐标和来源层

Ultralytics 的 `Results.boxes.xyxy` 通常是原图坐标；特征图对应的是 letterbox 后
的模型输入。适配器会根据 `result.orig_shape` 和 `imgsz` 把 bbox 映射回 letterbox
坐标，然后按框中心执行：

```python
torch.nn.functional.grid_sample(...)
```

如果只传一个 `embed_layers`，所有框都会从这一层采样。若传多个层，严格复现要求
Ultralytics 在 NMS 后的 `boxes` 对象里保留每个检测框来源层索引，字段名可为：

```text
level / levels / lvl / lvls / lvl_ind / lvl_inds
```

这些索引对应 `embed_layers` 的位置，而不是模型层号本身。例如
`embed_layers=[0, 10]` 时，`lvl_inds=1` 表示从第二个返回特征图采样。

## 使用方式

```python
from active_learning_yolo.adapters import predict_with_object_features
from active_learning_yolo.ppal import PPALSelector

predictions = predict_with_object_features(
    model,
    image_paths,
    embed_layers=[0, 10],
    imgsz=640,
    conf=0.05,
    verbose=False,
)
result = PPALSelector(budget=100).select(predictions)
```

如果多层特征暂时没有 `lvl_inds`，适配器默认会报错，避免静默退化成非严格实现。

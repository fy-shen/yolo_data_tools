# 面向 Ultralytics YOLO 的 PPAL 主动学习代码

该目录将原 MMDetection 项目中的主动学习逻辑重新组织为独立模块。PPAL 核心只
依赖 NumPy，Ultralytics 相关代码集中在适配层。

## 目录结构

```text
active_learning_yolo/
├── ppal/                         # PPAL 核心算法
│   ├── schemas.py                # 检测框、图片预测等通用数据结构
│   ├── uncertainty.py            # 分类熵和难度校准不确定性
│   ├── quality.py                # 类别质量 EMA
│   ├── distance.py               # 目标级特征距离
│   ├── diversity.py              # k-medoids 多样性采样
│   └── selector.py               # 两阶段 PPAL 选择流程
├── data/                         # 数据与标注状态
│   └── pool.py                   # 标注池、图片 txt 列表
├── adapters/                     # 检测器输出适配
│   └── ultralytics.py            # YOLO Results 和目标级特征采样
├── examples/                     # 可运行示例
├── tests/                        # 单元测试
├── docs/                         # 原理和集成说明
└── requirements.txt
```

## 严格 PPAL 目标级特征

每个检测框都要有独立的局部特征 `Detection.feature`。算法只比较相同预测类别的
目标，寻找最近目标特征距离，再按检测置信度加权。这与原 PPAL 的多样性部分一致。

```python
from active_learning_yolo.ppal import OBJECT_FEATURES, PPALSelector

selector = PPALSelector(budget=100, diversity_mode=OBJECT_FEATURES)
result = selector.select(predictions, class_weights=class_weights)
```

当前代码不再保留全局 embedding 路径。Ultralytics 必须返回 `[B,D,H,W]` 特征图，
适配器会按 PPAL 原版方式在检测框中心用 `grid_sample` 做双线性采样。

```python
from active_learning_yolo.adapters import predict_with_object_features
from active_learning_yolo.ppal import OBJECT_FEATURES, PPALSelector

predictions = predict_with_object_features(
    model,
    image_paths,
    embed_layers=[0, 10],
    imgsz=640,
    conf=0.05,
    verbose=False,
)
result = PPALSelector(
    budget=100,
    diversity_mode=OBJECT_FEATURES,
).select(predictions)
```

详细说明见 [docs/feature_modes.md](docs/feature_modes.md)。

## 命令行示例

准备一行一个图片路径的 `data/unlabeled.txt`：

```bash
conda run -n torch2.7.0 python \
  -m active_learning_yolo.examples.select_with_ultralytics \
  --model weights/yolo26s.pt \
  --unlabeled-list data/unlabeled.txt \
  --output work_dirs/round1/selected.txt \
  --budget 100 \
  --device 0 \
  --embed-layers 0 10
```

多层特征严格复现要求 Ultralytics 在 NMS 后的 `boxes` 对象中提供
`level/lvl/lvl_inds` 等字段，表示每个检测框来自 `--embed-layers` 中的第几个层。

## 类别难度

不修改 YOLO 训练代码时，可用逐类别验证 mAP 作为类别质量近似：

```python
from active_learning_yolo.adapters import class_qualities_from_metrics
from active_learning_yolo.ppal import compute_class_weights

metrics = model.val(data="dataset.yaml")
qualities = class_qualities_from_metrics(metrics)
class_weights = compute_class_weights(qualities)
```

严格复现则应在 YOLO 正样本分配后取得 `(class_id, confidence, IoU)`，传给
`ClassQualityEMA.update()`。

## 标注池

```python
from active_learning_yolo.data import AnnotationPool

pool = AnnotationPool(labeled=["a.jpg"], unlabeled=["b.jpg", "c.jpg"])
pool.request_annotation(["b.jpg"])
pool.mark_labeled(["b.jpg"])
pool.save("work_dirs/pool.json")
```

## 测试

```bash
conda run -n torch2.7.0 python -m unittest discover \
  -s active_learning_yolo/tests -v
```

## 当前边界

- 全局 embedding 模式已删除，只保留目标级局部特征。
- 多层严格采样需要 YOLO 返回每个检测框的来源层索引。
- 完整距离矩阵空间复杂度为 O(N^2)，候选池很大时需要分块或近似方法。
- 当前处理水平目标框，未覆盖分割、姿态和旋转框。

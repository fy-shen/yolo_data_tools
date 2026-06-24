# 可运行脚本

这里集中放置直接处理数据或运行流程的脚本。脚本按任务类型分目录，避免散落在库代码中。

建议分类：

- `ppal/`: PPAL 主动学习选择流程。
- `labels/`: 标签格式转换、类别映射、标注清洗。
- `video/`: 原始视频切帧、抽帧、帧列表生成。
- `yolo/`: 生成 YOLO txt、划分数据集、检查图片和标签一致性。

脚本应尽量只负责命令行参数、文件读写和调用库函数；可复用逻辑放回 `active_learning_yolo` 或后续的数据处理库目录。

## PPAL Football 数据集示例

从 football 目录下所有 `*-train.txt` 随机抽初始训练集，并把剩余训练集作为 PPAL 未标注池；验证集使用全部 `*-val.txt`：

```bash
conda run -n yolo python scripts/ppal/init_split.py \
  --train-glob '/media/sfy/disk11/dataset/football/*-train.txt' \
  --val-glob '/media/sfy/disk11/dataset/football/*-val.txt' \
  --dataset-root /media/sfy/disk11/dataset/football \
  --initial-size 1000 \
  --initial-output work_dirs/ppal/football/round0_train.txt \
  --pool-output work_dirs/ppal/football/round0_pool.txt \
  --all-output work_dirs/ppal/football/full_train.txt \
  --val-output work_dirs/ppal/football/full_val.txt \
  --seed 0
```

训练初始模型后，用该模型在未标注池中选择下一批，并同时输出剩余池和下一轮累计训练集：

```bash
conda run -n yolo python scripts/ppal/select_with_ultralytics.py \
  --model runs/detect/round0/weights/best.pt \
  --unlabeled-list work_dirs/ppal/football/round0_pool.txt \
  --labeled-list work_dirs/ppal/football/round0_train.txt \
  --output work_dirs/ppal/football/round1_selected.txt \
  --remaining-output work_dirs/ppal/football/round1_pool.txt \
  --updated-labeled-output work_dirs/ppal/football/round1_train.txt \
  --budget 1000 \
  --imgsz 640 \
  --batch 16 \
  --conf 0.05 \
  --device 0
```

每轮训练都可以继续使用 `full_val.txt` 作为验证集，与全量训练结果保持同一验证基准。

## PPAL 自动 5 轮训练示例

如果已经有 `round0_train.txt` 和 `round0_pool.txt`，可以直接运行自动迭代。下面命令会训练 5 次：

- `round0`: 使用初始 2% 训练；
- `round1~round4`: 每轮从剩余池中再加入约全训练池 2%；
- 最终 `round4_train.txt` 约为全训练池 10%。

```bash
conda run -n yolo python scripts/ppal/run_iterations.py \
  --data-template football_ppal_local.yaml \
  --initial-train round0_train.txt \
  --initial-pool round0_pool.txt \
  --work-dir work_dirs/ppal/football_auto \
  --base-model yolov8s.yaml \
  --pretrained ./weights/yolov8s.pt \
  --rounds 5 \
  --add-ratio 0.02 \
  --budget-base total \
  --epochs 50 \
  --imgsz 960 \
  --batch 36 \
  --device 0,1 \
  --workers 8 \
  --candidate-multiplier 4
```

如果服务器上希望每轮 data yaml 里的 train txt 仍然相对 `path`，把 `--work-dir` 放到数据集根目录下面，
并加 `--data-list-mode relative-to-data-path`。例如训练 txt 位于 `football_ppal_local.yaml` 的 `path` 内，
训练结果仍希望集中写到原来的 football project 下，可以这样指定：

```bash
conda run -n yolo python scripts/ppal/run_iterations.py \
  --data-template football_ppal_local.yaml \
  --initial-train /app/SFY/dynamic_yolo/datasets/football_dataset/round0_train.txt \
  --initial-pool /app/SFY/dynamic_yolo/datasets/football_dataset/round0_pool.txt \
  --work-dir /app/SFY/dynamic_yolo/datasets/football_dataset/ppal_auto \
  --data-list-mode relative-to-data-path \
  --base-model yolov8s.yaml \
  --pretrained ./weights/yolov8s.pt \
  --rounds 5 \
  --add-ratio 0.02 \
  --budget-base total \
  --epochs 50 \
  --imgsz 960 \
  --batch 36 \
  --device 0,1 \
  --workers 8 \
  --train-project /app/SFY/dynamic_yolo/ultralytics/runs/detect/football \
  --train-name-prefix yolov8s-960
```

此时训练目录会是 `.../football/yolov8s-960-round0`、`.../football/yolov8s-960-round1` 等；
脚本不会猜测路径，而是从 Ultralytics 训练后的 `trainer.save_dir` 读取实际目录，并在每轮
`work-dir/roundN/train_save_dir.txt` 中记录。默认 `--train-from previous`，即下一轮从上一轮
`best.pt` 继续训练；如果希望每轮都从同一个预训练模型重新训练，可加 `--train-from base`。

注意 `--data-list-mode` 只控制每轮 `data.yaml` 里 train txt 路径的写法。每轮 train/pool txt
内部的图片路径默认由 `--image-list-mode absolute` 处理：绝对路径保持不变，相对路径会按
`data-template` 的 `path` 转成绝对路径，避免 Ultralytics 按 `work-dir` 拼接相对图片路径。
如果确实要原样保留 txt 内容，可显式加 `--image-list-mode preserve`。

正式运行前可以先加 `--dry-run` 检查每轮 txt 数量和生成的 data yaml，不会启动训练和推理。

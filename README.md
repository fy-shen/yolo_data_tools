# yolo_data_tools

Data processing and active learning utilities for YOLO projects.

## Layout

```text
yolo_data_tools/
├── active_learning_yolo/      # PPAL 核心库和检测框架适配器
├── scripts/                   # 可直接运行的数据处理脚本
│   └── ppal/                  # PPAL 主动学习脚本
├── lib/ultralytics/           # Ultralytics source checkout, tracked as a gitlink/submodule
├── weights/                   # 本地测试权重
└── README.md
```

## Environment

Use the `yolo` conda environment from the repository root. The root repository is
intended to be used as scripts and source files, not installed as a Python package.

```bash
conda activate yolo
export PYTHONPATH="$PWD:$PWD/lib/ultralytics:$PYTHONPATH"
```

If you want `import ultralytics` to always point at this checkout without setting
`PYTHONPATH`, install only the local Ultralytics checkout in editable mode:

```bash
pip install -e lib/ultralytics
```

## PPAL Active Learning

The PPAL code is in `active_learning_yolo`. The runnable selection script is:

```bash
python scripts/ppal/select_with_ultralytics.py \
  --model weights/yolov8s.pt \
  --unlabeled-list data/unlabeled.txt \
  --output work_dirs/round1/selected.txt \
  --budget 100 \
  --device 0
```

See `active_learning_yolo/README.md` for the detailed PPAL flow and current strict object-feature requirements.

## Updating Ultralytics

`lib/ultralytics` points at the official Ultralytics repository. To update it
manually:

```bash
git -C lib/ultralytics fetch origin
git -C lib/ultralytics checkout main
git -C lib/ultralytics pull --ff-only origin main
```

Then review and test local integrations before committing the updated gitlink in
this repository.

If you need to modify Ultralytics source code and keep those changes, use a fork:

```bash
git -C lib/ultralytics remote rename origin upstream
git -C lib/ultralytics remote add origin git@github.com:fy-shen/ultralytics.git
```

In that setup, pull official changes from `upstream`, push your modified
Ultralytics commits to your fork, and commit the updated submodule pointer here.

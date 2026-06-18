"""使用修改版 Ultralytics YOLO 完成严格 PPAL 选择。"""

import argparse
from pathlib import Path

from active_learning_yolo.adapters.ultralytics import (
    predict_with_object_features,
)
from active_learning_yolo.data import read_image_list, write_image_list
from active_learning_yolo.ppal import OBJECT_FEATURES, PPALSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultralytics YOLO 主动学习")
    parser.add_argument("--model", required=True, help="YOLO 权重路径")
    parser.add_argument("--unlabeled-list", required=True,
                        help="一行一个未标注图片路径")
    parser.add_argument("--output", required=True, help="选中图片列表")
    parser.add_argument("--budget", required=True, type=int, help="标注预算")
    parser.add_argument("--device", default=None, help="例如 0、0,1 或 cpu")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--conf", default=0.05, type=float,
                        help="保留候选框的最低置信度")
    parser.add_argument(
        "--embed-layers", nargs="+", type=int, default=None,
        help="返回 [B,D,H,W] 特征图的层号，例如 0 10",
    )
    parser.add_argument(
        "--allow-missing-level-indices", action="store_true",
        help="仅单层特征调试使用；多层严格 PPAL 需要 boxes.lvl_inds",
    )
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("请先安装 ultralytics: pip install ultralytics") from exc

    image_paths = read_image_list(args.unlabeled_list)
    model = YOLO(args.model)
    kwargs = {
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "verbose": False,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    if args.embed_layers is None:
        raise SystemExit("严格 PPAL 需要显式指定 --embed-layers")
    predictions = predict_with_object_features(
        model,
        image_paths,
        image_ids=image_paths,
        embed_layers=args.embed_layers,
        require_level_indices=not args.allow_missing_level_indices,
        **kwargs,
    )
    selector = PPALSelector(
        budget=args.budget,
        candidate_multiplier=4,
        diversity_mode=OBJECT_FEATURES,
        seed=args.seed,
    )
    result = selector.select(predictions)
    write_image_list(args.output, result.selected_ids)
    print(f"未标注图片: {len(image_paths)}")
    print(f"不确定性候选池: {len(result.candidate_ids)}")
    print(f"本轮选中: {len(result.selected_ids)}")
    print(f"多样性特征: {result.diversity_mode}")
    print(f"输出: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

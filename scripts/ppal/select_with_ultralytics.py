"""使用 Ultralytics YOLO 原生 hook 完成 PPAL 选择。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    parser.add_argument("--output", required=True, help="本轮 PPAL 选中图片列表")
    parser.add_argument("--remaining-output", default=None, help="可选：本轮选择后剩余未标注池 txt")
    parser.add_argument("--labeled-list", default=None, help="可选：当前已标注训练集 txt")
    parser.add_argument("--updated-labeled-output", default=None, help="可选：已标注训练集 + 本轮选中结果 txt")
    parser.add_argument("--budget", required=True, type=int, help="标注预算")
    parser.add_argument("--device", default=None, help="例如 0、0,1 或 cpu")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--conf", default=0.05, type=float,
                        help="保留候选框的最低置信度")
    parser.add_argument("--candidate-multiplier", default=4, type=int, help="不确定性候选池大小 = budget * multiplier")
    parser.add_argument("--class-weights-json", default=None, help="可选：逐类别 PPAL 难度权重 JSON")
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
    predictions = predict_with_object_features(
        model,
        image_paths,
        image_ids=image_paths,
        **kwargs,
    )
    selector = PPALSelector(
        budget=args.budget,
        candidate_multiplier=args.candidate_multiplier,
        diversity_mode=OBJECT_FEATURES,
        seed=args.seed,
    )
    class_weights = None
    if args.class_weights_json:
        raw_weights = json.loads(Path(args.class_weights_json).read_text(encoding="utf-8"))
        class_weights = {int(key): float(value) for key, value in raw_weights.items()}
    result = selector.select(predictions, class_weights=class_weights)
    selected = list(result.selected_ids)
    selected_set = set(selected)
    remaining = [item for item in image_paths if item not in selected_set]

    write_image_list(args.output, selected)
    if args.remaining_output:
        write_image_list(args.remaining_output, remaining)
    if args.updated_labeled_output:
        labeled = read_image_list(args.labeled_list) if args.labeled_list else []
        write_image_list(args.updated_labeled_output, [*labeled, *selected])

    print(f"未标注输入: {len(image_paths)}")
    print(f"不确定性候选池: {len(result.candidate_ids)}")
    print(f"本轮选中: {len(selected)} -> {Path(args.output).resolve()}")
    if args.remaining_output:
        print(f"剩余未标注池: {len(remaining)} -> {Path(args.remaining_output).resolve()}")
    if args.updated_labeled_output:
        print(f"下一轮训练集: {len(read_image_list(args.updated_labeled_output))} -> {Path(args.updated_labeled_output).resolve()}")
    if args.class_weights_json:
        print(f"类别权重: {Path(args.class_weights_json).resolve()}")
    print(f"多样性特征: {result.diversity_mode}")


if __name__ == "__main__":
    main()

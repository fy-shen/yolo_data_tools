"""使用 Ultralytics YOLO 原生 hook 完成 PPAL 选择。"""

import argparse
from heapq import heappush, heapreplace
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning_yolo.adapters.ultralytics import (
    iter_predict_with_object_features,
)
from active_learning_yolo.data import read_image_list, write_image_list
from active_learning_yolo.ppal import OBJECT_FEATURES, PPALSelector
from active_learning_yolo.ppal.uncertainty import score_image_uncertainty


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
    parser.add_argument("--progress-interval", default=1000, type=int, help="流式筛选进度打印间隔；<=0 表示关闭")
    parser.add_argument("--predict-chunk-size", default=128, type=int, help="PPAL 推理每次传给 Ultralytics 的图片路径数量")
    parser.add_argument("--diversity-progress-interval", default=500000, type=int, help="多样性距离矩阵每处理多少图片对打印一次进度；<=0 表示关闭")
    parser.add_argument("--diversity-max-detections", default=0, type=int, help="多样性计算每张图最多使用多少个高置信框；<=0 表示不限制")
    parser.add_argument("--kmedoids-max-iter", default=100, type=int, help="k-medoids 最大迭代次数")
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def _load_class_weights(path: str | None) -> dict[int, float] | None:
    if path is None:
        return None
    raw_weights = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(key): float(value) for key, value in raw_weights.items()}


def _top_uncertain_predictions(
    model,
    image_paths: list[str],
    args: argparse.Namespace,
    predict_kwargs: dict,
    class_weights: dict[int, float] | None,
):
    candidate_limit = min(len(image_paths), args.budget * args.candidate_multiplier)
    if candidate_limit <= 0:
        return []

    heap = []
    progress_interval = int(args.progress_interval)
    prediction_iter = iter_predict_with_object_features(
        model,
        image_paths,
        image_ids=image_paths,
        predict_chunk_size=args.predict_chunk_size,
        **predict_kwargs,
    )
    for index, prediction in enumerate(prediction_iter, start=1):
        score = score_image_uncertainty(prediction, class_weights, args.conf)
        item = (score, index, prediction)
        if len(heap) < candidate_limit:
            heappush(heap, item)
        elif score > heap[0][0]:
            heapreplace(heap, item)

        if progress_interval > 0 and index % progress_interval == 0:
            print(
                f"stream select: processed={index}/{len(image_paths)} "
                f"top_candidates={len(heap)}/{candidate_limit}",
                flush=True,
            )

    return [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)]


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
    class_weights = _load_class_weights(args.class_weights_json)
    candidates = _top_uncertain_predictions(
        model, image_paths, args, kwargs, class_weights
    )
    selector = PPALSelector(
        budget=args.budget,
        candidate_multiplier=args.candidate_multiplier,
        diversity_mode=OBJECT_FEATURES,
        seed=args.seed,
        diversity_progress_interval=args.diversity_progress_interval,
        kmedoids_max_iter=args.kmedoids_max_iter,
        max_detections_per_image=(
            args.diversity_max_detections if args.diversity_max_detections > 0 else None
        ),
        progress_callback=lambda message: print(message, flush=True),
    )
    result = selector.select(candidates, class_weights=class_weights)
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

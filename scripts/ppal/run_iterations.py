"""自动执行 PPAL 多轮训练和数据选择。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from active_learning_yolo.adapters import class_qualities_from_metrics
from active_learning_yolo.data import read_image_list, write_image_list
from active_learning_yolo.ppal import compute_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPAL 自动多轮训练和选择")
    parser.add_argument("--data-template", required=True, help="原始 YOLO data yaml，保留 path/nc/names 等字段")
    parser.add_argument("--initial-train", required=True, help="round0 初始训练 txt")
    parser.add_argument("--initial-pool", required=True, help="round0 未标注池 txt")
    parser.add_argument("--val-list", default=None, help="固定验证集 txt；不传则沿用 data-template 的 val")
    parser.add_argument("--work-dir", required=True, help="输出所有轮次文件和训练结果的目录")
    parser.add_argument("--base-model", required=True, help="round0 训练使用的模型或 yaml/pt")
    parser.add_argument("--pretrained", default=None, help="可选：传给 YOLO.train(pretrained=...)，通常用于 base-model 是 yaml 的 round0")
    parser.add_argument(
        "--data-list-mode", choices=("absolute", "relative-to-data-path", "basename"), default="absolute",
        help="每轮 data.yaml 中 train/val txt 的写法；服务器数据根目录场景可用 relative-to-data-path 或 basename",
    )
    parser.add_argument(
        "--image-list-mode", choices=("absolute", "preserve"), default="absolute",
        help="写入每轮 train/pool txt 时如何处理图片路径；absolute 会把相对图片路径按 data-template.path 转为绝对路径",
    )
    parser.add_argument("--rounds", default=5, type=int, help="训练轮次数；round0 算第 1 次训练")
    parser.add_argument("--add-ratio", default=0.02, type=float, help="每轮新增数据比例")
    parser.add_argument(
        "--budget-base", choices=("total", "pool"), default="total",
        help="total 表示按初始 train+pool 总量计算每轮新增；pool 表示按当前剩余池计算",
    )
    parser.add_argument("--candidate-multiplier", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--patience", default=None, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", default=None, type=int)
    parser.add_argument(
        "--train-project", default=None,
        help="可选：YOLO 训练输出 project；默认每轮写入 work-dir/roundN",
    )
    parser.add_argument(
        "--train-name-prefix", default=None,
        help="可选：YOLO 训练 name 前缀；设置后每轮 name 为 <prefix>-roundN",
    )
    parser.add_argument("--conf", default=0.05, type=float, help="PPAL 推理保留框置信度")
    parser.add_argument("--select-batch", default=None, type=int, help="PPAL 推理 batch；默认沿用 --batch")
    parser.add_argument("--select-device", default=None, help="PPAL 推理设备；默认从 --device 取第一个设备")
    parser.add_argument("--select-progress-interval", default=1000, type=int, help="PPAL 流式筛选进度打印间隔；<=0 表示关闭")
    parser.add_argument("--select-predict-chunk-size", default=128, type=int, help="PPAL 推理每次传给 Ultralytics 的图片路径数量")
    parser.add_argument("--diversity-progress-interval", default=500000, type=int, help="多样性距离矩阵每处理多少图片对打印一次进度；<=0 表示关闭")
    parser.add_argument("--diversity-max-detections", default=0, type=int, help="多样性计算每张图最多使用多少个高置信框；<=0 表示不限制")
    parser.add_argument("--kmedoids-max-iter", default=100, type=int, help="k-medoids 最大迭代次数")
    parser.add_argument(
        "--class-weight-mode", choices=("map", "none"), default="map",
        help="map 表示每轮用 best.pt 在验证集上的逐类别 mAP 计算 PPAL 类别权重",
    )
    parser.add_argument("--class-weight-upper-bound", default=0.2, type=float)
    parser.add_argument("--class-weight-alpha", default=0.3, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--train-from", choices=("previous", "base"), default="previous",
        help="previous 用上一轮 best.pt 继续训练；base 每轮都从 base-model 重新训练",
    )
    parser.add_argument(
        "--train-arg", action="append", default=[], metavar="KEY=VALUE",
        help="额外传给 YOLO.train 的参数，可重复，例如 optimizer=auto lr0=0.01 cos_lr=True",
    )
    parser.add_argument("--dry-run", action="store_true", help="只生成计划和 data yaml，不执行训练/选择")
    return parser.parse_args()


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_train_args(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--train-arg 必须是 KEY=VALUE 格式: {item}")
        key, value = item.split("=", 1)
        parsed[key] = _parse_scalar(value)
    return parsed


def _load_template(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"data-template 格式无效: {path}")
    return data


def _format_list_path(path: Path, template: dict[str, Any], mode: str) -> str:
    resolved = path.resolve()
    if mode == "absolute":
        return str(resolved)
    if mode == "basename":
        return path.name
    data_root = template.get("path")
    if not data_root:
        raise SystemExit("--data-list-mode relative-to-data-path 需要 data-template 中包含 path 字段")
    return str(resolved.relative_to(Path(data_root).resolve()))


def _set_like_template(data: dict[str, Any], key: str, value: str) -> None:
    data[key] = [value] if isinstance(data.get(key), list) else value


def _data_root(template: dict[str, Any]) -> Path | None:
    value = template.get("path")
    return Path(value).resolve() if value else None


def _normalize_image_item(item: str, data_root: Path | None, mode: str) -> str:
    if mode == "preserve":
        return item
    path = Path(item)
    if path.is_absolute():
        return str(path)
    if data_root is None:
        raise SystemExit("--image-list-mode absolute 需要 data-template 中包含 path 字段，或改用 --image-list-mode preserve")
    # 每轮 train/pool txt 会写到 work-dir；相对图片路径必须先按数据集根目录解析，
    # 否则 Ultralytics 会按 txt 所在目录拼接，导致 work-dir 改变图片路径语义。
    return str((data_root / path).resolve())


def _read_images_for_work_list(path: str | Path, template: dict[str, Any], mode: str) -> list[str]:
    data_root = _data_root(template)
    return [_normalize_image_item(item, data_root, mode) for item in read_image_list(path)]


def _write_data_yaml(
    template: dict[str, Any],
    train_list: Path,
    val_list: str | None,
    output: Path,
    list_mode: str,
) -> None:
    data = dict(template)
    _set_like_template(data, "train", _format_list_path(train_list, template, list_mode))
    if val_list is not None:
        _set_like_template(data, "val", _format_list_path(Path(val_list), template, list_mode))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _train_round(args: argparse.Namespace, round_index: int, model_path: str, data_yaml: Path) -> Path:
    from ultralytics import YOLO

    round_dir = Path(args.work_dir) / f"round{round_index}"
    round_dir.mkdir(parents=True, exist_ok=True)

    if args.train_project is None:
        train_project = round_dir
        train_name = "train"
    else:
        train_project = Path(args.train_project)
        name_prefix = args.train_name_prefix or f"{Path(str(args.base_model)).stem}-{args.imgsz}"
        train_name = f"{name_prefix}-round{round_index}"

    train_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(train_project),
        "name": train_name,
        "exist_ok": True,
        "seed": args.seed,
    }
    if args.device is not None:
        train_args["device"] = args.device
    if args.workers is not None:
        train_args["workers"] = args.workers
    if args.patience is not None:
        train_args["patience"] = args.patience
    if args.pretrained is not None and (args.train_from == "base" or round_index == 0):
        train_args["pretrained"] = args.pretrained
    train_args.update(_parse_train_args(args.train_arg))

    print(f"[round{round_index}] train model={model_path} data={data_yaml}", flush=True)
    yolo = YOLO(model_path)
    yolo.train(**train_args)

    # 训练输出目录以 Ultralytics 实际 trainer.save_dir 为准，避免 project/name 被配置覆盖后找错权重。
    save_dir = Path(yolo.trainer.save_dir)
    (round_dir / "train_save_dir.txt").write_text(str(save_dir.resolve()) + "\n", encoding="utf-8")
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise RuntimeError(f"未找到训练权重: {best} 或 {last}")



def _first_device(device: str | None) -> str | None:
    if device is None:
        return None
    value = str(device)
    return value.split(",", 1)[0] if "," in value else value


def _validation_class_weights(
    args: argparse.Namespace,
    round_index: int,
    model_path: Path,
    data_yaml: Path,
) -> Path | None:
    if args.class_weight_mode == "none":
        return None

    from ultralytics import YOLO

    round_dir = Path(args.work_dir) / f"round{round_index}"
    val_args: dict[str, Any] = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(round_dir),
        "name": "val_class_weights",
        "exist_ok": True,
        "verbose": False,
    }
    device = args.select_device or _first_device(args.device)
    if device is not None:
        val_args["device"] = device
    if args.workers is not None:
        val_args["workers"] = args.workers

    print(f"[round{round_index}] val for class weights model={model_path}", flush=True)
    metrics = YOLO(str(model_path)).val(**val_args)
    qualities = class_qualities_from_metrics(metrics)
    weights = compute_class_weights(
        qualities,
        upper_bound=args.class_weight_upper_bound,
        alpha=args.class_weight_alpha,
    )

    round_dir.mkdir(parents=True, exist_ok=True)
    qualities_path = round_dir / "class_qualities.json"
    weights_path = round_dir / "class_weights.json"
    qualities_path.write_text(json.dumps(qualities, ensure_ascii=False, indent=2), encoding="utf-8")
    weights_path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[round{round_index}] class weights -> {weights_path}", flush=True)
    return weights_path

def _run_selection(
    args: argparse.Namespace,
    round_index: int,
    model_path: Path,
    train_list: Path,
    pool_list: Path,
    next_train: Path,
    next_pool: Path,
    selected_output: Path,
    budget: int,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ppal" / "select_with_ultralytics.py"),
        "--model", str(model_path),
        "--unlabeled-list", str(pool_list),
        "--labeled-list", str(train_list),
        "--output", str(selected_output),
        "--remaining-output", str(next_pool),
        "--updated-labeled-output", str(next_train),
        "--budget", str(budget),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.select_batch or args.batch),
        "--conf", str(args.conf),
        "--candidate-multiplier", str(args.candidate_multiplier),
        "--progress-interval", str(args.select_progress_interval),
        "--predict-chunk-size", str(args.select_predict_chunk_size),
        "--diversity-progress-interval", str(args.diversity_progress_interval),
        "--diversity-max-detections", str(args.diversity_max_detections),
        "--kmedoids-max-iter", str(args.kmedoids_max_iter),
        "--seed", str(args.seed + round_index),
    ]
    select_device = args.select_device or _first_device(args.device)
    if select_device is not None:
        cmd.extend(["--device", str(select_device)])
    class_weights = getattr(args, "_current_class_weights", None)
    if class_weights is not None:
        cmd.extend(["--class-weights-json", str(class_weights)])
    print(f"[round{round_index}] select budget={budget} pool={pool_list}", flush=True)
    subprocess.run(cmd, check=True)


def _budget(args: argparse.Namespace, total_count: int, pool_count: int) -> int:
    base = total_count if args.budget_base == "total" else pool_count
    return max(1, min(pool_count, round(base * args.add_ratio)))


def main() -> None:
    args = parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds 必须 >= 1")
    if not (0 < args.add_ratio < 1):
        raise SystemExit("--add-ratio 必须在 0 到 1 之间")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    template = _load_template(Path(args.data_template))

    round_train = work_dir / "round0_train.txt"
    round_pool = work_dir / "round0_pool.txt"
    write_image_list(round_train, _read_images_for_work_list(args.initial_train, template, args.image_list_mode))
    write_image_list(round_pool, _read_images_for_work_list(args.initial_pool, template, args.image_list_mode))

    initial_count = len(read_image_list(round_train))
    pool_count = len(read_image_list(round_pool))
    total_count = initial_count + pool_count
    print(f"初始训练集: {initial_count}")
    print(f"初始未标注池: {pool_count}")
    print(f"估计总训练池: {total_count}")

    current_model = args.base_model
    trained_model: Path | None = None
    for round_index in range(args.rounds):
        train_list = work_dir / f"round{round_index}_train.txt"
        pool_list = work_dir / f"round{round_index}_pool.txt"
        data_yaml = work_dir / f"round{round_index}" / "data.yaml"
        _write_data_yaml(template, train_list, args.val_list, data_yaml, args.data_list_mode)

        if args.dry_run:
            print(f"[dry-run] round{round_index} train={train_list} data={data_yaml}")
            trained_model = Path(current_model)
        else:
            model_for_train = current_model if args.train_from == "previous" else args.base_model
            trained_model = _train_round(args, round_index, model_for_train, data_yaml)
            current_model = str(trained_model)

        class_weights_path = None
        if not args.dry_run and trained_model is not None:
            class_weights_path = _validation_class_weights(args, round_index, trained_model, data_yaml)
        setattr(args, "_current_class_weights", class_weights_path)

        if round_index == args.rounds - 1:
            break

        pool_items = read_image_list(pool_list)
        budget = _budget(args, total_count, len(pool_items))
        next_train = work_dir / f"round{round_index + 1}_train.txt"
        next_pool = work_dir / f"round{round_index + 1}_pool.txt"
        selected = work_dir / f"round{round_index + 1}_selected.txt"
        if args.dry_run:
            current_train = read_image_list(train_list)
            selected_items = pool_items[:budget]
            selected_set = set(selected_items)
            write_image_list(selected, selected_items)
            write_image_list(next_train, [*current_train, *selected_items])
            write_image_list(next_pool, [item for item in pool_items if item not in selected_set])
            print(f"[dry-run] round{round_index} select budget={budget}")
        else:
            assert trained_model is not None
            _run_selection(args, round_index, trained_model, train_list, pool_list, next_train, next_pool, selected, budget)

    final_train = work_dir / f"round{args.rounds - 1}_train.txt"
    print(f"完成。最后训练列表: {final_train.resolve()} ({len(read_image_list(final_train))} images)")


if __name__ == "__main__":
    main()

"""从已有 train txt 划分 PPAL 初始训练集和未标注池。"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning_yolo.data import read_image_list, write_image_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="划分 PPAL 初始训练集和未标注池")
    parser.add_argument(
        "--train-list", action="append", default=[],
        help="训练集 txt，可重复传入多个",
    )
    parser.add_argument(
        "--train-glob", default=None,
        help="训练集 txt glob，例如 '/data/football/*-train.txt'",
    )
    parser.add_argument(
        "--dataset-root", default=None,
        help="相对路径的基准目录；默认使用每个 txt 文件所在目录",
    )
    parser.add_argument("--initial-output", required=True, help="初始训练集输出 txt")
    parser.add_argument("--pool-output", required=True, help="剩余 PPAL 未标注池输出 txt")
    parser.add_argument("--all-output", default=None, help="可选：合并去重后的全训练集输出 txt")
    parser.add_argument("--val-list", action="append", default=[], help="验证集 txt，可重复传入多个")
    parser.add_argument("--val-glob", default=None, help="验证集 txt glob，例如 '/data/football/*-val.txt'")
    parser.add_argument("--val-output", default=None, help="可选：合并去重后的全验证集输出 txt")
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--initial-size", type=int, help="初始训练集图片数量")
    size_group.add_argument("--initial-ratio", type=float, help="初始训练集比例，例如 0.02")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--path-mode", choices=("absolute", "preserve"), default="absolute",
        help="absolute 会把相对路径按 dataset-root/txt 所在目录转成绝对路径；preserve 保留原字符串",
    )
    parser.add_argument(
        "--no-stratify-source", action="store_true",
        help="不按来源 txt 等比例抽样，改为全局随机抽样",
    )
    return parser.parse_args()


def _collect_lists(items: list[str], pattern: str | None) -> list[Path]:
    import glob

    paths = [Path(item) for item in items]
    if pattern:
        paths.extend(Path(item) for item in sorted(glob.glob(pattern)))
    unique: list[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _collect_train_lists(args: argparse.Namespace) -> list[Path]:
    paths = _collect_lists(args.train_list, args.train_glob)
    if not paths:
        raise SystemExit("请通过 --train-list 或 --train-glob 指定训练集 txt")
    return paths


def _read_unique_images(
    list_paths: list[Path],
    dataset_root: Path | None,
    path_mode: str,
) -> list[str]:
    images: list[str] = []
    seen = set()
    for list_path in list_paths:
        for raw in read_image_list(list_path):
            image_path = _normalize_path(raw, list_path, dataset_root, path_mode)
            if image_path in seen:
                continue
            seen.add(image_path)
            images.append(image_path)
    return images


def _normalize_path(raw: str, list_path: Path, dataset_root: Path | None, path_mode: str) -> str:
    if path_mode == "preserve":
        return raw
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    base = dataset_root if dataset_root is not None else list_path.parent
    return str((base / path).resolve())


def _target_initial_count(total: int, args: argparse.Namespace) -> int:
    if args.initial_size is not None:
        if args.initial_size <= 0:
            raise SystemExit("--initial-size 必须大于 0")
        return min(args.initial_size, total)
    if not (0 < args.initial_ratio < 1):
        raise SystemExit("--initial-ratio 必须在 0 到 1 之间")
    return max(1, min(total, round(total * args.initial_ratio)))


def _stratified_sample(groups: dict[str, list[str]], n: int, rng: random.Random) -> list[str]:
    total = sum(len(items) for items in groups.values())
    quotas: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for name, items in groups.items():
        exact = n * len(items) / total
        quota = min(len(items), int(exact))
        quotas[name] = quota
        fractions.append((exact - quota, name))

    remaining = n - sum(quotas.values())
    for _fraction, name in sorted(fractions, reverse=True):
        if remaining <= 0:
            break
        if quotas[name] < len(groups[name]):
            quotas[name] += 1
            remaining -= 1

    selected: list[str] = []
    for name, items in groups.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:quotas[name]])
    rng.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
    train_lists = _collect_train_lists(args)

    groups: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for list_path in train_lists:
        for raw in read_image_list(list_path):
            image_path = _normalize_path(raw, list_path, dataset_root, args.path_mode)
            if image_path in seen:
                continue
            seen.add(image_path)
            groups[str(list_path)].append(image_path)

    all_images = [item for items in groups.values() for item in items]
    if not all_images:
        raise SystemExit("训练集 txt 为空")

    initial_count = _target_initial_count(len(all_images), args)
    if args.no_stratify_source:
        shuffled = list(all_images)
        rng.shuffle(shuffled)
        initial = shuffled[:initial_count]
    else:
        initial = _stratified_sample(groups, initial_count, rng)

    initial_set = set(initial)
    pool = [item for item in all_images if item not in initial_set]

    write_image_list(args.initial_output, initial)
    write_image_list(args.pool_output, pool)
    if args.all_output:
        write_image_list(args.all_output, all_images)

    val_lists = _collect_lists(args.val_list, args.val_glob)
    val_images = _read_unique_images(val_lists, dataset_root, args.path_mode) if val_lists else []
    if args.val_output:
        if not val_lists:
            raise SystemExit("指定 --val-output 时必须同时指定 --val-list 或 --val-glob")
        write_image_list(args.val_output, val_images)

    print(f"输入 train txt: {len(train_lists)}")
    print(f"全训练集去重: {len(all_images)}")
    print(f"初始训练集: {len(initial)} -> {Path(args.initial_output).resolve()}")
    print(f"PPAL 未标注池: {len(pool)} -> {Path(args.pool_output).resolve()}")
    if args.all_output:
        print(f"全训练集: {Path(args.all_output).resolve()}")
    if args.val_output:
        print(f"全验证集去重: {len(val_images)} -> {Path(args.val_output).resolve()}")


if __name__ == "__main__":
    main()

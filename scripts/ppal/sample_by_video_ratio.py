"""按视频来源占比从 YOLO 图片列表中随机抽样。"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning_yolo.data import read_image_list, write_image_list

DEFAULT_VIDEO_KEY_PATTERN = re.compile(r"^(?P<video>.+)_\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按视频占比分层随机抽样 YOLO 图片 txt")
    parser.add_argument("--input", required=True, help="输入图片列表 txt，一行一个图片路径")
    parser.add_argument("--output", required=True, help="抽样结果输出 txt")
    parser.add_argument("--remain-out", default=None, help="可选：未被抽中的剩余图片输出 txt")
    parser.add_argument("--ratio", default=0.1, type=float, help="抽样比例，默认 0.1")
    parser.add_argument("--seed", default=0, type=int, help="随机种子，默认 0")
    parser.add_argument(
        "--keep-input-order", action="store_true",
        help="输出时保持输入 txt 的相对顺序；默认会打乱输出",
    )
    parser.add_argument(
        "--video-key-regex", default=None,
        help=(
            "可选：从不含扩展名的文件名中提取视频名的正则，必须包含命名分组 video；"
            "默认移除末尾的 _数字帧索引"
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not (0 < args.ratio < 1):
        raise SystemExit("--ratio 必须在 0 到 1 之间，例如 0.1")
    if args.video_key_regex is not None:
        try:
            pattern = re.compile(args.video_key_regex)
        except re.error as exc:
            raise SystemExit(f"--video-key-regex 不是有效正则: {exc}") from exc
        if "video" not in pattern.groupindex:
            raise SystemExit("--video-key-regex 必须包含命名分组 (?P<video>...)")


def _video_key(image_path: str, pattern: re.Pattern[str] | None) -> str:
    stem = Path(image_path).stem
    pattern = pattern or DEFAULT_VIDEO_KEY_PATTERN
    match = pattern.match(stem)
    if match:
        return match.group("video")
    return stem


def _target_count(total: int, ratio: float) -> int:
    return max(1, min(total, round(total * ratio)))


def _allocate_counts(groups: dict[str, list[int]], target: int) -> dict[str, int]:
    total = sum(len(items) for items in groups.values())
    quotas: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []

    for name, items in groups.items():
        exact = target * len(items) / total
        quota = min(len(items), int(exact))
        quotas[name] = quota
        fractions.append((exact - quota, name))

    remaining = target - sum(quotas.values())
    for _fraction, name in sorted(fractions, reverse=True):
        if remaining <= 0:
            break
        if quotas[name] < len(groups[name]):
            quotas[name] += 1
            remaining -= 1

    return quotas


def _sample_indices_by_group(
    groups: dict[str, list[int]],
    quotas: dict[str, int],
    rng: random.Random,
) -> list[int]:
    selected_indices: list[int] = []
    for name, indices in groups.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        selected_indices.extend(shuffled[:quotas[name]])
    return selected_indices


def _print_stats(groups: dict[str, list[int]], quotas: dict[str, int], total: int, selected: int) -> None:
    print(f"total images: {total}")
    print(f"selected images: {selected}")
    print("video stats:")
    for name in sorted(groups):
        count = len(groups[name])
        quota = quotas[name]
        source_ratio = count / total
        selected_ratio = quota / selected if selected else 0
        print(
            f"  {name}: total={count}, selected={quota}, "
            f"source_ratio={source_ratio:.4f}, selected_ratio={selected_ratio:.4f}"
        )


def main() -> None:
    args = parse_args()
    _validate_args(args)

    pattern = re.compile(args.video_key_regex) if args.video_key_regex else None
    images = read_image_list(args.input)
    if not images:
        raise SystemExit("输入 txt 为空")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, image in enumerate(images):
        groups[_video_key(image, pattern)].append(index)

    target = _target_count(len(images), args.ratio)
    quotas = _allocate_counts(groups, target)

    rng = random.Random(args.seed)
    selected_indices = _sample_indices_by_group(groups, quotas, rng)
    selected_index_set = set(selected_indices)

    if args.keep_input_order:
        selected = [image for index, image in enumerate(images) if index in selected_index_set]
    else:
        rng.shuffle(selected_indices)
        selected = [images[index] for index in selected_indices]

    write_image_list(args.output, selected)

    if args.remain_out:
        remaining = [image for index, image in enumerate(images) if index not in selected_index_set]
        write_image_list(args.remain_out, remaining)

    _print_stats(groups, quotas, len(images), len(selected))
    print(f"save selected: {len(selected)} -> {Path(args.output).resolve()}")
    if args.remain_out:
        print(f"save remaining: {len(images) - len(selected)} -> {Path(args.remain_out).resolve()}")


if __name__ == "__main__":
    main()

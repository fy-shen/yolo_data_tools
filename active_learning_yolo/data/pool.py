"""标注池状态和 Ultralytics 图片列表管理。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..ppal.schemas import ImageId


@dataclass
class AnnotationPool:
    """维护已标注、未标注和正在标注的样本。"""

    labeled: list[ImageId] = field(default_factory=list)
    unlabeled: list[ImageId] = field(default_factory=list)
    pending: list[ImageId] = field(default_factory=list)

    def request_annotation(self, selected_ids: Iterable[ImageId]) -> None:
        selected = list(selected_ids)
        missing = [item for item in selected if item not in self.unlabeled]
        if missing:
            raise ValueError(f"样本不在未标注池中: {missing[:5]}")
        selected_set = set(selected)
        self.unlabeled = [x for x in self.unlabeled if x not in selected_set]
        self.pending.extend(selected)

    def mark_labeled(self, completed_ids: Iterable[ImageId]) -> None:
        completed = list(completed_ids)
        missing = [item for item in completed if item not in self.pending]
        if missing:
            raise ValueError(f"样本不在待标注池中: {missing[:5]}")
        completed_set = set(completed)
        self.pending = [x for x in self.pending if x not in completed_set]
        self.labeled.extend(completed)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "labeled": self.labeled,
            "unlabeled": self.unlabeled,
            "pending": self.pending,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AnnotationPool":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data.get("labeled", []), data.get("unlabeled", []),
                   data.get("pending", []))


def read_image_list(path: str | Path) -> list[str]:
    """读取一行一个图片路径的 txt 文件。"""
    return [line.strip() for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line.strip()]


def write_image_list(path: str | Path, image_paths: Iterable[str]) -> None:
    """写出可供 Ultralytics 数据 YAML 引用的图片列表。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(map(str, image_paths))
    output.write_text(content + ("\n" if content else ""), encoding="utf-8")

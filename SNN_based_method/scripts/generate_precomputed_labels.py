"""根据 CSV 中 ``fog_level=0`` 的 raw 为每个类别生成小型 label 池。

CLI example:
    python SNN_based_method/scripts/generate_precomputed_labels.py ^
        --data-paths D:/PYproject/SPADdata/0825 D:/PYproject/SPADdata/0826 ^
        --csv-paths D:/PYproject/SPADdata/0825/0825-group.csv D:/PYproject/SPADdata/0826/0826-group.csv ^
        --pages-per-group 128 --overwrite

Non-CLI example:
    python SNN_based_method/scripts/generate_precomputed_labels.py

输出约定:
    - 仅使用 CSV 中 ``fog_level=0`` 的 raw 文件。
    - 每个 ``target_class`` 只取对应 clean raw 的最后 5 个完整 group。
    - 文件结构为 ``<dataset>/label/<pages_per_group>/<class>/<class>_<index>.npy``。
    - 每个文件形状为 ``(2, 64, 64)``, 通道 0 为 depth, 通道 1 为 intensity。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from SNN_based_method.scripts.augment import clip_tof_to_valid_range
from SNN_based_method.scripts.data import (
    DEFAULT_LABEL_DIR_NAME,
    build_weak_label_from_group,
    collect_raw_records,
    infer_groupable_total_pages,
    read_raw_group,
    sanitize_label_class,
)


DEFAULT_LABELS_PER_CLASS = 5


@dataclass(frozen=True)
class GenerateLabelConfig:
    """预生成 label 池的运行配置。"""

    data_paths: Sequence[Path]
    csv_paths: Sequence[Path]
    pages_per_group: int = 128
    total_pages: int | None = None
    time_threshold: int = 128
    active_point: int = 1
    label_dir_name: str | Path = DEFAULT_LABEL_DIR_NAME
    labels_per_class: int = DEFAULT_LABELS_PER_CLASS
    recursive: bool = False
    skip_missing_csv_raw: bool = False
    overwrite: bool = False
    dry_run: bool = False
    progress_interval: int = 50


@dataclass(frozen=True)
class LabelSource:
    """一个类别 label 池的来源 clean raw。"""

    class_name: str
    raw_path: Path
    label_root: Path
    label_dir: Path
    num_groups: int
    group_indices: tuple[int, ...]


@dataclass
class GenerateLabelStats:
    """预生成 label 池的统计结果。"""

    raw_records: int = 0
    clean_records: int = 0
    classes: int = 0
    planned: int = 0
    generated: int = 0
    skipped_existing: int = 0
    label_roots: set[str] | None = None

    def __post_init__(self) -> None:
        if self.label_roots is None:
            self.label_roots = set()

    def as_dict(self) -> dict[str, object]:
        """转换为便于打印的普通字典。"""
        return {
            "raw_records": self.raw_records,
            "clean_records": self.clean_records,
            "classes": self.classes,
            "planned": self.planned,
            "generated": self.generated,
            "skipped_existing": self.skipped_existing,
            "label_roots": sorted(self.label_roots or set()),
        }


def label_root_from_record(
    raw_path: Path,
    csv_path_value: object,
    *,
    label_dir_name: str | Path,
    pages_per_group: int,
) -> Path:
    """解析 label 根目录: ``<dataset>/label/<pages_per_group>``。"""
    label_dir = Path(label_dir_name)
    if label_dir.is_absolute():
        root = label_dir
    elif csv_path_value:
        root = Path(str(csv_path_value)).resolve().parent / label_dir
    else:
        root = raw_path.resolve().parent / label_dir
    return root / str(int(pages_per_group))


def label_pool_dir(
    raw_path: Path,
    csv_path_value: object,
    class_name: str,
    *,
    label_dir_name: str | Path,
    pages_per_group: int,
) -> Path:
    """返回某个类别的 label 池目录。"""
    return (
        label_root_from_record(
            raw_path,
            csv_path_value,
            label_dir_name=label_dir_name,
            pages_per_group=pages_per_group,
        )
        / class_name
    )


def expected_label_paths_for_class(
    raw_path: Path,
    csv_path_value: object,
    target_class: object,
    *,
    pages_per_group: int,
    label_dir_name: str | Path = DEFAULT_LABEL_DIR_NAME,
    labels_per_class: int = DEFAULT_LABELS_PER_CLASS,
) -> list[Path]:
    """返回指定类别在当前数据集目录下应存在的 label 文件路径。"""
    class_name = sanitize_label_class(target_class)
    class_dir = label_pool_dir(
        Path(raw_path),
        csv_path_value,
        class_name,
        label_dir_name=label_dir_name,
        pages_per_group=pages_per_group,
    )
    return [class_dir / f"{class_name}_{index}.npy" for index in range(labels_per_class)]


def discover_expected_label_paths(
    data_paths: Sequence[str | Path],
    csv_paths: Sequence[str | Path],
    *,
    pages_per_group: int,
    label_dir_name: str | Path = DEFAULT_LABEL_DIR_NAME,
    labels_per_class: int = DEFAULT_LABELS_PER_CLASS,
    recursive: bool = False,
    skip_missing_csv_raw: bool = False,
) -> list[Path]:
    """从 CSV 推断本次训练需要的所有类别 label 文件。"""
    raw_records = collect_raw_records(
        data_paths,
        csv_paths=csv_paths,
        recursive=recursive,
        skip_missing_csv_raw=skip_missing_csv_raw,
    )
    expected: dict[str, Path] = {}
    for raw_path, metadata in raw_records:
        target_class = metadata.get("target_class")
        if target_class is None or str(target_class).strip() == "":
            continue
        for label_path in expected_label_paths_for_class(
            raw_path,
            metadata.get("csv_path"),
            target_class,
            pages_per_group=pages_per_group,
            label_dir_name=label_dir_name,
            labels_per_class=labels_per_class,
        ):
            expected[str(label_path.resolve()).lower()] = label_path
    return [expected[key] for key in sorted(expected)]


def labels_are_ready(
    data_paths: Sequence[str | Path],
    csv_paths: Sequence[str | Path],
    *,
    pages_per_group: int,
    label_dir_name: str | Path = DEFAULT_LABEL_DIR_NAME,
    labels_per_class: int = DEFAULT_LABELS_PER_CLASS,
    recursive: bool = False,
    skip_missing_csv_raw: bool = False,
) -> bool:
    """检查当前 pages_per_group 下训练需要的 label 是否都已存在。"""
    expected_paths = discover_expected_label_paths(
        data_paths,
        csv_paths,
        pages_per_group=pages_per_group,
        label_dir_name=label_dir_name,
        labels_per_class=labels_per_class,
        recursive=recursive,
        skip_missing_csv_raw=skip_missing_csv_raw,
    )
    return bool(expected_paths) and all(path.is_file() for path in expected_paths)


def ensure_precomputed_labels(config: GenerateLabelConfig) -> GenerateLabelStats:
    """若 label 池缺失则生成; 已完整存在时只返回 skipped 统计。"""
    expected_paths = discover_expected_label_paths(
        config.data_paths,
        config.csv_paths,
        pages_per_group=config.pages_per_group,
        label_dir_name=config.label_dir_name,
        labels_per_class=config.labels_per_class,
        recursive=config.recursive,
        skip_missing_csv_raw=config.skip_missing_csv_raw,
    )
    if expected_paths and all(path.is_file() for path in expected_paths) and not config.overwrite:
        stats = GenerateLabelStats(
            planned=len(expected_paths),
            skipped_existing=len(expected_paths),
        )
        stats.label_roots = {
            str(path.parents[1])
            for path in expected_paths
            if len(path.parents) >= 2
        }
        return stats
    return run_with_config(config)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="为每个 target_class 使用 fog_level=0 raw 的最后几组生成 label 池",
    )
    parser.add_argument(
        "--data-paths",
        nargs="+",
        type=Path,
        required=True,
        help="raw 文件或 raw 所在目录, 用于解析 CSV 中的 file_path",
    )
    parser.add_argument(
        "--csv-paths",
        nargs="+",
        type=Path,
        required=True,
        help="CSV 标注文件路径, 需包含 file_path/fog_level/target_class 列",
    )
    parser.add_argument("--pages-per-group", type=int, default=128, help="每个 label 对应的 raw page 数")
    parser.add_argument("--total-pages", type=int, default=None, help="每个 raw 文件最多使用的 page 数")
    parser.add_argument("--time-threshold", type=int, default=128, help="有效 ToF 上限")
    parser.add_argument("--active-point", type=int, default=1, help="N3 过滤所需的最小重复计数")
    parser.add_argument(
        "--label-dir-name",
        default=DEFAULT_LABEL_DIR_NAME,
        help="label 目录名; 相对路径按 CSV 所在目录解析, 默认 label",
    )
    parser.add_argument(
        "--labels-per-class",
        type=int,
        default=DEFAULT_LABELS_PER_CLASS,
        help="每个类别从 clean raw 最后取几个 group 生成 label, 默认 5",
    )
    parser.add_argument("--recursive", action="store_true", help="递归搜索 data-paths 中的 raw 文件")
    parser.add_argument(
        "--skip-missing-csv-raw",
        action="store_true",
        help="CSV 中 raw 缺失时跳过该行; 默认严格报错",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 .npy label")
    parser.add_argument("--dry-run", action="store_true", help="只打印将生成的 label 信息, 不写文件")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="每处理 N 个 label 打印一次进度",
    )
    return parser


def _is_clean_fog_level(value: object) -> bool:
    """判断 CSV fog_level 是否为 0。"""
    try:
        return float(str(value).strip()) == 0.0
    except ValueError:
        return str(value).strip().lower() in {"clean", "none"}


def _validate_config(config: GenerateLabelConfig) -> None:
    """在长时间生成前验证关键参数。"""
    if config.pages_per_group <= 0:
        raise ValueError("pages_per_group must be a positive integer")
    if config.total_pages is not None and config.total_pages <= 0:
        raise ValueError("total_pages must be positive when provided")
    if config.time_threshold <= 0:
        raise ValueError("time_threshold must be a positive integer")
    if config.active_point <= 0:
        raise ValueError("active_point must be a positive integer")
    if config.labels_per_class <= 0:
        raise ValueError("labels_per_class must be a positive integer")
    if not config.data_paths:
        raise ValueError("data_paths must not be empty")
    if not config.csv_paths:
        raise ValueError("csv_paths must not be empty")


def _collect_label_sources(config: GenerateLabelConfig) -> tuple[list[LabelSource], GenerateLabelStats]:
    """收集每个数据集目录下每个 class 的 clean raw 来源。"""
    raw_records = collect_raw_records(
        config.data_paths,
        csv_paths=config.csv_paths,
        recursive=config.recursive,
        skip_missing_csv_raw=config.skip_missing_csv_raw,
    )
    stats = GenerateLabelStats(raw_records=len(raw_records))
    class_sources: dict[tuple[str, str], LabelSource] = {}
    clean_counts: defaultdict[tuple[str, str], int] = defaultdict(int)

    for raw_path_value, metadata in raw_records:
        raw_path = Path(raw_path_value).resolve()
        target_class = metadata.get("target_class")
        if target_class is None or str(target_class).strip() == "":
            continue
        class_name = sanitize_label_class(target_class)
        label_root = label_root_from_record(
            raw_path,
            metadata.get("csv_path"),
            label_dir_name=config.label_dir_name,
            pages_per_group=config.pages_per_group,
        )
        source_key = (str(label_root.resolve()).lower(), class_name)
        stats.label_roots.add(str(label_root))

        if not _is_clean_fog_level(metadata.get("fog_level", "")):
            continue

        clean_counts[source_key] += 1
        if source_key in class_sources:
            continue

        resolved_pages = infer_groupable_total_pages(
            raw_path=raw_path,
            pages_per_group=config.pages_per_group,
            total_pages=config.total_pages,
        )
        num_groups = resolved_pages // int(config.pages_per_group)
        if num_groups < config.labels_per_class:
            raise ValueError(
                f"clean raw does not have enough groups for class {class_name}: "
                f"groups={num_groups}, required={config.labels_per_class}, raw={raw_path}"
            )
        group_indices = tuple(range(num_groups - config.labels_per_class, num_groups))
        class_sources[source_key] = LabelSource(
            class_name=class_name,
            raw_path=raw_path,
            label_root=label_root,
            label_dir=label_root / class_name,
            num_groups=num_groups,
            group_indices=group_indices,
        )

    stats.clean_records = sum(clean_counts.values())
    stats.classes = len(class_sources)

    all_class_keys = {
        (
            str(
                label_root_from_record(
                    Path(raw_path).resolve(),
                    metadata.get("csv_path"),
                    label_dir_name=config.label_dir_name,
                    pages_per_group=config.pages_per_group,
                ).resolve()
            ).lower(),
            sanitize_label_class(metadata["target_class"]),
        )
        for raw_path, metadata in raw_records
        if metadata.get("target_class") is not None
        and str(metadata.get("target_class")).strip() != ""
    }
    missing_clean = sorted(all_class_keys - set(class_sources))
    if missing_clean:
        missing_text = ", ".join(f"{root}/{class_name}" for root, class_name in missing_clean[:10])
        raise ValueError(
            "missing fog_level=0 raw for target_class label generation: "
            f"{missing_text}"
        )

    return list(class_sources.values()), stats


def _save_label(label_path: Path, label_array: np.ndarray, *, overwrite: bool, dry_run: bool) -> str:
    """保存单个 label, 返回 generated/skipped_existing/dry_run。"""
    if label_array.shape != (2, 64, 64):
        raise ValueError(f"label must have shape (2, 64, 64), got {label_array.shape}")
    if dry_run:
        return "dry_run"
    if label_path.exists() and not overwrite:
        return "skipped_existing"

    label_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(label_path, label_array.astype(np.float32, copy=False))
    return "generated"


def run_with_config(config: GenerateLabelConfig) -> GenerateLabelStats:
    """执行 label 池预生成流程。"""
    _validate_config(config)
    label_sources, stats = _collect_label_sources(config)
    progress_interval = max(1, int(config.progress_interval))
    processed = 0

    for source in sorted(label_sources, key=lambda item: (str(item.label_root), item.class_name)):
        for output_index, group_index in enumerate(source.group_indices):
            label_path = source.label_dir / f"{source.class_name}_{output_index}.npy"
            stats.planned += 1
            processed += 1
            if config.dry_run:
                print(
                    "example "
                    f"class={source.class_name} raw={source.raw_path} group={group_index} "
                    f"label={label_path} shape=(2,64,64)"
                )
                continue

            if label_path.exists() and not config.overwrite:
                stats.skipped_existing += 1
                continue

            raw_group = read_raw_group(
                source.raw_path,
                group_index=group_index,
                pages_per_group=config.pages_per_group,
                total_pages=source.num_groups * int(config.pages_per_group),
            )
            clipped_group = clip_tof_to_valid_range(raw_group, config.time_threshold)
            label_tensor = build_weak_label_from_group(
                clipped_group,
                config.time_threshold,
                active_point=config.active_point,
            )
            result = _save_label(
                label_path,
                label_tensor.numpy().astype(np.float32, copy=False),
                overwrite=config.overwrite,
                dry_run=False,
            )
            if result == "generated":
                stats.generated += 1
            elif result == "skipped_existing":
                stats.skipped_existing += 1

            if processed % progress_interval == 0:
                print(
                    f"processed={processed} generated={stats.generated} "
                    f"skipped_existing={stats.skipped_existing}"
                )

    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = GenerateLabelConfig(
        data_paths=args.data_paths,
        csv_paths=args.csv_paths,
        pages_per_group=args.pages_per_group,
        total_pages=args.total_pages,
        time_threshold=args.time_threshold,
        active_point=args.active_point,
        label_dir_name=args.label_dir_name,
        labels_per_class=args.labels_per_class,
        recursive=args.recursive,
        skip_missing_csv_raw=args.skip_missing_csv_raw,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        progress_interval=args.progress_interval,
    )
    stats = run_with_config(config)
    print(stats.as_dict())
    return 0


def main_without_cli() -> None:
    """无 CLI 调试入口; 默认只做 dry-run, 避免误写文件。"""
    # ===== Editable parameters =====
    data_paths = [
        Path(r"D:/PYproject/SPADdata/0825"),
        Path(r"D:/PYproject/SPADdata/0826"),
    ]
    csv_paths = [
        Path(r"D:/PYproject/SPADdata/0825/0825-group.csv"),
        Path(r"D:/PYproject/SPADdata/0826/0826-group.csv"),
    ]
    pages_per_group = 128
    total_pages = None
    time_threshold = 128
    active_point = 1
    label_dir_name = DEFAULT_LABEL_DIR_NAME
    labels_per_class = DEFAULT_LABELS_PER_CLASS
    overwrite = False
    dry_run = True

    # ===== Intermediate variables =====
    resolved_data_paths = [Path(path) for path in data_paths]
    resolved_csv_paths = [Path(path) for path in csv_paths]

    config = GenerateLabelConfig(
        data_paths=resolved_data_paths,
        csv_paths=resolved_csv_paths,
        pages_per_group=pages_per_group,
        total_pages=total_pages,
        time_threshold=time_threshold,
        active_point=active_point,
        label_dir_name=label_dir_name,
        labels_per_class=labels_per_class,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    stats = run_with_config(config)
    print(stats.as_dict())


if __name__ == "__main__":
    # Usage examples:
    #   python SNN_based_method/scripts/generate_precomputed_labels.py
    #       默认 dry-run, 打印每个 class 计划生成的最后 5 个 group。
    #   python SNN_based_method/scripts/generate_precomputed_labels.py --data-paths D:/data/0825 --csv-paths D:/data/0825/0825-group.csv
    #       生成 label/128/<class>/<class>_0.npy ... <class>_4.npy。
    #
    # Common parameters:
    #   --pages-per-group 128       必须与训练配置一致。
    #   --time-threshold 128        必须与训练配置一致。
    #   --label-dir-name label      相对路径放在每个 CSV 所在目录下。
    #   --labels-per-class 5        每个 class 用 clean raw 最后 5 组生成 label。
    #   --dry-run                   只打印计划, 不创建目录或文件。
    #
    # Outputs:
    #   <csv_parent>/label/<pages_per_group>/A/A_0.npy ... A_4.npy
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()

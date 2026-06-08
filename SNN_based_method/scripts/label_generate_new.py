"""基于目标/雾/背景 bin 先验生成 SPAD SNN label 池。

CLI example:
    python SNN_based_method/scripts/label_generate_new.py ^
        --pages-per-group 960 --label-dir-name label_prior --overwrite

Non-CLI example:
    python SNN_based_method/scripts/label_generate_new.py

输出约定:
    - 默认使用 SNNConfig 与训练脚本相同的 0825/0826 数据路径。
    - 文件结构为 ``<dataset>/label_prior/<P>/<class>/<class>_<index>.npy``。
    - 每个 label 为 ``(2, 64, 64)``: channel 0 为 depth, channel 1 为 confidence。
    - 非目标或低置信区域 depth/confidence 均为 0, 便于训练时使用 mask。
    - 诊断信息保存到 ``<dataset>/label_prior_debug/<P>/...``。
"""

from __future__ import annotations

import argparse
import csv
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

from SNN_based_method.config.SNN_config import SNNConfig
from SNN_based_method.scripts.train import apply_default_train_paths
from SNN_based_method.utils.data import (
    NUM_PIXELS,
    collect_raw_records,
    infer_groupable_total_pages,
    read_raw_group,
    sanitize_label_class,
)
from SNN_based_method.utils.generate_precomputed_labels import (
    DEFAULT_LABELS_PER_CLASS,
    label_root_from_record,
)


@dataclass(frozen=True)
class DatasetPrior:
    """单个数据集的目标/雾/背景 bin 先验。"""

    name: str
    target_center: int
    target_window: tuple[int, int]
    target_search_window: tuple[int, int]


@dataclass(frozen=True)
class NewLabelConfig:
    """新 label 生成器的运行配置。"""

    data_paths: Sequence[Path]
    csv_paths: Sequence[Path]
    pages_per_group: int = 960
    total_pages: int | None = None
    time_threshold: int = 128
    label_dir_name: str | Path = "label_prior"
    debug_dir_name: str | Path = "label_prior_debug"
    labels_per_class: int = DEFAULT_LABELS_PER_CLASS
    recursive: bool = False
    skip_missing_csv_raw: bool = False
    overwrite: bool = False
    dry_run: bool = False
    progress_interval: int = 20
    max_score_groups_per_raw: int = 0
    min_target_count: float = 3.0
    min_target_fraction: float = 0.003
    confidence_threshold: float = 0.10
    min_target_to_fog_ratio: float = 0.20
    min_target_to_background_ratio: float = 0.50
    alpha_fog: float = 0.5
    alpha_background: float = 1.0
    lambda_fog: float = 0.5
    lambda_background: float = 1.0
    local_half_width: int = 3
    min_component_area: int = 8
    min_mask_area_ratio: float = 0.001
    max_mask_area_ratio: float = 0.35


@dataclass(frozen=True)
class GroupCandidate:
    """一个可生成 label 的候选 raw group。"""

    class_name: str
    raw_path: Path
    csv_path_value: object
    label_root: Path
    label_dir: Path
    debug_root: Path
    debug_dir: Path
    group_index: int
    total_pages: int
    prior: DatasetPrior
    score: float
    target_sum: float
    fog_sum: float
    background_sum: float


@dataclass
class LabelStats:
    """生成过程的总体统计。"""

    raw_records: int = 0
    clean_records: int = 0
    classes: int = 0
    planned: int = 0
    generated: int = 0
    skipped_existing: int = 0
    rejected: int = 0
    dry_run: int = 0
    label_roots: set[str] | None = None
    summary_paths: set[str] | None = None

    def __post_init__(self) -> None:
        if self.label_roots is None:
            self.label_roots = set()
        if self.summary_paths is None:
            self.summary_paths = set()

    def as_dict(self) -> dict[str, object]:
        """转换为可打印的普通字典。"""
        return {
            "raw_records": self.raw_records,
            "clean_records": self.clean_records,
            "classes": self.classes,
            "planned": self.planned,
            "generated": self.generated,
            "skipped_existing": self.skipped_existing,
            "rejected": self.rejected,
            "dry_run": self.dry_run,
            "label_roots": sorted(self.label_roots or set()),
            "summary_paths": sorted(self.summary_paths or set()),
        }


DATASET_PRIORS: dict[str, DatasetPrior] = {
    "0825": DatasetPrior(
        name="0825",
        target_center=60,
        target_window=(54, 66),
        target_search_window=(50, 70),
    ),
    "0826": DatasetPrior(
        name="0826",
        target_center=66,
        target_window=(60, 72),
        target_search_window=(56, 76),
    ),
}
FOG_WINDOW = (40, 50)
BACKGROUND_WINDOW = (90, 95)
EPS = 1.0e-6


def _window_len(window: tuple[int, int]) -> int:
    """返回闭区间 bin 窗口长度。"""
    return int(window[1]) - int(window[0]) + 1


def _dataset_prior_for_record(raw_path: Path, csv_path_value: object) -> DatasetPrior:
    """根据 raw/csv 路径判断使用 0825 还是 0826 的 bin 先验。"""
    candidates = [str(raw_path)]
    if csv_path_value is not None:
        candidates.append(str(csv_path_value))
    normalized = " ".join(candidates).replace("\\", "/").lower()
    for key, prior in DATASET_PRIORS.items():
        if key in normalized:
            return prior
    raise ValueError(f"cannot infer dataset prior from path: {raw_path}")


def _is_clean_fog_level(value: object) -> bool:
    """判断 CSV fog_level 是否可作为 clean label 来源。"""
    try:
        return float(str(value).strip()) == 0.0
    except ValueError:
        return str(value).strip().lower() in {"clean", "none"}


def _group_indices_to_score(num_groups: int, max_groups: int) -> list[int]:
    """返回需要评分的 group 索引; max_groups<=0 表示全量评分。"""
    if num_groups <= 0:
        return []
    if max_groups <= 0 or max_groups >= num_groups:
        return list(range(num_groups))
    positions = np.linspace(0, num_groups - 1, num=max_groups)
    return sorted({int(round(value)) for value in positions})


def _count_window(group_tof: np.ndarray, window: tuple[int, int]) -> int:
    """统计 group 中落入闭区间窗口的 ToF 数。"""
    lo, hi = window
    return int(((group_tof >= lo) & (group_tof <= hi)).sum())


def _score_group(group_tof: np.ndarray, prior: DatasetPrior, config: NewLabelConfig) -> tuple[float, float, float, float]:
    """按目标/雾/背景统计给一个 group 打分。"""
    target_sum = float(_count_window(group_tof, prior.target_window))
    fog_sum = float(_count_window(group_tof, FOG_WINDOW))
    background_sum = float(_count_window(group_tof, BACKGROUND_WINDOW))
    target_len = float(_window_len(prior.target_window))
    fog_baseline = fog_sum / float(_window_len(FOG_WINDOW)) * target_len
    background_baseline = background_sum / float(_window_len(BACKGROUND_WINDOW)) * target_len
    corrected = target_sum - config.alpha_fog * fog_baseline - config.alpha_background * background_baseline
    corrected = max(corrected, 0.0)
    score = corrected / (config.lambda_fog * fog_sum + config.lambda_background * background_sum + EPS)
    return score, target_sum, fog_sum, background_sum


def _pixel_indices(pages_per_group: int) -> np.ndarray:
    """生成 group_tof.ravel() 对应的像素索引。"""
    return np.repeat(np.arange(NUM_PIXELS, dtype=np.int32), int(pages_per_group))


def _histogram_by_pixel(
    group_tof: np.ndarray,
    *,
    time_threshold: int,
    pixel_indices: np.ndarray,
) -> np.ndarray:
    """生成 ``[time_threshold + 1, 4096]`` 的 per-pixel ToF 直方图。"""
    if group_tof.shape[0] != NUM_PIXELS:
        raise ValueError(f"group_tof must have shape [4096, P], got {group_tof.shape}")
    values = group_tof.reshape(-1)
    if values.shape[0] != pixel_indices.shape[0]:
        raise ValueError(
            f"pixel index length mismatch: values={values.shape[0]}, "
            f"indices={pixel_indices.shape[0]}"
        )
    valid = (values >= 1) & (values <= time_threshold)
    linear = values[valid].astype(np.int64) * NUM_PIXELS + pixel_indices[valid].astype(np.int64)
    counts = np.bincount(
        linear,
        minlength=(int(time_threshold) + 1) * NUM_PIXELS,
    )
    return counts.reshape(int(time_threshold) + 1, NUM_PIXELS).astype(np.float32, copy=False)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """移除 64x64 二值 mask 中的小连通域。"""
    if min_area <= 1:
        return mask
    mask_2d = mask.reshape(64, 64).astype(bool, copy=True)
    visited = np.zeros_like(mask_2d, dtype=bool)
    keep = np.zeros_like(mask_2d, dtype=bool)
    for y in range(64):
        for x in range(64):
            if visited[y, x] or not mask_2d[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < 64 and 0 <= nx < 64 and not visited[ny, nx] and mask_2d[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) >= min_area:
                for cy, cx in component:
                    keep[cy, cx] = True
    return keep.reshape(NUM_PIXELS)


def _build_prior_label(
    group_tof: np.ndarray,
    *,
    prior: DatasetPrior,
    config: NewLabelConfig,
    pixel_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    """为单个 group 生成基于 bin 先验的 label 与诊断信息。"""
    hist = _histogram_by_pixel(
        group_tof,
        time_threshold=config.time_threshold,
        pixel_indices=pixel_indices,
    )
    pix = np.arange(NUM_PIXELS)
    target_lo, target_hi = prior.target_window
    search_lo, search_hi = prior.target_search_window

    target_count = hist[target_lo : target_hi + 1].sum(axis=0)
    fog_count = hist[FOG_WINDOW[0] : FOG_WINDOW[1] + 1].sum(axis=0)
    background_count = hist[BACKGROUND_WINDOW[0] : BACKGROUND_WINDOW[1] + 1].sum(axis=0)

    target_len = float(_window_len(prior.target_window))
    fog_baseline = fog_count / float(_window_len(FOG_WINDOW)) * target_len
    background_baseline = background_count / float(_window_len(BACKGROUND_WINDOW)) * target_len
    corrected_target = (
        target_count
        - config.alpha_fog * fog_baseline
        - config.alpha_background * background_baseline
    )
    corrected_target = np.maximum(corrected_target, 0.0)

    def _mode_depth_for_peak(peak_bin: np.ndarray) -> np.ndarray:
        """以 peak_bin 为中心做局部加权平均, 得到亚 bin 精度的众数深度。"""
        local_count = np.zeros(NUM_PIXELS, dtype=np.float32)
        local_weighted_sum = np.zeros(NUM_PIXELS, dtype=np.float32)
        for offset in range(-int(config.local_half_width), int(config.local_half_width) + 1):
            bins = np.clip(peak_bin + offset, 1, int(config.time_threshold))
            counts = hist[bins, pix]
            local_count += counts
            local_weighted_sum += counts * bins.astype(np.float32)
        return local_weighted_sum / np.maximum(local_count, EPS)

    # 目标深度: 在 target_search_window 内取每像素众数 (峰值 bin), 再局部加权细化
    search_hist = hist[search_lo : search_hi + 1]
    peak_bin = search_hist.argmax(axis=0).astype(np.int32) + int(search_lo)
    depth = _mode_depth_for_peak(peak_bin)

    # 背景深度: 在全有效范围 [1, time_threshold] 内取每像素众数, 给非目标区域提供监督
    # (避免背景 depth=0 导致整片区域无梯度; 雾后向散射众数 ~40 bin)
    full_hist = hist[1 : int(config.time_threshold) + 1]
    full_peak_bin = full_hist.argmax(axis=0).astype(np.int32) + 1
    background_depth = _mode_depth_for_peak(full_peak_bin)

    confidence = corrected_target / (
        corrected_target
        + config.lambda_fog * fog_count
        + config.lambda_background * background_count
        + EPS
    )
    min_count = max(float(config.min_target_count), float(config.min_target_fraction) * float(config.pages_per_group))
    mask = (
        (corrected_target >= min_count)
        & (confidence >= float(config.confidence_threshold))
        & (peak_bin >= target_lo)
        & (peak_bin <= target_hi)
        & (target_count >= config.min_target_to_fog_ratio * fog_count)
        & (target_count >= config.min_target_to_background_ratio * background_count)
    )
    mask = _remove_small_components(mask, int(config.min_component_area))

    # 目标区用 target-window 众数深度, 背景区用全范围众数深度 (不再置 0)
    # 这样深度通道逐像素稠密, 训练 mask (d_gt>0) 自动转为全 1, 背景获得监督
    depth = np.where(mask, depth, background_depth)
    confidence = np.where(mask, confidence, 0.0)
    label = np.stack(
        [depth.reshape(64, 64), confidence.reshape(64, 64)],
        axis=0,
    ).astype(np.float32, copy=False)
    diagnostics: dict[str, np.ndarray | float | int] = {
        "mask": mask.reshape(64, 64).astype(np.uint8, copy=False),
        "confidence": confidence.reshape(64, 64).astype(np.float32, copy=False),
        "peak_bin": peak_bin.reshape(64, 64).astype(np.int16, copy=False),
        "target_count": target_count.reshape(64, 64).astype(np.float32, copy=False),
        "fog_count": fog_count.reshape(64, 64).astype(np.float32, copy=False),
        "background_count": background_count.reshape(64, 64).astype(np.float32, copy=False),
        "corrected_target": corrected_target.reshape(64, 64).astype(np.float32, copy=False),
        "mask_area_ratio": float(mask.mean()),
        "depth_mean": float(depth[mask].mean()) if mask.any() else 0.0,
        "depth_std": float(depth[mask].std()) if mask.any() else 0.0,
        "confidence_mean": float(confidence[mask].mean()) if mask.any() else 0.0,
        "target_count_mean": float(target_count[mask].mean()) if mask.any() else 0.0,
        "fog_count_mean": float(fog_count[mask].mean()) if mask.any() else 0.0,
        "background_count_mean": float(background_count[mask].mean()) if mask.any() else 0.0,
    }
    return label, diagnostics


def _validate_config(config: NewLabelConfig) -> None:
    """在长时间读 raw 前验证配置。"""
    if config.pages_per_group <= 0:
        raise ValueError("pages_per_group must be positive")
    if config.time_threshold <= 0:
        raise ValueError("time_threshold must be positive")
    if config.labels_per_class <= 0:
        raise ValueError("labels_per_class must be positive")
    if not config.data_paths:
        raise ValueError("data_paths must not be empty")
    if not config.csv_paths:
        raise ValueError("csv_paths must not be empty")


def _collect_candidates(config: NewLabelConfig) -> tuple[dict[tuple[str, str], list[GroupCandidate]], LabelStats]:
    """收集并评分每个 class 的 clean group 候选。"""
    raw_records = collect_raw_records(
        config.data_paths,
        csv_paths=config.csv_paths,
        recursive=config.recursive,
        skip_missing_csv_raw=config.skip_missing_csv_raw,
    )
    stats = LabelStats(raw_records=len(raw_records))
    candidates: dict[tuple[str, str], list[GroupCandidate]] = defaultdict(list)
    all_class_keys: set[tuple[str, str]] = set()
    clean_records = 0

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
        debug_root = label_root_from_record(
            raw_path,
            metadata.get("csv_path"),
            label_dir_name=config.debug_dir_name,
            pages_per_group=config.pages_per_group,
        )
        source_key = (str(label_root.resolve()).lower(), class_name)
        all_class_keys.add(source_key)
        stats.label_roots.add(str(label_root))
        if not _is_clean_fog_level(metadata.get("fog_level", "")):
            continue
        clean_records += 1
        prior = _dataset_prior_for_record(raw_path, metadata.get("csv_path"))
        total_pages = infer_groupable_total_pages(
            raw_path,
            pages_per_group=config.pages_per_group,
            total_pages=config.total_pages,
        )
        num_groups = total_pages // int(config.pages_per_group)
        for group_index in _group_indices_to_score(num_groups, config.max_score_groups_per_raw):
            raw_group = read_raw_group(
                raw_path,
                group_index=group_index,
                pages_per_group=config.pages_per_group,
                total_pages=total_pages,
            )
            score, target_sum, fog_sum, background_sum = _score_group(raw_group, prior, config)
            candidates[source_key].append(
                GroupCandidate(
                    class_name=class_name,
                    raw_path=raw_path,
                    csv_path_value=metadata.get("csv_path"),
                    label_root=label_root,
                    label_dir=label_root / class_name,
                    debug_root=debug_root,
                    debug_dir=debug_root / class_name,
                    group_index=group_index,
                    total_pages=total_pages,
                    prior=prior,
                    score=score,
                    target_sum=target_sum,
                    fog_sum=fog_sum,
                    background_sum=background_sum,
                )
            )

    stats.clean_records = clean_records
    stats.classes = len(all_class_keys)
    missing = sorted(all_class_keys - set(candidates.keys()))
    if missing:
        missing_text = ", ".join(f"{root}/{class_name}" for root, class_name in missing[:10])
        raise ValueError(f"missing clean fog_level=0 records for label generation: {missing_text}")
    return candidates, stats


def _summary_row(
    candidate: GroupCandidate,
    output_index: int,
    label_path: Path,
    diagnostics: dict[str, np.ndarray | float | int],
    status: str,
) -> dict[str, object]:
    """构造 summary.csv 的一行。"""
    return {
        "status": status,
        "class": candidate.class_name,
        "output_index": output_index,
        "dataset": candidate.prior.name,
        "raw_path": str(candidate.raw_path),
        "group_index": candidate.group_index,
        "label_path": str(label_path),
        "score": candidate.score,
        "raw_target_sum": candidate.target_sum,
        "raw_fog_sum": candidate.fog_sum,
        "raw_background_sum": candidate.background_sum,
        "target_center": candidate.prior.target_center,
        "target_window": f"{candidate.prior.target_window[0]}-{candidate.prior.target_window[1]}",
        "fog_window": f"{FOG_WINDOW[0]}-{FOG_WINDOW[1]}",
        "background_window": f"{BACKGROUND_WINDOW[0]}-{BACKGROUND_WINDOW[1]}",
        "mask_area_ratio": diagnostics.get("mask_area_ratio", 0.0),
        "depth_mean": diagnostics.get("depth_mean", 0.0),
        "depth_std": diagnostics.get("depth_std", 0.0),
        "confidence_mean": diagnostics.get("confidence_mean", 0.0),
        "target_count_mean": diagnostics.get("target_count_mean", 0.0),
        "fog_count_mean": diagnostics.get("fog_count_mean", 0.0),
        "background_count_mean": diagnostics.get("background_count_mean", 0.0),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """写 summary.csv。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_with_config(config: NewLabelConfig) -> LabelStats:
    """执行新 label 生成流程。"""
    _validate_config(config)
    candidates_by_class, stats = _collect_candidates(config)
    pixel_indices = _pixel_indices(config.pages_per_group)
    rows_by_debug_root: dict[Path, list[dict[str, object]]] = defaultdict(list)
    progress_interval = max(1, int(config.progress_interval))
    processed = 0

    for source_key in sorted(candidates_by_class):
        candidates = sorted(
            candidates_by_class[source_key],
            key=lambda item: item.score,
            reverse=True,
        )
        accepted = 0
        for candidate in candidates:
            if accepted >= int(config.labels_per_class):
                break
            output_index = accepted
            label_path = candidate.label_dir / f"{candidate.class_name}_{output_index}.npy"
            debug_path = candidate.debug_dir / f"{candidate.class_name}_{output_index}.npz"
            processed += 1
            if label_path.exists() and not config.overwrite:
                stats.planned += 1
                stats.skipped_existing += 1
                accepted += 1
                rows_by_debug_root[candidate.debug_root].append(
                    _summary_row(candidate, output_index, label_path, {}, "skipped_existing")
                )
                continue
            if config.dry_run:
                stats.planned += 1
                stats.dry_run += 1
                accepted += 1
                rows_by_debug_root[candidate.debug_root].append(
                    _summary_row(candidate, output_index, label_path, {}, "dry_run")
                )
                print(
                    f"dry_run class={candidate.class_name} dataset={candidate.prior.name} "
                    f"group={candidate.group_index} score={candidate.score:.6f} label={label_path}"
                )
                continue

            raw_group = read_raw_group(
                candidate.raw_path,
                group_index=candidate.group_index,
                pages_per_group=config.pages_per_group,
                total_pages=candidate.total_pages,
            )
            label, diagnostics = _build_prior_label(
                raw_group,
                prior=candidate.prior,
                config=config,
                pixel_indices=pixel_indices,
            )
            mask_area_ratio = float(diagnostics.get("mask_area_ratio", 0.0))
            if (
                mask_area_ratio < float(config.min_mask_area_ratio)
                or mask_area_ratio > float(config.max_mask_area_ratio)
            ):
                stats.rejected += 1
                rejected_path = candidate.debug_dir / (
                    f"rejected_{candidate.class_name}_{candidate.group_index}.npz"
                )
                candidate.debug_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    rejected_path,
                    label=label.astype(np.float32, copy=False),
                    **diagnostics,
                )
                rows_by_debug_root[candidate.debug_root].append(
                    _summary_row(candidate, output_index, label_path, diagnostics, "rejected")
                )
                continue

            candidate.label_dir.mkdir(parents=True, exist_ok=True)
            np.save(label_path, label.astype(np.float32, copy=False))
            candidate.debug_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                debug_path,
                label=label.astype(np.float32, copy=False),
                **diagnostics,
            )
            stats.planned += 1
            stats.generated += 1
            accepted += 1
            rows_by_debug_root[candidate.debug_root].append(
                _summary_row(candidate, output_index, label_path, diagnostics, "generated")
            )
            if processed % progress_interval == 0:
                print(
                    f"processed={processed} generated={stats.generated} "
                    f"skipped={stats.skipped_existing} rejected={stats.rejected} "
                    f"label={label_path}"
                )
        if accepted < int(config.labels_per_class):
            raise ValueError(
                f"not enough accepted labels for class {source_key[1]}: "
                f"accepted={accepted}, required={config.labels_per_class}, "
                f"candidates={len(candidates)}, rejected={stats.rejected}"
            )

    if not config.dry_run:
        for debug_root, rows in rows_by_debug_root.items():
            summary_path = debug_root / "summary.csv"
            _write_summary_csv(summary_path, rows)
            stats.summary_paths.add(str(summary_path))
    return stats


def _config_from_args(args: argparse.Namespace) -> NewLabelConfig:
    """从 CLI 参数和 SNNConfig 组合新 label 生成配置。"""
    cfg = SNNConfig.load(args.config) if args.config else SNNConfig()
    cfg = apply_default_train_paths(cfg)
    data_paths = [Path(path) for path in (args.data_paths if args.data_paths else cfg.data_paths or [])]
    csv_paths = [Path(path) for path in (args.csv_paths if args.csv_paths else cfg.csv_paths or [])]
    pages_per_group = int(args.pages_per_group if args.pages_per_group is not None else cfg.pages_per_group)
    total_pages = args.total_pages if args.total_pages is not None else cfg.total_pages
    time_threshold = int(args.time_threshold if args.time_threshold is not None else cfg.time_threshold)
    return NewLabelConfig(
        data_paths=data_paths,
        csv_paths=csv_paths,
        pages_per_group=pages_per_group,
        total_pages=total_pages,
        time_threshold=time_threshold,
        label_dir_name=args.label_dir_name,
        debug_dir_name=args.debug_dir_name,
        labels_per_class=int(args.labels_per_class),
        recursive=bool(args.recursive or cfg.recursive),
        skip_missing_csv_raw=bool(args.skip_missing_csv_raw or cfg.skip_missing_csv_raw),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        progress_interval=int(args.progress_interval),
        max_score_groups_per_raw=int(args.max_score_groups_per_raw),
        min_target_count=float(args.min_target_count),
        min_target_fraction=float(args.min_target_fraction),
        confidence_threshold=float(args.confidence_threshold),
        min_target_to_fog_ratio=float(args.min_target_to_fog_ratio),
        min_target_to_background_ratio=float(args.min_target_to_background_ratio),
        alpha_fog=float(args.alpha_fog),
        alpha_background=float(args.alpha_background),
        lambda_fog=float(args.lambda_fog),
        lambda_background=float(args.lambda_background),
        local_half_width=int(args.local_half_width),
        min_component_area=int(args.min_component_area),
        min_mask_area_ratio=float(args.min_mask_area_ratio),
        max_mask_area_ratio=float(args.max_mask_area_ratio),
    )


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(
        description="按 0825/0826 目标 bin 先验生成 mask 化 SPAD label 池",
    )
    parser.add_argument("--config", type=Path, default=None, help="可选 SNNConfig JSON; 未提供时使用默认训练路径")
    parser.add_argument("--data-paths", nargs="+", type=Path, default=None, help="raw 文件或目录")
    parser.add_argument("--csv-paths", nargs="+", type=Path, default=None, help="CSV 样本清单")
    parser.add_argument("--pages-per-group", type=int, default=None, help="每个 label 对应的 page 数 P")
    parser.add_argument("--total-pages", type=int, default=None, help="每个 raw 文件最多使用的 page 数")
    parser.add_argument("--time-threshold", type=int, default=None, help="有效 ToF 上限")
    parser.add_argument("--label-dir-name", default="label_prior", help="输出 label 目录名, 默认 label_prior")
    parser.add_argument("--debug-dir-name", default="label_prior_debug", help="输出 debug 目录名")
    parser.add_argument("--labels-per-class", type=int, default=DEFAULT_LABELS_PER_CLASS, help="每类保留 top K label")
    parser.add_argument("--recursive", action="store_true", help="递归搜索 data-paths")
    parser.add_argument("--skip-missing-csv-raw", action="store_true", help="跳过 CSV 中缺失 raw")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在 label")
    parser.add_argument("--dry-run", action="store_true", help="只评分和打印计划, 不写 label")
    parser.add_argument("--progress-interval", type=int, default=20, help="每处理 N 个 label 打印一次进度")
    parser.add_argument("--max-score-groups-per-raw", type=int, default=0, help="每个 raw 最多评分几个 group; 0 表示全量")
    parser.add_argument("--min-target-count", type=float, default=3.0, help="每像素目标窗口最小扣除后计数")
    parser.add_argument("--min-target-fraction", type=float, default=0.003, help="每像素目标计数下限占 P 的比例")
    parser.add_argument("--confidence-threshold", type=float, default=0.10, help="目标置信度阈值")
    parser.add_argument("--min-target-to-fog-ratio", type=float, default=0.20, help="target_count/fog_count 下限")
    parser.add_argument("--min-target-to-background-ratio", type=float, default=0.50, help="target_count/background_count 下限")
    parser.add_argument("--alpha-fog", type=float, default=0.5, help="雾窗口基线扣除权重")
    parser.add_argument("--alpha-background", type=float, default=1.0, help="背景窗口基线扣除权重")
    parser.add_argument("--lambda-fog", type=float, default=0.5, help="confidence 中雾窗口惩罚权重")
    parser.add_argument("--lambda-background", type=float, default=1.0, help="confidence 中背景窗口惩罚权重")
    parser.add_argument("--local-half-width", type=int, default=3, help="peak bin 周围加权中心半宽")
    parser.add_argument("--min-component-area", type=int, default=8, help="mask 小连通域删除阈值")
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.001, help="label mask 面积比例下限")
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.35, help="label mask 面积比例上限")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    stats = run_with_config(config)
    print(stats.as_dict())
    return 0


def main_without_cli() -> None:
    """无参数调试入口; 默认 dry-run, 避免误写。"""
    editable_args = [
        "--pages-per-group",
        "960",
        "--label-dir-name",
        "label_prior",
        "--dry-run",
        "--max-score-groups-per-raw",
        "8",
    ]
    raise SystemExit(main(editable_args))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()

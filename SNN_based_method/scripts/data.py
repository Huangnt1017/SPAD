"""SPAD SNN 训练使用的数据集与 DataLoader 工具。

raw 文件读取约定:
1. 按 uint16 读取原始 page, 每个 page 尺寸为 ``64x64``。
2. 大于 ``time_threshold`` 的 ToF 值置 0, 作为无效触发。
3. 按 ``pages_per_group`` 分组为 ``[G, 4096, P]``。

数据集会把每组 reshape 为 ``[P, 1, 64, 64]``。自定义 collate 会把
batch 堆叠成时间维优先格式:

    frames: ``[P, B, 1, 64, 64]``

弱标签是可选的 ``[B, 2, 64, 64]`` 图像标签, 通道 0 为深度, 通道 1
为强度。
"""

from __future__ import annotations

import random
import csv
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from SNN_based_method.scripts.augment import (
        SpadRawTrainAugmentation,
        clip_tof_to_valid_range,
    )
except ImportError:
    from augment import SpadRawTrainAugmentation, clip_tof_to_valid_range


NUM_PIXELS = 64 * 64
RAW_VALUE_BYTES = 2
DEFAULT_SEED = 42


@dataclass(frozen=True)
class RawGroupSample:
    """一个 raw 分组样本的索引信息。"""

    raw_path: str
    group_index: int
    total_pages: int


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """统一设置 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """根据 PyTorch worker seed 设置单个 DataLoader worker 的随机种子。"""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collect_raw_paths(
    paths: str | Path | Sequence[str | Path],
    *,
    recursive: bool = False,
) -> list[Path]:
    """从文件或目录中收集排序后的 ``.raw`` 文件。

    Args:
        paths: raw 文件路径、目录路径或路径序列。
        recursive: 为 True 时递归搜索目录。

    Returns:
        去重并排序后的 raw 文件路径。
    """
    if isinstance(paths, (str, Path)):
        candidate_paths: Iterable[str | Path] = [paths]
    else:
        candidate_paths = paths

    raw_paths: list[Path] = []
    for path_value in candidate_paths:
        path = Path(path_value)
        if path.is_file() and path.suffix.lower() == ".raw":
            raw_paths.append(path)
            continue
        if path.is_dir():
            pattern = "**/*.raw" if recursive else "*.raw"
            raw_paths.extend(sorted(path.glob(pattern)))

    unique_paths = {str(path.resolve()): path.resolve() for path in raw_paths}
    return [unique_paths[key] for key in sorted(unique_paths)]


def _as_path_list(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    """把单个路径或路径序列统一为 ``Path`` 列表。"""
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _raw_path_candidates(file_path_value: str | Path) -> list[Path]:
    """根据 CSV 中的 ``file_path`` 字段生成可能的 raw 文件名。"""
    value_path = Path(str(file_path_value).strip())
    candidates = [value_path]
    if value_path.suffix.lower() != ".raw":
        candidates.append(value_path.with_suffix(".raw"))

    unique: dict[str, Path] = {}
    for candidate in candidates:
        key = str(candidate)
        if key not in unique:
            unique[key] = candidate
    return list(unique.values())


def _resolve_csv_raw_path(
    file_path_value: str | Path,
    *,
    csv_path: Path,
    data_roots: Sequence[Path],
) -> Path:
    """把 CSV 中的文件名解析到真实存在的 ``.raw`` 路径。"""
    if str(file_path_value).strip() == "":
        raise ValueError(f"Empty file_path in CSV: {csv_path}")

    candidates = _raw_path_candidates(file_path_value)
    search_roots = [csv_path.parent]
    for data_root in data_roots:
        root = data_root if data_root.is_dir() else data_root.parent
        if root not in search_roots:
            search_roots.append(root)

    attempted: list[Path] = []
    for candidate in candidates:
        if candidate.is_absolute():
            attempted.append(candidate)
            if candidate.is_file():
                return candidate.resolve()
            continue

        for root in search_roots:
            resolved = root / candidate
            attempted.append(resolved)
            if resolved.is_file():
                return resolved.resolve()

    attempted_text = ", ".join(str(path) for path in attempted[:6])
    raise FileNotFoundError(
        f"Cannot resolve CSV raw file '{file_path_value}' from {csv_path}. "
        f"Tried: {attempted_text}"
    )


def collect_raw_records(
    paths: str | Path | Sequence[str | Path],
    *,
    csv_paths: str | Path | Sequence[str | Path] | None = None,
    recursive: bool = False,
    skip_missing_csv_raw: bool = False,
) -> list[tuple[Path, dict[str, Any]]]:
    """收集 raw 文件及可选 CSV 元信息。

    当 ``csv_paths`` 非空时，CSV 的 ``file_path`` 列是样本清单，只读取 CSV
    中列出的 raw 文件，并把 ``fog_level`` / ``target_class`` 等列作为元信息
    带入 batch。否则保持旧行为：从 ``paths`` 直接收集 ``.raw`` 文件。
    """
    if not csv_paths:
        return [(path, {}) for path in collect_raw_paths(paths, recursive=recursive)]

    data_roots = _as_path_list(paths)
    csv_path_list = _as_path_list(csv_paths)
    records: list[tuple[Path, dict[str, Any]]] = []

    for csv_path_value in csv_path_list:
        csv_path = csv_path_value.resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            if reader.fieldnames is None or "file_path" not in reader.fieldnames:
                raise ValueError(f"CSV must contain a file_path column: {csv_path}")

            for row_index, row in enumerate(reader, start=2):
                try:
                    raw_path = _resolve_csv_raw_path(
                        row.get("file_path", ""),
                        csv_path=csv_path,
                        data_roots=data_roots,
                    )
                except FileNotFoundError as exc:
                    if skip_missing_csv_raw:
                        warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
                        continue
                    raise
                metadata = {
                    key: value
                    for key, value in row.items()
                    if key is not None and value is not None
                }
                metadata["csv_path"] = str(csv_path)
                metadata["csv_row"] = row_index
                records.append((raw_path, metadata))

    if not records:
        raise ValueError(f"no CSV raw records were found in: {csv_path_list}")
    return records


def infer_groupable_total_pages(
    raw_path: str | Path,
    pages_per_group: int,
    total_pages: int | None = None,
) -> int:
    """推断可整分组的有效 page 总数。"""
    raw_path = Path(raw_path)
    if pages_per_group <= 0:
        raise ValueError("pages_per_group must be a positive integer")
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw file not found: {raw_path}")

    total_values = raw_path.stat().st_size // RAW_VALUE_BYTES
    available_pages = total_values // NUM_PIXELS
    if available_pages == 0:
        raise ValueError(f"raw file has no complete 64x64 page: {raw_path}")

    if total_pages is None:
        resolved_pages = available_pages
    else:
        if total_pages <= 0:
            raise ValueError("total_pages must be positive when provided")
        if total_pages > available_pages:
            raise ValueError(
                f"total_pages {total_pages} exceeds file pages "
                f"{available_pages}: {raw_path}"
            )
        resolved_pages = int(total_pages)

    resolved_pages -= resolved_pages % pages_per_group
    if resolved_pages <= 0:
        raise ValueError(
            f"raw file does not contain one complete group "
            f"(pages_per_group={pages_per_group}): {raw_path}"
        )
    return resolved_pages


def build_group_samples(
    raw_paths: Sequence[str | Path],
    pages_per_group: int,
    total_pages: int | None = None,
) -> list[RawGroupSample]:
    """把 raw 文件列表展开成分组级样本记录。"""
    samples: list[RawGroupSample] = []
    for raw_path_value in raw_paths:
        raw_path = Path(raw_path_value).resolve()
        resolved_pages = infer_groupable_total_pages(
            raw_path=raw_path,
            pages_per_group=pages_per_group,
            total_pages=total_pages,
        )
        num_groups = resolved_pages // pages_per_group
        samples.extend(
            RawGroupSample(
                raw_path=str(raw_path),
                group_index=group_index,
                total_pages=resolved_pages,
            )
            for group_index in range(num_groups)
        )
    if not samples:
        raise ValueError("no group samples were built from raw_paths")
    return samples


def raw2frame(
    filename: str | Path,
    pages_per_group: int,
    total_pages: int,
    time_threshold: int,
) -> np.ndarray:
    """读取 SPAD raw 文件并返回未裁剪的 ``[G, 4096, P]`` 分组数组。

    注意: 这里保留 raw 中原始 ToF 值, 不提前把 ``> time_threshold`` 置 0。
    训练增强需要先在原始 ToF 层面操作, 再由 Dataset 统一把小于 1 或大于
    ``time_threshold`` 的值置 0。
    """
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"raw file not found: {filename}")
    if pages_per_group <= 0:
        raise ValueError("pages_per_group must be a positive integer")
    if total_pages <= 0:
        raise ValueError("total_pages must be a positive integer")
    if time_threshold <= 0:
        raise ValueError("time_threshold must be a positive integer")
    if total_pages % pages_per_group != 0:
        raise ValueError("total_pages must be divisible by pages_per_group")

    available_pages = (filename.stat().st_size // RAW_VALUE_BYTES) // NUM_PIXELS
    if available_pages == 0:
        return np.empty((0, NUM_PIXELS, pages_per_group), dtype=np.uint16)
    if total_pages > available_pages:
        raise ValueError(f"total_pages {total_pages} exceeds file pages {available_pages}")

    count = int(total_pages) * NUM_PIXELS
    flat_data = np.fromfile(filename, dtype=np.uint16, count=count)
    if flat_data.size != count:
        raise IOError(f"File too short: expected {count} values, got {flat_data.size}")

    frames = flat_data.reshape(total_pages, 64, 64)
    flat_frames = frames.reshape(total_pages, NUM_PIXELS)
    num_groups = total_pages // pages_per_group
    grouped = flat_frames.reshape(num_groups, pages_per_group, NUM_PIXELS)
    return np.transpose(grouped, (0, 2, 1)).astype(np.uint16, copy=False)


def read_raw_group(
    filename: str | Path,
    *,
    group_index: int,
    pages_per_group: int,
    total_pages: int,
) -> np.ndarray:
    """只读取单个 raw group, 返回未裁剪的 ``[4096, P]`` 数组。

    随机训练时如果每次都读取完整 raw 文件, DataLoader worker 很容易成为
    GPU 的等待点。该函数按 group 的字节偏移直接读取连续 page, 更适合
    shuffle 后的分组级采样。
    """
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"raw file not found: {filename}")
    if pages_per_group <= 0:
        raise ValueError("pages_per_group must be a positive integer")
    if total_pages <= 0:
        raise ValueError("total_pages must be a positive integer")
    if group_index < 0:
        raise ValueError("group_index must be non-negative")

    start_page = int(group_index) * int(pages_per_group)
    end_page = start_page + int(pages_per_group)
    if end_page > int(total_pages):
        raise IndexError(
            f"group_index {group_index} exceeds total_pages={total_pages} "
            f"with pages_per_group={pages_per_group}"
        )

    count = int(pages_per_group) * NUM_PIXELS
    byte_offset = start_page * NUM_PIXELS * RAW_VALUE_BYTES
    with filename.open("rb") as file_obj:
        file_obj.seek(byte_offset)
        flat_data = np.fromfile(file_obj, dtype=np.uint16, count=count)
    if flat_data.size != count:
        raise IOError(f"File too short for group {group_index}: {filename}")

    frames = flat_data.reshape(pages_per_group, 64, 64)
    flat_frames = frames.reshape(pages_per_group, NUM_PIXELS)
    return flat_frames.T.astype(np.uint16, copy=False)


def n3_filter(points: np.ndarray, min_count: int) -> np.ndarray:
    """按重复次数过滤点云行, 输出 ``[x, y, tof, count]``。"""
    if points.size == 0:
        return np.empty((0, 4), dtype=np.int64)
    unique_points, counts = np.unique(points, axis=0, return_counts=True)
    mask = counts >= min_count
    return np.column_stack((unique_points[mask], counts[mask]))


def max_count_maps(filtered_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从带计数点云生成深度图和强度图。"""
    intensity_map = np.zeros((64, 64), dtype=np.int32)
    depth_map = np.zeros((64, 64), dtype=np.int32)
    for x_coord, y_coord, tof_value, count in filtered_points:
        x_idx = int(x_coord) - 1
        y_idx = int(y_coord) - 1
        if count > intensity_map[x_idx, y_idx]:
            intensity_map[x_idx, y_idx] = int(count)
            depth_map[x_idx, y_idx] = int(tof_value)
    return depth_map, intensity_map


def group_to_time_first_frames(
    group_tof: np.ndarray,
    *,
    normalize: bool = False,
    time_threshold: int | None = None,
) -> torch.Tensor:
    """把 ``[4096, P]`` 分组转换为 ``[P, 1, 64, 64]`` 张量。"""
    if group_tof.ndim != 2 or group_tof.shape[0] != NUM_PIXELS:
        raise ValueError("group_tof must have shape [4096, P]")

    pages_per_group = group_tof.shape[1]
    # 数据流: [4096, P] -> [P, 64, 64] -> [P, 1, 64, 64]
    frames = group_tof.T.reshape(pages_per_group, 64, 64)
    frames_tensor = torch.from_numpy(frames.astype(np.float32, copy=False))
    frames_tensor = frames_tensor.unsqueeze(1).contiguous()

    if normalize:
        if time_threshold is None or time_threshold <= 0:
            raise ValueError("time_threshold must be positive when normalize=True")
        frames_tensor = frames_tensor / float(time_threshold)
    return frames_tensor


def time_first_to_model_input(frames: torch.Tensor) -> torch.Tensor:
    """把 ``[P, B, 1, 64, 64]`` 帧序列转换为 ``[B, 4096, P]``。

    ``SNN.py`` 和 ``SNN_new.py`` 当前接收扁平像素格式, DataLoader 侧则保留
    时间维优先的 5D 形式, 因此这里做统一转换。
    """
    if frames.dim() != 5 or frames.shape[2:] != (1, 64, 64):
        raise ValueError("frames must have shape [P, B, 1, 64, 64]")

    # 数据流: [P, B, 1, 64, 64] -> [B, 64, 64, P] -> [B, 4096, P]
    return (
        frames.squeeze(2)
        .permute(1, 2, 3, 0)
        .reshape(frames.shape[1], NUM_PIXELS, frames.shape[0])
    )


def build_point_cloud_from_group(
    group_tof: np.ndarray,
    time_threshold: int,
) -> np.ndarray:
    """从一个分组生成 ``[N, 3]`` 点云行 ``(x, y, tof)``。"""
    if group_tof.ndim != 2 or group_tof.shape[0] != NUM_PIXELS:
        raise ValueError("group_tof must have shape [4096, P]")

    frames = group_tof.T.reshape(group_tof.shape[1], 64, 64)
    valid_mask = (frames > 0) & (frames <= time_threshold)
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.uint16)

    indices = np.argwhere(valid_mask)
    values = frames[valid_mask]
    return np.column_stack(
        (
            indices[:, 1] + 1,
            indices[:, 2] + 1,
            values.astype(np.uint16, copy=False),
        )
    )


def build_weak_label_from_group(
    group_tof: np.ndarray,
    time_threshold: int,
    *,
    active_point: int = 1,
) -> torch.Tensor:
    """从一个分组生成弱监督 ``[2, 64, 64]`` 标签。"""
    point_cloud = build_point_cloud_from_group(group_tof, time_threshold)
    if point_cloud.size == 0:
        label = np.zeros((2, 64, 64), dtype=np.float32)
        return torch.from_numpy(label)

    filtered_points = n3_filter(point_cloud, active_point)
    if filtered_points.size == 0:
        label = np.zeros((2, 64, 64), dtype=np.float32)
        return torch.from_numpy(label)

    depth_map, intensity_map = max_count_maps(filtered_points)
    intensity_map = np.clip(
        intensity_map.astype(np.float32, copy=False) / float(group_tof.shape[1]),
        0.0,
        1.0,
    )
    label = np.stack([depth_map, intensity_map], axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(label)


class SpadRawGroupDataset(Dataset):
    """面向 SNN 训练的 SPAD raw 分组数据集。

    每个样本包含:
        frames: ``[P, 1, 64, 64]``
        label: ``[2, 64, 64]``，仅当 ``return_label=True`` 时存在

    Args:
        raw_paths: 需要读取的 raw 文件。
        pages_per_group: 每个样本包含的 page 数, 即 ``P``。
        total_pages: 每个 raw 文件使用的 page 数; 为 None 时使用所有完整分组。
        time_threshold: 有效 ToF 上限, 大于该值的输入会被置 0。
        return_label: 是否为每个分组生成弱标签。
        normalize: 是否将输入除以 ``time_threshold``。
        shuffle_pages: 是否随机打乱样本内部的 ``P`` 维。
        cache_size: 在内存中缓存的 raw 分组数组数量。
        transform: 作用于 ``frames`` 的可选变换。
        label_transform: 作用于 ``label`` 的可选变换。
    """

    def __init__(
        self,
        raw_paths: Sequence[str | Path],
        *,
        pages_per_group: int,
        total_pages: int | None = None,
        time_threshold: int = 150,
        return_label: bool = True,
        normalize: bool = False,
        shuffle_pages: bool = False,
        active_point: int = 1,
        cache_size: int = 2,
        raw_load_mode: str = "group",
        raw_metadata: Sequence[Mapping[str, Any]] | None = None,
        samples: Sequence[RawGroupSample] | None = None,
        raw_group_transform: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        label_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if pages_per_group <= 0:
            raise ValueError("pages_per_group must be a positive integer")
        if time_threshold <= 0:
            raise ValueError("time_threshold must be a positive integer")

        self.raw_paths = [Path(path).resolve() for path in raw_paths]
        if not self.raw_paths:
            raise ValueError("raw_paths must not be empty")
        if raw_metadata is not None and len(raw_metadata) != len(self.raw_paths):
            raise ValueError("raw_metadata length must match raw_paths length")

        self.pages_per_group = int(pages_per_group)
        self.total_pages = total_pages
        self.time_threshold = int(time_threshold)
        self.return_label = bool(return_label)
        self.normalize = bool(normalize)
        self.shuffle_pages = bool(shuffle_pages)
        self.active_point = int(active_point)
        self.cache_size = max(0, int(cache_size))
        self.raw_load_mode = str(raw_load_mode).lower()
        if self.raw_load_mode not in {"group", "file_cache"}:
            raise ValueError("raw_load_mode must be 'group' or 'file_cache'")
        self.raw_group_transform = raw_group_transform
        self.transform = transform
        self.label_transform = label_transform
        if raw_metadata is None:
            metadata_items: Sequence[Mapping[str, Any]] = [{} for _ in self.raw_paths]
        else:
            metadata_items = raw_metadata
        self.raw_metadata_by_path = {
            str(path): dict(metadata)
            for path, metadata in zip(self.raw_paths, metadata_items)
        }

        if samples is None:
            self.samples = build_group_samples(
                raw_paths=self.raw_paths,
                pages_per_group=self.pages_per_group,
                total_pages=self.total_pages,
            )
        else:
            self.samples = list(samples)
            if not self.samples:
                raise ValueError("samples must not be empty")

        self._group_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_raw_groups(self, sample: RawGroupSample) -> np.ndarray:
        cache_key = sample.raw_path
        if cache_key in self._group_cache:
            grouped = self._group_cache.pop(cache_key)
            self._group_cache[cache_key] = grouped
            return grouped

        grouped = raw2frame(
            sample.raw_path,
            pages_per_group=self.pages_per_group,
            total_pages=sample.total_pages,
            time_threshold=self.time_threshold,
        )

        if self.cache_size > 0:
            self._group_cache[cache_key] = grouped
            while len(self._group_cache) > self.cache_size:
                self._group_cache.popitem(last=False)
        return grouped

    def _load_raw_group(self, sample: RawGroupSample) -> np.ndarray:
        if self.raw_load_mode == "group":
            return read_raw_group(
                sample.raw_path,
                group_index=sample.group_index,
                pages_per_group=self.pages_per_group,
                total_pages=sample.total_pages,
            )
        return self._load_raw_groups(sample)[sample.group_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        raw_group_tof = self._load_raw_group(sample)

        if self.raw_group_transform is not None:
            group_tof, label_group_tof = self.raw_group_transform(raw_group_tof)
        else:
            group_tof = clip_tof_to_valid_range(raw_group_tof, self.time_threshold)
            label_group_tof = group_tof

        frames = group_to_time_first_frames(
            group_tof,
            normalize=self.normalize,
            time_threshold=self.time_threshold,
        )
        if self.shuffle_pages:
            permutation = torch.randperm(frames.shape[0])
            frames = frames[permutation].contiguous()

        if self.transform is not None:
            frames = self.transform(frames)

        item: dict[str, Any] = {
            "frames": frames,
            "raw_path": sample.raw_path,
            "raw_name": Path(sample.raw_path).stem,
            "group_index": sample.group_index,
            "total_pages": sample.total_pages,
        }
        metadata = self.raw_metadata_by_path.get(sample.raw_path, {})
        if metadata:
            item["metadata"] = metadata
            if "fog_level" in metadata:
                item["fog_level"] = metadata["fog_level"]
            if "target_class" in metadata:
                item["target_class"] = metadata["target_class"]

        if self.return_label:
            label = build_weak_label_from_group(
                label_group_tof,
                time_threshold=self.time_threshold,
                active_point=self.active_point,
            )
            if self.label_transform is not None:
                label = self.label_transform(label)
            item["label"] = label

        return item


def spad_time_first_collate(
    batch: Sequence[dict[str, Any]],
    *,
    include_model_input: bool = False,
) -> dict[str, Any]:
    """把样本列表整理为时间维优先的 SNN batch。

    Returns:
        ``frames`` 形状为 ``[P, B, 1, 64, 64]``。如果存在标签,
        ``label`` 形状为 ``[B, 2, 64, 64]``。当 ``include_model_input=True``
        时额外生成 ``model_input=[B, 4096, P]``，把形状整理工作放到
        DataLoader worker 中完成。
    """
    if not batch:
        raise ValueError("batch must not be empty")

    frames = torch.stack([item["frames"] for item in batch], dim=1).contiguous()
    collated: dict[str, Any] = {
        "frames": frames,
        "raw_path": [item["raw_path"] for item in batch],
        "raw_name": [item["raw_name"] for item in batch],
        "group_index": torch.tensor([item["group_index"] for item in batch], dtype=torch.long),
        "total_pages": torch.tensor([item["total_pages"] for item in batch], dtype=torch.long),
    }
    if include_model_input:
        collated["model_input"] = time_first_to_model_input(frames).contiguous()
    if "metadata" in batch[0]:
        collated["metadata"] = [item.get("metadata", {}) for item in batch]
    if "fog_level" in batch[0]:
        collated["fog_level"] = [item.get("fog_level") for item in batch]
    if "target_class" in batch[0]:
        collated["target_class"] = [item.get("target_class") for item in batch]

    if "label" in batch[0]:
        collated["label"] = torch.stack([item["label"] for item in batch], dim=0).contiguous()
    return collated


def make_spad_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    include_model_input: bool = False,
    seed: int = DEFAULT_SEED,
    drop_last: bool = False,
) -> DataLoader:
    """创建输出 ``frames=[P, B, 1, 64, 64]`` 的 DataLoader。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": (
            partial(spad_time_first_collate, include_model_input=True)
            if include_model_input
            else spad_time_first_collate
        ),
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    return DataLoader(dataset, **loader_kwargs)


def split_indices(
    num_samples: int,
    split_ratios: Sequence[float] = (0.7, 0.2, 0.1),
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[list[int], list[int], list[int]]:
    """把样本索引划分为 train/val/test 三部分。"""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must contain train, val and test ratios")

    train_ratio, val_ratio, test_ratio = (float(value) for value in split_ratios)
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    train_count = int(round(num_samples * train_ratio))
    val_count = int(round(num_samples * val_ratio))
    test_count = num_samples - train_count - val_count

    if test_count < 0:
        val_count = max(0, num_samples - train_count)
        test_count = num_samples - train_count - val_count

    train_indices = indices[:train_count].tolist()
    val_indices = indices[train_count:train_count + val_count].tolist()
    test_indices = indices[train_count + val_count:train_count + val_count + test_count].tolist()
    return train_indices, val_indices, test_indices


def create_spad_dataloaders(
    paths: str | Path | Sequence[str | Path],
    *,
    csv_paths: str | Path | Sequence[str | Path] | None = None,
    skip_missing_csv_raw: bool = False,
    pages_per_group: int,
    total_pages: int | None = None,
    time_threshold: int = 150,
    batch_size: int = 4,
    split_ratios: Sequence[float] = (0.7, 0.2, 0.1),
    seed: int = DEFAULT_SEED,
    return_label: bool = True,
    normalize: bool = False,
    shuffle_pages: bool = False,
    augment_train: bool = False,
    tof_shift_max: int = 15,
    tof_shift_prob: float = 1.0,
    page_dropout: bool = False,
    page_dropout_prob: float = 0.1,
    active_point: int = 1,
    cache_size: int = 2,
    raw_load_mode: str = "group",
    recursive: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    include_model_input: bool = False,
    drop_last: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, SpadRawGroupDataset]:
    """从 raw 文件或目录构建 train/val/test DataLoader。

    返回的 loader 都会产出 ``batch["frames"]``，形状为 ``[P, B, 1, 64, 64]``。
    """
    raw_records = collect_raw_records(
        paths,
        csv_paths=csv_paths,
        recursive=recursive,
        skip_missing_csv_raw=skip_missing_csv_raw,
    )
    if not raw_records:
        raise ValueError(f"no .raw files found in paths: {paths}")
    raw_paths = [record[0] for record in raw_records]
    raw_metadata = [record[1] for record in raw_records]

    dataset = SpadRawGroupDataset(
        raw_paths=raw_paths,
        pages_per_group=pages_per_group,
        total_pages=total_pages,
        time_threshold=time_threshold,
        return_label=return_label,
        normalize=normalize,
        shuffle_pages=False,
        active_point=active_point,
        cache_size=cache_size,
        raw_load_mode=raw_load_mode,
        raw_metadata=raw_metadata,
    )
    train_indices, val_indices, test_indices = split_indices(
        len(dataset),
        split_ratios=split_ratios,
        seed=seed,
    )
    train_transform = None
    if augment_train or page_dropout:
        train_transform = SpadRawTrainAugmentation(
            t_max=time_threshold,
            tof_shift_max=tof_shift_max if augment_train else 0,
            tof_shift_prob=tof_shift_prob if augment_train else 0.0,
            page_dropout=page_dropout,
            page_dropout_prob=page_dropout_prob,
        )
    train_dataset = SpadRawGroupDataset(
        raw_paths=raw_paths,
        pages_per_group=pages_per_group,
        total_pages=total_pages,
        time_threshold=time_threshold,
        return_label=return_label,
        normalize=normalize,
        shuffle_pages=shuffle_pages,
        active_point=active_point,
        cache_size=cache_size,
        raw_load_mode=raw_load_mode,
        raw_metadata=raw_metadata,
        samples=[dataset.samples[index] for index in train_indices],
        raw_group_transform=train_transform,
    )

    train_loader = make_spad_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        include_model_input=include_model_input,
        seed=seed,
        drop_last=drop_last,
    )
    val_loader = make_spad_dataloader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        include_model_input=include_model_input,
        seed=seed + 1,
    )
    test_loader = make_spad_dataloader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        include_model_input=include_model_input,
        seed=seed + 2,
    )
    return train_loader, val_loader, test_loader, dataset


def create_spad_dataloader(
    paths: str | Path | Sequence[str | Path],
    *,
    csv_paths: str | Path | Sequence[str | Path] | None = None,
    skip_missing_csv_raw: bool = False,
    pages_per_group: int,
    total_pages: int | None = None,
    time_threshold: int = 150,
    batch_size: int = 4,
    shuffle: bool = True,
    seed: int = DEFAULT_SEED,
    return_label: bool = True,
    normalize: bool = False,
    shuffle_pages: bool = False,
    active_point: int = 1,
    cache_size: int = 2,
    raw_load_mode: str = "group",
    recursive: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    include_model_input: bool = False,
    drop_last: bool = False,
) -> tuple[DataLoader, SpadRawGroupDataset]:
    """从 raw 文件或目录构建单个 DataLoader。"""
    raw_records = collect_raw_records(
        paths,
        csv_paths=csv_paths,
        recursive=recursive,
        skip_missing_csv_raw=skip_missing_csv_raw,
    )
    if not raw_records:
        raise ValueError(f"no .raw files found in paths: {paths}")
    raw_paths = [record[0] for record in raw_records]
    raw_metadata = [record[1] for record in raw_records]

    dataset = SpadRawGroupDataset(
        raw_paths=raw_paths,
        pages_per_group=pages_per_group,
        total_pages=total_pages,
        time_threshold=time_threshold,
        return_label=return_label,
        normalize=normalize,
        shuffle_pages=shuffle_pages,
        active_point=active_point,
        cache_size=cache_size,
        raw_load_mode=raw_load_mode,
        raw_metadata=raw_metadata,
    )
    loader = make_spad_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        include_model_input=include_model_input,
        seed=seed,
        drop_last=drop_last,
    )
    return loader, dataset


if __name__ == "__main__":
    # 使用示例:
    # loader, dataset = create_spad_dataloader(
    #     r"E:\path\to\raw_dir",
    #     pages_per_group=512,
    #     time_threshold=128,
    #     batch_size=8,
    # )
    # batch = next(iter(loader))
    # print(batch["frames"].shape)  # torch.Size([512, 8, 1, 64, 64])
    pass

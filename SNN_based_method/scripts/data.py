"""Dataset and DataLoader helpers for SPAD SNN training.

The raw reader follows ``models/raw2frame.py``:

1. Read uint16 raw pages as ``[total_pages, 64, 64]``.
2. Set values above ``time_threshold`` to ``0``.
3. Group pages into ``[G, 4096, P]``.

This module then reshapes each group to ``[P, 1, 64, 64]``.  The custom
collate function stacks samples as time-first batches:

    frames: ``[P, B, 1, 64, 64]``

Labels are optional weak image labels shaped ``[B, 2, 64, 64]`` where
channel 0 is depth and channel 1 is intensity.
"""

from __future__ import annotations

import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from models.raw2frame import max_count_maps, n3_filter, raw2frame
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from models.raw2frame import max_count_maps, n3_filter, raw2frame


NUM_PIXELS = 64 * 64
RAW_VALUE_BYTES = 2
DEFAULT_SEED = 42


@dataclass(frozen=True)
class RawGroupSample:
    """Index information for one grouped raw sample."""

    raw_path: str
    group_index: int
    total_pages: int


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Seed one DataLoader worker from PyTorch's worker seed."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collect_raw_paths(
    paths: str | Path | Sequence[str | Path],
    *,
    recursive: bool = False,
) -> list[Path]:
    """Collect sorted ``.raw`` files from files and directories.

    Args:
        paths: A raw file path, a directory, or a sequence of paths.
        recursive: If ``True``, search directories recursively.

    Returns:
        Sorted raw file paths.
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


def infer_groupable_total_pages(
    raw_path: str | Path,
    pages_per_group: int,
    total_pages: int | None = None,
) -> int:
    """Infer a valid page count that can be reshaped into complete groups."""
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
    """Expand raw files into group-level sample records."""
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


def group_to_time_first_frames(
    group_tof: np.ndarray,
    *,
    normalize: bool = False,
    time_threshold: int | None = None,
) -> torch.Tensor:
    """Convert a ``[4096, P]`` group to ``[P, 1, 64, 64]`` tensor."""
    if group_tof.ndim != 2 or group_tof.shape[0] != NUM_PIXELS:
        raise ValueError("group_tof must have shape [4096, P]")

    pages_per_group = group_tof.shape[1]
    # [4096, P] -> [P, 64, 64] -> [P, 1, 64, 64]
    frames = group_tof.T.reshape(pages_per_group, 64, 64)
    frames_tensor = torch.from_numpy(frames.astype(np.float32, copy=False))
    frames_tensor = frames_tensor.unsqueeze(1).contiguous()

    if normalize:
        if time_threshold is None or time_threshold <= 0:
            raise ValueError("time_threshold must be positive when normalize=True")
        frames_tensor = frames_tensor / float(time_threshold)
    return frames_tensor


def time_first_to_model_input(frames: torch.Tensor) -> torch.Tensor:
    """Convert ``[P, B, 1, 64, 64]`` frames to ``[B, 4096, P]``.

    ``SNN.py`` and ``SNN_new.py`` currently accept the flattened format, while
    many SNN layers prefer the explicit time-first 5D layout.
    """
    if frames.dim() != 5 or frames.shape[2:] != (1, 64, 64):
        raise ValueError("frames must have shape [P, B, 1, 64, 64]")

    # [P, B, 1, 64, 64] -> [B, 64, 64, P] -> [B, 4096, P]
    return (
        frames.squeeze(2)
        .permute(1, 2, 3, 0)
        .reshape(frames.shape[1], NUM_PIXELS, frames.shape[0])
    )


def build_point_cloud_from_group(
    group_tof: np.ndarray,
    time_threshold: int,
) -> np.ndarray:
    """Build ``[N, 3]`` point cloud rows ``(x, y, tof)`` from one group."""
    if group_tof.ndim != 2 or group_tof.shape[0] != NUM_PIXELS:
        raise ValueError("group_tof must have shape [4096, P]")

    frames = group_tof.T.reshape(group_tof.shape[1], 64, 64)
    valid_mask = (frames > 0) & (frames < time_threshold)
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
    """Generate a weak ``[2, 64, 64]`` label from one grouped sample."""
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
    """Group-level SPAD raw dataset for SNN training.

    Each item contains:
        frames: ``[P, 1, 64, 64]``
        label: ``[2, 64, 64]`` when ``return_label=True``

    Args:
        raw_paths: Raw files to read.
        pages_per_group: Number of pages per sample group, i.e. ``P``.
        total_pages: Optional page count per raw file. If ``None``, all
            complete groups in each file are used.
        time_threshold: Values greater than this threshold are set to zero by
            ``raw2frame``. Labels also use this threshold as the valid upper
            bound.
        return_label: Whether to generate weak labels from each group.
        normalize: Whether to divide input frames by ``time_threshold``.
        shuffle_pages: Whether to randomly permute the ``P`` dimension.
        cache_size: Number of raw files whose grouped arrays are kept in RAM.
        transform: Optional transform applied to ``frames``.
        label_transform: Optional transform applied to ``label``.
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
        samples: Sequence[RawGroupSample] | None = None,
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

        self.pages_per_group = int(pages_per_group)
        self.total_pages = total_pages
        self.time_threshold = int(time_threshold)
        self.return_label = bool(return_label)
        self.normalize = bool(normalize)
        self.shuffle_pages = bool(shuffle_pages)
        self.active_point = int(active_point)
        self.cache_size = max(0, int(cache_size))
        self.transform = transform
        self.label_transform = label_transform

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

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        grouped = self._load_raw_groups(sample)
        group_tof = grouped[sample.group_index]

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

        if self.return_label:
            label = build_weak_label_from_group(
                group_tof,
                time_threshold=self.time_threshold,
                active_point=self.active_point,
            )
            if self.label_transform is not None:
                label = self.label_transform(label)
            item["label"] = label

        return item


def spad_time_first_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate samples into a time-first SNN batch.

    Returns:
        Dict with ``frames`` shaped ``[P, B, 1, 64, 64]``. If labels are
        present, ``label`` is shaped ``[B, 2, 64, 64]``.
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
    seed: int = DEFAULT_SEED,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader whose ``frames`` batch is ``[P, B, 1, 64, 64]``."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=spad_time_first_collate,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def split_indices(
    num_samples: int,
    split_ratios: Sequence[float] = (0.7, 0.2, 0.1),
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[list[int], list[int], list[int]]:
    """Split sample indices into train, val and test sets."""
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
    pages_per_group: int,
    total_pages: int | None = None,
    time_threshold: int = 150,
    batch_size: int = 4,
    split_ratios: Sequence[float] = (0.7, 0.2, 0.1),
    seed: int = DEFAULT_SEED,
    return_label: bool = True,
    normalize: bool = False,
    shuffle_pages: bool = False,
    active_point: int = 1,
    cache_size: int = 2,
    recursive: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, SpadRawGroupDataset]:
    """Build train/val/test DataLoaders from raw files or directories.

    The returned loaders all yield ``batch["frames"]`` with shape
    ``[P, B, 1, 64, 64]``.
    """
    raw_paths = collect_raw_paths(paths, recursive=recursive)
    if not raw_paths:
        raise ValueError(f"no .raw files found in paths: {paths}")

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
    )
    train_indices, val_indices, test_indices = split_indices(
        len(dataset),
        split_ratios=split_ratios,
        seed=seed,
    )

    train_loader = make_spad_dataloader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        drop_last=drop_last,
    )
    val_loader = make_spad_dataloader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed + 1,
    )
    test_loader = make_spad_dataloader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed + 2,
    )
    return train_loader, val_loader, test_loader, dataset


def create_spad_dataloader(
    paths: str | Path | Sequence[str | Path],
    *,
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
    recursive: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = False,
) -> tuple[DataLoader, SpadRawGroupDataset]:
    """Build one DataLoader from raw files or directories."""
    raw_paths = collect_raw_paths(paths, recursive=recursive)
    if not raw_paths:
        raise ValueError(f"no .raw files found in paths: {paths}")

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
    )
    loader = make_spad_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        drop_last=drop_last,
    )
    return loader, dataset


if __name__ == "__main__":
    # Example:
    # loader, dataset = create_spad_dataloader(
    #     r"E:\path\to\raw_dir",
    #     pages_per_group=500,
    #     time_threshold=150,
    #     batch_size=4,
    # )
    # batch = next(iter(loader))
    # print(batch["frames"].shape)  # torch.Size([500, 4, 1, 64, 64])
    pass

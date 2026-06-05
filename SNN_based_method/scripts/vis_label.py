"""label 可视化工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def visualize_labels(
	label_root: str | Path,
	pages_per_group: str | None = None,
	class_: str | None = None,
) -> None:
	"""可视化 label 文件夹中的深度图与强度图。

	Args:
		label_root: label 根目录路径。
		pages_per_group: pages_per_group 子文件夹名，为空则遍历全部。
		class_: class 子文件夹名，为空则遍历全部。

	Returns:
		None.

	Raises:
		FileNotFoundError: label_root 或子文件夹不存在。
		NotADirectoryError: 期望为文件夹但实际不是。
		ValueError: label 文件为空或形状不符合要求。
	"""
	root_path = Path(label_root)
	if not root_path.exists():
		raise FileNotFoundError(f"label 根目录不存在: {root_path}")
	if not root_path.is_dir():
		raise NotADirectoryError(f"label 根目录不是文件夹: {root_path}")

	page_dirs = _resolve_page_dirs(root_path, pages_per_group)
	for page_dir in page_dirs:
		class_dirs = _resolve_class_dirs(page_dir, class_)
		for class_dir in class_dirs:
			_visualize_class_labels(page_dir, class_dir)


def _resolve_page_dirs(label_root: Path, pages_per_group: str | None) -> list[Path]:
	if pages_per_group is not None:
		page_dir = label_root / pages_per_group
		if not page_dir.exists():
			raise FileNotFoundError(f"pages_per_group 子目录不存在: {page_dir}")
		if not page_dir.is_dir():
			raise NotADirectoryError(f"pages_per_group 不是文件夹: {page_dir}")
		return [page_dir]

	page_dirs = sorted(p for p in label_root.iterdir() if p.is_dir())
	if not page_dirs:
		raise ValueError(f"label 根目录下没有 pages_per_group 子目录: {label_root}")
	return page_dirs


def _resolve_class_dirs(page_dir: Path, class_name: str | None) -> list[Path]:
	if class_name is not None:
		class_dir = page_dir / class_name
		if not class_dir.exists():
			raise FileNotFoundError(f"class 子目录不存在: {class_dir}")
		if not class_dir.is_dir():
			raise NotADirectoryError(f"class 不是文件夹: {class_dir}")
		return [class_dir]

	class_dirs = sorted(p for p in page_dir.iterdir() if p.is_dir())
	if not class_dirs:
		raise ValueError(f"pages_per_group 下没有 class 子目录: {page_dir}")
	return class_dirs


def _visualize_class_labels(page_dir: Path, class_dir: Path) -> None:
	label_paths = sorted(class_dir.glob("*.npy"))
	if not label_paths:
		raise ValueError(f"class 目录下没有 label 文件: {class_dir}")
	if len(label_paths) != 5:
		print(
			"提示: class 下 label 数量不是 5 个, "
			f"实际为 {len(label_paths)} 个: {class_dir}"
		)

	label_arrays = [_load_label_array(label_path) for label_path in label_paths]
	_plot_label_group(page_dir.name, class_dir.name, label_paths, label_arrays)


def _load_label_array(label_path: Path) -> np.ndarray:
	label_array = np.load(label_path)
	if label_array.shape != (2, 64, 64):
		raise ValueError(
			"label 形状必须为 (2, 64, 64), "
			f"实际为 {label_array.shape}: {label_path}"
		)
	return label_array


def _plot_label_group(
	page_name: str,
	class_name: str,
	label_paths: Iterable[Path],
	label_arrays: Iterable[np.ndarray],
) -> None:
	label_list = list(label_paths)
	array_list = list(label_arrays)
	row_count = len(label_list)

	fig, axes = plt.subplots(
		nrows=row_count,
		ncols=2,
		figsize=(6, max(3, row_count * 2.6)),
		squeeze=False,
	)
	fig.suptitle(f"{page_name}/{class_name}")

	for row_index, (label_path, label_array) in enumerate(
		zip(label_list, array_list)
	):
		depth_map = label_array[0]
		intensity_map = _normalize_intensity(label_array[1])

		depth_im = axes[row_index, 0].imshow(
			depth_map, cmap="turbo", vmin=0.0, vmax=128.0
		)
		axes[row_index, 0].set_title(f"{label_path.name} depth")
		axes[row_index, 0].axis("off")
		fig.colorbar(depth_im, ax=axes[row_index, 0], fraction=0.046, pad=0.04)

		intensity_im = axes[row_index, 1].imshow(intensity_map, cmap="inferno")
		axes[row_index, 1].set_title(f"{label_path.name} intensity")
		axes[row_index, 1].axis("off")
		fig.colorbar(
			intensity_im, ax=axes[row_index, 1], fraction=0.046, pad=0.04
		)

	fig.tight_layout(rect=(0, 0, 1, 0.96))
	plt.show()
	plt.close(fig)


def _normalize_intensity(intensity_map: np.ndarray) -> np.ndarray:
	intensity_map = intensity_map.astype(np.float32, copy=False)
	min_value = float(np.min(intensity_map))
	max_value = float(np.max(intensity_map))
	if max_value == min_value:
		return np.zeros_like(intensity_map)
	return (intensity_map - min_value) / (max_value - min_value)


if __name__ == "__main__":
    demo_label_root = r"D:\\PYproject\\SPADdata\\0825\\label"
    visualize_labels(demo_label_root, "128", "K")
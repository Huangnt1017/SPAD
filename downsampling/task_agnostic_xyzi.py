"""强度感知、任务无关 SampleNet-XYZI 训练所需的数据与损失工具。

论文名称：SampleNet: Differentiable Point Cloud Sampling
官方 GitHub：https://github.com/itailang/SampleNet
复现状态：本模块的强度加权覆盖损失和无标签训练流程是 SPAD 本地扩展，
不属于 SampleNet 官方 GitHub 源码。

BibTeX::

    @inproceedings{lang2020samplenet,
      title={SampleNet: Differentiable Point Cloud Sampling},
      author={Lang, Itai and Manor, Asaf and Avidan, Shai},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and
                 Pattern Recognition},
      pages={7578--7588},
      year={2020}
    }

本模块不读取类别目录语义，也不产生分类标签。训练路径始终为：
``原始 XYZI -> 8192 个候选点 -> SampleNet-XYZI -> 任务无关损失``。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data import load_point_cloud_auto, parse_formal_window_filename

from .common import DownsampleOutput


def scan_formal_point_files(
    data_root: Path,
    max_files: int = 0,
) -> List[Path]:
    """递归扫描严格合法的正式三页窗口 TXT 文件。

    目录名只用于保持相对路径，不解释为 A--Z 类别标签。

    Args:
        data_root: 正式数据根目录。
        max_files: 大于零时只返回排序后的前若干文件；零表示全部。

    Returns:
        按相对路径排序的绝对文件路径。
    """

    root = data_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在或不是目录: {root}")
    if max_files < 0:
        raise ValueError(f"max_files 必须大于等于 0，实际为 {max_files}")

    formal_files = [
        path.resolve()
        for path in root.rglob("*.txt")
        if path.is_file() and parse_formal_window_filename(path.name) is not None
    ]
    formal_files.sort(key=lambda path: path.relative_to(root).as_posix())
    if max_files > 0:
        formal_files = formal_files[:max_files]
    if not formal_files:
        raise FileNotFoundError(
            "未扫描到正式点云文件；要求文件名严格匹配 "
            "yyyy-mm-dd_hh-mm-ss_Delay-0_Width-200-i-(i+2).txt，"
            f"数据根目录: {root}"
        )
    return formal_files


def split_formal_files(
    files: Sequence[Path],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    """按稳定路径哈希划分无标签训练集和验证集。"""

    if len(files) < 2:
        raise ValueError("训练至少需要 2 个正式文件，才能生成训练集和验证集")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio 必须位于 (0, 1)，实际为 {val_ratio}")

    def stable_key(path: Path) -> bytes:
        payload = f"{seed}:{path.as_posix()}".encode("utf-8")
        return hashlib.sha256(payload).digest()

    ordered = sorted((Path(path) for path in files), key=stable_key)
    val_count = max(1, min(len(ordered) - 1, round(len(ordered) * val_ratio)))
    val_files = sorted(ordered[:val_count])
    train_files = sorted(ordered[val_count:])
    return train_files, val_files


def _stable_sample_seed(relative_path: Path, seed: int) -> int:
    """由相对路径和全局种子生成跨进程稳定的 NumPy 种子。"""

    payload = f"{seed}:{relative_path.as_posix()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def load_candidate_points(
    file_path: Path,
    data_root: Path,
    candidate_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """从正式文件无放回选择固定数量候选点。

    返回候选点及其在原始 TXT 中的行索引。正式文件点数不足时直接失败，
    避免补点导致导出索引虽不同、实际原始行却重复。
    """

    if candidate_points <= 0:
        raise ValueError(
            f"candidate_points 必须为正整数，实际为 {candidate_points}"
        )
    root = data_root.expanduser().resolve()
    resolved_path = file_path.expanduser().resolve()
    try:
        relative_path = resolved_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"点云文件必须位于数据根目录内: file={resolved_path}, root={root}"
        ) from exc
    if parse_formal_window_filename(resolved_path.name) is None:
        raise ValueError(f"拒绝非正式文件名: {resolved_path.name}")

    points = load_point_cloud_auto(str(resolved_path))
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(
            f"点云必须为 (N, 4) XYZI，实际为 {points.shape}: {resolved_path}"
        )
    if not np.isfinite(points).all():
        raise ValueError(f"点云包含 NaN 或 Inf: {resolved_path}")
    num_raw_points = int(points.shape[0])
    if num_raw_points < candidate_points:
        raise ValueError(
            f"正式文件点数 {num_raw_points} 少于 candidate_points={candidate_points}: "
            f"{resolved_path}"
        )

    generator = np.random.default_rng(_stable_sample_seed(relative_path, seed))
    source_indices = generator.choice(
        num_raw_points,
        size=candidate_points,
        replace=False,
    ).astype(np.int64, copy=False)
    selected_points = np.ascontiguousarray(points[source_indices], dtype=np.float32)
    return selected_points, source_indices


class FormalXYZICandidateDataset(Dataset[Dict[str, object]]):
    """无标签正式 XYZI 候选点数据集。"""

    def __init__(
        self,
        data_root: Path,
        files: Sequence[Path],
        candidate_points: int = 8192,
        seed: int = 42,
    ) -> None:
        if not files:
            raise ValueError("files 不能为空")
        if candidate_points <= 0:
            raise ValueError("candidate_points 必须为正整数")
        self.data_root = data_root.expanduser().resolve()
        self.files = [Path(path).expanduser().resolve() for path in files]
        self.candidate_points = int(candidate_points)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, object]:
        file_path = self.files[index]
        points, source_indices = load_candidate_points(
            file_path=file_path,
            data_root=self.data_root,
            candidate_points=self.candidate_points,
            seed=self.seed,
        )
        relative_path = file_path.relative_to(self.data_root).as_posix()
        return {
            "points": torch.from_numpy(points),
            "source_indices": torch.from_numpy(source_indices),
            "relative_path": relative_path,
        }


def log1p_intensity_weighted_coverage_loss(
    input_points: torch.Tensor,
    projected_points: torch.Tensor,
    chunk_size: int = 512,
    eps: float = 1e-6,
) -> torch.Tensor:
    """计算 ``log1p`` 强度加权的输入到软投影覆盖距离。

    距离在按输入范围归一化的 XYZ 空间计算。查询点使用
    ``projected_points``，因此损失保持从软投影权重到采样网络的梯度。
    全零强度样本退化为均匀覆盖损失。
    """

    if input_points.ndim != 3 or input_points.shape[-1] != 4:
        raise ValueError(
            f"input_points 必须为 (B, N, 4)，实际为 {tuple(input_points.shape)}"
        )
    if projected_points.ndim != 3 or projected_points.shape[-1] != 4:
        raise ValueError(
            "projected_points 必须为 (B, K, 4)，"
            f"实际为 {tuple(projected_points.shape)}"
        )
    if projected_points.shape[0] != input_points.shape[0]:
        raise ValueError("input_points 与 projected_points 批大小不一致")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正整数，实际为 {chunk_size}")
    if eps <= 0:
        raise ValueError(f"eps 必须为正数，实际为 {eps}")

    input_xyz = input_points[..., :3]
    xyz_min = input_xyz.amin(dim=1, keepdim=True)
    xyz_scale = (input_xyz.amax(dim=1, keepdim=True) - xyz_min).clamp_min(eps)
    normalized_input_xyz = (input_xyz - xyz_min) / xyz_scale
    normalized_projected_xyz = (projected_points[..., :3] - xyz_min) / xyz_scale

    intensity_weights = torch.log1p(input_points[..., 3].clamp_min(0.0))
    weighted_sum = torch.zeros(
        input_points.shape[0],
        dtype=input_points.dtype,
        device=input_points.device,
    )
    uniform_sum = torch.zeros_like(weighted_sum)

    for start in range(0, input_points.shape[1], chunk_size):
        end = min(start + chunk_size, input_points.shape[1])
        min_distance = torch.cdist(
            normalized_input_xyz[:, start:end, :],
            normalized_projected_xyz,
            p=2,
        ).amin(dim=-1)
        weighted_sum = weighted_sum + (
            min_distance * intensity_weights[:, start:end]
        ).sum(dim=1)
        uniform_sum = uniform_sum + min_distance.sum(dim=1)

    weight_sum = intensity_weights.sum(dim=1)
    weighted_mean = weighted_sum / weight_sum.clamp_min(eps)
    uniform_mean = uniform_sum / max(int(input_points.shape[1]), 1)
    per_sample_loss = torch.where(
        weight_sum > eps,
        weighted_mean,
        uniform_mean,
    )
    return per_sample_loss.mean()


def task_agnostic_samplenet_loss(
    input_points: torch.Tensor,
    output: DownsampleOutput,
    geometry_weight: float = 1.0,
    intensity_weight: float = 1.0,
    projection_weight: float = 1.0,
    intensity_chunk_size: int = 512,
) -> Dict[str, torch.Tensor]:
    """组合任务无关 SampleNet-XYZI 三项损失。"""

    weights = (geometry_weight, intensity_weight, projection_weight)
    if any(weight < 0 for weight in weights):
        raise ValueError("三项损失权重均必须大于等于 0")
    if output.generated_points is None:
        raise ValueError("SampleNet 输出缺少 generated_points")
    if output.projected_points is None:
        raise ValueError("SampleNet 输出缺少 projected_points")
    if output.generated_points.shape[:2] != output.projected_points.shape[:2]:
        raise ValueError("generated_points 与 projected_points 点数不一致")

    try:
        geometry_loss = output.aux_losses["simplification"]
        projection_loss = output.aux_losses["projection_temperature"]
    except KeyError as exc:
        raise ValueError("SampleNet 输出缺少现有几何或温度损失") from exc
    intensity_loss = log1p_intensity_weighted_coverage_loss(
        input_points=input_points,
        projected_points=output.projected_points,
        chunk_size=intensity_chunk_size,
    )
    total_loss = (
        geometry_weight * geometry_loss
        + intensity_weight * intensity_loss
        + projection_weight * projection_loss
    )
    return {
        "total": total_loss,
        "geometry": geometry_loss,
        "intensity_coverage": intensity_loss,
        "projection_temperature": projection_loss,
    }


"""SPAD ``xyzi`` 点云降采样的公共张量工具。

统一数据契约：

- 输入点云：``(B, N, 4)``，最后一维依次为 ``x, y, z, intensity``；
- 输出索引：``(B, K)``，每个样本内索引互异；
- 输出点云：``(B, K, 4)``，始终从原始输入按索引提取。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class DownsampleOutput:
    """统一的降采样输出。

    Attributes:
        points: 从原始输入按 ``indices`` 提取的硬点子集，形状 ``(B, K, 4)``。
        indices: 每个样本内互异的原始点索引，形状 ``(B, K)``。
        projected_points: 可选的可微软投影点，形状 ``(B, K, 4)``。
        generated_points: 可选的采样网络生成查询点，通常为 ``(B, K, 3)``。
        features: 可选的选点特征，通常为 ``(B, C, K)``。
        scores: 可选的逐输入点分数，形状 ``(B, N)``。
        aux_losses: 学习型采样器提供的辅助标量损失。
    """

    points: torch.Tensor
    indices: torch.Tensor
    projected_points: Optional[torch.Tensor] = None
    generated_points: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    aux_losses: Dict[str, torch.Tensor] = field(default_factory=dict)


def validate_xyzi_points(points: torch.Tensor, num_samples: int) -> None:
    """尽早校验点云张量与目标点数。

    Args:
        points: ``(B, N, 4)`` 浮点张量。
        num_samples: 每个样本要选择的点数 ``K``。

    Raises:
        TypeError: 输入不是浮点 ``torch.Tensor``。
        ValueError: 布局、数值或目标点数不合法。
    """

    if not isinstance(points, torch.Tensor):
        raise TypeError(f"points must be a torch.Tensor, got {type(points)!r}")
    if not points.is_floating_point():
        raise TypeError(f"points must use a floating dtype, got {points.dtype}")
    if points.ndim != 3 or points.shape[-1] != 4:
        raise ValueError(
            "points must have shape (B, N, 4) with xyzi channels, "
            f"got {tuple(points.shape)}"
        )
    if points.shape[0] <= 0:
        raise ValueError("points must contain at least one batch item")
    if points.shape[1] <= 0:
        raise ValueError("points must contain at least one point per sample")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        raise TypeError(f"num_samples must be an int, got {type(num_samples)!r}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_samples > points.shape[1]:
        raise ValueError(
            f"num_samples={num_samples} exceeds input point count N={points.shape[1]}"
        )
    if not torch.isfinite(points).all():
        raise ValueError("points contains NaN or Inf values")


def normalize_xyzi(points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """逐样本归一化 ``xyz``，并对计数执行 ``log1p`` 后归一化。

    该函数只生成采样器内部使用的特征，不修改最终输出的原始点值。

    Args:
        points: ``(B, N, 4)``。
        eps: 防止除零的正数。

    Returns:
        ``(B, N, 4)``，四个通道均位于 ``[0, 1]``。
    """

    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if points.ndim != 3 or points.shape[-1] != 4:
        raise ValueError(f"points must have shape (B, N, 4), got {tuple(points.shape)}")

    xyz = points[..., :3]
    intensity = torch.log1p(points[..., 3:4].clamp_min(0.0))

    xyz_min = xyz.amin(dim=1, keepdim=True)
    xyz_max = xyz.amax(dim=1, keepdim=True)
    xyz_norm = (xyz - xyz_min) / (xyz_max - xyz_min).clamp_min(eps)

    intensity_min = intensity.amin(dim=1, keepdim=True)
    intensity_max = intensity.amax(dim=1, keepdim=True)
    intensity_norm = (intensity - intensity_min) / (
        intensity_max - intensity_min
    ).clamp_min(eps)

    return torch.cat((xyz_norm, intensity_norm), dim=-1)


def gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """按批次索引从原始 ``xyzi`` 点云提取点。

    Args:
        points: ``(B, N, 4)``。
        indices: ``(B, K)``，整数索引。

    Returns:
        ``(B, K, 4)``。
    """

    if indices.ndim != 2 or indices.shape[0] != points.shape[0]:
        raise ValueError(
            "indices must have shape (B, K) and match points batch size, "
            f"got points={tuple(points.shape)}, indices={tuple(indices.shape)}"
        )
    if indices.dtype != torch.long:
        raise TypeError(f"indices must use torch.long, got {indices.dtype}")
    if indices.device != points.device:
        raise ValueError("points and indices must be on the same device")
    if indices.numel() > 0:
        if int(indices.amin()) < 0 or int(indices.amax()) >= points.shape[1]:
            raise IndexError(
                f"indices must be in [0, {points.shape[1] - 1}]"
            )

    gather_index = indices.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    return torch.gather(points, dim=1, index=gather_index)


def gather_neighbors(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """为 channel-first 特征提取 KNN 邻居。

    Args:
        features: ``(B, C, N)``。
        indices: ``(B, Q, K)``。

    Returns:
        ``(B, C, Q, K)``。
    """

    if features.ndim != 3:
        raise ValueError(f"features must have shape (B, C, N), got {features.shape}")
    if indices.ndim != 3 or indices.shape[0] != features.shape[0]:
        raise ValueError(
            "indices must have shape (B, Q, K) and match features batch size"
        )
    if indices.dtype != torch.long:
        raise TypeError(f"indices must use torch.long, got {indices.dtype}")

    batch_size = features.shape[0]
    features_bnc = features.transpose(1, 2).contiguous()
    batch_indices = torch.arange(
        batch_size,
        device=features.device,
    ).view(batch_size, 1, 1)
    neighbors = features_bnc[batch_indices, indices]
    return neighbors.permute(0, 3, 1, 2).contiguous()


def knn_indices_chunked(
    xyz: torch.Tensor,
    k: int,
    chunk_size: int = 1024,
    exclude_self: bool = True,
) -> torch.Tensor:
    """以查询分块方式计算欧氏 KNN，避免一次保存完整查询距离矩阵。

    Args:
        xyz: ``(B, N, D)``，通常为归一化 ``xyz``。
        k: 每个点的近邻数。
        chunk_size: 单次处理的查询点数。
        exclude_self: 是否排除中心点自身。

    Returns:
        ``(B, N, k)`` 的近邻索引。
    """

    if xyz.ndim != 3:
        raise ValueError(f"xyz must have shape (B, N, D), got {tuple(xyz.shape)}")
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive int, got {k}")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive int, got {chunk_size}")

    batch_size, num_points, _ = xyz.shape
    max_neighbors = num_points - 1 if exclude_self else num_points
    if k > max_neighbors:
        raise ValueError(
            f"k={k} exceeds available neighbors {max_neighbors} for N={num_points}"
        )

    neighbor_parts = []
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        query = xyz[:, start:end, :]
        distances = torch.cdist(query, xyz, p=2)

        if exclude_self:
            local_count = end - start
            batch_axis = torch.arange(batch_size, device=xyz.device)[:, None]
            query_axis = torch.arange(local_count, device=xyz.device)[None, :]
            source_axis = torch.arange(start, end, device=xyz.device)[None, :]
            distances[batch_axis, query_axis, source_axis] = torch.inf

        indices = distances.topk(k=k, dim=-1, largest=False, sorted=False).indices
        neighbor_parts.append(indices)

    return torch.cat(neighbor_parts, dim=1)


def deduplicate_and_fill_indices(
    primary_indices: torch.Tensor,
    candidate_priority: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """保留首次命中的索引，并按候选优先级补齐为互异的 ``K`` 点。

    ``candidate_priority`` 越小越优先。实现先把每个输入点压缩成一个
    选择键，再在输入点维度执行一次 Top-K，因此不会产生重复索引。

    Args:
        primary_indices: ``(B, L)``，允许重复。
        candidate_priority: ``(B, N)``，越小越优先。
        num_samples: 输出点数 ``K``。

    Returns:
        ``(B, K)``，每个样本内索引互异。
    """

    if primary_indices.ndim != 2:
        raise ValueError("primary_indices must have shape (B, L)")
    if candidate_priority.ndim != 2:
        raise ValueError("candidate_priority must have shape (B, N)")
    if primary_indices.shape[0] != candidate_priority.shape[0]:
        raise ValueError("primary_indices and candidate_priority batch sizes differ")
    if primary_indices.dtype != torch.long:
        raise TypeError("primary_indices must use torch.long")
    if primary_indices.device != candidate_priority.device:
        raise ValueError("primary_indices and candidate_priority must share a device")

    batch_size, primary_count = primary_indices.shape
    num_candidates = candidate_priority.shape[1]
    if not 0 < num_samples <= num_candidates:
        raise ValueError(
            f"num_samples must be in [1, {num_candidates}], got {num_samples}"
        )
    if primary_indices.numel() > 0:
        if int(primary_indices.amin()) < 0 or int(primary_indices.amax()) >= num_candidates:
            raise IndexError("primary_indices contains an out-of-range index")

    first_position = torch.full(
        (batch_size, num_candidates),
        fill_value=primary_count,
        dtype=torch.long,
        device=primary_indices.device,
    )
    positions = torch.arange(
        primary_count,
        device=primary_indices.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1)

    if hasattr(first_position, "scatter_reduce_"):
        first_position.scatter_reduce_(
            dim=1,
            index=primary_indices,
            src=positions,
            reduce="amin",
            include_self=True,
        )
    else:  # pragma: no cover - 兼容旧版 PyTorch。
        for position in range(primary_count):
            point_index = primary_indices[:, position : position + 1]
            old_value = first_position.gather(1, point_index)
            new_value = torch.minimum(
                old_value,
                torch.full_like(old_value, position),
            )
            first_position.scatter_(1, point_index, new_value)

    priority = candidate_priority.detach()
    finite_priority = torch.where(
        torch.isfinite(priority),
        priority,
        torch.full_like(priority, torch.finfo(priority.dtype).max),
    )
    priority_min = finite_priority.amin(dim=1, keepdim=True)
    priority_max = finite_priority.amax(dim=1, keepdim=True)
    priority_norm = (finite_priority - priority_min) / (
        priority_max - priority_min
    ).clamp_min(torch.finfo(priority.dtype).eps)

    candidate_index = torch.arange(
        num_candidates,
        device=priority.device,
        dtype=priority.dtype,
    ).unsqueeze(0)
    tie_break = candidate_index / max(num_candidates, 1) * 1e-6
    selection_key = primary_count + priority_norm + tie_break

    primary_mask = first_position < primary_count
    selection_key = torch.where(
        primary_mask,
        first_position.to(selection_key.dtype),
        selection_key,
    )

    return selection_key.topk(
        k=num_samples,
        dim=1,
        largest=False,
        sorted=True,
    ).indices


def assert_unique_indices(indices: torch.Tensor) -> None:
    """验证每个批次样本的索引均互异，主要用于测试和导出前检查。"""

    if indices.ndim != 2:
        raise ValueError(f"indices must have shape (B, K), got {indices.shape}")
    sorted_indices = indices.sort(dim=1).values
    if sorted_indices.shape[1] > 1:
        duplicate_mask = sorted_indices[:, 1:] == sorted_indices[:, :-1]
        if duplicate_mask.any():
            bad_batches = duplicate_mask.any(dim=1).nonzero(as_tuple=False).flatten()
            raise ValueError(
                "indices contains duplicates in batch items "
                f"{bad_batches.tolist()}"
            )

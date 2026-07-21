"""面向 SPAD ``xyzi`` 点云的 SampleNet 风格可微降采样器。

论文名称：SampleNet: Differentiable Point Cloud Sampling
官方 GitHub：https://github.com/itailang/SampleNet
审计版本：``3d20c7a62f6788cc56b68d5367ff25a8a2c13fad``
对照源码：``registration/src/samplenet.py``、
``registration/src/soft_projection.py`` 和 ``registration/src/sputils.py``。
复现状态：本文件保留“采样网络 + 软投影 + 硬匹配”的方法骨架，但编码器、
解码器、温度参数化、XYZI 投影和去重补点均为 SPAD 本地适配，不能标注为
官方 GitHub 的逐层、逐行或 checkpoint 兼容复现。完整差异见
``downsampling/SOURCE_AUDIT.md``。

BibTeX::

    @inproceedings{lang2020samplenet,
      title={SampleNet: Differentiable Point Cloud Sampling},
      author={Lang, Itai and Manor, Asaf and Avidan, Shai},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and
                 Pattern Recognition},
      pages={7578--7588},
      year={2020}
    }

实现保留 SampleNet 的核心思路：PointNet 风格采样网络生成三维查询点，
训练时以温度控制的邻域软投影保持可微，推理时映射为互异的原始点索引。
本文件为适配本项目数据契约的独立 PyTorch 实现，不依赖外部 CUDA 算子。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .common import (
    DownsampleOutput,
    deduplicate_and_fill_indices,
    gather_neighbors,
    gather_points,
    normalize_xyzi,
    validate_xyzi_points,
)


class SampleNetXYZI(nn.Module):
    """SampleNet 的 SPAD ``xyzi`` 本地适配。

    编码器读取四维 ``xyzi``，解码器仅生成归一化三维查询点。邻域距离
    始终在 ``xyz`` 空间计算，同一软投影权重用于聚合完整的原始 ``xyzi``。

    Args:
        num_samples: 输出点数 ``K``。
        projection_neighbors: 每个生成点用于软投影的输入近邻数。
        feature_dim: PointNet 风格逐点编码的最终通道数。
        hidden_dim: 全连接解码器隐层宽度。
        initial_temperature: 软投影初始温度。
        min_temperature: 温度严格正下界。
        coverage_weight: 简化损失中输入覆盖项的权重。
        distance_chunk_size: 分块计算生成点到输入点距离时的查询块大小。
        eps: 数值稳定常数。
    """

    def __init__(
        self,
        num_samples: int = 1024,
        projection_neighbors: int = 8,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.01,
        coverage_weight: float = 1.0,
        distance_chunk_size: int = 256,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive int, got {num_samples}")
        if not isinstance(projection_neighbors, int) or projection_neighbors <= 0:
            raise ValueError(
                "projection_neighbors must be a positive int, "
                f"got {projection_neighbors}"
            )
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if min_temperature <= 0:
            raise ValueError("min_temperature must be positive")
        if initial_temperature <= min_temperature:
            raise ValueError(
                "initial_temperature must be greater than min_temperature"
            )
        if coverage_weight < 0:
            raise ValueError("coverage_weight must be non-negative")
        if not isinstance(distance_chunk_size, int) or distance_chunk_size <= 0:
            raise ValueError("distance_chunk_size must be a positive int")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.num_samples = num_samples
        self.projection_neighbors = projection_neighbors
        self.coverage_weight = float(coverage_weight)
        self.distance_chunk_size = distance_chunk_size
        self.min_temperature = float(min_temperature)
        self.eps = float(eps)

        self.encoder = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_samples * 3),
        )

        temperature_offset = initial_temperature - min_temperature
        inverse_softplus = math.log(math.expm1(temperature_offset))
        self._temperature_unconstrained = nn.Parameter(
            torch.tensor(inverse_softplus, dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        """返回带严格正下界的当前软投影温度。"""

        return self.min_temperature + F.softplus(self._temperature_unconstrained)

    def _generate_queries(self, normalized_points: torch.Tensor) -> torch.Tensor:
        """由 ``xyzi`` 全局特征生成 ``(B, K, 3)`` 归一化查询点。"""

        features = self.encoder(normalized_points.transpose(1, 2).contiguous())
        pooled = torch.cat(
            (features.amax(dim=-1), features.mean(dim=-1)),
            dim=1,
        )
        query_logits = self.decoder(pooled)
        return torch.sigmoid(
            query_logits.view(points_batch_size(normalized_points), self.num_samples, 3)
        )

    def _soft_project(
        self,
        generated_xyz: torch.Tensor,
        normalized_xyz: torch.Tensor,
        original_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """分块完成软投影，并同时收集硬投影所需的距离与索引。"""

        batch_size, num_input_points, _ = normalized_xyz.shape
        neighbor_count = min(self.projection_neighbors, num_input_points)
        temperature = self.temperature.to(
            device=original_points.device,
            dtype=original_points.dtype,
        )

        projected_parts = []
        primary_index_parts = []
        generated_min_distance_parts = []
        input_min_distance = torch.full(
            (batch_size, num_input_points),
            fill_value=torch.inf,
            dtype=original_points.dtype,
            device=original_points.device,
        )
        original_channel_first = original_points.transpose(1, 2).contiguous()

        for start in range(0, self.num_samples, self.distance_chunk_size):
            end = min(start + self.distance_chunk_size, self.num_samples)
            query_chunk = generated_xyz[:, start:end, :]
            distances = torch.cdist(query_chunk, normalized_xyz, p=2)

            nearest_distances, nearest_indices = distances.topk(
                k=neighbor_count,
                dim=-1,
                largest=False,
                sorted=True,
            )
            projection_weights = torch.softmax(
                -nearest_distances.square() / temperature.clamp_min(self.eps),
                dim=-1,
            )

            neighbor_points = gather_neighbors(
                original_channel_first,
                nearest_indices,
            ).permute(0, 2, 3, 1)
            projected_chunk = (
                projection_weights.unsqueeze(-1) * neighbor_points
            ).sum(dim=2)

            projected_parts.append(projected_chunk)
            primary_index_parts.append(nearest_indices[..., 0])
            generated_min_distance_parts.append(nearest_distances[..., 0])
            input_min_distance = torch.minimum(
                input_min_distance,
                distances.amin(dim=1),
            )

        return (
            torch.cat(projected_parts, dim=1),
            torch.cat(primary_index_parts, dim=1),
            torch.cat(generated_min_distance_parts, dim=1),
            input_min_distance,
        )

    def forward(self, points: torch.Tensor) -> DownsampleOutput:
        """生成可微软投影以及互异的原始硬点子集。

        训练时应把 ``output.projected_points`` 送入固定任务网络；导出和评价时
        使用 ``output.points`` 与 ``output.indices``。
        """

        validate_xyzi_points(points, self.num_samples)
        normalized_points = normalize_xyzi(points, eps=self.eps)
        generated_xyz = self._generate_queries(normalized_points)

        (
            projected_points,
            primary_indices,
            generated_min_distance,
            input_min_distance,
        ) = self._soft_project(
            generated_xyz=generated_xyz,
            normalized_xyz=normalized_points[..., :3],
            original_points=points,
        )

        hard_indices = deduplicate_and_fill_indices(
            primary_indices=primary_indices,
            candidate_priority=input_min_distance,
            num_samples=self.num_samples,
        )
        hard_points = gather_points(points, hard_indices)

        mean_nearest = generated_min_distance.mean()
        max_nearest = generated_min_distance.amax(dim=1).mean()
        coverage = input_min_distance.mean()
        simplification_loss = (
            mean_nearest
            + max_nearest
            + self.coverage_weight * coverage
        )
        projection_loss = self.temperature.square()

        aux_losses: Dict[str, torch.Tensor] = {
            "simplification": simplification_loss,
            "projection_temperature": projection_loss,
        }

        return DownsampleOutput(
            points=hard_points,
            indices=hard_indices,
            projected_points=projected_points,
            generated_points=generated_xyz,
            scores=-input_min_distance,
            aux_losses=aux_losses,
        )

    @staticmethod
    def sampler_loss(
        output: DownsampleOutput,
        simplification_weight: float = 1.0,
        projection_weight: float = 1.0,
    ) -> torch.Tensor:
        """按给定权重组合 SampleNet 的两个辅助损失。"""

        if simplification_weight < 0 or projection_weight < 0:
            raise ValueError("sampler loss weights must be non-negative")
        try:
            simplification = output.aux_losses["simplification"]
            projection = output.aux_losses["projection_temperature"]
        except KeyError as exc:
            raise ValueError("output does not contain SampleNet auxiliary losses") from exc
        return (
            simplification_weight * simplification
            + projection_weight * projection
        )


def points_batch_size(points: torch.Tensor) -> int:
    """返回批大小；单独封装以保持查询 reshape 的含义清晰。"""

    return int(points.shape[0])

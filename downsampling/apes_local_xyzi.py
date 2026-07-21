"""面向 SPAD ``xyzi`` 点云的 APES-Local 风格降采样器。

论文名称：Attention-Based Point Cloud Edge Sampling
官方 GitHub：https://github.com/JunweiZheng93/APES
审计版本：``988aa892980261d8685dc6734422ed4c0da25a52``
对照源码：``apes/models/utils/layers.py`` 中的 ``Embedding``、
``N2PAttention`` 和 ``LocalDownSample``。
复现状态：本文件复现了局部注意力标准差打分与 Top-K 选点核心，但邻域图、
Embedding、N2PAttention、特征聚合及 XYZI 输入均有本地改造，不能标注为
官方 GitHub 的完整网络或 checkpoint 兼容复现。完整差异见
``downsampling/SOURCE_AUDIT.md``。

BibTeX::

    @inproceedings{wu2023attention,
      title={Attention-Based Point Cloud Edge Sampling},
      author={Wu, Chengzhi and Zheng, Junwei and Pfrommer, Julius and
              Beyerer, Jürgen},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and
                 Pattern Recognition},
      pages={5333--5343},
      year={2023}
    }

实现使用归一化 ``xyz`` 构造局部 KNN，以四维 ``xyzi`` 编码点特征，
再根据邻域注意力分布的标准差选择 Top-K 输入点。硬输出始终是原始点
子集；``features`` 保留可微的注意力聚合路径，供固定任务头训练采样器。
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .common import (
    DownsampleOutput,
    gather_neighbors,
    gather_points,
    knn_indices_chunked,
    normalize_xyzi,
    validate_xyzi_points,
)


class APESLocalXYZI(nn.Module):
    """APES 局部注意力边缘采样的 SPAD ``xyzi`` 本地适配。

    Args:
        num_samples: 输出点数 ``K``。
        num_neighbors: 局部 KNN 近邻数，不包含中心点自身。
        embedding_dim: 注意力特征维度。
        knn_chunk_size: KNN 查询分块大小。
        negative_slope: LeakyReLU 负半轴斜率。
        eps: 数值稳定常数。
    """

    def __init__(
        self,
        num_samples: int = 1024,
        num_neighbors: int = 32,
        embedding_dim: int = 128,
        knn_chunk_size: int = 1024,
        negative_slope: float = 0.2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive int, got {num_samples}")
        if not isinstance(num_neighbors, int) or num_neighbors <= 0:
            raise ValueError(
                f"num_neighbors must be a positive int, got {num_neighbors}"
            )
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if not isinstance(knn_chunk_size, int) or knn_chunk_size <= 0:
            raise ValueError("knn_chunk_size must be a positive int")
        if negative_slope < 0:
            raise ValueError("negative_slope must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.num_samples = num_samples
        self.num_neighbors = num_neighbors
        self.embedding_dim = embedding_dim
        self.knn_chunk_size = knn_chunk_size
        self.eps = float(eps)

        mid_channels = max(embedding_dim // 2, 16)
        self.embedding = nn.Sequential(
            nn.Conv1d(4, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv1d(mid_channels, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(negative_slope, inplace=True),
        )
        self.query_conv = nn.Conv1d(
            embedding_dim,
            embedding_dim,
            kernel_size=1,
            bias=False,
        )
        self.key_conv = nn.Conv2d(
            embedding_dim,
            embedding_dim,
            kernel_size=1,
            bias=False,
        )
        self.value_conv = nn.Conv2d(
            embedding_dim,
            embedding_dim,
            kernel_size=1,
            bias=False,
        )
        self.output_norm = nn.BatchNorm1d(embedding_dim)
        self.output_act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, points: torch.Tensor) -> DownsampleOutput:
        """选择局部注意力边缘分数最高的 ``K`` 个原始点。

        ``output.features`` 为 ``(B, C, K)`` 可微特征。训练时任务损失应
        作用于该特征路径；导出点云时使用 ``output.points``。
        """

        validate_xyzi_points(points, self.num_samples)
        num_points = points.shape[1]
        if num_points <= 1:
            raise ValueError("APES-Local requires at least two input points")

        neighbor_count = min(self.num_neighbors, num_points - 1)
        normalized_points = normalize_xyzi(points, eps=self.eps)
        features = self.embedding(
            normalized_points.transpose(1, 2).contiguous()
        )

        # KNN 是离散拓扑，不需要保留输入坐标的梯度图。
        neighbor_indices = knn_indices_chunked(
            normalized_points[..., :3].detach(),
            k=neighbor_count,
            chunk_size=self.knn_chunk_size,
            exclude_self=True,
        )
        neighbor_features = gather_neighbors(features, neighbor_indices)
        relative_features = neighbor_features - features.unsqueeze(-1)

        query = self.query_conv(features).unsqueeze(-1)
        key = self.key_conv(relative_features)
        value = self.value_conv(relative_features)

        attention_logits = (query * key).sum(dim=1) / math.sqrt(self.embedding_dim)
        attention = torch.softmax(attention_logits, dim=-1)

        # APES-Local 使用局部注意力分布离散程度作为边缘显著性。
        scores = attention.std(dim=-1, unbiased=False)
        indices = scores.topk(
            k=self.num_samples,
            dim=1,
            largest=True,
            sorted=False,
        ).indices

        aggregated_features = (
            attention.unsqueeze(1) * value
        ).sum(dim=-1)
        gather_index = indices.unsqueeze(1).expand(
            -1,
            self.embedding_dim,
            -1,
        )
        sampled_features = torch.gather(
            aggregated_features,
            dim=2,
            index=gather_index,
        )
        sampled_features = self.output_act(
            self.output_norm(sampled_features)
        )
        sampled_points = gather_points(points, indices)

        return DownsampleOutput(
            points=sampled_points,
            indices=indices,
            features=sampled_features,
            scores=scores,
        )


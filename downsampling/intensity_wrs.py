"""基于 SPAD 光子计数的加权无放回随机降采样。

论文名称：Weighted random sampling with a reservoir
论文 DOI：https://doi.org/10.1016/j.ipl.2005.11.003
GitHub 来源：未发现论文作者发布的官方 GitHub 仓库；本文件不是 GitHub
逐行移植，而是 Algorithm A-Res 键采样的批量 PyTorch 等价实现。
复现状态：``log(U) / w`` 与论文的 ``U ** (1 / w)`` 排序等价；
``log1p`` 强度归一化、``gamma`` 权重和整批 ``topk`` 是 SPAD 本地适配。

BibTeX::

    @article{efraimidis2006weighted,
      title={Weighted random sampling with a reservoir},
      author={Efraimidis, Pavlos S. and Spirakis, Paul G.},
      journal={Information Processing Letters},
      volume={97},
      number={5},
      pages={181--185},
      year={2006},
      doi={10.1016/j.ipl.2005.11.003}
    }
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .common import DownsampleOutput, gather_points, normalize_xyzi, validate_xyzi_points


class IntensityWeightedRandomSampler(nn.Module):
    """按归一化光子计数执行加权无放回随机采样。

    对每个点生成 Efraimidis--Spirakis 键 ``log(u) / w``，选择最大的
    ``K`` 个键。``torch.topk`` 天然返回互异索引。

    Args:
        num_samples: 输出点数 ``K``。
        gamma: 强度权重指数；越大越偏向高计数点。
        eps: 数值稳定常数。
    """

    def __init__(
        self,
        num_samples: int = 1024,
        gamma: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive int, got {num_samples}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.num_samples = num_samples
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(
        self,
        points: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> DownsampleOutput:
        """从 ``(B, N, 4)`` 输入选择 ``K`` 个原始点。

        Args:
            points: ``(B, N, 4)``，通道顺序为 ``xyzi``。
            generator: 可选随机数生成器；其设备必须与 ``points`` 一致。

        Returns:
            :class:`DownsampleOutput`，其中 ``scores`` 为加权随机键。
        """

        validate_xyzi_points(points, self.num_samples)

        normalized = normalize_xyzi(points, eps=self.eps)
        intensity = normalized[..., 3]
        weights = (intensity + self.eps).pow(self.gamma)

        uniform = torch.rand(
            weights.shape,
            dtype=points.dtype,
            device=points.device,
            generator=generator,
        ).clamp_min(self.eps)
        scores = torch.log(uniform) / weights
        indices = scores.topk(
            k=self.num_samples,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        sampled_points = gather_points(points, indices)

        return DownsampleOutput(
            points=sampled_points,
            indices=indices,
            scores=scores,
        )

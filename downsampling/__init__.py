"""SPAD ``xyzi`` 点云降采样模块。

所有采样器接收 ``(B, N, 4)``，并通过 :class:`DownsampleOutput` 返回
``(B, K, 4)`` 原始点子集及 ``(B, K)`` 唯一索引。
"""

from .common import DownsampleOutput, assert_unique_indices, normalize_xyzi
from .apes_local_xyzi import APESLocalXYZI
from .factory import available_downsamplers, build_downsampler
from .intensity_wrs import IntensityWeightedRandomSampler
from .samplenet_xyzi import SampleNetXYZI

__all__ = [
    "APESLocalXYZI",
    "DownsampleOutput",
    "IntensityWeightedRandomSampler",
    "SampleNetXYZI",
    "available_downsamplers",
    "assert_unique_indices",
    "build_downsampler",
    "normalize_xyzi",
]

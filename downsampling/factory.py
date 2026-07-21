"""降采样器名称注册与统一构建入口。"""

from __future__ import annotations

from typing import Any, Tuple

from torch import nn

from .apes_local_xyzi import APESLocalXYZI
from .intensity_wrs import IntensityWeightedRandomSampler
from .samplenet_xyzi import SampleNetXYZI


_AVAILABLE_SAMPLERS: Tuple[str, ...] = (
    "i_wrs",
    "samplenet_xyzi",
    "apes_local_xyzi",
)


def available_downsamplers() -> Tuple[str, ...]:
    """返回工厂支持的规范化采样器名称。"""

    return _AVAILABLE_SAMPLERS


def build_downsampler(
    name: str,
    num_samples: int = 1024,
    **kwargs: Any,
) -> nn.Module:
    """按名称构建 SPAD ``xyzi`` 降采样器。

    Args:
        name: ``i_wrs``、``samplenet_xyzi`` 或 ``apes_local_xyzi``；
            同时接受少量便捷别名。
        num_samples: 输出点数 ``K``。
        **kwargs: 透传给对应采样器构造函数。

    Returns:
        构建完成的 ``nn.Module``。

    Raises:
        TypeError: ``name`` 不是字符串。
        ValueError: 名称不受支持。
    """

    if not isinstance(name, str):
        raise TypeError(f"name must be a string, got {type(name)!r}")

    normalized_name = name.strip().lower().replace("-", "_")
    aliases = {
        "iwrs": "i_wrs",
        "intensity_wrs": "i_wrs",
        "samplenet": "samplenet_xyzi",
        "sample_net_xyzi": "samplenet_xyzi",
        "apes": "apes_local_xyzi",
        "apes_local": "apes_local_xyzi",
    }
    normalized_name = aliases.get(normalized_name, normalized_name)

    if normalized_name == "i_wrs":
        return IntensityWeightedRandomSampler(
            num_samples=num_samples,
            **kwargs,
        )
    if normalized_name == "samplenet_xyzi":
        return SampleNetXYZI(
            num_samples=num_samples,
            **kwargs,
        )
    if normalized_name == "apes_local_xyzi":
        return APESLocalXYZI(
            num_samples=num_samples,
            **kwargs,
        )

    supported = ", ".join(_AVAILABLE_SAMPLERS)
    raise ValueError(f"Unsupported downsampler '{name}'. Supported: {supported}")

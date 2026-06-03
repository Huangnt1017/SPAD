"""SPAD SNN 模型兼容入口。

本项目现在只使用环境中安装的官方 ``spikingjelly`` 包，并统一走
``spikingjelly.activation_based`` API。保留本文件是为了兼容旧代码中的
``from SNN_based_method.SNN import SPADSpikeNet`` 导入路径；实际实现位于
``SNN_based_method.SNN_new``。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from SNN_based_method.SNN_new import (  # noqa: E402
    LearnableTofEmbedding,
    MultiScaleDSConv,
    SPADSpikeNet,
    SpatialRefineHead,
    SpikeBlock,
    build_node,
    encode_tof,
    run_5d_memory_benchmark,
)

__all__ = [
    "SPADSpikeNet",
    "build_node",
    "encode_tof",
    "LearnableTofEmbedding",
    "MultiScaleDSConv",
    "SpikeBlock",
    "SpatialRefineHead",
    "run_5d_memory_benchmark",
]

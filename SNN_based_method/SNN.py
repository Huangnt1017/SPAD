"""SPAD SNN 模型兼容入口。

本项目现在只使用环境中安装的官方 ``spikingjelly`` 包，并统一走
``spikingjelly.activation_based`` API。保留本文件是为了兼容旧代码中的
``from SNN_based_method.SNN import ...`` 导入路径，并统一导出默认 SNN
后端以及显式 RNN / ConvLSTM / ConvGRU 版本。
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
from SNN_based_method.SNN_c_RNN import (  # noqa: E402
    SNN_c_RNN,
    SPADSpikeRNN,
    SpikeBlockRNN,
    SpikingRecurrentCell,
)
from SNN_based_method.SNN_c_LSTM import (  # noqa: E402
    ConvLSTMCell,
    LSTMBlock,
    SNN_c_LSTM,
    SPADSpikeLSTM,
)
from SNN_based_method.SNN_c_GRU import (  # noqa: E402
    ConvGRUCell,
    GRUBlock,
    SNN_c_GRU,
    SPADSpikeGRU,
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
    "SNN_c_RNN",
    "SPADSpikeRNN",
    "SpikingRecurrentCell",
    "SpikeBlockRNN",
    "SNN_c_LSTM",
    "SPADSpikeLSTM",
    "ConvLSTMCell",
    "LSTMBlock",
    "SNN_c_GRU",
    "SPADSpikeGRU",
    "ConvGRUCell",
    "GRUBlock",
]

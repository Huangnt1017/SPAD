"""SNN 模型实现包。

按需导出常用模型类，避免单纯导入包时就触发外部依赖。
"""

__all__ = [
    "SPADSpikeNet",
    "SNN_c_RNN",
    "SPADSpikeRNN",
    "SNN_c_LSTM",
    "SPADSpikeLSTM",
    "SNN_c_GRU",
    "SPADSpikeGRU",
]


def __getattr__(name: str):
    """按需导入模型实现。"""
    if name == "SPADSpikeNet":
        from SNN_based_method.model.SNN_new import SPADSpikeNet

        return SPADSpikeNet
    if name in {"SNN_c_RNN", "SPADSpikeRNN"}:
        from SNN_based_method.model.SNN_c_RNN import SNN_c_RNN, SPADSpikeRNN

        return {"SNN_c_RNN": SNN_c_RNN, "SPADSpikeRNN": SPADSpikeRNN}[name]
    if name in {"SNN_c_LSTM", "SPADSpikeLSTM"}:
        from SNN_based_method.model.SNN_c_LSTM import SNN_c_LSTM, SPADSpikeLSTM

        return {"SNN_c_LSTM": SNN_c_LSTM, "SPADSpikeLSTM": SPADSpikeLSTM}[name]
    if name in {"SNN_c_GRU", "SPADSpikeGRU"}:
        from SNN_based_method.model.SNN_c_GRU import SNN_c_GRU, SPADSpikeGRU

        return {"SNN_c_GRU": SNN_c_GRU, "SPADSpikeGRU": SPADSpikeGRU}[name]
    raise AttributeError(f"module 'SNN_based_method.model' has no attribute {name!r}")

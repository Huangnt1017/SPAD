"""SNN_based_method 包入口。

当前目录按职责拆分为:
    - ``scripts``: 训练、测试、可视化等可执行入口
    - ``model``: SNN / RNN / LSTM / GRU 模型实现
    - ``utils``: 数据、loss、运行时辅助工具
    - ``config``: 统一配置对象

根目录保留少量兼容导出，但通过懒加载避免包导入时立即触发重量级依赖。
"""

__all__ = ["SNNConfig", "SPADSpikeNet", "SPADImagingLoss", "ImageMetrics"]


def __getattr__(name: str):
    """按需暴露常用符号，避免 ``import SNN_based_method`` 触发重导入。"""
    if name == "SNNConfig":
        from SNN_based_method.config.SNN_config import SNNConfig

        return SNNConfig
    if name == "SPADSpikeNet":
        from SNN_based_method.model.SNN_new import SPADSpikeNet

        return SPADSpikeNet
    if name == "SPADImagingLoss":
        from SNN_based_method.utils.loss import SPADImagingLoss

        return SPADImagingLoss
    if name == "ImageMetrics":
        from SNN_based_method.utils.loss import ImageMetrics

        return ImageMetrics
    raise AttributeError(f"module 'SNN_based_method' has no attribute {name!r}")

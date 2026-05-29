"""SNN 训练与推理的统一配置。

主要导出:
    SNNConfig: 统一管理数据、模型、损失、训练和测试参数的 dataclass。
    SINUSOIDAL_DEFAULT, LUT_RBF_16, LUT_SIN_16, LUT_RBF_32: 常用预设配置。
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _as_list(values: Sequence[str] | None) -> list[str]:
    """把可选字符串序列转换为 JSON 友好的 list。"""
    if values is None:
        return []
    return [str(value) for value in values]


@dataclass
class SNNConfig:
    """SPAD SNN 的全局配置对象。

    同一个对象会用于模型构建、loss/metric 构建、raw 数据加载、checkpoint
    保存和命令行脚本参数管理。
    """

    # ---- 数据 ----
    data_paths: list[str] | None = None
    """raw 文件路径, 或包含 ``.raw`` 文件的目录。"""

    pages_per_group: int = 500
    """单个训练样本包含的 raw page 数, 即 ``P``。"""

    total_pages: Optional[int] = None
    """每个 raw 文件使用的 page 数; ``None`` 表示使用全部完整分组。"""

    time_threshold: int = 150
    """大于该 ToF bin 的值视为无效并置 0。"""

    recursive: bool = False
    """是否递归搜索数据目录中的 ``.raw`` 文件。"""

    return_label: bool = True
    """是否从分组 raw 数据生成弱标签 ``[B, 2, 64, 64]``。"""

    normalize_input: bool = False
    """是否在 Dataset 内将 ToF 输入除以 ``time_threshold``。"""

    shuffle_pages: bool = False
    """加载时是否随机打乱每个样本内部的 P 维。"""

    active_point: int = 1
    """弱标签点云过滤使用的最小重复次数。"""

    cache_size: int = 2
    """Dataset 在内存中缓存的 raw 分组数组数量。"""

    split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1)
    """train/val/test 划分比例。"""

    # ---- Dataloader ----
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: Optional[bool] = None
    drop_last: bool = False
    seed: int = 42

    # ---- 编码 ----
    encoding_mode: str = "sinusoidal"
    """``sinusoidal`` or ``lut``."""

    n_freq: int = 8
    embed_dim: int = 16
    lut_init: str = "sinusoidal"
    lut_max_norm: Optional[float] = None

    # ---- 网络 ----
    model_backend: str = "legacy"
    """``new`` 使用 SNN_new.py; ``legacy`` 使用 SNN.py。"""

    C: int = 32
    chunk_size: int = 128
    spike_mode: str = "plif"
    num_blocks: int = 3
    refine_mid: int = 8

    # ---- 损失权重 ----
    w_gt: float = 0.3
    w_ssim: float = 0.1
    w_var: float = 1.0
    w_sparse: float = 0.05
    w_smooth: float = 0.1
    w_lut_smooth: float = 0.01
    w_lut_norm: float = 0.005

    # ---- 损失超参数 ----
    sigma_target: float = 4.0
    rho_target: float = 0.15
    beta_smooth: float = 5.0
    ssim_kernel_size: int = 7
    depth_range: float = 150.0
    intensity_range: float = 1.0

    # ---- 优化器 / 调度器 ----
    epochs: int = 20
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    amp: bool = False

    # ---- 运行时 / 实验产物 ----
    device: str = "auto"
    output_dir: str = "SNN_based_method/artifacts"
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    save_every: int = 1

    @property
    def t_max(self) -> int:
        """模型和 loss 代码使用的 ToF 上限别名。"""
        return self.time_threshold

    @property
    def C_enc(self) -> int:
        """stem 层之前的编码通道数。"""
        if self.encoding_mode == "lut":
            return self.embed_dim
        return 2 * self.n_freq + 1

    def resolved_device(self) -> torch.device:
        """解析配置中的设备; ``auto`` 会优先映射到可用 CUDA。"""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)

    def build_model(self) -> torch.nn.Module:
        """根据当前配置构建 ``SPADSpikeNet`` 模型。"""
        backend = self.model_backend.lower()
        if backend in {"new", "activation", "activation_based"}:
            from SNN_based_method.SNN_new import SPADSpikeNet
        elif backend in {"legacy", "clock", "clock_driven"}:
            from SNN_based_method.SNN import SPADSpikeNet
        else:
            raise ValueError("model_backend must be 'new' or 'legacy'")

        return SPADSpikeNet(
            C=self.C,
            chunk_size=self.chunk_size,
            spike_mode=self.spike_mode,
            t_max=self.time_threshold,
            n_freq=self.n_freq,
            num_blocks=self.num_blocks,
            encoding_mode=self.encoding_mode,
            embed_dim=self.embed_dim,
            lut_init=self.lut_init,
        )

    def build_loss(self) -> torch.nn.Module:
        """构建标准 SNN 成像损失。"""
        from SNN_based_method.loss import SPADImagingLoss

        return SPADImagingLoss(
            w_gt=self.w_gt,
            w_ssim=self.w_ssim,
            w_var=self.w_var,
            w_sparse=self.w_sparse,
            w_smooth=self.w_smooth,
            w_lut_smooth=self.w_lut_smooth,
            w_lut_norm=self.w_lut_norm,
            sigma_target=self.sigma_target,
            rho_target=self.rho_target,
            beta_smooth=self.beta_smooth,
            ssim_kernel_size=self.ssim_kernel_size,
            depth_range=self.depth_range,
            intensity_range=self.intensity_range,
        )

    def build_metrics(self):
        """构建验证和测试使用的图像指标。"""
        from SNN_based_method.loss import ImageMetrics

        return ImageMetrics(
            depth_range=self.depth_range,
            intensity_range=self.intensity_range,
            ssim_kernel_size=self.ssim_kernel_size,
        )

    def build_dataloaders(self):
        """根据配置中的 raw 路径构建 train/val/test DataLoader。"""
        if not self.data_paths:
            raise ValueError("data_paths is empty; pass --data-paths or use a config JSON")

        from SNN_based_method.scripts.data import create_spad_dataloaders

        return create_spad_dataloaders(
            self.data_paths,
            pages_per_group=self.pages_per_group,
            total_pages=self.total_pages,
            time_threshold=self.time_threshold,
            batch_size=self.batch_size,
            split_ratios=self.split_ratios,
            seed=self.seed,
            return_label=self.return_label,
            normalize=self.normalize_input,
            shuffle_pages=self.shuffle_pages,
            active_point=self.active_point,
            cache_size=self.cache_size,
            recursive=self.recursive,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def build_dataloader(self, *, shuffle: bool = False):
        """根据所有配置 raw 路径构建单个 DataLoader。"""
        if not self.data_paths:
            raise ValueError("data_paths is empty; pass --data-paths or use a config JSON")

        from SNN_based_method.scripts.data import create_spad_dataloader

        return create_spad_dataloader(
            self.data_paths,
            pages_per_group=self.pages_per_group,
            total_pages=self.total_pages,
            time_threshold=self.time_threshold,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.seed,
            return_label=self.return_label,
            normalize=self.normalize_input,
            shuffle_pages=self.shuffle_pages,
            active_point=self.active_point,
            cache_size=self.cache_size,
            recursive=self.recursive,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """导出可 JSON 序列化的字典。"""
        data = asdict(self)
        data["data_paths"] = _as_list(self.data_paths)
        data["split_ratios"] = list(self.split_ratios)
        return data

    def save(self, path: str | Path) -> None:
        """将当前配置保存为 JSON。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(self.to_dict(), file_obj, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "SNNConfig":
        """从 JSON 文件加载配置。"""
        with Path(path).open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        if "split_ratios" in data:
            data["split_ratios"] = tuple(data["split_ratios"])
        return cls(**data)

    def clone_with(self, **updates: Any) -> "SNNConfig":
        """创建一个覆盖指定字段的新配置。"""
        data = self.to_dict()
        data.update(updates)
        if "split_ratios" in data:
            data["split_ratios"] = tuple(data["split_ratios"])
        return SNNConfig(**data)

    def summary(self) -> str:
        """返回适合打印的人类可读配置摘要。"""
        lines = ["SNNConfig:"]
        lines.append(f"  data_paths={self.data_paths}")
        lines.append(
            f"  pages_per_group={self.pages_per_group}, "
            f"time_threshold={self.time_threshold}, batch_size={self.batch_size}"
        )
        lines.append(
            f"  model_backend={self.model_backend}, encoding={self.encoding_mode}, "
            f"C_enc={self.C_enc}, C={self.C}, chunk_size={self.chunk_size}"
        )
        lines.append(
            f"  epochs={self.epochs}, lr={self.lr}, weight_decay={self.weight_decay}, "
            f"device={self.resolved_device()}"
        )
        return "\n".join(lines)


SINUSOIDAL_DEFAULT = SNNConfig()
"""默认正弦编码配置。"""

LUT_RBF_16 = SNNConfig(encoding_mode="lut", embed_dim=16, lut_init="rbf")
"""16 维 LUT 编码, RBF 初始化。"""

LUT_SIN_16 = SNNConfig(encoding_mode="lut", embed_dim=16, lut_init="sinusoidal")
"""16 维 LUT 编码, 正弦初始化。"""

LUT_RBF_32 = SNNConfig(encoding_mode="lut", embed_dim=32, lut_init="rbf")
"""32 维 LUT 编码, RBF 初始化。"""


if __name__ == "__main__":
    cfg = SNNConfig()
    print(cfg.summary())
    model = cfg.build_model()
    criterion = cfg.build_loss()
    metrics = cfg.build_metrics()
    print(f"model={model.__class__.__name__}")
    print(f"criterion={criterion.__class__.__name__}")
    print(f"metrics={metrics.__class__.__name__}")

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

    csv_paths: list[str] | None = None
    """可选 CSV 样本清单路径; 存在时按 ``file_path`` 列选择 raw 文件。"""

    skip_missing_csv_raw: bool = False
    """CSV 中 raw 文件缺失时是否跳过; 默认严格报错, 避免静默丢样本。"""

    pages_per_group: int = 32 * 4
    """单个训练样本包含的 raw page 数, 即 ``P``。"""

    total_pages: Optional[int] = None
    """每个 raw 文件使用的 page 数; ``None`` 表示使用全部完整分组。"""

    time_threshold: int = 128
    """大于该 ToF bin 的值视为无效并置 0。"""

    recursive: bool = False
    """是否递归搜索数据目录中的 ``.raw`` 文件。"""

    return_label: bool = True
    """是否从分组 raw 数据生成弱标签 ``[B, 2, 64, 64]``。"""

    normalize_input: bool = False
    """是否在 Dataset 内将 ToF 输入除以 ``time_threshold``。"""

    shuffle_pages: bool = False
    """训练时是否随机打乱每个样本内部的 P 维; val/test 不使用。"""

    augment_train: bool = False
    """是否在训练集启用 raw group 级数据增强。"""

    tof_shift_max: int = 15
    """训练增强的最大整数 ToF 偏移; 增强后小于 1 或大于 time_threshold 的值置 0。"""

    tof_shift_prob: float = 1.0
    """训练增强中每个样本执行 ToF 偏移的概率。"""

    page_dropout: bool = False
    """训练增强中是否随机丢弃整页 raw page。"""

    page_dropout_prob: float = 0.1
    """PageDropout 中每页被置 0 的概率。"""

    active_point: int = 1
    """弱标签点云过滤使用的最小重复次数。"""

    cache_size: int = 2
    """Dataset 在内存中缓存的 raw 分组数组数量。"""

    raw_load_mode: str = "group"
    """raw 分组读取模式: ``group`` 只读当前 group, ``file_cache`` 缓存整文件。"""

    split_ratios: tuple[float, float, float] = (0.8, 0.2, 0.0)
    """train/val/test 划分比例。"""

    # ---- Dataloader ----
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: Optional[bool] = None
    persistent_workers: bool = True
    """num_workers > 0 时保持 DataLoader worker 常驻, 减少 epoch 间空窗。"""

    prefetch_factor: int = 4
    """每个 DataLoader worker 预取的 batch 数; 仅 num_workers > 0 时生效。"""

    precompute_model_input: bool = False
    """是否在 DataLoader worker 中额外生成 ``model_input=[B,4096,P]``。"""

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
    model_backend: str = "new"
    """模型后端; 本项目统一使用官方 ``spikingjelly.activation_based``。"""

    C: int = 32
    chunk_size: int = 32
    spike_mode: str = "plif"
    spike_backend: str = "auto"
    """spikingjelly 神经元后端: ``auto`` 优先 cupy, 也可显式 ``cupy``/``torch``。"""

    num_blocks: int = 2
    refine_mid: int = 8
    return_sequence: bool = True
    """是否返回完整 gate/tof/valid 时间序列; 训练 var/sparse loss 时需要开启。"""

    # ---- 损失权重 ----
    w_gt: float = 0.3
    w_ssim: float = 0.5
    w_var: float = 0.15
    w_sparse: float = 0.02
    w_smooth: float = 0.03
    w_lut_smooth: float = 0.01
    w_lut_norm: float = 0.005

    # ---- 损失超参数 ----
    sigma_target: float = 4.0
    rho_target: float = 0.15
    beta_smooth: float = 5.0
    ssim_kernel_size: int = 7
    ssim_smooth_kernel_size: int = 3
    """SSIM 前对预测和标签做轻量均值滤波的窗口; 1 表示关闭。"""

    gt_use_mask: bool = False
    """GT L1 是否仅在 depth_gt > 0 区域计算; 干净 GT 默认全图计算。"""

    ssim_use_mask: bool = False
    """SSIM 是否仅在 depth_gt > 0 区域计算; 干净 GT 默认全图计算。"""

    depth_range: float = 128.0
    intensity_range: float = 1.0

    # ---- 优化器 / 调度器 ----
    epochs: int = 20
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    grad_accum_steps: int = 8
    """梯度累积步数; 实际等效 batch_size = batch_size * grad_accum_steps。"""

    amp: bool = False
    tf32: bool = True
    """允许 Ampere/Ada GPU 使用 TF32 加速 matmul/conv。"""

    cudnn_benchmark: bool = True
    """输入尺寸固定时启用 cuDNN benchmark, 选择更快卷积实现。"""

    cuda_prefetch: bool = True
    """CUDA 训练/验证时用独立 stream 预取下一批, 降低数据搬运空窗。"""

    progress_interval: int = 20
    """训练/验证进度条每 N 个 batch 同步一次 loss, 降低 CPU-GPU 同步频率。"""

    # ---- 运行时 / 实验产物 ----
    device: str = "auto"
    log_dir: str = "logs/SNN"
    """训练日志、测试 summary 和预测结果的输出根目录。"""

    checkpoint_dir: str = "checkpoints/SNN"
    """训练 checkpoint 的输出根目录。"""

    output_dir: str = "SNN_based_method/artifacts"
    """旧版统一输出目录; 新脚本优先使用 ``log_dir`` 和 ``checkpoint_dir``。"""

    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    save_every: int = 1

    def __post_init__(self) -> None:
        """把旧后端名归一化为官方 activation_based 实现, 并校验枚举项。"""
        backend = str(self.model_backend).lower()
        if backend in {"legacy", "clock", "clock_driven"}:
            self.model_backend = "new"
        elif backend in {"activation", "activation_based"}:
            self.model_backend = "new"

        self.raw_load_mode = str(self.raw_load_mode).lower()
        if self.raw_load_mode not in {"group", "file_cache"}:
            raise ValueError("raw_load_mode must be 'group' or 'file_cache'")

        self.spike_backend = str(self.spike_backend).lower()
        if self.spike_backend not in {"auto", "cupy", "torch"}:
            raise ValueError("spike_backend must be 'auto', 'cupy' or 'torch'")

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
            # 旧配置文件可能还保存为 legacy/clock_driven。这里仅做兼容映射,
            # 实际仍使用官方 spikingjelly.activation_based 实现。
            from SNN_based_method.SNN_new import SPADSpikeNet
        else:
            raise ValueError("model_backend must be 'new'/'activation_based'")

        return SPADSpikeNet(
            C=self.C,
            chunk_size=self.chunk_size,
            spike_mode=self.spike_mode,
            spike_backend=self.spike_backend,
            t_max=self.time_threshold,
            n_freq=self.n_freq,
            num_blocks=self.num_blocks,
            encoding_mode=self.encoding_mode,
            embed_dim=self.embed_dim,
            lut_init=self.lut_init,
            refine_mid=self.refine_mid,
            return_sequence=self.return_sequence,
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
            ssim_smooth_kernel_size=self.ssim_smooth_kernel_size,
            gt_use_mask=self.gt_use_mask,
            ssim_use_mask=self.ssim_use_mask,
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
            csv_paths=self.csv_paths,
            skip_missing_csv_raw=self.skip_missing_csv_raw,
            pages_per_group=self.pages_per_group,
            total_pages=self.total_pages,
            time_threshold=self.time_threshold,
            batch_size=self.batch_size,
            split_ratios=self.split_ratios,
            seed=self.seed,
            return_label=self.return_label,
            normalize=self.normalize_input,
            shuffle_pages=self.shuffle_pages,
            augment_train=self.augment_train,
            tof_shift_max=self.tof_shift_max,
            tof_shift_prob=self.tof_shift_prob,
            page_dropout=self.page_dropout,
            page_dropout_prob=self.page_dropout_prob,
            active_point=self.active_point,
            cache_size=self.cache_size,
            raw_load_mode=self.raw_load_mode,
            recursive=self.recursive,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            include_model_input=self.precompute_model_input,
            drop_last=self.drop_last,
        )

    def build_dataloader(self, *, shuffle: bool = False):
        """根据所有配置 raw 路径构建单个 DataLoader。"""
        if not self.data_paths:
            raise ValueError("data_paths is empty; pass --data-paths or use a config JSON")

        from SNN_based_method.scripts.data import create_spad_dataloader

        return create_spad_dataloader(
            self.data_paths,
            csv_paths=self.csv_paths,
            skip_missing_csv_raw=self.skip_missing_csv_raw,
            pages_per_group=self.pages_per_group,
            total_pages=self.total_pages,
            time_threshold=self.time_threshold,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.seed,
            return_label=self.return_label,
            normalize=self.normalize_input,
            shuffle_pages=False,
            active_point=self.active_point,
            cache_size=self.cache_size,
            raw_load_mode=self.raw_load_mode,
            recursive=self.recursive,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            include_model_input=self.precompute_model_input,
            drop_last=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """导出可 JSON 序列化的字典。"""
        data = asdict(self)
        data["data_paths"] = _as_list(self.data_paths)
        data["csv_paths"] = _as_list(self.csv_paths)
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
        lines.append(f"  csv_paths={self.csv_paths}")
        lines.append(
            f"  pages_per_group={self.pages_per_group}, "
            f"time_threshold={self.time_threshold}, batch_size={self.batch_size}"
        )
        lines.append(
            f"  dataloader: num_workers={self.num_workers}, pin_memory={self.pin_memory}, "
            f"persistent_workers={self.persistent_workers}, prefetch_factor={self.prefetch_factor}, "
            f"precompute_model_input={self.precompute_model_input}, raw_load_mode={self.raw_load_mode}"
        )
        lines.append(
            f"  augment_train={self.augment_train}, tof_shift_max={self.tof_shift_max}, "
            f"tof_shift_prob={self.tof_shift_prob}, page_dropout={self.page_dropout}, "
            f"page_dropout_prob={self.page_dropout_prob}, shuffle_pages={self.shuffle_pages}"
        )
        lines.append(
            f"  model_backend={self.model_backend}, encoding={self.encoding_mode}, "
            f"C_enc={self.C_enc}, C={self.C}, chunk_size={self.chunk_size}, "
            f"spike_backend={self.spike_backend}, refine_mid={self.refine_mid}, "
            f"return_sequence={self.return_sequence}"
        )
        lines.append(
            f"  loss_weights: gt={self.w_gt}, ssim={self.w_ssim}, "
            f"var={self.w_var}, sparse={self.w_sparse}, smooth={self.w_smooth}"
        )
        lines.append(
            f"  epochs={self.epochs}, lr={self.lr}, weight_decay={self.weight_decay}, "
            f"grad_accum_steps={self.grad_accum_steps}, device={self.resolved_device()}"
        )
        lines.append(
            f"  runtime: amp={self.amp}, tf32={self.tf32}, cuda_prefetch={self.cuda_prefetch}, "
            f"cudnn_benchmark={self.cudnn_benchmark}, progress_interval={self.progress_interval}"
        )
        lines.append(f"  log_dir={self.log_dir}, checkpoint_dir={self.checkpoint_dir}")
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

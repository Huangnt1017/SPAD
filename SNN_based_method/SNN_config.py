"""SPAD SNN 模型统一配置.

所有可调参数集中在 SNNConfig 中管理, 模型和损失函数通过 config 构建:
    cfg = SNNConfig(encoding_mode="lut", embed_dim=16, lut_init="rbf")
    model = cfg.build_model()
    criterion = cfg.build_loss()
    metrics = cfg.build_metrics()

LUT 相关参数 (embed_dim / lut_init / w_lut_smooth / w_lut_norm)
仅在 encoding_mode="lut" 时生效, sinusoidal 模式下自动忽略.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class SNNConfig:
    """SPAD SNN 全局配置, 按逻辑分组.

    分组:
      - 数据: t_max, H, W
      - 编码: encoding_mode, n_freq / embed_dim+lut_init
      - 网络: C, chunk_size, spike_mode, num_blocks, refine_mid
      - 损失权重: w_gt, w_ssim, w_var, w_sparse, w_smooth, w_lut_*
      - 损失超参: sigma_target, rho_target, beta_smooth, ssim_kernel_size
    """

    # ── 数据 ──────────────────────────────────────────────
    t_max: int = 150
    """最大有效 ToF bin (1~t_max 有效, 0=无效)"""

    # ── 编码 ──────────────────────────────────────────────
    encoding_mode: str = "sinusoidal"
    """编码方式: "sinusoidal" (固定正弦) 或 "lut" (可学习查表)"""

    n_freq: int = 8
    """正弦编码频率对数 (sinusoidal 模式输出 2*n_freq+1 通道)"""

    # 以下参数仅 encoding_mode="lut" 时生效
    embed_dim: int = 16
    """LUT 嵌入维度 (自由调整, stem 层自动适配输入通道)"""

    lut_init: str = "sinusoidal"
    """LUT 初始化方式: "sinusoidal" / "rbf" / "random" """

    lut_max_norm: Optional[float] = None
    """LUT embedding 最大 L2 范数约束 (None=不限)"""

    # ── 网络结构 ──────────────────────────────────────────
    C: int = 32
    """SpikeBlock 工作通道数 (全程不变)"""

    chunk_size: int = 128
    """每 chunk 帧数 (= PLIF timestep, 影响显存)"""

    spike_mode: str = "plif"
    """脉冲神经元类型: "plif" / "lif" / "if" """

    num_blocks: int = 3
    """SpikeBlock 堆叠层数"""

    refine_mid: int = 8
    """空间精修头中间通道数"""

    # ── 损失权重 ──────────────────────────────────────────
    w_gt: float = 0.3
    """L1 (MAE) loss 权重"""

    w_ssim: float = 0.1
    """SSIM loss 权重"""

    w_var: float = 1.0
    """Gate 方差 loss 权重"""

    w_sparse: float = 0.05
    """Gate 稀疏性 loss 权重"""

    w_smooth: float = 0.1
    """边缘保持平滑 loss 权重"""

    # 以下权重仅 encoding_mode="lut" 时生效
    w_lut_smooth: float = 0.01
    """LUT 相邻 bin 平滑正则权重"""

    w_lut_norm: float = 0.005
    """LUT 范数一致性正则权重"""

    # ── 损失超参 ──────────────────────────────────────────
    sigma_target: float = 4.0
    """方差 loss 中目标 sigma (bin 单位, 对应回波 FWHM)"""

    rho_target: float = 0.15
    """稀疏 loss 中目标激活率"""

    beta_smooth: float = 5.0
    """平滑 loss 的边缘衰减系数"""

    ssim_kernel_size: int = 7
    """SSIM 高斯窗口大小 (64×64 分辨率推荐 7)"""

    depth_range: float = 150.0
    """Depth 动态范围, 用于 SSIM loss 和评估指标归一化"""

    intensity_range: float = 1.0
    """Intensity 动态范围"""

    # ── 编码通道数 (只读, 自动计算) ────────────────────────

    @property
    def C_enc(self) -> int:
        """编码输出通道数, stem 层的输入维度."""
        if self.encoding_mode == "lut":
            return self.embed_dim
        return 2 * self.n_freq + 1

    # ── 构建方法 ──────────────────────────────────────────

    def build_model(self):
        """根据配置构建 SPADSpikeNet 模型实例.

        Returns:
            SPADSpikeNet 实例 (未移至 GPU, 需调用方自行 .to(device))
        """
        from SNN import SPADSpikeNet
        return SPADSpikeNet(
            C=self.C,
            chunk_size=self.chunk_size,
            spike_mode=self.spike_mode,
            t_max=self.t_max,
            n_freq=self.n_freq,
            num_blocks=self.num_blocks,
            encoding_mode=self.encoding_mode,
            embed_dim=self.embed_dim,
            lut_init=self.lut_init,
        )

    def build_loss(self):
        """根据配置构建 SPADImagingLoss 实例.

        LUT 正则权重在 sinusoidal 模式下传入但不会生效
        (SPADImagingLoss 内部检查 result dict 中是否有 lut_* 键).

        Returns:
            SPADImagingLoss 实例
        """
        from loss import SPADImagingLoss
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
        )

    def build_metrics(self):
        """根据配置构建 ImageMetrics 评估工具.

        Returns:
            ImageMetrics 实例
        """
        from loss import ImageMetrics
        return ImageMetrics(
            depth_range=self.depth_range,
            intensity_range=self.intensity_range,
            ssim_kernel_size=self.ssim_kernel_size,
        )

    # ── 序列化 ────────────────────────────────────────────

    def to_dict(self) -> dict:
        """导出为普通 dict (可直接 json.dumps)."""
        return asdict(self)

    def save(self, path: str):
        """保存配置到 JSON 文件.

        Args:
            path: JSON 文件路径
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SNNConfig":
        """从 JSON 文件加载配置.

        Args:
            path: JSON 文件路径

        Returns:
            SNNConfig 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def summary(self) -> str:
        """打印可读的配置摘要."""
        lines = ["SNNConfig:"]

        lines.append("  [数据]")
        lines.append(f"    t_max={self.t_max}")

        lines.append("  [编码]")
        lines.append(f"    encoding_mode={self.encoding_mode}")
        if self.encoding_mode == "sinusoidal":
            lines.append(f"    n_freq={self.n_freq} → {self.C_enc} 通道")
        else:
            lines.append(f"    embed_dim={self.embed_dim}, lut_init={self.lut_init}")
            if self.lut_max_norm is not None:
                lines.append(f"    lut_max_norm={self.lut_max_norm}")

        lines.append("  [网络]")
        lines.append(f"    C={self.C}, chunk_size={self.chunk_size}, "
                      f"spike_mode={self.spike_mode}, num_blocks={self.num_blocks}")

        lines.append("  [损失权重]")
        weights = f"    w_gt={self.w_gt}, w_ssim={self.w_ssim}, w_var={self.w_var}, " \
                  f"w_sparse={self.w_sparse}, w_smooth={self.w_smooth}"
        lines.append(weights)
        if self.encoding_mode == "lut":
            lines.append(f"    w_lut_smooth={self.w_lut_smooth}, w_lut_norm={self.w_lut_norm}")

        lines.append("  [损失超参]")
        lines.append(f"    sigma_target={self.sigma_target}, rho_target={self.rho_target}, "
                      f"beta_smooth={self.beta_smooth}")

        return "\n".join(lines)


# ── 预置配置 ──────────────────────────────────────────────

SINUSOIDAL_DEFAULT = SNNConfig()
"""默认正弦编码配置."""

LUT_RBF_16 = SNNConfig(
    encoding_mode="lut",
    embed_dim=16,
    lut_init="rbf",
)
"""LUT 编码, 16维, RBF 初始化."""

LUT_SIN_16 = SNNConfig(
    encoding_mode="lut",
    embed_dim=16,
    lut_init="sinusoidal",
)
"""LUT 编码, 16维, 正弦初始化."""

LUT_RBF_32 = SNNConfig(
    encoding_mode="lut",
    embed_dim=32,
    lut_init="rbf",
)
"""LUT 编码, 32维, RBF 初始化."""


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    presets = {
        "SINUSOIDAL_DEFAULT": SINUSOIDAL_DEFAULT,
        "LUT_RBF_16": LUT_RBF_16,
        "LUT_SIN_16": LUT_SIN_16,
        "LUT_RBF_32": LUT_RBF_32,
    }

    for name, cfg in presets.items():
        print(f"\n{'='*50}")
        print(f"预置: {name}")
        print(cfg.summary())

        model = cfg.build_model()
        criterion = cfg.build_loss()
        metrics = cfg.build_metrics()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数: {n_params:,}")
        print(f"  编码通道: {cfg.C_enc}")

    # 序列化往返测试
    print(f"\n{'='*50}")
    print("序列化往返测试:")
    test_cfg = SNNConfig(encoding_mode="lut", embed_dim=24, lut_init="rbf", w_gt=0.5)
    save_path = os.path.join(os.path.dirname(__file__), "_test_config.json")
    test_cfg.save(save_path)
    loaded = SNNConfig.load(save_path)
    assert loaded.encoding_mode == "lut"
    assert loaded.embed_dim == 24
    assert loaded.w_gt == 0.5
    os.remove(save_path)
    print("  save/load 往返一致 OK")

    print("\nAll tests passed!")

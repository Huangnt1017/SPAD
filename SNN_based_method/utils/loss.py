"""SPAD 浓雾场景 SNN 成像模型的损失函数与评估指标。

这些损失只依赖单个时间窗口的数据，不需要跨场景配对。弱标签可来自
histogram peak 或重复点计数生成的噪声 GT。

包含:
- ``WeakGTLoss``: 对弱 GT 的 L1/MAE 约束。
- ``DepthRegressionLoss``: 有效深度区域的 MSE/Charbonnier 约束, 直接对齐 RMSE/PSNR。
- ``SSIMLoss``: 结构相似性损失 ``1 - SSIM``。
- ``GatedMomentVarianceLoss``: 约束 gate 选中光子的 ToF 方差。
- ``SpikeSparsityLoss``: gate 稀疏性正则。
- ``IntensityAwareSmoothnessLoss``: 强度引导的边缘保持平滑项。
- ``SPADImagingLoss``: 组合训练损失。
- ``ImageMetrics``: MAE / RMSE / SSIM / PSNR 评估工具，不参与梯度。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _spatial_gradient(x):
    """Compute spatial gradient magnitude for [B, 1, H, W] tensor."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)


def _gaussian_kernel_2d(kernel_size=11, sigma=1.5, channels=1, device=None):
    """生成 2D 高斯卷积核, 用于 SSIM 的局部统计计算.

    Args:
        kernel_size: 窗口大小 (奇数)
        sigma: 高斯标准差
        channels: 输入通道数 (用于 groups 卷积)
        device: 目标设备

    Returns:
        [channels, 1, kernel_size, kernel_size] 高斯核, 已归一化
    """
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    coords -= kernel_size // 2
    gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    # 外积构造 2D 核
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]       # [K, K]
    gauss_2d = gauss_2d / gauss_2d.sum()
    # [channels, 1, K, K] 用于 groups=channels 的 depthwise 卷积
    kernel = gauss_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


def _compute_ssim(pred, target, mask=None, kernel_size=11, sigma=1.5,
                  data_range=1.0, k1=0.01, k2=0.03):
    """计算单通道 SSIM map.

    Args:
        pred:   [B, 1, H, W] 预测图
        target: [B, 1, H, W] 真值图
        mask:   [B, 1, H, W] or None, 有效像素 mask (1=有效)
        kernel_size: 高斯窗口大小
        sigma: 高斯标准差
        data_range: 数据动态范围 (target 的 max - min)
        k1, k2: SSIM 稳定常数

    Returns:
        ssim_val: 标量, masked 区域的平均 SSIM
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    kernel = _gaussian_kernel_2d(kernel_size, sigma, channels=1, device=pred.device)
    pad_size = kernel_size // 2

    # 局部均值
    mu_pred = F.conv2d(pred, kernel, padding=pad_size)
    mu_target = F.conv2d(target, kernel, padding=pad_size)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    # 局部方差与协方差
    sigma_pred_sq = F.conv2d(pred ** 2, kernel, padding=pad_size) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, kernel, padding=pad_size) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, kernel, padding=pad_size) - mu_cross

    # SSIM 公式: (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / ((mu_x^2+mu_y^2+C1)(sigma_x^2+sigma_y^2+C2))
    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    ssim_map = numerator / (denominator + 1e-8)             # [B, 1, H, W]

    if mask is not None:
        return (ssim_map * mask).sum() / (mask.sum() + 1e-6)
    return ssim_map.mean()


class WeakGTLoss(nn.Module):
    """归一化后的 GT L1 约束, 可选择是否仅在有效深度区域计算。"""

    def __init__(
        self,
        w_depth=1.0,
        w_intensity=1.0,
        depth_range=128.0,
        intensity_range=1.0,
        use_mask: bool = False,
    ):
        super().__init__()
        self.w_depth = w_depth
        self.w_intensity = w_intensity
        self.depth_range = float(depth_range)
        self.intensity_range = float(intensity_range)
        self.use_mask = bool(use_mask)

    def forward(self, result, gt):
        """
        Args:
            result: model output dict with 'output' [B,2,H,W] (refined)
            gt:     [B, 2, H, W] where ch0=depth_gt, ch1=intensity_gt
        """
        d_pred = (result["output"][:, 0:1] / self.depth_range).clamp(0.0, 1.0)
        i_pred = (result["output"][:, 1:2] / self.intensity_range).clamp(0.0, 1.0)
        d_gt = (gt[:, 0:1] / self.depth_range).clamp(0.0, 1.0)
        i_gt = (gt[:, 1:2] / self.intensity_range).clamp(0.0, 1.0)

        if self.use_mask:
            mask = (d_gt > 0).float()
            normalizer = mask.sum() + 1e-6
            loss_d = (torch.abs(d_pred - d_gt) * mask).sum() / normalizer
            loss_i = (torch.abs(i_pred - i_gt) * mask).sum() / normalizer
        else:
            loss_d = torch.abs(d_pred - d_gt).mean()
            loss_i = torch.abs(i_pred - i_gt).mean()

        return self.w_depth * loss_d + self.w_intensity * loss_i


class DepthRegressionLoss(nn.Module):
    """有效深度区域的归一化 depth 回归项, 用于直接压低 RMSE/提升 PSNR。"""

    def __init__(
        self,
        depth_range=128.0,
        mode: str = "mse",
        use_mask: bool = True,
        charbonnier_eps: float = 1.0e-3,
    ):
        super().__init__()
        self.depth_range = float(depth_range)
        self.mode = str(mode).lower()
        if self.mode not in {"mse", "charbonnier", "l1"}:
            raise ValueError("depth_reg_mode must be 'mse', 'charbonnier' or 'l1'")
        self.use_mask = bool(use_mask)
        self.charbonnier_eps = float(charbonnier_eps)
        if self.charbonnier_eps <= 0:
            raise ValueError("depth_reg_charbonnier_eps must be positive")

    def forward(self, result, gt):
        """计算 depth 通道的归一化回归损失。"""
        d_pred = (result["output"][:, 0:1] / self.depth_range).clamp(0.0, 1.0)
        d_gt_raw = gt[:, 0:1]
        d_gt = (d_gt_raw / self.depth_range).clamp(0.0, 1.0)

        if self.use_mask:
            mask = (d_gt_raw > 0).float()
        else:
            mask = torch.ones_like(d_gt)
        normalizer = mask.sum().clamp(min=1.0)

        error = d_pred - d_gt
        if self.mode == "mse":
            penalty = error ** 2
        elif self.mode == "charbonnier":
            eps = self.charbonnier_eps
            penalty = torch.sqrt(error ** 2 + eps ** 2) - eps
        else:
            penalty = error.abs()

        return (penalty * mask).sum() / normalizer


class GatedMomentVarianceLoss(nn.Module):
    """Penalize if gate-selected photons scatter too far from predicted depth.

    Target echo is narrow (~4 bin FWHM). If the variance of selected photon
    timestamps around predicted depth is large, the gate is selecting wrong photons.
    """

    def __init__(self, sigma_target=4.0, depth_range=128.0):
        super().__init__()
        self.sigma2 = (float(sigma_target) / float(depth_range)) ** 2
        self.depth_range = float(depth_range)

    def forward(self, result):
        gate = result["gate"]             # [P, B, 1, H, W]
        tof = result["tof"]               # [P, B, H, W]
        valid = result["valid"]           # [P, B, H, W]
        depth = result["depth_coarse"]    # [B, 1, H, W]  use coarse for variance

        depth_exp = depth.squeeze(1).unsqueeze(0) / self.depth_range  # [1, B, H, W]
        tof_norm = tof / self.depth_range
        gate_sq = gate.squeeze(2)                                # [T, B, H, W]

        gv = gate_sq * valid                                     # [T, B, H, W]
        residual2 = (tof_norm - depth_exp) ** 2                  # [T, B, H, W]

        weighted_var = (gv * residual2).sum(0) / (gv.sum(0) + 1e-6)  # [B, H, W]
        excess = F.relu(weighted_var - self.sigma2)

        return excess.mean()


class SpikeSparsityLoss(nn.Module):
    """约束 gate 平均激活率, 避免单边阈值项过早失去梯度。"""

    def __init__(
        self,
        rho_target=0.15,
        mode: str = "band",
        rho_min: float | None = 0.03,
        rho_max: float | None = 0.15,
    ):
        super().__init__()
        self.rho_target = float(rho_target)
        self.mode = str(mode).lower()
        if self.mode not in {"upper", "target", "band"}:
            raise ValueError("sparse_mode must be 'upper', 'target' or 'band'")
        self.rho_min = None if rho_min is None else float(rho_min)
        self.rho_max = None if rho_max is None else float(rho_max)
        if self.rho_target < 0:
            raise ValueError("rho_target must be non-negative")
        if self.rho_min is not None and self.rho_min < 0:
            raise ValueError("rho_min must be non-negative")
        if self.rho_max is not None and self.rho_max < 0:
            raise ValueError("rho_max must be non-negative")
        if (
            self.rho_min is not None
            and self.rho_max is not None
            and self.rho_min > self.rho_max
        ):
            raise ValueError("rho_min must be <= rho_max")

    def mean_rate(self, result):
        """返回有效 ToF 位置上的平均 gate 激活率。"""
        gate = result["gate"]       # [T, B, 1, H, W]
        valid = result["valid"]     # [T, B, H, W]

        gate_sq = gate.squeeze(2)   # [T, B, H, W]
        return (gate_sq * valid).sum() / (valid.sum() + 1e-6)

    def penalty_from_mean(self, mean_rate):
        """根据平均激活率计算 sparse 正则值。"""
        if self.mode == "upper":
            return F.relu(mean_rate - self.rho_target)
        if self.mode == "target":
            return torch.abs(mean_rate - self.rho_target)

        rho_min = self.rho_min if self.rho_min is not None else self.rho_target
        rho_max = self.rho_max if self.rho_max is not None else self.rho_target
        lower = torch.as_tensor(rho_min, dtype=mean_rate.dtype, device=mean_rate.device)
        upper = torch.as_tensor(rho_max, dtype=mean_rate.dtype, device=mean_rate.device)
        return F.relu(lower - mean_rate) + F.relu(mean_rate - upper)

    def band_components_from_mean(self, mean_rate):
        """返回下限/上限惩罚分量, 仅用于日志诊断。"""
        if self.mode != "band":
            zero = torch.zeros_like(mean_rate)
            if self.mode == "upper":
                return zero, F.relu(mean_rate - self.rho_target)
            target_gap = torch.abs(mean_rate - self.rho_target)
            return target_gap, target_gap

        rho_min = self.rho_min if self.rho_min is not None else self.rho_target
        rho_max = self.rho_max if self.rho_max is not None else self.rho_target
        lower = torch.as_tensor(rho_min, dtype=mean_rate.dtype, device=mean_rate.device)
        upper = torch.as_tensor(rho_max, dtype=mean_rate.dtype, device=mean_rate.device)
        return F.relu(lower - mean_rate), F.relu(mean_rate - upper)

    def forward(self, result):
        mean_rate = self.mean_rate(result)
        return self.penalty_from_mean(mean_rate)




class SSIMLoss(nn.Module):
    """结构相似性损失 (1 - SSIM), 对 depth 和 intensity 分别计算.

    SSIM 捕获局部亮度/对比度/结构信息, 弥补纯像素级 L1 的不足.
    可选对 GT 做有效区域 mask, 并可在 SSIM 前做轻量平滑以降低标签噪声影响。

    Args:
        w_depth: depth SSIM loss 权重
        w_intensity: intensity SSIM loss 权重
        kernel_size: 高斯窗口大小 (推荐 11, 图片 64×64 时可用 7)
        depth_range: depth 的数据动态范围 (默认 150, 对应 tof bin 最大值)
        intensity_range: intensity 的动态范围 (默认 1.0, 归一化后)
    """

    def __init__(
        self,
        w_depth=1.0,
        w_intensity=1.0,
        kernel_size=7,
        depth_range=128.0,
        intensity_range=1.0,
        use_mask: bool = False,
        smooth_kernel_size: int = 3,
    ):
        super().__init__()
        self.w_depth = w_depth
        self.w_intensity = w_intensity
        self.kernel_size = kernel_size
        self.depth_range = depth_range
        self.intensity_range = intensity_range
        self.use_mask = bool(use_mask)
        self.smooth_kernel_size = int(smooth_kernel_size)

    def _smooth_for_ssim(self, x: torch.Tensor) -> torch.Tensor:
        """SSIM 前的轻量低通滤波, 降低少量标签噪声对结构项的影响。"""
        if self.smooth_kernel_size <= 1:
            return x
        kernel_size = self.smooth_kernel_size
        if kernel_size % 2 == 0:
            raise ValueError("smooth_kernel_size must be odd or <= 1")
        padding = kernel_size // 2
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, result, gt):
        """
        Args:
            result: model output dict with 'output' [B, 2, H, W]
            gt:     [B, 2, H, W] where ch0=depth_gt, ch1=intensity_gt

        Returns:
            (1 - SSIM) 加权合并的标量 loss
        """
        d_pred = (result["output"][:, 0:1] / self.depth_range).clamp(0.0, 1.0)
        i_pred = (result["output"][:, 1:2] / self.intensity_range).clamp(0.0, 1.0)
        d_gt = (gt[:, 0:1] / self.depth_range).clamp(0.0, 1.0)
        i_gt = (gt[:, 1:2] / self.intensity_range).clamp(0.0, 1.0)

        d_pred = self._smooth_for_ssim(d_pred)
        i_pred = self._smooth_for_ssim(i_pred)
        d_gt = self._smooth_for_ssim(d_gt)
        i_gt = self._smooth_for_ssim(i_gt)

        mask = (d_gt > 0).float() if self.use_mask else None

        ssim_d = _compute_ssim(d_pred, d_gt, mask=mask,
                               kernel_size=self.kernel_size,
                               data_range=1.0)
        ssim_i = _compute_ssim(i_pred, i_gt, mask=mask,
                               kernel_size=self.kernel_size,
                               data_range=1.0)

        loss = self.w_depth * (1.0 - ssim_d) + self.w_intensity * (1.0 - ssim_i)
        return loss


class IntensityAwareSmoothnessLoss(nn.Module):
    """Edge-preserving depth smoothness guided by intensity gradients.

    Depth should be smooth where intensity is smooth (background),
    and can have discontinuities where intensity jumps (target boundary).
    """

    def __init__(self, beta=5.0, depth_range=128.0, intensity_range=1.0):
        super().__init__()
        self.beta = beta
        self.depth_range = float(depth_range)
        self.intensity_range = float(intensity_range)

    def forward(self, result):
        depth = (result["output"][:, 0:1] / self.depth_range).clamp(0.0, 1.0)
        intensity = (result["output"][:, 1:2] / self.intensity_range).clamp(0.0, 1.0)

        grad_d = _spatial_gradient(depth)
        grad_i = _spatial_gradient(intensity)

        weight = torch.exp(-self.beta * grad_i)
        return (grad_d * weight).mean()


class SPADImagingLoss(nn.Module):
    """组合训练损失函数.

    L = w_gt * L_GT + w_depth_reg * L_depth_reg + w_ssim * L_SSIM
        + w_var * L_var + w_sparse * L_sparse + w_smooth * L_smooth
        [+ w_lut_smooth * L_lut_smooth + w_lut_norm * L_lut_norm]   (仅 LUT 编码模式)

    Args:
        w_gt: L1 (MAE) loss 权重
        w_depth_reg: 有效 depth 区域 MSE/Charbonnier 回归项权重
        w_ssim: SSIM loss 权重 (结构相似性)
        w_var: gate 方差 loss 权重
        w_sparse: gate 稀疏性 loss 权重
        w_smooth: 平滑 loss 权重
        w_lut_smooth: LUT 相邻 bin 平滑正则权重 (仅 encoding_mode="lut" 时生效)
        w_lut_norm: LUT 范数一致性正则权重 (仅 encoding_mode="lut" 时生效)
        sigma_target: 方差 loss 中目标 sigma (bin 单位)
        rho_target: 稀疏 loss 中目标激活率
        beta_smooth: 平滑 loss 的边缘衰减系数
        ssim_kernel_size: SSIM 高斯窗口大小
        depth_range: depth 动态范围, 用于 SSIM 计算
        intensity_range: intensity 动态范围, 默认为归一化占比 1.0
    """

    def __init__(
        self,
        w_gt=0.6,
        w_depth_reg=0.5,
        w_ssim=0.25,
        w_var=0.2,
        w_sparse=0.01,
        w_smooth=0.02,
        w_lut_smooth=0.01,
        w_lut_norm=0.005,
        sigma_target=4.0,
        rho_target=0.08,
        sparse_mode="band",
        rho_min=0.03,
        rho_max=0.12,
        beta_smooth=5.0,
        ssim_kernel_size=7,
        ssim_smooth_kernel_size=3,
        gt_use_mask=False,
        ssim_use_mask=False,
        depth_reg_mode="mse",
        depth_reg_use_mask=True,
        depth_reg_charbonnier_eps=1.0e-3,
        depth_range=128.0,
        intensity_range=1.0,
    ):
        super().__init__()
        self.w_gt = w_gt
        self.w_depth_reg = w_depth_reg
        self.w_ssim = w_ssim
        self.w_var = w_var
        self.w_sparse = w_sparse
        self.w_smooth = w_smooth
        self.w_lut_smooth = w_lut_smooth
        self.w_lut_norm = w_lut_norm

        self.gt_loss = WeakGTLoss(
            depth_range=depth_range,
            intensity_range=intensity_range,
            use_mask=gt_use_mask,
        )
        self.depth_reg_loss = DepthRegressionLoss(
            depth_range=depth_range,
            mode=depth_reg_mode,
            use_mask=depth_reg_use_mask,
            charbonnier_eps=depth_reg_charbonnier_eps,
        )
        self.ssim_loss = SSIMLoss(
            kernel_size=ssim_kernel_size,
            depth_range=depth_range,
            intensity_range=intensity_range,
            use_mask=ssim_use_mask,
            smooth_kernel_size=ssim_smooth_kernel_size,
        )
        self.var_loss = GatedMomentVarianceLoss(
            sigma_target=sigma_target,
            depth_range=depth_range,
        )
        self.sparse_loss = SpikeSparsityLoss(
            rho_target=rho_target,
            mode=sparse_mode,
            rho_min=rho_min,
            rho_max=rho_max,
        )
        self.smooth_loss = IntensityAwareSmoothnessLoss(
            beta=beta_smooth,
            depth_range=depth_range,
            intensity_range=intensity_range,
        )

    def forward(self, result, gt=None):
        """
        Args:
            result: dict from SPADSpikeNet.forward(),
                    LUT 模式下额外包含 'lut_smooth' 和 'lut_norm' 正则 loss
            gt:     [B, 2, H, W] or None (if no GT available, skip L_GT and L_SSIM)

        Returns:
            total loss (scalar), and dict of individual loss components
        """
        losses = {}
        total = torch.tensor(0.0, device=result["depth"].device)

        if gt is not None and self.w_gt > 0:
            l_gt = self.gt_loss(result, gt)
            losses["gt"] = l_gt.detach()
            losses["weighted_gt"] = (self.w_gt * l_gt).detach()
            total = total + self.w_gt * l_gt

        if gt is not None and self.w_depth_reg > 0:
            l_depth_reg = self.depth_reg_loss(result, gt)
            losses["depth_reg"] = l_depth_reg.detach()
            losses["weighted_depth_reg"] = (self.w_depth_reg * l_depth_reg).detach()
            total = total + self.w_depth_reg * l_depth_reg

        if gt is not None and self.w_ssim > 0:
            l_ssim = self.ssim_loss(result, gt)
            losses["ssim"] = l_ssim.detach()
            losses["weighted_ssim"] = (self.w_ssim * l_ssim).detach()
            total = total + self.w_ssim * l_ssim

        has_sequence = all(key in result for key in ("gate", "tof", "valid"))
        if self.w_var > 0 and has_sequence:
            l_var = self.var_loss(result)
            losses["var"] = l_var.detach()
            losses["weighted_var"] = (self.w_var * l_var).detach()
            total = total + self.w_var * l_var
        elif self.w_var > 0:
            losses["var_skipped"] = 1.0

        if self.w_sparse > 0 and has_sequence:
            sparse_rate = self.sparse_loss.mean_rate(result)
            sparse_lower, sparse_upper = self.sparse_loss.band_components_from_mean(sparse_rate)
            l_sparse = self.sparse_loss.penalty_from_mean(sparse_rate)
            losses["sparse_rate"] = sparse_rate.detach()
            losses["sparse_lower"] = sparse_lower.detach()
            losses["sparse_upper"] = sparse_upper.detach()
            losses["sparse"] = l_sparse.detach()
            losses["weighted_sparse"] = (self.w_sparse * l_sparse).detach()
            total = total + self.w_sparse * l_sparse
        elif self.w_sparse > 0:
            losses["sparse_skipped"] = 1.0

        if self.w_smooth > 0:
            l_smooth = self.smooth_loss(result)
            losses["smooth"] = l_smooth.detach()
            losses["weighted_smooth"] = (self.w_smooth * l_smooth).detach()
            total = total + self.w_smooth * l_smooth

        # LUT 编码正则 (仅当 model 使用 encoding_mode="lut" 时 result 中才有这些键)
        if "lut_smooth" in result and self.w_lut_smooth > 0:
            l_lut_s = result["lut_smooth"]
            losses["lut_smooth"] = l_lut_s.detach()
            losses["weighted_lut_smooth"] = (self.w_lut_smooth * l_lut_s).detach()
            total = total + self.w_lut_smooth * l_lut_s

        if "lut_norm" in result and self.w_lut_norm > 0:
            l_lut_n = result["lut_norm"]
            losses["lut_norm"] = l_lut_n.detach()
            losses["weighted_lut_norm"] = (self.w_lut_norm * l_lut_n).detach()
            total = total + self.w_lut_norm * l_lut_n

        losses["total"] = total.detach()
        return total, losses


# ─── 评估指标 (不参与梯度, 仅用于 validation/test) ────────────────────

class ImageMetrics:
    """图像质量评估工具, 计算 depth 和 intensity 的 MAE / RMSE / SSIM / PSNR.

    用法:
        metrics = ImageMetrics(depth_range=128.0)
        scores = metrics.compute(result, gt)
        # scores = {"depth_mae": ..., "depth_rmse": ..., "depth_ssim": ..., "depth_psnr": ...,
        #           "intensity_mae": ..., ...}

    所有计算在有效像素 (d_gt > 0) 上进行, 无效区域被 mask 排除.
    """

    def __init__(self, depth_range=128.0, intensity_range=1.0, ssim_kernel_size=7):
        """
        Args:
            depth_range: depth 数据动态范围, 用于 SSIM 和 PSNR
            intensity_range: intensity 动态范围
            ssim_kernel_size: SSIM 窗口大小
        """
        self.depth_range = depth_range
        self.intensity_range = intensity_range
        self.ssim_kernel_size = ssim_kernel_size

    @torch.no_grad()
    def compute_tensors(self, result, gt):
        """计算全部图像质量指标, 返回 device 上的标量张量。

        Args:
            result: model output dict with 'output' [B, 2, H, W]
            gt:     [B, 2, H, W] (ch0=depth, ch1=intensity)

        Returns:
            dict: 各通道各指标的标量张量
        """
        d_pred = result["output"][:, 0:1]           # [B, 1, H, W]
        i_pred = result["output"][:, 1:2]           # [B, 1, H, W]
        d_gt = gt[:, 0:1]
        i_gt = gt[:, 1:2]

        mask = (d_gt > 0).float()                   # [B, 1, H, W]
        num_valid = mask.sum().clamp(min=1.0)

        scores = {}

        # ── Depth 指标 (归一化到 [0, 1] 后计算, 与图像指标量纲统一) ──
        d_pred_norm = d_pred / self.depth_range
        d_gt_norm = d_gt / self.depth_range
        d_err = (d_pred_norm - d_gt_norm) * mask
        scores["depth_mae"] = d_err.abs().sum() / num_valid
        d_mse = (d_err ** 2).sum() / num_valid
        scores["depth_rmse"] = torch.sqrt(d_mse)
        scores["depth_ssim"] = _compute_ssim(
            d_pred_norm, d_gt_norm, mask=mask,
            kernel_size=self.ssim_kernel_size,
            data_range=1.0,
        )
        # PSNR = 10 * log10(1^2 / MSE), 归一化后 data_range=1.0
        scores["depth_psnr"] = self._psnr(d_mse, 1.0)

        # ── Intensity 指标 (已在 [0, 1] 范围, 直接计算) ──
        i_pred_norm = i_pred / self.intensity_range
        i_gt_norm = i_gt / self.intensity_range
        i_err = (i_pred_norm - i_gt_norm) * mask
        scores["intensity_mae"] = i_err.abs().sum() / num_valid
        i_mse = (i_err ** 2).sum() / num_valid
        scores["intensity_rmse"] = torch.sqrt(i_mse)
        scores["intensity_ssim"] = _compute_ssim(
            i_pred_norm, i_gt_norm, mask=mask,
            kernel_size=self.ssim_kernel_size,
            data_range=1.0,
        )
        # PSNR = 10 * log10(1^2 / MSE)
        scores["intensity_psnr"] = self._psnr(i_mse, 1.0)

        return scores

    @torch.no_grad()
    def compute(self, result, gt):
        """计算全部图像质量指标, 返回 Python float, 兼容测试脚本。"""
        tensor_scores = self.compute_tensors(result, gt)
        return {
            key: float(value.detach().cpu().item())
            for key, value in tensor_scores.items()
        }

    @staticmethod
    def _psnr(mse, data_range):
        """标准 PSNR 公式: 10 * log10(MAX^2 / MSE). MSE≈0 时返回 inf."""
        if isinstance(mse, torch.Tensor):
            data_range_tensor = torch.as_tensor(
                data_range,
                dtype=mse.dtype,
                device=mse.device,
            )
            psnr = 10.0 * torch.log10(data_range_tensor ** 2 / mse.clamp_min(1e-10))
            return torch.where(
                mse < 1e-10,
                torch.full_like(psnr, float("inf")),
                psnr,
            )
        if mse < 1e-10:
            return float("inf")
        return 10.0 * math.log10(data_range ** 2 / mse)

    def format_scores(self, scores):
        """格式化输出评估指标为可读字符串.

        Args:
            scores: compute() 返回的 dict

        Returns:
            格式化的多行字符串
        """
        lines = []
        for channel in ("depth", "intensity"):
            mae = scores[f"{channel}_mae"]
            rmse = scores[f"{channel}_rmse"]
            ssim = scores[f"{channel}_ssim"]
            psnr = scores[f"{channel}_psnr"]
            lines.append(
                f"  {channel:10s} | MAE={mae:.4f} | RMSE={rmse:.4f} "
                f"| SSIM={ssim:.4f} | PSNR={psnr:.2f}dB"
            )
        return "\n".join(lines)

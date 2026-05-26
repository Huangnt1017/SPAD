"""Loss functions for SPAD dense-fog SNN imaging model.

All losses operate on single-window data (no cross-scene pairing required).
Compatible with noisy GT from histogram-peak extraction.

包含:
- WeakGTLoss: L1 loss (MAE)
- SSIMLoss: 结构相似性 loss (1 - SSIM), 捕获局部结构质量
- GatedMomentVarianceLoss: gate 选中光子的时间方差惩罚
- SpikeSparsityLoss: gate 稀疏性正则
- IntensityAwareSmoothnessLoss: 强度引导的边缘保持平滑
- SPADImagingLoss: 组合训练 loss
- ImageMetrics: 评估工具 (MAE / RMSE / SSIM / PSNR), 不参与梯度
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
    """L1 loss against noisy GT depth and intensity."""

    def __init__(self, w_depth=1.0, w_intensity=1.0):
        super().__init__()
        self.w_depth = w_depth
        self.w_intensity = w_intensity

    def forward(self, result, gt):
        """
        Args:
            result: model output dict with 'output' [B,2,H,W] (refined)
            gt:     [B, 2, H, W] where ch0=depth_gt, ch1=intensity_gt
        """
        d_pred = result["output"][:, 0:1]
        i_pred = result["output"][:, 1:2]
        d_gt = gt[:, 0:1]
        i_gt = gt[:, 1:2]

        mask = (d_gt > 0).float()

        loss_d = (torch.abs(d_pred - d_gt) * mask).sum() / (mask.sum() + 1e-6)
        loss_i = (torch.abs(i_pred - i_gt) * mask).sum() / (mask.sum() + 1e-6)

        return self.w_depth * loss_d + self.w_intensity * loss_i


class GatedMomentVarianceLoss(nn.Module):
    """Penalize if gate-selected photons scatter too far from predicted depth.

    Target echo is narrow (~4 bin FWHM). If the variance of selected photon
    timestamps around predicted depth is large, the gate is selecting wrong photons.
    """

    def __init__(self, sigma_target=4.0):
        super().__init__()
        self.sigma2 = sigma_target ** 2

    def forward(self, result):
        gate = result["gate"]             # [P, B, 1, H, W]
        tof = result["tof"]               # [P, B, H, W]
        valid = result["valid"]           # [P, B, H, W]
        depth = result["depth_coarse"]    # [B, 1, H, W]  use coarse for variance

        depth_exp = depth.squeeze(1).unsqueeze(0)                # [1, B, H, W]
        gate_sq = gate.squeeze(2)                                # [T, B, H, W]

        gv = gate_sq * valid                                     # [T, B, H, W]
        residual2 = (tof - depth_exp) ** 2                       # [T, B, H, W]

        weighted_var = (gv * residual2).sum(0) / (gv.sum(0) + 1e-6)  # [B, H, W]
        excess = F.relu(weighted_var - self.sigma2)

        return excess.mean()


class SpikeSparsityLoss(nn.Module):
    """Encourage gate sparsity: target photons are rare in dense fog."""

    def __init__(self, rho_target=0.15):
        super().__init__()
        self.rho_target = rho_target

    def forward(self, result):
        gate = result["gate"]       # [T, B, 1, H, W]
        valid = result["valid"]     # [T, B, H, W]

        gate_sq = gate.squeeze(2)   # [T, B, H, W]
        mean_rate = (gate_sq * valid).sum() / (valid.sum() + 1e-6)

        return F.relu(mean_rate - self.rho_target)


class SSIMLoss(nn.Module):
    """结构相似性损失 (1 - SSIM), 对 depth 和 intensity 分别计算.

    SSIM 捕获局部亮度/对比度/结构信息, 弥补纯像素级 L1 的不足.
    仅在有 GT 且 mask 覆盖区域计算; 数据范围由 depth_range / intensity_range 指定.

    Args:
        w_depth: depth SSIM loss 权重
        w_intensity: intensity SSIM loss 权重
        kernel_size: 高斯窗口大小 (推荐 11, 图片 64×64 时可用 7)
        depth_range: depth 的数据动态范围 (默认 150, 对应 tof bin 最大值)
        intensity_range: intensity 的动态范围 (默认 1.0, 归一化后)
    """

    def __init__(self, w_depth=1.0, w_intensity=1.0,
                 kernel_size=7, depth_range=150.0, intensity_range=1.0):
        super().__init__()
        self.w_depth = w_depth
        self.w_intensity = w_intensity
        self.kernel_size = kernel_size
        self.depth_range = depth_range
        self.intensity_range = intensity_range

    def forward(self, result, gt):
        """
        Args:
            result: model output dict with 'output' [B, 2, H, W]
            gt:     [B, 2, H, W] where ch0=depth_gt, ch1=intensity_gt

        Returns:
            (1 - SSIM) 加权合并的标量 loss
        """
        d_pred = result["output"][:, 0:1]       # [B, 1, H, W]
        i_pred = result["output"][:, 1:2]       # [B, 1, H, W]
        d_gt = gt[:, 0:1]
        i_gt = gt[:, 1:2]

        mask = (d_gt > 0).float()               # [B, 1, H, W]

        ssim_d = _compute_ssim(d_pred, d_gt, mask=mask,
                               kernel_size=self.kernel_size,
                               data_range=self.depth_range)
        ssim_i = _compute_ssim(i_pred, i_gt, mask=mask,
                               kernel_size=self.kernel_size,
                               data_range=self.intensity_range)

        loss = self.w_depth * (1.0 - ssim_d) + self.w_intensity * (1.0 - ssim_i)
        return loss


class IntensityAwareSmoothnessLoss(nn.Module):
    """Edge-preserving depth smoothness guided by intensity gradients.

    Depth should be smooth where intensity is smooth (background),
    and can have discontinuities where intensity jumps (target boundary).
    """

    def __init__(self, beta=5.0):
        super().__init__()
        self.beta = beta

    def forward(self, result):
        depth = result["output"][:, 0:1]       # [B, 1, H, W]  refined
        intensity = result["output"][:, 1:2]   # [B, 1, H, W]  refined

        grad_d = _spatial_gradient(depth)
        grad_i = _spatial_gradient(intensity)

        weight = torch.exp(-self.beta * grad_i)
        return (grad_d * weight).mean()


class SPADImagingLoss(nn.Module):
    """组合训练损失函数.

    L = w_gt * L_GT + w_ssim * L_SSIM + w_var * L_var + w_sparse * L_sparse + w_smooth * L_smooth
        [+ w_lut_smooth * L_lut_smooth + w_lut_norm * L_lut_norm]   (仅 LUT 编码模式)

    Args:
        w_gt: L1 (MAE) loss 权重
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
    """

    def __init__(
        self,
        w_gt=0.3,
        w_ssim=0.1,
        w_var=1.0,
        w_sparse=0.05,
        w_smooth=0.1,
        w_lut_smooth=0.01,
        w_lut_norm=0.005,
        sigma_target=4.0,
        rho_target=0.15,
        beta_smooth=5.0,
        ssim_kernel_size=7,
        depth_range=150.0,
    ):
        super().__init__()
        self.w_gt = w_gt
        self.w_ssim = w_ssim
        self.w_var = w_var
        self.w_sparse = w_sparse
        self.w_smooth = w_smooth
        self.w_lut_smooth = w_lut_smooth
        self.w_lut_norm = w_lut_norm

        self.gt_loss = WeakGTLoss()
        self.ssim_loss = SSIMLoss(kernel_size=ssim_kernel_size, depth_range=depth_range)
        self.var_loss = GatedMomentVarianceLoss(sigma_target=sigma_target)
        self.sparse_loss = SpikeSparsityLoss(rho_target=rho_target)
        self.smooth_loss = IntensityAwareSmoothnessLoss(beta=beta_smooth)

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
            losses["gt"] = l_gt.item()
            total = total + self.w_gt * l_gt

        if gt is not None and self.w_ssim > 0:
            l_ssim = self.ssim_loss(result, gt)
            losses["ssim"] = l_ssim.item()
            total = total + self.w_ssim * l_ssim

        if self.w_var > 0:
            l_var = self.var_loss(result)
            losses["var"] = l_var.item()
            total = total + self.w_var * l_var

        if self.w_sparse > 0:
            l_sparse = self.sparse_loss(result)
            losses["sparse"] = l_sparse.item()
            total = total + self.w_sparse * l_sparse

        if self.w_smooth > 0:
            l_smooth = self.smooth_loss(result)
            losses["smooth"] = l_smooth.item()
            total = total + self.w_smooth * l_smooth

        # LUT 编码正则 (仅当 model 使用 encoding_mode="lut" 时 result 中才有这些键)
        if "lut_smooth" in result and self.w_lut_smooth > 0:
            l_lut_s = result["lut_smooth"]
            losses["lut_smooth"] = l_lut_s.item()
            total = total + self.w_lut_smooth * l_lut_s

        if "lut_norm" in result and self.w_lut_norm > 0:
            l_lut_n = result["lut_norm"]
            losses["lut_norm"] = l_lut_n.item()
            total = total + self.w_lut_norm * l_lut_n

        losses["total"] = total.item()
        return total, losses


# ─── 评估指标 (不参与梯度, 仅用于 validation/test) ────────────────────

class ImageMetrics:
    """图像质量评估工具, 计算 depth 和 intensity 的 MAE / RMSE / SSIM / PSNR.

    用法:
        metrics = ImageMetrics(depth_range=150.0)
        scores = metrics.compute(result, gt)
        # scores = {"depth_mae": ..., "depth_rmse": ..., "depth_ssim": ..., "depth_psnr": ...,
        #           "intensity_mae": ..., ...}

    所有计算在有效像素 (d_gt > 0) 上进行, 无效区域被 mask 排除.
    """

    def __init__(self, depth_range=150.0, intensity_range=1.0, ssim_kernel_size=7):
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
    def compute(self, result, gt):
        """计算全部图像质量指标.

        Args:
            result: model output dict with 'output' [B, 2, H, W]
            gt:     [B, 2, H, W] (ch0=depth, ch1=intensity)

        Returns:
            dict: 各通道各指标的标量值
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
        scores["depth_mae"] = d_err.abs().sum().item() / num_valid.item()
        d_mse = (d_err ** 2).sum().item() / num_valid.item()
        scores["depth_rmse"] = math.sqrt(d_mse)
        scores["depth_ssim"] = _compute_ssim(
            d_pred_norm, d_gt_norm, mask=mask,
            kernel_size=self.ssim_kernel_size,
            data_range=1.0,
        ).item()
        # PSNR = 10 * log10(1^2 / MSE), 归一化后 data_range=1.0
        scores["depth_psnr"] = self._psnr(d_mse, 1.0)

        # ── Intensity 指标 (已在 [0, 1] 范围, 直接计算) ──
        i_pred_norm = i_pred / self.intensity_range
        i_gt_norm = i_gt / self.intensity_range
        i_err = (i_pred_norm - i_gt_norm) * mask
        scores["intensity_mae"] = i_err.abs().sum().item() / num_valid.item()
        i_mse = (i_err ** 2).sum().item() / num_valid.item()
        scores["intensity_rmse"] = math.sqrt(i_mse)
        scores["intensity_ssim"] = _compute_ssim(
            i_pred_norm, i_gt_norm, mask=mask,
            kernel_size=self.ssim_kernel_size,
            data_range=1.0,
        ).item()
        # PSNR = 10 * log10(1^2 / MSE)
        scores["intensity_psnr"] = self._psnr(i_mse, 1.0)

        return scores

    @staticmethod
    def _psnr(mse, data_range):
        """标准 PSNR 公式: 10 * log10(MAX^2 / MSE). MSE≈0 时返回 inf."""
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

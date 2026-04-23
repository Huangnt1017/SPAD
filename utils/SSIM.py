import torch
from torch import nn
import numpy as np
import os
from pathlib import Path
from torch.nn import functional as F
from typing import Literal, Callable, Sequence, Union


# ===== 优化的 SSIMLoss（处理全零情况）=====
class AverageConvnd(nn.Module):
    kernel: torch.Tensor

    def __init__(self, size: Sequence[Union[int, float]]):
        super().__init__()
        self.size = list(size)
        assert len(self.size) == 3, "Only 3D conv supported for [N, C, D, H, W] input"
        self.register_buffer("kernel", self._make_average_kernel(self.size))
        self.conv_fn = F.conv3d

    def _make_average_kernel(self, size: Sequence[Union[int, float]]) -> torch.Tensor:
        kernel = torch.ones(size) / torch.as_tensor(size).prod()
        kernel = kernel.view(1, 1, *kernel.shape)
        return kernel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in_channels = input.size(1)
        kernel = self.kernel.expand(in_channels, 1, *self.kernel.shape[2:]).type_as(input)
        padding = tuple(k // 2 for k in self.size)
        return self.conv_fn(
            input,
            kernel,
            bias=None,
            stride=1,
            padding=padding,
            dilation=1,
            groups=in_channels
        )


class GaussianConvnd(nn.Module):
    kernel: torch.Tensor

    def __init__(self, size: Sequence[Union[int, float]], *, sigma: float = 1.5):
        super().__init__()
        self.size = list(size)
        assert len(self.size) == 3, "Only 3D conv supported for [N, C, D, H, W] input"
        self.sigma = sigma
        self.register_buffer("kernel", self._make_gaussian_kernel(self.size, sigma=sigma))
        self.conv_fn = F.conv3d

    def _make_gaussian_kernel(self, size, sigma=1.5):
        coords = [torch.arange(s, dtype=torch.float32) - (s - 1) / 2.0 for s in size]
        grids = torch.meshgrid(*coords, indexing="ij")
        kernel = torch.exp(-torch.stack(grids).pow(2).sum(0) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        return kernel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in_channels = input.size(1)
        kernel = self.kernel.expand(in_channels, 1, *self.kernel.shape[2:]).type_as(input)
        padding = tuple(k // 2 for k in self.size)
        return self.conv_fn(
            input,
            kernel,
            bias=None,
            stride=1,
            padding=padding,
            dilation=1,
            groups=in_channels
        )

def calculate_ssim_for_point_clouds(gt_dir, ds_dir, mode="ds", letters=None):
    # mode配置
    if mode == "ds":
        x_range = (20, 35)
        y_range = (5, 20)
        z_range = (75, 90)
        grid_size = (15, 15, 15)
    elif mode == "all":
        x_range = (1, 64)
        y_range = (1, 64)
        z_range = (1, 190)
        grid_size = (64, 64, 190)
    else:
        raise ValueError("mode must be 'ds' or 'all'")

    gt_data, ds_data = load_full_point_clouds(gt_dir, ds_dir, letters=letters)

    if not gt_data:
        print("No point cloud data found!")
        return {}

    ssim_loss = SSIMLoss(kernel_size=3, kernel="gauss", reduction="none")
    H, W, D = grid_size

    # 合并后的体素化逻辑
    def to_voxel_grid(points):
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 4:
            raise ValueError("points must have shape [N, >=4], with intensity in column 3")

        in_range = (
            (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1]) &
            (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1]) &
            (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])
        )
        pts = pts[in_range]

        grid = np.zeros((H, W, D), dtype=np.float32)
        kept = pts.shape[0]
        if kept == 0:
            return grid, kept

        idx_x = ((pts[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * (H - 1)).astype(np.int64)
        idx_y = ((pts[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * (W - 1)).astype(np.int64)
        idx_z = ((pts[:, 2] - z_range[0]) / (z_range[1] - z_range[0]) * (D - 1)).astype(np.int64)

        np.maximum.at(grid, (idx_x, idx_y, idx_z), pts[:, 3])
        return grid, kept

    results = {}
    valid_letters = []

    for letter in sorted(gt_data.keys()):
        if letter not in ds_data:
            continue

        gt_points = gt_data[letter]
        ds_points = ds_data[letter]

        # 按你的要求：仅保留这一处滤波
        gt_points = gt_points[gt_points[:, 3] > 15]

        if len(gt_points) == 0 or len(ds_points) == 0:
            print(f"Skipping {letter}: empty points after filtering")
            continue

        try:
            gt_grid, gt_kept = to_voxel_grid(gt_points)
            ds_grid, ds_kept = to_voxel_grid(ds_points)
        except Exception as e:
            print(f"Error voxelizing {letter}: {e}")
            continue

        if gt_kept == 0 or ds_kept == 0:
            print(f"Skipping {letter}: no in-range points after filtering")
            continue

        # 在获得 gt_grid 和 ds_grid 后，进行联合归一化
        # 计算两个网格的全局最小值和最大值
        all_values = np.concatenate([gt_grid.flatten(), ds_grid.flatten()])
        global_min = all_values.min()
        global_max = all_values.max()

        if global_max - global_min > 1e-8:  # 避免范围过小
            gt_grid = (gt_grid - global_min) / (global_max - global_min)
            ds_grid = (ds_grid - global_min) / (global_max - global_min)
        else:
            # 如果所有值都相同（比如全是0），则两个网格都设为全零
            gt_grid = np.zeros_like(gt_grid)
            ds_grid = np.zeros_like(ds_grid)

        gt_tensor = torch.from_numpy(gt_grid).unsqueeze(0).unsqueeze(0)
        ds_tensor = torch.from_numpy(ds_grid).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            gt_density = F.avg_pool3d(gt_tensor, kernel_size=3, stride=1, padding=1)
            ds_density = F.avg_pool3d(ds_tensor, kernel_size=3, stride=1, padding=1)

        mask = (gt_density > 0.01) | (ds_density > 0.01)
        valid_voxels = int(mask.sum().item())

        if valid_voxels == 0:
            ssim_val = 0.0
        else:
            ssim_map = 1 - ssim_loss(gt_tensor, ds_tensor)
            valid_ssim = ssim_map[mask]
            ssim_val = float(valid_ssim.mean().item()) if valid_ssim.numel() > 0 else 0.0

        results[letter] = {
            "ssim": ssim_val,
            "grid_shape": gt_grid.shape,
            "valid_voxels": valid_voxels
        }
        valid_letters.append(letter)
        # print(f"Letter {letter}: SSIM = {ssim_val:.4f}, Valid voxels: {valid_voxels}")

    if valid_letters:
        avg_ssim = float(np.mean([results[l]["ssim"] for l in valid_letters]))
        total_valid_voxels = int(sum(results[l]["valid_voxels"] for l in valid_letters))
        print(f"\nAverage SSIM: {avg_ssim:.4f}, Total valid voxels: {total_valid_voxels}")
    else:
        print("No valid results.")

    return results


def load_full_point_clouds(gt_dir, ds_dir, letters=None):
    if letters is None:
        letters = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

    gt_data = {}
    ds_data = {}

    for letter in letters:
        gt_file = os.path.join(gt_dir, f"{letter}.txt")
        ds_file = os.path.join(ds_dir, f"{letter}.txt")

        if not (os.path.exists(gt_file) and os.path.exists(ds_file)):
            print(f"Files not found for {letter}")
            continue

        try:
            gt_points = np.loadtxt(gt_file, delimiter=",")
            ds_points = np.loadtxt(ds_file, delimiter=",")
        except Exception as e:
            print(f"Skipping {letter}: load failed, {e}")
            continue

        if gt_points.shape[1] < 4 or ds_points.shape[1] < 4:
            print(f"Skipping {letter}: invalid format, need at least 4 columns")
            continue

        if len(gt_points) == 0 or len(ds_points) == 0:
            print(f"Skipping {letter}: empty file")
            continue

        gt_data[letter] = gt_points.astype(np.float32, copy=False)
        ds_data[letter] = ds_points.astype(np.float32, copy=False)
        # print(f"Loaded {letter}: GT {gt_data[letter].shape}, DS {ds_data[letter].shape}")

    return gt_data, ds_data


class SSIMLoss(nn.Module):
    r"""
    Creates a criterion that measures the SSIM (structural similarity index measure) between
    each element in the input and target.

    Supports 1d, 2d, 3d input.

    Attributes:
        kernel_size: The size of the sliding window. Must be an int, or a shape with 1, 2 or 3 dimensions.
        *,
        kernel_type: Type of kernel ("avg" or "gauss") or a Custom callable object.
        reduction: Reduction method ("mean", "sum", or "none").
        data_range: Dynamic range of input tensors.
        k1,k2: Stabilization constants for SSIM calculation.
    """

    conv: Callable[..., torch.Tensor]
    data_range: torch.Tensor

    def __init__(
            self,
            kernel_size: Union[int, Sequence[Union[int,float]]],
            *,
            kernel: Union[Literal["avg", "gauss"], Callable[[torch.Tensor], torch.Tensor]] = "gauss",
            reduction: Literal["mean", "sum", "none"] = "mean",
            data_range: float = 1.0,
            k1: float = 0.01, k2: float = 0.03
    ):
        super().__init__()
        self.kernel_size = [kernel_size] * 3 if isinstance(kernel_size, int) else kernel_size
        self.kernel = kernel
        self.reduction = reduction
        self.register_buffer("data_range", torch.as_tensor(data_range))
        self.k1, self.k2 = k1, k2

        npts = torch.as_tensor(self.kernel_size).prod()
        self.cov_norm = npts / (npts - 1)

        if self.kernel == "avg":
            self.conv = AverageConvnd(self.kernel_size)
        elif self.kernel == "gauss":
            self.conv = GaussianConvnd(self.kernel_size, sigma=1.5)
        elif callable(self.kernel):
            self.conv = self.kernel
        else:
            raise ValueError("`kernel` only supports 'avg', 'gauss' or Callable object.")

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            batch_data_range: torch.Tensor = None,
    ):
        device = input.device
        data_range = self.data_range.to(device)

        if batch_data_range is not None:
            if input.size(0)  != batch_data_range.size(0):
                raise ValueError("`input` and `batch_data_range` must have the same batchsize.")
            data_range = batch_data_range.to(device).view(input.size(0),*([1] * (input.ndim - 1)))
        else:
            data_range = data_range.view(*([1] * (input.ndim))).expand(input.size(0),*([1] * (input.ndim - 1)))

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        ux = self.conv(input)
        uy = self.conv(target)
        uxx = self.conv(input**2)
        uyy = self.conv(target**2)
        uxy = self.conv(input * target)

        vx = self.cov_norm * (uxx - ux**2)
        vy = self.cov_norm * (uyy - uy**2)
        vxy = self.cov_norm * (uxy - ux * uy)

        A1 = 2 * ux * uy + C1
        A2 = 2 * vxy + C2
        B1 = ux**2 + uy**2 + C1
        B2 = vx + vy + C2
        ssim_map = (A1 * A2) / (B1 * B2 + 1e-8)

        if self.reduction == "mean":
            return 1 - ssim_map.mean()
        elif self.reduction == "sum":
            return ssim_map.size(0) - ssim_map.mean(tuple(range(1, ssim_map.ndim))).sum()
        else:
            return 1 - ssim_map
        # 这里输出的是1-SSIM，所以是LOSS，

import random

random.seed(42)  # 设置随机数种子
if __name__ == "__main__":
    gt_dir = r"E:\essay\硕士\点云降采样+补全\CVPR\data\raw_data"
    ds_dir = r"E:\essay\硕士\点云降采样+补全\CVPR\data\ds_data\fpps"
    mode = "ds"  # 可选: "ds" 或 "all"
    results = calculate_ssim_for_point_clouds(gt_dir, ds_dir, mode=mode)
    print("filename:", ds_dir)
    print("MODE:", mode)
    print("Final Results:", results, "\n")
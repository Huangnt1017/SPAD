import torch
import numpy as np
import json
from typing import Tuple, List, Dict, Optional

def load_xyzi(file_path: str) -> np.ndarray:
    """Read xyzi txt as numpy array (Utility)"""
    try:
        data = np.loadtxt(file_path, dtype=int)
        if data.ndim == 1 and data.size > 0: data = data.reshape(1, -1)
        elif data.size == 0: return np.zeros((0, 4), dtype=int)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return np.zeros((0, 4), dtype=int)

def save_xyzi(data: np.ndarray, save_path: str):
    """Save numpy array as xyzi txt (Utility)"""
    np.savetxt(save_path, data, fmt='%d', delimiter=' ')


def _randint_inclusive(low: int, high: int, generator: Optional[torch.Generator] = None) -> int:
    """Sample an integer from [low, high] with an optional deterministic generator."""
    if low > high:
        raise ValueError(f"Invalid randint range: [{low}, {high}]")
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())

def augment_pytorch_batch(
    points: torch.Tensor,
    label_class: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[List[Dict]]]:
    """
    Batch augmentation for SPAD point clouds.

    Features:
    - Input: (B, N, 4) FloatTensor.
    - Target layer (fixed source region): x[20,35), y[5,25), z[80,85).
      Random translation in x/y/z (no rotation).
    - Fog layer (fixed source region): x[1,65), y[1,65), z[35,65).
      Random translation only in z.
    - Constraints:
      1) Fog is always in front of target by at least 5 bins.
      2) Target z max does not exceed 110.

    Args:
        points: Input point clouds (B, N, 4).
        label_class: Class name string. If None, metadata is not returned.
        seed: Optional base seed. If provided, each sample i uses (seed + i),
              which guarantees deterministic augmentation across runs.

    Returns:
        augmented_points: (B, N, 4) Tensor.
        metadata: List of dicts (if label_class provided) else None.
    """
    if points.ndim != 3 or points.shape[-1] < 4:
        raise ValueError(f"points shape must be (B, N, >=4), got {tuple(points.shape)}")

    device = points.device
    B, N, _ = points.shape

    # Source regions (upper bounds are exclusive)
    tgt_x = (20, 35)
    tgt_y = (5, 20)
    tgt_z = (80, 85)

    fog_x = (1, 65)
    fog_y = (1, 65)
    fog_z = (35, 65)

    # Global constraints (inclusive bounds)
    xy_limit = (1, 64)
    target_z_limit = (60, 110)
    fog_z_limit = (1, 105)
    min_gap_bins = 5

    # Shift ranges for target translation
    dx_range = (xy_limit[0] - tgt_x[0], xy_limit[1] - (tgt_x[1] - 1))
    dy_range = (xy_limit[0] - tgt_y[0], xy_limit[1] - (tgt_y[1] - 1))
    dz_target_range = (target_z_limit[0] - tgt_z[0], target_z_limit[1] - (tgt_z[1] - 1))

    # Base shift range for fog z translation
    dz_fog_range = (fog_z_limit[0] - fog_z[0], fog_z_limit[1] - (fog_z[1] - 1))

    aug_points = points.clone()
    meta_list: List[Dict] = []

    # Process batch
    for i in range(B):
        pc = points[i]
        xyz = pc[:, :3]

        target_mask = (xyz[:, 0] >= tgt_x[0]) & (xyz[:, 0] < tgt_x[1]) & \
                      (xyz[:, 1] >= tgt_y[0]) & (xyz[:, 1] < tgt_y[1]) & \
                      (xyz[:, 2] >= tgt_z[0]) & (xyz[:, 2] < tgt_z[1])

        fog_mask = (xyz[:, 0] >= fog_x[0]) & (xyz[:, 0] < fog_x[1]) & \
                   (xyz[:, 1] >= fog_y[0]) & (xyz[:, 1] < fog_y[1]) & \
                   (xyz[:, 2] >= fog_z[0]) & (xyz[:, 2] < fog_z[1])

        sample_generator: Optional[torch.Generator] = None
        if seed is not None:
            sample_generator = torch.Generator(device="cpu")
            sample_generator.manual_seed(int(seed) + i)

        dx = _randint_inclusive(dx_range[0], dx_range[1], sample_generator)
        dy = _randint_inclusive(dy_range[0], dy_range[1], sample_generator)

        # Sample target and fog z shifts with gap constraint:
        # target_z_min_new - fog_z_max_new >= min_gap_bins
        sampled = False
        dz_target = 0
        dz_fog = 0
        for _ in range(20):
            dz_target = _randint_inclusive(dz_target_range[0], dz_target_range[1], sample_generator)
            dz_fog_max_by_gap = tgt_z[0] + dz_target - min_gap_bins - (fog_z[1] - 1)
            dz_fog_low = dz_fog_range[0]
            dz_fog_high = min(dz_fog_range[1], dz_fog_max_by_gap)

            if dz_fog_low <= dz_fog_high:
                dz_fog = _randint_inclusive(dz_fog_low, dz_fog_high, sample_generator)
                sampled = True
                break

        if not sampled:
            raise RuntimeError("Unable to sample valid target/fog shifts under current constraints.")

        # Move target: random x/y/z translation, no rotation
        if target_mask.any():
            aug_points[i, target_mask, 0] += dx
            aug_points[i, target_mask, 1] += dy
            aug_points[i, target_mask, 2] += dz_target

            aug_points[i, target_mask, 0] = torch.clamp(aug_points[i, target_mask, 0], xy_limit[0], xy_limit[1])
            aug_points[i, target_mask, 1] = torch.clamp(aug_points[i, target_mask, 1], xy_limit[0], xy_limit[1])
            aug_points[i, target_mask, 2] = torch.clamp(aug_points[i, target_mask, 2], target_z_limit[0], target_z_limit[1])

        # Move fog: global z-only translation
        if fog_mask.any():
            aug_points[i, fog_mask, 2] += dz_fog
            aug_points[i, fog_mask, 2] = torch.clamp(aug_points[i, fog_mask, 2], fog_z_limit[0], fog_z_limit[1])

        target_x_new = (tgt_x[0] + dx, tgt_x[1] + dx)
        target_y_new = (tgt_y[0] + dy, tgt_y[1] + dy)
        target_z_new_inclusive = (tgt_z[0] + dz_target, (tgt_z[1] - 1) + dz_target)
        fog_z_new_inclusive = (fog_z[0] + dz_fog, (fog_z[1] - 1) + dz_fog)

        meta_list.append({
            "label": label_class,
            "target_shift": [int(dx), int(dy), int(dz_target)],
            "fog_shift_z": int(dz_fog),
            "target_x_range": [int(target_x_new[0]), int(target_x_new[1])],
            "target_y_range": [int(target_y_new[0]), int(target_y_new[1])],
            "target_z_range": [int(target_z_new_inclusive[0]), int(target_z_new_inclusive[1])],
            "fog_z_range": [int(fog_z_new_inclusive[0]), int(fog_z_new_inclusive[1])],
            "fog_ahead_gap_bins": int(target_z_new_inclusive[0] - fog_z_new_inclusive[1])
        })

    return aug_points, meta_list

def _draw_3d_box_wireframe(ax, x_range, y_range, z_range, color='r', linewidth=1.5, alpha=0.8):
    """在3D Axes上绘制线框box，用于可视化目标/烟雾3D区域"""
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range

    # 8个顶点
    verts = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # 底面
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # 顶面
    ]
    # 12条边
    edges = [
        (0,1),(1,2),(2,3),(3,0),   # 底面
        (4,5),(5,6),(6,7),(7,4),   # 顶面
        (0,4),(1,5),(2,6),(3,7),   # 竖边
    ]
    for i, j in edges:
        ax.plot3D([verts[i][0], verts[j][0]],
                  [verts[i][1], verts[j][1]],
                  [verts[i][2], verts[j][2]],
                  color=color, linewidth=linewidth, alpha=alpha)


if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决 OpenMP 多副本冲突

    import matplotlib
    matplotlib.use('TkAgg')  # 使用TkAgg后端以支持交互显示
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    print("=== PyTorch Batch Augmentation Test (Real Data) ===\n")

    # ── 1. 加载真实数据 ──────────────────────────────────────────
    data_path = r"D:\PYproject\SPADdata\2025-04-30-dpc\G\2025-04-30_18-53-59_Delay-0_Width-200-1-3.txt"
    raw_data = np.loadtxt(data_path, dtype=int, delimiter=',')
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)
    print(f"原始数据: {raw_data.shape[0]} 个点, 维度={raw_data.shape[1]} (xyzi)")

    # ── 2. 构建 batch=1 的输入张量 ──────────────────────────────
    B = 1
    points = torch.from_numpy(raw_data).float().unsqueeze(0)  # (1, N, 4)
    N = points.shape[1]
    print(f"输入张量形状: {points.shape}  (B={B}, N={N})")

    # ── 3. 执行增强 ─────────────────────────────────────────────
    label_class = "A"
    aug_points, meta = augment_pytorch_batch(points, label_class=label_class, seed=42)

    # ── 4. 输出增强后的 label：3D box 位置及类别 ──────────────────
    print("\n" + "=" * 62)
    print("  增强后标签 — 3D Box 位置及类别")
    print("=" * 62)
    m = meta[0]
    print(f"  类别 (label)        : {m['label']}")
    print(f"  目标位移 dx/dy/dz   : {m['target_shift']}")
    print(f"  目标 3D Box X 范围  : [{m['target_x_range'][0]}, {m['target_x_range'][1]})")
    print(f"  目标 3D Box Y 范围  : [{m['target_y_range'][0]}, {m['target_y_range'][1]})")
    print(f"  目标 3D Box Z 范围  : [{m['target_z_range'][0]}, {m['target_z_range'][1]}] (含)")
    print(f"  烟雾位移 dz          : {m['fog_shift_z']}")
    print(f"  烟雾 3D Box Z 范围  : [{m['fog_z_range'][0]}, {m['fog_z_range'][1]}] (含)")
    print(f"  烟雾在目标前方 bins  : {m['fog_ahead_gap_bins']}")

    # ── 5. 准备可视化数据 ────────────────────────────────────────
    orig = points[0].cpu().numpy()      # (N, 4) 原始数据
    aug  = aug_points[0].cpu().numpy()  # (N, 4) 增强后数据

    # 源区域（固定，用于原始数据 box 标注）
    src_tgt_x, src_tgt_y, src_tgt_z = (20, 35), (5, 20), (80, 85)
    src_fog_x, src_fog_y, src_fog_z = (1, 65), (1, 65), (35, 65)

    # 增强后区域（从 metadata 获取）
    aug_tgt_x = (m['target_x_range'][0], m['target_x_range'][1])
    aug_tgt_y = (m['target_y_range'][0], m['target_y_range'][1])
    aug_tgt_z = (m['target_z_range'][0], m['target_z_range'][1] + 1)   # metadata 上界含
    aug_fog_x = src_fog_x       # 烟雾 x/y 不变
    aug_fog_y = src_fog_y
    aug_fog_z = (m['fog_z_range'][0], m['fog_z_range'][1] + 1)         # metadata 上界含

    # ── 6. 参照 plot_pc 风格绘制原始 vs 增强对比图 ──────────────
    # 自定义 colormap：浅蓝 → 深红 (与 plot_pc 一致)
    light_blue = (113/255, 178/255, 255/255)
    dark_red   = (255/255, 0/255, 0/255)
    cmap_custom = LinearSegmentedColormap.from_list('lightblue_to_darkred', [light_blue, dark_red])
    norm = Normalize(vmin=1, vmax=750)

    # 强度驱动的逐点透明度 (与 plot_pc 一致)
    int_orig = orig[:, 3].astype(np.float32)
    int_aug = aug[:, 3].astype(np.float32)

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f"SPAD Point Cloud Augmentation  |  label={label_class}  |  {N} points",
                 fontsize=13, fontweight='bold')

    # ---- 子图1: 原始数据 ----
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("Original", fontsize=12, color='navy')

    ax1.scatter(orig[:, 0], orig[:, 1], orig[:, 2],
                c=int_orig, s=2, cmap=cmap_custom, alpha=0.5)

    # 叠加 3D Box 线框
    _draw_3d_box_wireframe(ax1, src_tgt_x, src_tgt_y, src_tgt_z, color='red', linewidth=1.5)
    _draw_3d_box_wireframe(ax1, src_fog_x, src_fog_y, src_fog_z, color='cyan', linewidth=1.0)

    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_xlim(1, 64); ax1.set_ylim(1, 64); ax1.set_zlim(1, 110)
    ax1.view_init(elev=10, azim=-7)

    # ---- 子图2: 增强后数据 ----
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("Augmented", fontsize=12, color='darkgreen')

    ax2.scatter(aug[:, 0], aug[:, 1], aug[:, 2],
                c=int_aug, s=2, cmap=cmap_custom, alpha=0.5)

    # 叠加增强后 3D Box 线框
    _draw_3d_box_wireframe(ax2, aug_tgt_x, aug_tgt_y, aug_tgt_z, color='red', linewidth=1.5)
    _draw_3d_box_wireframe(ax2, aug_fog_x, aug_fog_y, aug_fog_z, color='cyan', linewidth=1.0)

    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_xlim(1, 64); ax2.set_ylim(1, 64); ax2.set_zlim(1, 110)
    ax2.view_init(elev=10, azim=-7)

    plt.tight_layout()
    plt.show()

    print("\nDone.")

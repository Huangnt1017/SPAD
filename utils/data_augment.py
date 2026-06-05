import torch
import numpy as np
import json
from typing import Tuple, List, Dict, Optional


AUGMENT_SEED_STRIDE = 1_000_003
"""不同增强副本之间的随机种子步长，选用大素数降低重复采样概率。"""

TARGET_ROTATION_DEGREE_RANGE = (-180.0, 180.0)
"""target 绕 z 轴旋转的角度范围，单位为 degree。"""

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
    """从闭区间 [low, high] 采样整数，可传入确定性随机生成器。"""
    if low > high:
        raise ValueError(f"Invalid randint range: [{low}, {high}]")
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def _uniform_float(low: float, high: float, generator: Optional[torch.Generator] = None) -> float:
    """从闭区间近似采样浮点数，可传入确定性随机生成器。"""
    if low > high:
        raise ValueError(f"Invalid uniform range: [{low}, {high}]")
    if low == high:
        return float(low)
    value = torch.rand(1, generator=generator).item()
    return float(low + value * (high - low))


def _rotate_xy_about_center(xy: torch.Tensor, center_xy: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    """在 xy 平面绕 center_xy 旋转，等价于绕 z 轴旋转。"""
    angle_radians = float(np.deg2rad(angle_degrees))
    cos_theta = torch.as_tensor(np.cos(angle_radians), dtype=xy.dtype, device=xy.device)
    sin_theta = torch.as_tensor(np.sin(angle_radians), dtype=xy.dtype, device=xy.device)

    relative = xy - center_xy
    rotated_x = relative[:, 0] * cos_theta - relative[:, 1] * sin_theta
    rotated_y = relative[:, 0] * sin_theta + relative[:, 1] * cos_theta
    return torch.stack((rotated_x, rotated_y), dim=1) + center_xy


def resolve_num_aug(num_aug: int, apply_augment: bool = True) -> int:
    """规范化单个样本的增强副本数。

    Args:
        num_aug: 启用增强时，每个原始样本生成的增强样本份数。
        apply_augment: 当前数据集是否启用增强；关闭增强时固定返回 1。

    Returns:
        Dataset 索引展开时使用的有效副本数。

    Raises:
        ValueError: 启用增强但 num_aug 不是正整数。
    """
    if not apply_augment:
        return 1

    try:
        value = int(num_aug)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"num_aug must be a positive integer when augmentation is enabled, got: {num_aug}") from exc

    if value <= 0:
        raise ValueError(f"num_aug must be positive when augmentation is enabled, got: {num_aug}")
    return value


def augment_pytorch_batch(
    points: torch.Tensor,
    label_class: Optional[str] = None,
    seed: Optional[int] = None,
    num_aug: int = 1,
) -> Tuple[torch.Tensor, Optional[List[Dict]]]:
    """
    Batch augmentation for SPAD point clouds.

    Features:
    - Input: (B, N, 4) FloatTensor.
    - Target layer (fixed source region): x[20,35), y[5,20), z[80,85).
      Random z-axis rotation around the target center, then translation in x/y/z.
    - Fog layer (fixed source region): x[1,65), y[1,65), z[35,65).
      Random translation only in z.
    - Constraints:
      1) Fog is always in front of target by at least 5 bins.
      2) Target z max does not exceed 110.

    Args:
        points: Input point clouds (B, N, 4).
        label_class: 类别名称，仅写入增强元信息。
        seed: Optional base seed. If provided, each sample i uses (seed + i),
              which guarantees deterministic augmentation across runs.
        num_aug: 每个输入样本生成的增强副本数。

    Returns:
        augmented_points: (B * num_aug, N, 4) 张量。
        metadata: 与 augmented_points 对齐的元信息列表。
    """
    if points.ndim != 3 or points.shape[-1] < 4:
        raise ValueError(f"points shape must be (B, N, >=4), got {tuple(points.shape)}")

    effective_num_aug = resolve_num_aug(num_aug, apply_augment=True)
    if effective_num_aug > 1:
        aug_batches: List[torch.Tensor] = []
        meta_batches: List[Dict] = []
        for aug_index in range(effective_num_aug):
            aug_seed = None if seed is None else int(seed) + aug_index * AUGMENT_SEED_STRIDE
            aug_points, aug_meta = augment_pytorch_batch(
                points,
                label_class=label_class,
                seed=aug_seed,
                num_aug=1,
            )
            aug_batches.append(aug_points)
            if aug_meta is not None:
                meta_batches.extend(aug_meta)
        return torch.cat(aug_batches, dim=0), meta_batches

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

    # target 的训练标签沿用固定连续框，旋转时绕该框中心点旋转。
    target_rotation_center = (
        (tgt_x[0] + tgt_x[1]) * 0.5,
        (tgt_y[0] + tgt_y[1]) * 0.5,
    )
    target_source_corners_xy = torch.tensor(
        [
            [tgt_x[0], tgt_y[0]],
            [tgt_x[1] - 1, tgt_y[0]],
            [tgt_x[1] - 1, tgt_y[1] - 1],
            [tgt_x[0], tgt_y[1] - 1],
        ],
        dtype=torch.float32,
    )

    # Shift range for target z translation
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

        target_rotation_degrees = _uniform_float(
            TARGET_ROTATION_DEGREE_RANGE[0],
            TARGET_ROTATION_DEGREE_RANGE[1],
            sample_generator,
        )

        # 先计算旋转后 target 的实际 xy 外接范围，再采样平移量，避免边界处被 clamp 截断。
        center_xy_cpu = torch.tensor(target_rotation_center, dtype=torch.float32)
        rotated_corners_xy = _rotate_xy_about_center(
            target_source_corners_xy,
            center_xy_cpu,
            target_rotation_degrees,
        )
        rotated_xy_min = rotated_corners_xy.min(dim=0).values
        rotated_xy_max = rotated_corners_xy.max(dim=0).values
        dx_range = (
            int(np.ceil(xy_limit[0] - float(rotated_xy_min[0]))),
            int(np.floor(xy_limit[1] - float(rotated_xy_max[0]))),
        )
        dy_range = (
            int(np.ceil(xy_limit[0] - float(rotated_xy_min[1]))),
            int(np.floor(xy_limit[1] - float(rotated_xy_max[1]))),
        )

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

        # Move target: z 轴旋转与 x/y/z 平移在同一次增强中完成。
        if target_mask.any():
            center_xy = torch.as_tensor(target_rotation_center, dtype=pc.dtype, device=device)
            shift_xy = torch.as_tensor([dx, dy], dtype=pc.dtype, device=device)
            rotated_xy = _rotate_xy_about_center(
                pc[target_mask, :2],
                center_xy,
                target_rotation_degrees,
            )

            aug_points[i, target_mask, 0:2] = rotated_xy + shift_xy
            aug_points[i, target_mask, 2] = pc[target_mask, 2] + dz_target

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
        target_rotated_x_range = (
            float(rotated_xy_min[0]) + dx,
            float(rotated_xy_max[0]) + dx,
        )
        target_rotated_y_range = (
            float(rotated_xy_min[1]) + dy,
            float(rotated_xy_max[1]) + dy,
        )

        meta_list.append({
            "label": label_class,
            "target_shift": [int(dx), int(dy), int(dz_target)],
            "target_rotation_degrees": float(target_rotation_degrees),
            "target_rotation_center": [
                float(target_rotation_center[0] + dx),
                float(target_rotation_center[1] + dy),
            ],
            "target_rotated_x_range": [float(target_rotated_x_range[0]), float(target_rotated_x_range[1])],
            "target_rotated_y_range": [float(target_rotated_y_range[0]), float(target_rotated_y_range[1])],
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


def _draw_rotated_box_wireframe(
    ax,
    x_range,
    y_range,
    z_range,
    center_xy,
    angle_degrees,
    color='r',
    linewidth=1.5,
    alpha=0.8,
):
    """绘制绕 z 轴旋转后的 3D 线框 box。"""
    base_xy = torch.tensor(
        [
            [x_range[0], y_range[0]],
            [x_range[1], y_range[0]],
            [x_range[1], y_range[1]],
            [x_range[0], y_range[1]],
        ],
        dtype=torch.float32,
    )
    center_xy_tensor = torch.tensor(center_xy, dtype=torch.float32)
    rotated_xy = _rotate_xy_about_center(base_xy, center_xy_tensor, angle_degrees).cpu().numpy()

    z0, z1 = z_range
    verts = [
        [rotated_xy[0, 0], rotated_xy[0, 1], z0],
        [rotated_xy[1, 0], rotated_xy[1, 1], z0],
        [rotated_xy[2, 0], rotated_xy[2, 1], z0],
        [rotated_xy[3, 0], rotated_xy[3, 1], z0],
        [rotated_xy[0, 0], rotated_xy[0, 1], z1],
        [rotated_xy[1, 0], rotated_xy[1, 1], z1],
        [rotated_xy[2, 0], rotated_xy[2, 1], z1],
        [rotated_xy[3, 0], rotated_xy[3, 1], z1],
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot3D(
            [verts[i][0], verts[j][0]],
            [verts[i][1], verts[j][1]],
            [verts[i][2], verts[j][2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


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
    aug_points, meta = augment_pytorch_batch(points, label_class=label_class, seed=41)

    # ── 4. 输出增强后的 label：3D box 位置及类别 ──────────────────
    print("\n" + "=" * 62)
    print("  增强后标签 — 3D Box 位置及类别")
    print("=" * 62)
    m = meta[0]
    print(f"  类别 (label)        : {m['label']}")
    print(f"  目标位移 dx/dy/dz   : {m['target_shift']}")
    print(f"  目标绕 z 轴旋转角度 : {m['target_rotation_degrees']:.2f}°")
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
    aug_tgt_center_xy = (m['target_rotation_center'][0], m['target_rotation_center'][1])
    aug_tgt_angle = m['target_rotation_degrees']
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

    # 叠加增强后 3D Box 线框：target 使用真实旋转后的线框，fog 仍为轴对齐框。
    _draw_rotated_box_wireframe(
        ax2,
        aug_tgt_x,
        aug_tgt_y,
        aug_tgt_z,
        center_xy=aug_tgt_center_xy,
        angle_degrees=aug_tgt_angle,
        color='red',
        linewidth=1.5,
    )
    _draw_3d_box_wireframe(ax2, aug_fog_x, aug_fog_y, aug_fog_z, color='cyan', linewidth=1.0)

    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_xlim(1, 64); ax2.set_ylim(1, 64); ax2.set_zlim(1, 110)
    ax2.view_init(elev=10, azim=-7)

    plt.tight_layout()
    plt.show()

    print("\nDone.")

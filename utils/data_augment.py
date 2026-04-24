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

if __name__ == "__main__":
    print("=== PyTorch Batch Augmentation Test ===")
    
    # Simulate (B=8, N=1024, C=4) Data
    B, N, C = 8, 1024, 4
    dummy_points = torch.randint(0, 100, (B, N, C)).float()
    dummy_points[:, :, 3] = torch.randint(0, 10, (B, N)).float() # Intensity
    
    # Inject target-layer points
    dummy_points[:, :100, 0] = torch.randint(20, 35, (B, 100)).float()
    dummy_points[:, :100, 1] = torch.randint(5, 25, (B, 100)).float()
    dummy_points[:, :100, 2] = torch.randint(80, 85, (B, 100)).float()

    # Inject fog-layer points
    dummy_points[:, 100:400, 0] = torch.randint(1, 65, (B, 300)).float()
    dummy_points[:, 100:400, 1] = torch.randint(1, 65, (B, 300)).float()
    dummy_points[:, 100:400, 2] = torch.randint(35, 65, (B, 300)).float()

    print(f"Input Shape: {dummy_points.shape}")
    
    # Test augmentation
    out_points, meta = augment_pytorch_batch(dummy_points, label_class="A")
    
    print(f"Output Shape: {out_points.shape}")
    if meta:
        print("\nSample Metadata (Batch 0):")
        print(json.dumps(meta[0], indent=2))
        print("\nSample Metadata (Batch 1):")
        print(json.dumps(meta[1], indent=2))
        
    print("\nTest Finished.")
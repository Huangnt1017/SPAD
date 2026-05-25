"""
SPAD 单 txt 文件推理 + 可视化脚本。

用途:
- 输入单个 .txt 点云文件 (raw 整数 xyzi 格式, 与 utils.data.load_point_cloud_auto 兼容);
- 做一遍训练管线一致的数据增强 (target shift + fog shift);
- 喂进训练好的 baseline, 拿预测类别 + 预测中心点;
- 用 plot_pc 同款 3D 散点风格画增强后的点云, 叠加 pred bbox (红) 和 GT bbox (绿);
- 图像文件名: <pred_class>_<完整 ckpt stem>_<YYYYMMDD_HHMM>.png
  (例如 ``U_pointnet_20260522_003326_448064_best_202605220205.png``),
  保留完整 ckpt 标识以便回溯, 时间戳精确到分钟避免无意义堆积。

数据流:
  txt → (N, 4) int xyzi (raw scale)
      → augment_single_point_cloud → (N, 4) raw scale
      → normalize_points → (N, 4) [0, 1]            ← model 输入
      → model(pts) → logits, pred_center_norm [B, 3]
      → 解码: norm 中心 + 固定半宽 → raw scale 6 维 bbox
      → matplotlib 出图 (raw scale 坐标系)

模型输出位于 [0, 1] 归一化空间 (训练时 dataset 用 normalize_points + normalize_bbox);
可视化需要 raw scale, 因此本脚本里独立实现 ``denormalize_corners_to_raw`` 做反归一化,
不复用 ``decode_normalized_boxes_3d`` (那个是给 [-1, 1] 空间设计的, 不对应此处的 [0, 1])。
"""

from __future__ import annotations

# 必须在 import torch / matplotlib 之前设置: Windows conda 环境下 mkl 与 matplotlib
# 都链了 libiomp5md.dll, 进程退出时会 OMP Error #15 终止 (图来不及落盘)。这条 env
# 是 Intel 官方给的 workaround, 在我们这种纯推理小脚本里完全安全。
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: F401  (Normalize 仅占位, 与 plot_pc 风格一致)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (注册 3D projection)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_model, resolve_path, set_seed
from utils.data import SPAD_NORM_BOUNDS, load_point_cloud_auto, normalize_points
from utils.data_augment import _draw_3d_box_wireframe  # 复用已有 3D wireframe 绘制
from utils.loss import center_to_corners


# ============================================================================
# Logger
# ============================================================================

def setup_logger(log_dir: Path) -> Tuple[logging.Logger, Path]:
	"""创建单样本测试的 logger (file + stderr)。"""
	log_dir.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	log_file = log_dir / f"test1_{timestamp}.log"
	logger_name = f"spad_test1_{timestamp}"
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.INFO)
	logger.propagate = False
	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
	file_handler = logging.FileHandler(log_file, encoding="utf-8")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	return logger, log_file


# ============================================================================
# 单样本数据增强 (与训练管线 utils/data_augment.augment_pytorch_batch 等价的单样本版本)
# ============================================================================

def _randint_inclusive(low: int, high: int, generator: Optional[torch.Generator] = None) -> int:
	"""从闭区间 [low, high] 采一个整数。"""
	if low > high:
		raise ValueError(f"Invalid randint range: [{low}, {high}]")
	return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def augment_single_point_cloud(points: torch.Tensor, seed: Optional[int] = None) -> Dict[str, object]:
	"""对单个点云做 target + fog 偏移 (单样本版增强)。

	Args:
		points: (N, >=4) raw scale xyzi (int / float 均可), 不会被原地修改。
		seed: 复现用的 RNG 种子; None 时用 torch 全局 RNG。

	Returns:
		dict:
			- aug_points: (N, 4) 增强后的 raw scale 点云 (clone, 与输入同 dtype/device)
			- meta: 增强信息, 含 target_box / fog_box 的最终 (raw scale) 范围与平移量
	"""
	if points.ndim != 2 or points.shape[-1] < 4:
		raise ValueError(f"points shape must be (N, >=4), got {tuple(points.shape)}")

	aug_points = points.clone()
	xyz = aug_points[:, :3]

	# === 默认 SPAD 目标 / 雾区初始范围 (与 utils/data_augment.augment_pytorch_batch 完全一致) ===
	tgt_x = (20, 35)
	tgt_y = (5, 20)
	tgt_z = (80, 85)
	fog_x = (1, 65)
	fog_y = (1, 65)
	fog_z = (35, 65)
	xy_limit = (1, 64)
	target_z_limit = (60, 110)
	fog_z_limit = (1, 105)
	min_gap_bins = 5

	# 平移可行域 (避免推出物理量程)
	dx_range = (xy_limit[0] - tgt_x[0], xy_limit[1] - (tgt_x[1] - 1))
	dy_range = (xy_limit[0] - tgt_y[0], xy_limit[1] - (tgt_y[1] - 1))
	dz_target_range = (target_z_limit[0] - tgt_z[0], target_z_limit[1] - (tgt_z[1] - 1))
	dz_fog_range = (fog_z_limit[0] - fog_z[0], fog_z_limit[1] - (fog_z[1] - 1))

	sample_generator: Optional[torch.Generator] = None
	if seed is not None:
		sample_generator = torch.Generator(device="cpu")
		sample_generator.manual_seed(int(seed))

	# === 用 raw scale 阈值生成 target / fog 区域 mask ===
	target_mask = (
		(xyz[:, 0] >= tgt_x[0]) & (xyz[:, 0] < tgt_x[1])
		& (xyz[:, 1] >= tgt_y[0]) & (xyz[:, 1] < tgt_y[1])
		& (xyz[:, 2] >= tgt_z[0]) & (xyz[:, 2] < tgt_z[1])
	)
	fog_mask = (
		(xyz[:, 0] >= fog_x[0]) & (xyz[:, 0] < fog_x[1])
		& (xyz[:, 1] >= fog_y[0]) & (xyz[:, 1] < fog_y[1])
		& (xyz[:, 2] >= fog_z[0]) & (xyz[:, 2] < fog_z[1])
	)

	# 采 dx/dy + 受 gap 约束的 dz_target/dz_fog (最多 20 次重采, 与 batch 增强一致)
	dx = _randint_inclusive(dx_range[0], dx_range[1], sample_generator)
	dy = _randint_inclusive(dy_range[0], dy_range[1], sample_generator)
	valid = False
	dz_target = 0
	dz_fog = 0
	for _ in range(20):
		dz_target = _randint_inclusive(dz_target_range[0], dz_target_range[1], sample_generator)
		dz_fog_max_by_gap = tgt_z[0] + dz_target - min_gap_bins - (fog_z[1] - 1)
		dz_fog_low = dz_fog_range[0]
		dz_fog_high = min(dz_fog_range[1], dz_fog_max_by_gap)
		if dz_fog_low <= dz_fog_high:
			dz_fog = _randint_inclusive(dz_fog_low, dz_fog_high, sample_generator)
			valid = True
			break
	if not valid:
		raise RuntimeError("Unable to sample valid target/fog shifts under current constraints.")

	# === 目标区: x/y/z 三轴平移 + 限幅 ===
	if target_mask.any():
		aug_points[target_mask, 0] += dx
		aug_points[target_mask, 1] += dy
		aug_points[target_mask, 2] += dz_target
		aug_points[target_mask, 0] = torch.clamp(aug_points[target_mask, 0], xy_limit[0], xy_limit[1])
		aug_points[target_mask, 1] = torch.clamp(aug_points[target_mask, 1], xy_limit[0], xy_limit[1])
		aug_points[target_mask, 2] = torch.clamp(aug_points[target_mask, 2], target_z_limit[0], target_z_limit[1])

	# === 雾区: 仅 z 轴平移 + 限幅 ===
	if fog_mask.any():
		aug_points[fog_mask, 2] += dz_fog
		aug_points[fog_mask, 2] = torch.clamp(aug_points[fog_mask, 2], fog_z_limit[0], fog_z_limit[1])

	# 重新计算平移后的 target/fog 真值框 (与训练时 meta 写入逻辑一致, raw scale)
	target_x_new = [int(tgt_x[0] + dx), int(tgt_x[1] + dx)]
	target_y_new = [int(tgt_y[0] + dy), int(tgt_y[1] + dy)]
	target_z_new = [int(tgt_z[0] + dz_target), int((tgt_z[1] - 1) + dz_target)]
	fog_z_new = [int(fog_z[0] + dz_fog), int((fog_z[1] - 1) + dz_fog)]

	return {
		"aug_points": aug_points,
		"meta": {
			"target_shift": [int(dx), int(dy), int(dz_target)],
			"fog_shift_z": int(dz_fog),
			"target_box": {
				"x_range": target_x_new,
				"y_range": target_y_new,
				"z_range": target_z_new,
			},
			"fog_box": {
				"x_range": [int(fog_x[0]), int(fog_x[1])],
				"y_range": [int(fog_y[0]), int(fog_y[1])],
				"z_range": fog_z_new,
			},
			"fog_ahead_gap_bins": int(target_z_new[0] - fog_z_new[1]),
		},
	}


# ============================================================================
# 归一化反变换: 把模型输出的 [0, 1] bbox 解回 raw scale (与 utils.data.normalize_bbox 互逆)
# ============================================================================

def denormalize_corners_to_raw(corners_norm: torch.Tensor) -> torch.Tensor:
	"""把 6 维 bbox 角点从归一化 [0, 1] 反解到 raw scale。

	Args:
		corners_norm: (..., 6) [xmin, xmax, ymin, ymax, zmin, zmax], 归一化空间。

	Returns:
		raw: (..., 6) 同形状, 物理坐标 (x/y 范围 [1, 64], z 范围 [1, 110])。
	"""
	x_min, x_max = SPAD_NORM_BOUNDS["x"]
	y_min, y_max = SPAD_NORM_BOUNDS["y"]
	z_min, z_max = SPAD_NORM_BOUNDS["z"]

	raw = corners_norm.clone().to(torch.float32)
	raw[..., 0] = raw[..., 0] * (x_max - x_min) + x_min
	raw[..., 1] = raw[..., 1] * (x_max - x_min) + x_min
	raw[..., 2] = raw[..., 2] * (y_max - y_min) + y_min
	raw[..., 3] = raw[..., 3] * (y_max - y_min) + y_min
	raw[..., 4] = raw[..., 4] * (z_max - z_min) + z_min
	raw[..., 5] = raw[..., 5] * (z_max - z_min) + z_min
	return raw


# ============================================================================
# 可视化: 复刻 data_read/raw2pointcloud.py::plot_pc 的 'all' mode + 叠加 bbox
# ============================================================================

def plot_points_with_boxes(
	aug_points_np: np.ndarray,
	pred_box_raw: np.ndarray,
	gt_box_raw: Optional[np.ndarray],
	fog_box_raw: Optional[np.ndarray],
	pred_class: str,
	pred_score: float,
	save_path: Path,
) -> None:
	"""画增强后的点云 + 预测/真值/雾 bbox, 用 plot_pc 'all' mode 同款风格。

	plot_pc 风格细节 (data_read/raw2pointcloud.py::plot_pc):
	- 自定义 colormap 从浅蓝 (113,178,255) → 深红 (255,0,0)
	- 强度非线性化 i^0.7 后归一化得到逐点 alpha
	- 三轴: ax X = SPAD Y; ax Y = SPAD Z; ax Z = |SPAD X - 65| (即 64 - (x - 1))
	- 视角 elev=10, azim=-7

	bbox 同样要按上面的坐标置换后画 (raw scale 输入 [xmin,xmax,ymin,ymax,zmin,zmax]
	对应 SPAD 物理 (X, Y, Z), 画到 ax (Y, Z, X) 上)。

	Args:
		aug_points_np: (N, 4) raw scale xyzi (int 或 float 均可)。
		pred_box_raw: (6,) [xmin, xmax, ymin, ymax, zmin, zmax] 模型预测框 (raw scale)。
		gt_box_raw: (6,) augment 给出的 target 真值框 (raw scale), None 时不画。
		fog_box_raw: (6,) augment 给出的 fog 区域框 (raw scale), None 时不画。
		pred_class: 模型预测类别字符串, 用作图标题与文件名一部分。
		pred_score: 模型 softmax 置信度, 标在标题。
		save_path: 输出 PNG 路径; 上级目录会自动创建。
	"""
	pc = np.asarray(aug_points_np)
	xyz = pc[:, :3].astype(np.int32, copy=False)
	intensity = pc[:, 3].astype(np.int32, copy=False)

	fig = plt.figure(figsize=(12, 9))
	ax = fig.add_subplot(111, projection="3d")
	# 与 plot_pc 'all' mode 一致: 坐标轴顺序 (Y, Z, |X-65|)
	ax.set_xlabel("Y")
	ax.set_ylabel("Z")
	ax.set_zlabel("X")

	# === plot_pc 'all' mode colormap & 透明度 ===
	light_blue = (113 / 255, 178 / 255, 255 / 255)
	dark_red = (255 / 255, 0 / 255, 0 / 255)
	cmap_custom = LinearSegmentedColormap.from_list("lightblue_to_darkred", [light_blue, dark_red])

	i_nl = intensity.astype(np.float32) ** 0.7   # 非线性化强度
	denom = float(i_nl.max() - i_nl.min())
	alpha = (i_nl - i_nl.min()) / denom if denom > 0 else 0.6

	# 注意轴置换: scatter(ax_X=spad_Y, ax_Y=spad_Z, ax_Z=|spad_X-65|)
	sc = ax.scatter(
		xyz[:, 1], xyz[:, 2], np.abs(xyz[:, 0] - 65),
		c=intensity, s=2, cmap=cmap_custom, alpha=alpha,
	)
	ax.set_xlim(0, 64)
	ax.set_ylim(0, 190)
	ax.set_zlim(0, 64)
	ax.view_init(elev=10, azim=-7)

	# === 叠加 bbox: 注意 raw bbox 是 (X, Y, Z) 坐标, 但 ax 是 (Y, Z, |X-65|) ===
	def _draw_box_in_plot_pc_axes(raw_box: np.ndarray, color: str, label: str, linewidth: float = 1.8):
		"""raw_box=[xmin,xmax,ymin,ymax,zmin,zmax] (SPAD 物理 X/Y/Z)。

		画到 ax 坐标系: ax_X 用 SPAD_Y, ax_Y 用 SPAD_Z, ax_Z 用 |SPAD_X - 65|。
		为了与点云的轴置换一致, 这里需要把 raw_box 的 X 端点也做 |x - 65| 变换 (单调下降,
		min/max 会反转, 但 _draw_3d_box_wireframe 内部用 verts 直接连边, 不依赖 min/max 顺序)。
		"""
		x0, x1 = float(raw_box[0]), float(raw_box[1])
		y0, y1 = float(raw_box[2]), float(raw_box[3])
		z0, z1 = float(raw_box[4]), float(raw_box[5])
		# X 反转: ax_Z = |spad_X - 65|, 即 spad_X 端点 → ax_Z 端点
		ax_z0 = abs(x0 - 65)
		ax_z1 = abs(x1 - 65)
		# 调用项目里已有的线框绘制 (按 ax 坐标顺序 ax_X=Y, ax_Y=Z, ax_Z=|X-65|)
		_draw_3d_box_wireframe(
			ax,
			x_range=(y0, y1),
			y_range=(z0, z1),
			z_range=(ax_z0, ax_z1),
			color=color, linewidth=linewidth, alpha=0.95,
		)
		# 额外加文字标签 (放在 bbox 第一个角点附近)
		ax.text(y0, z0, ax_z0, label, color=color, fontsize=10, weight="bold")

	def _box_center_raw(raw_box: np.ndarray) -> Tuple[float, float, float]:
		"""从 [xmin, xmax, ymin, ymax, zmin, zmax] 求 (cx, cy, cz) 中心 (SPAD raw scale)。"""
		return (
			0.5 * (float(raw_box[0]) + float(raw_box[1])),
			0.5 * (float(raw_box[2]) + float(raw_box[3])),
			0.5 * (float(raw_box[4]) + float(raw_box[5])),
		)

	# pred box (红)
	_draw_box_in_plot_pc_axes(pred_box_raw, color="red", label=f"pred:{pred_class}")
	# GT box (绿)
	if gt_box_raw is not None:
		_draw_box_in_plot_pc_axes(gt_box_raw, color="lime", label="gt")
	# fog box (灰)
	if fog_box_raw is not None:
		_draw_box_in_plot_pc_axes(fog_box_raw, color="gray", label="fog", linewidth=1.0)

	cbar = fig.colorbar(sc, location="left", shrink=0.5, fraction=0.05, pad=0.03)
	cbar.set_label("Intensity of points")

	# === 标题: 第一行类别+置信度, 第二行 pred / actual 中心点坐标 (X Y Z, 保留 1 位小数) ===
	# 用 raw scale 物理坐标 (X∈[1,64], Y∈[1,64], Z∈[1,110]), 与 bbox 在图上的坐标一致,
	# 这样读图时能直接验证"框在不在中心"。actual=GT 中心, 没有 gt 时只显示 pred。
	pred_cx, pred_cy, pred_cz = _box_center_raw(pred_box_raw)
	title_lines = [f"pred = {pred_class}    score = {pred_score:.3f}"]
	if gt_box_raw is not None:
		gt_cx, gt_cy, gt_cz = _box_center_raw(gt_box_raw)
		title_lines.append(
			f"pred[{pred_cx:.1f} {pred_cy:.1f} {pred_cz:.1f}]    "
			f"actual[{gt_cx:.1f} {gt_cy:.1f} {gt_cz:.1f}]"
		)
	else:
		title_lines.append(f"pred[{pred_cx:.1f} {pred_cy:.1f} {pred_cz:.1f}]")
	ax.set_title("\n".join(title_lines))

	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(save_path, dpi=160, bbox_inches="tight")
	plt.show()
	plt.close(fig)


# ============================================================================
# 主流程: 单文件读 → 增强 → 推理 → 画图
# ============================================================================

def run_single_test(args: argparse.Namespace) -> Dict[str, str]:
	"""单 txt 文件: 增强 + 推理 + 出图。

	Returns:
		dict {log_file, image_path, pred_class, pred_score}
	"""
	project_root = Path(__file__).resolve().parents[1]
	input_file = resolve_path(args.input, project_root)
	checkpoint_path = resolve_path(args.checkpoint, project_root)
	log_dir = resolve_path(args.log_dir, project_root)
	output_dir = resolve_path(args.output_dir, project_root)
	# seed=None (默认) → 每次随机, 数据增强每次不同, 方便反复 click Run 看不同样本;
	# 传 --seed N → 固定 RNG, 同一文件 + 同一 seed 永远产生同一增强 (复现 / debug 用)。
	# 注意: utils/data.py 模块级 (line 22-25) 在 import 时就调用了
	# torch.manual_seed(42) / np.random.seed(42), 所以"不传 seed"实际还是固定的;
	# 这里在 seed=None 分支主动调用 torch.seed() / np.random.seed(None) / random.seed()
	# 把全局 RNG 重置回系统熵源, 才能真正做到每次 click Run 增强结果不同。
	if args.seed is not None:
		set_seed(int(args.seed))
	else:
		torch.seed()              # 用系统熵源给 torch 全局 RNG 重新 seed
		torch.cuda.seed_all() if torch.cuda.is_available() else None
		np.random.seed(None)      # numpy 同理 (虽然本文件不直接用, 但保持一致)
		import random as _random
		_random.seed()            # stdlib random 也重置

	# device 解析: auto → 有 CUDA 用 CUDA, 否则 CPU
	if args.device == "auto":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	elif args.device == "cuda" and not torch.cuda.is_available():
		device = torch.device("cpu")
	else:
		device = torch.device(args.device)

	logger, log_file = setup_logger(log_dir)
	logger.info("=== Single TXT File Test ===")
	logger.info("input=%s", input_file)
	logger.info("checkpoint=%s", checkpoint_path)
	logger.info("device=%s", device)

	if not input_file.exists():
		raise FileNotFoundError(f"input file not found: {input_file}")
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

	# === 1) 读 txt → raw (N, 4) ===
	points_np = load_point_cloud_auto(str(input_file))    # int32 raw xyzi
	points_raw = torch.from_numpy(points_np).to(torch.float32)
	logger.info("loaded points shape=%s", tuple(points_raw.shape))

	# === 2) 增强 (raw scale 内做) ===
	if args.augment:
		aug_out = augment_single_point_cloud(points_raw, seed=args.seed)
		aug_points_raw = aug_out["aug_points"]    # (N, 4) raw scale float
		aug_meta = aug_out["meta"]
		logger.info("augment target_shift=%s fog_shift_z=%d",
		            aug_meta["target_shift"], aug_meta["fog_shift_z"])
	else:
		aug_points_raw = points_raw.clone()
		aug_meta = None
		logger.info("augment disabled")

	# === 3) 归一化 (与训练 dataset 一致), 喂模型 ===
	aug_points_np = aug_points_raw.cpu().numpy().astype(np.float32, copy=False)
	normed_np = normalize_points(aug_points_np)        # (N, 4) [0, 1]
	normed = torch.from_numpy(normed_np).to(device)    # 模型在 device 上

	# === 4) 模型加载 + 前向 ===
	checkpoint = torch.load(checkpoint_path, map_location=device)
	class_to_idx = checkpoint.get("class_to_idx")
	if not isinstance(class_to_idx, dict):
		raise KeyError("Checkpoint does not contain class_to_idx mapping.")
	idx_to_class = {idx: name for name, idx in class_to_idx.items()}
	num_classes = len(class_to_idx)

	# args.model 在 build_parser 里限制了 choices, 这里直接传
	model = build_model(args.model, num_classes=num_classes, project_root=project_root).to(device)
	state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
	model.load_state_dict(state_dict)
	model.eval()

	with torch.no_grad():
		# 模型期待 (B, N, 4)
		logits, box_pred = model(normed.unsqueeze(0))
		probs = torch.softmax(logits, dim=1)
		pred_score, pred_idx = probs.max(dim=1)
		pred_class = idx_to_class.get(int(pred_idx.item()), str(int(pred_idx.item())))

		# === 5) 解码 pred bbox: norm 中心 → norm 角点 → raw scale ===
		box_pred_t = torch.as_tensor(box_pred, device=device, dtype=logits.dtype)
		if box_pred_t.shape[-1] == 3:
			# center-only 新约定 (memory: project_bbox_refactor)
			pred_corners_norm = center_to_corners(box_pred_t, device=device, dtype=logits.dtype).squeeze(0)
		elif box_pred_t.shape[-1] == 6:
			# 兼容老模型 (3DETR 等仍输出 6 维)
			pred_corners_norm = box_pred_t.squeeze(0)
		else:
			raise ValueError(f"Unexpected box_pred trailing dim: {tuple(box_pred_t.shape)}")
		pred_corners_raw = denormalize_corners_to_raw(pred_corners_norm).detach().cpu().numpy()

	pred_class_str = str(pred_class)
	pred_score_val = float(pred_score.item())
	logger.info("pred_class=%s score=%.4f", pred_class_str, pred_score_val)
	logger.info("pred_box_raw=%s", json.dumps([round(float(v), 3) for v in pred_corners_raw.tolist()]))

	# pred 中心 (raw scale) — 由 6 维角点取中点; 与 plot 中标的红星位置一致。
	pred_center_raw = np.array([
		0.5 * (pred_corners_raw[0] + pred_corners_raw[1]),
		0.5 * (pred_corners_raw[2] + pred_corners_raw[3]),
		0.5 * (pred_corners_raw[4] + pred_corners_raw[5]),
	], dtype=np.float32)
	logger.info("pred_center_raw=%s", json.dumps([round(float(v), 3) for v in pred_center_raw.tolist()]))

	# === 6) 准备 GT / fog 框 (从 aug_meta 拿, 已是 raw scale) ===
	gt_box_raw = None
	fog_box_raw = None
	gt_center_raw: Optional[np.ndarray] = None
	if aug_meta is not None:
		tb = aug_meta["target_box"]
		gt_box_raw = np.array([
			tb["x_range"][0], tb["x_range"][1],
			tb["y_range"][0], tb["y_range"][1],
			tb["z_range"][0], tb["z_range"][1],
		], dtype=np.float32)
		fb = aug_meta["fog_box"]
		fog_box_raw = np.array([
			fb["x_range"][0], fb["x_range"][1],
			fb["y_range"][0], fb["y_range"][1],
			fb["z_range"][0], fb["z_range"][1],
		], dtype=np.float32)
		gt_center_raw = np.array([
			0.5 * (gt_box_raw[0] + gt_box_raw[1]),
			0.5 * (gt_box_raw[2] + gt_box_raw[3]),
			0.5 * (gt_box_raw[4] + gt_box_raw[5]),
		], dtype=np.float32)
		# 中心点偏移 (L2 距离), 直观看模型定位精度; 也是 plot 中橙色虚线段的长度。
		center_offset = float(np.linalg.norm(pred_center_raw - gt_center_raw))
		logger.info("gt_box_raw=%s", json.dumps(gt_box_raw.tolist()))
		logger.info("gt_center_raw=%s", json.dumps([round(float(v), 3) for v in gt_center_raw.tolist()]))
		logger.info("center_offset_raw=%.3f (raw L2 distance between pred and gt centers)", center_offset)

	# === 7) 出图: 文件名 = <预测类别>_<完整 ckpt stem>_<时间戳到分钟>.png ===
	# - pred_class: 模型预测的字母 (字符串)
	# - ckpt_stem:  整个 .pth 文件名 (去掉 .pth 扩展), 例如
	#               'pointnet_20260522_003326_448064_best'; 不做截断, 保留训练时
	#               的完整时间戳和 best/last 标识, 便于在挑选样本时回溯到准确 ckpt。
	# - YYYYMMDD_HHMM: 测试时刻精确到分钟 (不要秒/微秒), 同一分钟内对同一 ckpt
	#                  同一 pred 类别再跑会覆盖, 这是预期行为 (避免堆积重复样本)。
	timestamp = datetime.now().strftime("%Y%m%d_%H%M")
	safe_pred = "".join(c for c in pred_class_str if c.isalnum() or c in "_-") or "unknown"
	# stem 已不含 .pth, 但若用户传了非 .pth 文件名也直接用 stem; 路径分隔符/特殊字符
	# 都已被 stem 过滤, 这里不再二次清洗以保留完整可识别度。
	ckpt_stem = checkpoint_path.stem
	image_path = output_dir / f"{safe_pred}_{ckpt_stem}_{timestamp}.png"
	plot_points_with_boxes(
		aug_points_np=aug_points_np,
		pred_box_raw=pred_corners_raw,
		gt_box_raw=gt_box_raw,
		fog_box_raw=fog_box_raw,
		pred_class=pred_class_str,
		pred_score=pred_score_val,
		save_path=image_path,
	)
	logger.info("saved figure to %s", image_path)

	return {
		"log_file": str(log_file),
		"image_path": str(image_path),
		"pred_class": pred_class_str,
		"pred_score": pred_score_val,
	}


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
	"""单 txt 文件推理 + 可视化 CLI。"""
	parser = argparse.ArgumentParser(description="SPAD single-TXT-file inference + visualization")
	parser.add_argument(
		"--input", type=str, 
		default=r"D:\\PYproject\\SPADdata\\2025-04-30-dpc\\D\\2025-04-30_18-51-28_Delay-0_Width-200-6-8.txt",
		help="路径到单个点云 .txt (或 .npy/.npz, 与 utils.data.load_point_cloud_auto 一致)",
	)
	parser.add_argument(
		"--checkpoint", type=str,
		# 默认指向 2026-05-21 bbox-refactor 之后训练的 dgcnn ckpt (box_head 已是 3 维中心);
		# 早于 2026-05-21 的 ckpt (例如 dgcnn_20260426_*) 仍是 6 维角点 box_head, 与当前
		# baseline/DGCNN.py 接口不兼容, load_state_dict 会 size mismatch。
		default=r"D:\PYproject\SPAD\checkpoints\dgcnn_20260521_044755_814832_best.pth",
		help="训练 checkpoint 路径 (须为 bbox-refactor 之后的 3 维 box_head ckpt)",
	)
	# 与 scripts/test.py 的 --model choices 列表保持一致, 覆盖 baseline/ 下所有
	# 已注册的模型 (build_model 支持的全部 name)。
	parser.add_argument(
		"--model", type=str, default="dgcnn",
		choices=["dgcnn", "pointnet", "pointnet2", "pointnet2msg", "pointtransformer", "pointtransv2", "pointtransv3",
			"pointmlp", "pointbert", "pointmae", "pointrwkv", "spt", "upp",
		],
	)
	parser.add_argument("--output-dir", type=str, default=r"D:\PYproject\SPAD\logs\test1",
	                    help="输出图像保存目录, 文件名=<pred_class>_<ckpt_stem>_<YYYYMMDD_HHMM>.png")
	parser.add_argument("--log-dir", type=str, default="logs")
	parser.add_argument(
		"--seed", type=int, default=None,
		help="可选随机 seed; 默认 None = 每次都不同 (适合反复 click Run 看不同增强样本); "
		     "传整数则固定 RNG 用于复现",
	)
	parser.add_argument("--device", type=str, default="auto",
	                    choices=["auto", "cpu", "cuda"])
	# 数据增强开关 (默认开启, 与训练管线一致); 用 --no-augment 关掉做纯前向调试
	parser.add_argument("--augment", dest="augment", action="store_true",
	                    help="启用 target+fog 增强 (默认)")
	parser.add_argument("--no-augment", dest="augment", action="store_false",
	                    help="禁用增强, 只跑原始点云的纯前向")
	parser.set_defaults(augment=True)
	return parser


def main(argv=None) -> None:
	"""CLI 入口。"""
	parser = build_parser()
	args = parser.parse_args(argv)
	run_single_test(args)


if __name__ == "__main__":
	# 用法示例 (PowerShell, 单 txt 文件推理 + 可视化):
	#   $env:PYTHONPATH = "D:\PYproject\SPAD"
	#   & "D:\anaconda3\envs\pytorch\python.exe" "D:\PYproject\SPAD\scripts\test1.py" `
	#       --model dgcnn `
	#       --checkpoint "D:\PYproject\SPAD\checkpoints\dgcnn_xxx_best.pth" `
	#       --input "D:\PYproject\SPADdata\2025-04-30-dpc\A\sample_0001.txt"
	#
	# 常用参数 (完整列表见 build_parser):
	#   --input <path>       必填, 单个 .txt 点云路径 (或 .npy/.npz)
	#   --checkpoint <path>  必填, 训练产出的 best.pth
	#   --model <name>       与 ckpt 一致 (默认 dgcnn); choices 与 test.py 对齐
	#   --output-dir <dir>   输出 PNG 目录 (默认 logs/test1)
	#                        文件名 = <pred_class>_<ckpt_stem>_<YYYYMMDD_HHMM>.png
	#                        例: U_pointnet_20260522_003326_448064_best_202605220205.png
	#   --no-augment         关闭增强 (默认开)
	#   --seed 42            固定 seed (默认 None = 每次随机, 反复点 Run 增强结果不同)
	#
	# 输出:
	#   logs/test1_<时间戳>.log                    pred class / score / pred_box / gt_box 数值
	#   <output-dir>/<pred_class>_<ckpt_stem>_<YYYYMMDD_HHMM>.png
	#                                             增强后的点云 + pred bbox(红) + gt bbox(绿) + fog bbox(灰)
	main()

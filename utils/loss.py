from __future__ import annotations

"""
SPAD 多任务损失模块。

模块目的：
- 统一处理分类输出与 3D 轴对齐框输出的损失计算。
- 提供框格式标准化、体积与 IoU 计算等几何工具函数。

主要导出内容：
- to_box_tensor / canonicalize_boxes_3d / box_iou_3d_aligned
- split_cls_and_box_predictions
- build_spad_boxes_from_meta
- PointCloudMultiTaskLoss
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


TensorLike = Union[torch.Tensor, Sequence[float]]
DEFAULT_SPAD_BOX_BOUNDS = ((1.0, 64.0), (1.0, 64.0), (60.0, 110.0))

# ----------------------------------------------------------------------------
# 固定 bbox 尺寸 (center-only 回归约定)
# ----------------------------------------------------------------------------
# SPAD 数据集 GT 框尺寸固定 (由 utils/data_augment.py 中 tgt_x=(20,35) / tgt_y=(5,20) /
# tgt_z=(80,85) 决定; z 在 meta 里以 inclusive [80,84] 存储, 故 raw 宽度为 4)。
# 既然尺寸固定, 模型只需回归中心点 (cx, cy, cz), 测试时用 ±half_size 重建 6 维 bbox。
#
# SPAD 数据集中目标框原始尺寸 (由 data_augment.py 中 tgt_x/y/z 定义):
#   tgt_x = (20, 35)  upper-exclusive → 连续宽度 15, 半宽 7.5
#   tgt_y = (5, 20)   upper-exclusive → 连续宽度 15, 半宽 7.5
#   tgt_z = (80, 85)  upper-exclusive → 连续宽度  5, 半宽 2.5
# 注: data_augment 中 meta 存储 z 为 inclusive [80, 84], 但在连续空间中
#     对应 [80, 85), 宽度 = 5, 与 x/y 的 exclusive 语义一致。
# 归一化分母: SPAD_NORM_BOUNDS 跨度 (63, 63, 109)。
FIXED_BBOX_RAW_HALF_SIZE: Tuple[float, float, float] = (7.5, 7.5, 2.5)
FIXED_BBOX_HALF_SIZE_NORMALIZED: Tuple[float, float, float] = (
    7.5 / 63.0,    # ≈ 0.11905
    7.5 / 63.0,    # ≈ 0.11905
    2.5 / 109.0,   # ≈ 0.02294
)


def corners_to_center(boxes: TensorLike,
                      device: Optional[torch.device] = None,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """从 6 维角点框 [xmin,xmax,ymin,ymax,zmin,zmax] 提取中心点 [cx,cy,cz]。"""
    box_tensor = canonicalize_boxes_3d(boxes, device=device, dtype=dtype)
    centers = (box_tensor[..., 1::2] + box_tensor[..., 0::2]) * 0.5  # (..., 3)
    return centers


def center_to_corners(centers: TensorLike,
                      half_size: Sequence[float] = FIXED_BBOX_HALF_SIZE_NORMALIZED,
                      device: Optional[torch.device] = None,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """以固定半宽从中心点 [cx,cy,cz] 重建 6 维 bbox [xmin,xmax,ymin,ymax,zmin,zmax]。

    Args:
        centers: (..., 3) 中心点张量。
        half_size: (hx, hy, hz) 三轴半宽 (归一化空间), 默认用 SPAD 数据集固定半宽。

    Returns:
        (..., 6) bbox 张量, 顺序 [xmin,xmax,ymin,ymax,zmin,zmax]。
    """
    centers_tensor = torch.as_tensor(centers, dtype=dtype, device=device)
    if centers_tensor.shape[-1] != 3:
        raise ValueError(f"center_to_corners expects last dim = 3, got {tuple(centers_tensor.shape)}")
    hs = torch.as_tensor(half_size, dtype=centers_tensor.dtype, device=centers_tensor.device)
    if hs.shape != (3,):
        raise ValueError(f"half_size must be length-3, got {tuple(hs.shape)}")
    cx, cy, cz = centers_tensor[..., 0], centers_tensor[..., 1], centers_tensor[..., 2]
    hx, hy, hz = hs[0], hs[1], hs[2]
    corners = torch.stack(
        (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz),
        dim=-1,
    )
    return corners


def to_box_tensor(boxes: TensorLike, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
	"""
	Convert boxes into a tensor with trailing shape (..., 6).

	Supported formats:
	- (..., 6): [xmin, xmax, ymin, ymax, zmin, zmax]
	- (..., 3, 2): [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

	Args:
		boxes: 输入框数据，支持 Tensor 或 Python 序列。
		device: 目标设备；若为 None 则沿用 torch.as_tensor 的默认行为。
		dtype: 目标张量类型。

	Returns:
		形状为 (..., 6) 的张量，最后一维按 [xmin, xmax, ymin, ymax, zmin, zmax] 排布。

	Raises:
		ValueError: 输入无法解释为单框 6 个值，或最终形状不是 (...,6)/(...,3,2)。
	"""
	box_tensor = torch.as_tensor(boxes, dtype=dtype, device=device)

	if box_tensor.ndim == 1:
		if box_tensor.numel() != 6:
			raise ValueError(f"Expected 6 values for one box, got {box_tensor.numel()}.")
		box_tensor = box_tensor.unsqueeze(0)
	elif box_tensor.ndim >= 2 and box_tensor.shape[-2:] == (3, 2):
		box_tensor = box_tensor.reshape(*box_tensor.shape[:-2], 6)

	if box_tensor.shape[-1] != 6:
		raise ValueError(f"Expected box shape (...,6) or (...,3,2), got {tuple(box_tensor.shape)}.")

	return box_tensor


def canonicalize_boxes_3d(boxes: TensorLike, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
	"""
	将 3D 框规范化为每个轴都满足 min <= max。

	Args:
		boxes: 输入框，支持 (...,6) 或 (...,3,2)。
		device: 输出设备。
		dtype: 输出类型。

	Returns:
		形状为 (..., 6) 的规范化框。

	Raises:
		ValueError: 输入框形状非法。
	"""
	box_tensor = to_box_tensor(boxes, device=device, dtype=dtype)

	mins = torch.minimum(box_tensor[..., 0::2], box_tensor[..., 1::2])
	maxs = torch.maximum(box_tensor[..., 0::2], box_tensor[..., 1::2])

	return torch.stack(
		(
			mins[..., 0],
			maxs[..., 0],
			mins[..., 1],
			maxs[..., 1],
			mins[..., 2],
			maxs[..., 2],
		),
		dim=-1,
	)


def decode_normalized_boxes_3d(
	boxes: TensorLike,
	bounds: Sequence[Tuple[float, float]] = DEFAULT_SPAD_BOX_BOUNDS,
	device: Optional[torch.device] = None,
	dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
	"""
	Decode normalized 3D boxes from [-1, 1] into absolute SPAD coordinates.

	Args:
		boxes: 输入框，支持 (..., 6) 或 (..., 3, 2)。
		bounds: 三轴绝对坐标范围，顺序为 (x_min/x_max, y_min/y_max, z_min/z_max)。
		device: 输出设备。
		dtype: 输出类型。

	Returns:
		解码后的绝对坐标框，形状为 (..., 6)。

	Raises:
		ValueError: bounds 不是三轴范围，或输入框形状非法。
	"""
	if len(bounds) != 3:
		raise ValueError(f"bounds must contain 3 axis ranges, got {len(bounds)}")

	box_tensor = canonicalize_boxes_3d(boxes, device=device, dtype=dtype)
	bounds_tensor = torch.as_tensor(bounds, dtype=dtype, device=box_tensor.device)
	if bounds_tensor.shape != (3, 2):
		raise ValueError(f"bounds must have shape (3,2), got {tuple(bounds_tensor.shape)}")

	mins = bounds_tensor[:, 0]
	maxs = bounds_tensor[:, 1]
	span = maxs - mins

	normalized = box_tensor.clamp(min=-1.0, max=1.0)
	decoded = torch.stack(
		(
			(normalized[..., 0] + 1.0) * 0.5 * span[0] + mins[0],
			(normalized[..., 1] + 1.0) * 0.5 * span[0] + mins[0],
			(normalized[..., 2] + 1.0) * 0.5 * span[1] + mins[1],
			(normalized[..., 3] + 1.0) * 0.5 * span[1] + mins[1],
			(normalized[..., 4] + 1.0) * 0.5 * span[2] + mins[2],
			(normalized[..., 5] + 1.0) * 0.5 * span[2] + mins[2],
		),
		dim=-1,
	)
	return canonicalize_boxes_3d(decoded, device=box_tensor.device, dtype=dtype)


def box_volume_3d(boxes: TensorLike, eps: float = 1e-8) -> torch.Tensor:
	"""
	计算轴对齐 3D 框体积。

	Args:
		boxes: 输入框，支持 (...,6) 或 (...,3,2)。
		eps: 数值稳定项，避免零体积导致下游除零。

	Returns:
		每个框对应的体积张量。
	"""
	box_tensor = canonicalize_boxes_3d(boxes)
	sizes = (box_tensor[..., 1::2] - box_tensor[..., 0::2]).clamp(min=0.0)
	return sizes.prod(dim=-1) + eps


def box_iou_3d_aligned(pred_boxes: TensorLike, gt_boxes: TensorLike, eps: float = 1e-8) -> torch.Tensor:
	"""
	Pairwise aligned IoU: pred_boxes[i] vs gt_boxes[i].

	Args:
		pred_boxes: (..., 6) or (..., 3, 2)
		gt_boxes:   (..., 6) or (..., 3, 2)
		eps: 数值稳定项。

	Returns:
		对齐 IoU，形状与输入前导维一致。

	Raises:
		ValueError: 预测框与真值框形状不一致。
	"""
	pred = canonicalize_boxes_3d(pred_boxes)
	gt = canonicalize_boxes_3d(gt_boxes, device=pred.device, dtype=pred.dtype)

	if pred.shape != gt.shape:
		raise ValueError(f"Shape mismatch for IoU: pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")

	inter_min = torch.maximum(pred[..., 0::2], gt[..., 0::2])
	inter_max = torch.minimum(pred[..., 1::2], gt[..., 1::2])
	inter_size = (inter_max - inter_min).clamp(min=0.0)
	inter_vol = inter_size.prod(dim=-1)

	pred_vol = (pred[..., 1::2] - pred[..., 0::2]).clamp(min=0.0).prod(dim=-1)
	gt_vol = (gt[..., 1::2] - gt[..., 0::2]).clamp(min=0.0).prod(dim=-1)
	union = pred_vol + gt_vol - inter_vol

	return torch.where(union > 0, inter_vol / (union + eps), torch.zeros_like(union))


def box_diou_3d_aligned(pred_boxes: TensorLike, gt_boxes: TensorLike, eps: float = 1e-8) -> torch.Tensor:
	"""3D Distance-IoU: 当框不重叠 (IoU=0) 时仍有中心距离梯度。

	DIoU = IoU - (center_dist² / enclosing_diagonal²)
	- IoU > 0 时: 正常 IoU 梯度 + 中心对齐惩罚
	- IoU = 0 时: 中心距离项仍推动预测框向 GT 靠拢
	  (解决 SPAD z 轴极窄导致 IoU 长期为 0、梯度消失的问题)

	Args:
		pred_boxes: (..., 6)
		gt_boxes:   (..., 6)
		eps: 数值稳定项。

	Returns:
		DIoU 值 ∈ [-1, 1], 形状与前导维一致。
	"""
	pred = canonicalize_boxes_3d(pred_boxes)
	gt = canonicalize_boxes_3d(gt_boxes, device=pred.device, dtype=pred.dtype)

	if pred.shape != gt.shape:
		raise ValueError(f"Shape mismatch for DIoU: pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")

	# 标准 IoU
	inter_min = torch.maximum(pred[..., 0::2], gt[..., 0::2])
	inter_max = torch.minimum(pred[..., 1::2], gt[..., 1::2])
	inter_size = (inter_max - inter_min).clamp(min=0.0)
	inter_vol = inter_size.prod(dim=-1)

	pred_vol = (pred[..., 1::2] - pred[..., 0::2]).clamp(min=0.0).prod(dim=-1)
	gt_vol = (gt[..., 1::2] - gt[..., 0::2]).clamp(min=0.0).prod(dim=-1)
	union = pred_vol + gt_vol - inter_vol
	iou = torch.where(union > 0, inter_vol / (union + eps), torch.zeros_like(union))

	# 中心距离平方
	pred_center = (pred[..., 0::2] + pred[..., 1::2]) * 0.5   # (..., 3)
	gt_center = (gt[..., 0::2] + gt[..., 1::2]) * 0.5
	center_dist_sq = ((pred_center - gt_center) ** 2).sum(dim=-1)

	# 最小外接框对角线平方
	enclose_min = torch.minimum(pred[..., 0::2], gt[..., 0::2])
	enclose_max = torch.maximum(pred[..., 1::2], gt[..., 1::2])
	diag_sq = ((enclose_max - enclose_min) ** 2).sum(dim=-1).clamp(min=eps)

	return iou - center_dist_sq / diag_sq


def split_cls_and_box_predictions(model_outputs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
	"""
	Parse model outputs into classification logits and optional box predictions.

	Supported output forms:
	- Tensor logits
	- Tuple/List: (logits, box_preds, ...)
	- Dict with common keys:
	  logits keys: logits / cls_logits / class_logits / pred_logits
	  box keys: boxes / pred_boxes / bbox / bbox_pred / box_pred

	Args:
		model_outputs: 模型前向输出。

	Returns:
		(logits, box_preds)。当模型不提供框分支时 box_preds 为 None。

	Raises:
		ValueError: tuple/list 为空。
		TypeError: 输出类型不支持，或 tuple/list 第一个元素不是 logits 张量。
		KeyError: dict 中缺失可识别 logits 键。
	"""
	if torch.is_tensor(model_outputs):
		return model_outputs, None

	if isinstance(model_outputs, (tuple, list)):
		if len(model_outputs) == 0:
			raise ValueError("model_outputs is empty.")
		logits = model_outputs[0]
		if not torch.is_tensor(logits):
			raise TypeError("The first tuple/list element of model_outputs must be logits tensor.")
		box_preds = model_outputs[1] if len(model_outputs) > 1 and torch.is_tensor(model_outputs[1]) else None
		return logits, box_preds

	if isinstance(model_outputs, Mapping):
		logits = None
		for key in ("logits", "cls_logits", "class_logits", "pred_logits"):
			value = model_outputs.get(key)
			if torch.is_tensor(value):
				logits = value
				break

		if logits is None:
			raise KeyError("Cannot find logits in model output dict.")

		box_preds = None
		for key in ("boxes", "pred_boxes", "bbox", "bbox_pred", "box_pred"):
			value = model_outputs.get(key)
			if torch.is_tensor(value):
				box_preds = value
				break
		return logits, box_preds

	raise TypeError(f"Unsupported model output type: {type(model_outputs)}")


def build_spad_boxes_from_meta(meta: Mapping[str, Any], device: Optional[torch.device] = None) -> torch.Tensor:
	"""
	Build [B, 6] boxes from SPAD dataset metadata.

	Expected keys are either:
	- target_x_new / target_y_new / target_z_new
	or
	- target_x_range / target_y_range / target_z_range

	Args:
		meta: batch 级元信息字典，来自 DataLoader 拼接后的样本元信息。
		device: 输出张量设备。

	Returns:
		[B, 6] 规范化框张量，列顺序为 [xmin, xmax, ymin, ymax, zmin, zmax]。

	Raises:
		KeyError: 缺失任一轴的范围键。
		ValueError: 任一轴形状不是 [B,2]，或 x/y/z 轴 batch 维不一致。
	"""

	def _pick_axis(keys: Tuple[str, str]) -> Any:
		for key in keys:
			if key in meta:
				return meta[key]
		raise KeyError(f"Missing axis keys, expected one of: {keys}")

	def _to_range_tensor(values: Any) -> torch.Tensor:
		# 数据流：单样本 [2] 先提升为 [1,2]，再与批量 [B,2] 统一处理。
		if isinstance(values, (list, tuple)) and len(values) == 2 and any(torch.is_tensor(item) for item in values):
			stacked_items = [torch.as_tensor(item, dtype=torch.float32, device=device) for item in values]
			tensor = torch.stack(stacked_items, dim=-1)
			if tensor.ndim == 2 and tensor.shape[0] == 2 and tensor.shape[1] != 2:
				tensor = tensor.transpose(0, 1)
			if tensor.ndim == 1 and tensor.numel() == 2:
				tensor = tensor.unsqueeze(0)
			if tensor.ndim != 2 or tensor.shape[1] != 2:
				raise ValueError(f"Expected axis range tensor shape [B,2], got {tuple(tensor.shape)}")
			return tensor

		tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
		if tensor.ndim == 1:
			if tensor.numel() != 2:
				raise ValueError(f"Axis range must have 2 values, got {tensor.numel()}.")
			tensor = tensor.unsqueeze(0)
		if tensor.ndim != 2 or tensor.shape[1] != 2:
			raise ValueError(f"Expected axis range tensor shape [B,2], got {tuple(tensor.shape)}")
		return tensor

	x_range = _to_range_tensor(_pick_axis(("target_x_new", "target_x_range")))
	y_range = _to_range_tensor(_pick_axis(("target_y_new", "target_y_range")))
	z_range = _to_range_tensor(_pick_axis(("target_z_new", "target_z_range")))

	if not (x_range.shape[0] == y_range.shape[0] == z_range.shape[0]):
		raise ValueError("Batch dimension mismatch among x/y/z box ranges in meta.")

	boxes = torch.cat((x_range, y_range, z_range), dim=1)
	return canonicalize_boxes_3d(boxes)


class PointCloudMultiTaskLoss(nn.Module):
	"""
	点云分类 + 3D 中心点回归的多任务损失 (center-only 约定)。

	GT 框尺寸固定 (见 FIXED_BBOX_HALF_SIZE_NORMALIZED), 模型只回归中心点 [cx,cy,cz];
	测试/可视化时再用固定半宽重建 6 维 bbox。

	默认使用固定权重 λ_cls · L_cls + λ_depth · L_depth (auto_balance=False)。
	也可通过 auto_balance=True 启用 Kendall et al. 同方差不确定性自动平衡:
	  L = exp(-s_cls) * L_cls + s_cls + exp(-s_box) * L_box + s_box

	这里优化的是 s = log(sigma^2), 而不是 sigma^2 本身。s 可以为负,
	但实际方差 sigma^2 = exp(s) 始终为正。由于 +s 正则项存在, 总优化目标
	在分项 loss 很小时可能为负, 这不是负方差, 也不表示反向传播出错。

	其中 L_depth 为 SPAD Soft-histogram depth loss:
	  L_depth = Σ_d Σ_k w_k · (ĉ_d - (c_d^gt + k · δ_d))²
	直接建模 SPAD 物理过程: 时间 bin 量化 (δ_d) + 高斯脉冲展宽 (w_k)。
	w_k = exp(-k² / (2σ²)) / Z 为高斯权重, Z 为归一化常数。

	输入约定:
	- model_outputs 第二项: [B, 3] 中心点 [cx,cy,cz]
	- box_targets: [B, 6] 角点框 (内部转中心)
	"""

	def __init__(
		self,
		cls_weight: float = 1.0,
		box_weight: float = 1.0,
		label_smoothing: float = 0.0,
		half_size: Sequence[float] = FIXED_BBOX_HALF_SIZE_NORMALIZED,
		auto_balance: bool = False,
		sh_k: int = 2,
		sh_sigma: float = 1.5,
	):
		"""
		Args:
			cls_weight: 分类损失固定权重 λ_cls。
			box_weight: 深度回归损失固定权重 λ_depth。
			label_smoothing: 分类标签平滑系数。
			half_size: 固定 bbox 半宽 (归一化空间)。
			auto_balance: 是否启用 Kendall 自适应权重 (默认 False, 使用固定权重)。
			sh_k: Soft-histogram 窗口半径 K (总窗口 2K+1)。
			sh_sigma: Soft-histogram 高斯宽度 σ (单位: bin 数)。
		"""
		super().__init__()
		self.cls_weight = float(cls_weight)
		self.box_weight = float(box_weight)
		self.auto_balance = auto_balance
		self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
		self.register_buffer(
			"_half_size",
			torch.tensor(half_size, dtype=torch.float32),
			persistent=False,
		)

		# Soft-histogram 参数
		self.sh_k = int(sh_k)
		self.sh_sigma = float(sh_sigma)
		# 预计算高斯权重 (2K+1,), 归一化使和为 1
		k_vals = torch.arange(-sh_k, sh_k + 1, dtype=torch.float32)
		weights = torch.exp(-k_vals.pow(2) / (2 * sh_sigma ** 2))
		weights = weights / weights.sum()
		self.register_buffer("_sh_weights", weights, persistent=False)

		# Kendall et al. (CVPR 2018) 同方差不确定性自适应权重 (2 项: cls + box)
		if auto_balance:
			self.log_var_cls = nn.Parameter(torch.zeros(()))
			self.log_var_box = nn.Parameter(torch.zeros(()))

	def forward(
		self,
		model_outputs: Any,
		cls_targets: torch.Tensor,
		box_targets: Optional[TensorLike] = None,
		box_valid_mask: Optional[torch.Tensor] = None,
	) -> Dict[str, torch.Tensor]:
		"""
		计算多任务损失并返回分项指标。

		Args:
			model_outputs: 模型输出, 第二项为 [B,3] 中心点预测 (或 None)。
			cls_targets: 分类标签 [B]。
			box_targets: GT 框, 仍按 [B,6] 角点形式传入 (内部转中心)。
			box_valid_mask: 有效样本掩码 [B]。

		Returns:
			字典: total_loss / cls_loss / box_depth_loss /
			       box_iou_mean (用固定半宽重建后的 3D IoU, 仅监控)。
		"""
		logits, center_preds = split_cls_and_box_predictions(model_outputs)
		cls_targets = cls_targets.long().to(logits.device)

		cls_loss = self.cls_criterion(logits, cls_targets)
		if self.auto_balance:
			total_loss = torch.exp(-self.log_var_cls) * cls_loss + self.log_var_cls
		else:
			total_loss = self.cls_weight * cls_loss

		out: Dict[str, torch.Tensor] = {
			"total_loss": total_loss,
			"cls_loss": cls_loss,
			"box_depth_loss": torch.zeros((), device=logits.device),
			"box_iou_mean": torch.zeros((), device=logits.device),
		}

		if box_targets is None or center_preds is None:
			out["total_loss"] = total_loss
			return out

		# 模型输出可能是 [B,3] (新约定) 或历史 [B,6] (后向兼容); 都收敛到中心 [B,3]。
		center_preds_t = torch.as_tensor(center_preds, dtype=logits.dtype, device=logits.device)
		if center_preds_t.shape[-1] == 6:
			pred_centers = corners_to_center(center_preds_t, device=logits.device, dtype=logits.dtype)
		elif center_preds_t.shape[-1] == 3:
			pred_centers = center_preds_t
		else:
			raise ValueError(
				f"Expected center pred trailing dim 3 (or 6 for legacy), got {tuple(center_preds_t.shape)}"
			)

		gt_corners = canonicalize_boxes_3d(box_targets, device=logits.device, dtype=logits.dtype)
		gt_centers = corners_to_center(gt_corners, device=logits.device, dtype=logits.dtype)

		if pred_centers.shape != gt_centers.shape:
			raise ValueError(
				f"Center pred/target shape mismatch: pred={tuple(pred_centers.shape)}, gt={tuple(gt_centers.shape)}"
			)

		if box_valid_mask is None:
			valid_mask = torch.ones(pred_centers.shape[0], dtype=torch.bool, device=logits.device)
		else:
			valid_mask = box_valid_mask.to(logits.device).bool()

		if valid_mask.numel() != pred_centers.shape[0]:
			raise ValueError(
				f"box_valid_mask length mismatch: mask={valid_mask.numel()}, centers={pred_centers.shape[0]}"
			)

		if valid_mask.any():
			pred_c_valid = pred_centers[valid_mask]
			gt_c_valid = gt_centers[valid_mask]

			half_size = self._half_size.to(device=logits.device, dtype=logits.dtype)

			# SPAD Soft-histogram depth loss (Deng et al., Optics Letters 2026)
			# 直接建模 SPAD 物理过程: 时间 bin 量化 + 高斯脉冲展宽。
			# L_depth = Σ_d Σ_k w_k · (ĉ_d - (c_d^gt + k · δ_d))²
			# 其中 δ_d 为各维度的 bin 宽度 (归一化空间), w_k 为高斯权重。
			#
			# 物理解释:
			# - SPAD 激光雷达的回波信号在时间 bin 上是离散量化的
			# - 高斯脉冲展宽 (σ bins) 导致 GT 深度存在不确定性
			# - Soft-histogram 允许预测落在 GT 附近的多个 bin 上, 受较小惩罚
			sh_weights = self._sh_weights.to(device=logits.device, dtype=logits.dtype)

			# 归一化空间的 bin 宽度: δ_d = 1 / (bins - 1)
			# x/y: 64 bins → δ = 1/63 ≈ 0.01587
			# z: 109 bins → δ = 1/108 ≈ 0.00926
			delta_t = torch.tensor([1.0 / 63.0, 1.0 / 63.0, 1.0 / 108.0],
			                      device=logits.device, dtype=logits.dtype)  # (3,)

			# 对每个 k ∈ [-K, K], 计算加权 MSE
			# sh_weights shape: (2K+1,), 索引 i 对应 k = i - K
			box_depth_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
			for i in range(2 * self.sh_k + 1):
				k = i - self.sh_k  # 当前偏移 bin 数
				w_k = sh_weights[i]  # 高斯权重
				# GT 中心偏移 k 个 bin: c_d^gt + k · δ_d
				gt_shifted = gt_c_valid + k * delta_t.unsqueeze(0)  # (B_valid, 3)
				# MSE: (ĉ_d - (c_d^gt + k · δ_d))²
				mse_k = (pred_c_valid - gt_shifted).pow(2)  # (B_valid, 3)
				# 加权求和: w_k · MSE
				box_depth_loss = box_depth_loss + w_k * mse_k.sum(dim=-1).mean()

			# 标准 IoU 仅用于监控 (不参与反传)
			with torch.no_grad():
				pred_box_recon = center_to_corners(pred_c_valid, half_size=half_size,
				                                  device=logits.device, dtype=logits.dtype)
				gt_box_recon = center_to_corners(gt_c_valid, half_size=half_size,
				                                device=logits.device, dtype=logits.dtype)
				iou_per_sample = box_iou_3d_aligned(pred_box_recon, gt_box_recon)

			if self.auto_balance:
				total_loss = (
					total_loss
					+ torch.exp(-self.log_var_box) * box_depth_loss + self.log_var_box
				)
			else:
				total_loss = total_loss + self.box_weight * box_depth_loss

			out["box_depth_loss"] = box_depth_loss
			out["box_iou_mean"] = iou_per_sample.mean()

		out["total_loss"] = total_loss
		return out


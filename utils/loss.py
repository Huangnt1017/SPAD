from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorLike = Union[torch.Tensor, Sequence[float]]


def to_box_tensor(boxes: TensorLike, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
	"""
	Convert boxes into a tensor with trailing shape (..., 6).

	Supported formats:
	- (..., 6): [xmin, xmax, ymin, ymax, zmin, zmax]
	- (..., 3, 2): [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
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
	"""Ensure each axis in 3D boxes is ordered as min <= max."""
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


def box_volume_3d(boxes: TensorLike, eps: float = 1e-8) -> torch.Tensor:
	"""Compute volume for axis-aligned 3D boxes in [xmin,xmax,ymin,ymax,zmin,zmax] format."""
	box_tensor = canonicalize_boxes_3d(boxes)
	sizes = (box_tensor[..., 1::2] - box_tensor[..., 0::2]).clamp(min=0.0)
	return sizes.prod(dim=-1) + eps


def box_iou_3d_aligned(pred_boxes: TensorLike, gt_boxes: TensorLike, eps: float = 1e-8) -> torch.Tensor:
	"""
	Pairwise aligned IoU: pred_boxes[i] vs gt_boxes[i].

	Args:
		pred_boxes: (..., 6) or (..., 3, 2)
		gt_boxes:   (..., 6) or (..., 3, 2)
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


def split_cls_and_box_predictions(model_outputs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
	"""
	Parse model outputs into classification logits and optional box predictions.

	Supported output forms:
	- Tensor logits
	- Tuple/List: (logits, box_preds, ...)
	- Dict with common keys:
	  logits keys: logits / cls_logits / class_logits / pred_logits
	  box keys: boxes / pred_boxes / bbox / bbox_pred / box_pred
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
	"""

	def _pick_axis(keys: Tuple[str, str]) -> Any:
		for key in keys:
			if key in meta:
				return meta[key]
		raise KeyError(f"Missing axis keys, expected one of: {keys}")

	def _to_range_tensor(values: Any) -> torch.Tensor:
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
	Multi-task loss for point cloud classification + single 3D box regression.

	Total loss:
	  L = cls_weight * L_cls + box_l1_weight * L_smooth_l1 + box_iou_weight * (1 - IoU)
	"""

	def __init__(
		self,
		cls_weight: float = 1.0,
		box_l1_weight: float = 1.0,
		box_iou_weight: float = 1.0,
		label_smoothing: float = 0.0,
	):
		super().__init__()
		self.cls_weight = float(cls_weight)
		self.box_l1_weight = float(box_l1_weight)
		self.box_iou_weight = float(box_iou_weight)
		self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

	def forward(
		self,
		model_outputs: Any,
		cls_targets: torch.Tensor,
		box_targets: Optional[TensorLike] = None,
		box_valid_mask: Optional[torch.Tensor] = None,
	) -> Dict[str, torch.Tensor]:
		logits, box_preds = split_cls_and_box_predictions(model_outputs)
		cls_targets = cls_targets.long().to(logits.device)

		cls_loss = self.cls_criterion(logits, cls_targets)
		total_loss = self.cls_weight * cls_loss

		out: Dict[str, torch.Tensor] = {
			"total_loss": total_loss,
			"cls_loss": cls_loss,
			"box_l1_loss": torch.zeros((), device=logits.device),
			"box_iou_loss": torch.zeros((), device=logits.device),
			"box_iou_mean": torch.zeros((), device=logits.device),
		}

		if box_targets is None or box_preds is None:
			out["total_loss"] = total_loss
			return out

		pred_boxes = canonicalize_boxes_3d(box_preds, device=logits.device, dtype=logits.dtype)
		gt_boxes = canonicalize_boxes_3d(box_targets, device=logits.device, dtype=logits.dtype)

		if pred_boxes.shape != gt_boxes.shape:
			raise ValueError(
				f"Box prediction/target shape mismatch: pred={tuple(pred_boxes.shape)}, gt={tuple(gt_boxes.shape)}"
			)

		if box_valid_mask is None:
			valid_mask = torch.ones(pred_boxes.shape[0], dtype=torch.bool, device=logits.device)
		else:
			valid_mask = box_valid_mask.to(logits.device).bool()

		if valid_mask.numel() != pred_boxes.shape[0]:
			raise ValueError(
				f"box_valid_mask length mismatch: mask={valid_mask.numel()}, boxes={pred_boxes.shape[0]}"
			)

		if valid_mask.any():
			pred_valid = pred_boxes[valid_mask]
			gt_valid = gt_boxes[valid_mask]

			box_l1_loss = F.smooth_l1_loss(pred_valid, gt_valid, reduction="mean")
			iou = box_iou_3d_aligned(pred_valid, gt_valid)
			box_iou_loss = 1.0 - iou.mean()

			total_loss = total_loss + self.box_l1_weight * box_l1_loss + self.box_iou_weight * box_iou_loss

			out["box_l1_loss"] = box_l1_loss
			out["box_iou_loss"] = box_iou_loss
			out["box_iou_mean"] = iou.mean()

		out["total_loss"] = total_loss
		return out


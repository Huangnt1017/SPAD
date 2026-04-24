import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
	balanced_accuracy_score,
	classification_report,
	confusion_matrix,
	precision_recall_fscore_support,
)
from tqdm import tqdm

from utils.data import create_spad_classification_dataloaders
from utils.loss import (
	box_iou_3d_aligned,
	build_spad_boxes_from_meta,
	canonicalize_boxes_3d,
	split_cls_and_box_predictions,
)
from scipts.train import build_model, compute_topk_hits, prepare_model_inputs, resolve_path, set_seed

# 测试脚本只负责从 checkpoint 读取模型，再把分类与 box 相关指标统一算出来并落盘。

def setup_logger(log_dir: Path, model_name: str) -> Tuple[logging.Logger, Path, str]:
	# 测试阶段也保留文件日志，便于记录最终指标、混淆矩阵路径和 AP 结果。
	log_dir.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	log_file = log_dir / f"test_{model_name}_{timestamp}.log"

	logger_name = f"spad_test_{model_name}_{timestamp}"
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

	return logger, log_file, timestamp


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict:
	# 分类指标都来自真实标签 y_true 与预测标签 y_pred 的对比，混淆矩阵是后续画图和分项统计的基础。
	labels = np.arange(num_classes)
	cm = confusion_matrix(y_true, y_pred, labels=labels)

	precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
		y_true,
		y_pred,
		labels=labels,
		average="macro",
		zero_division=0,
	)

	per_class_acc: Dict[int, float] = {}
	for cls_idx in labels:
		denom = cm[cls_idx].sum()
		per_class_acc[int(cls_idx)] = float(cm[cls_idx, cls_idx] / denom) if denom > 0 else 0.0

	metrics = {
		"accuracy": float((y_true == y_pred).mean()),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"precision_macro": float(precision_macro),
		"recall_macro": float(recall_macro),
		"f1_macro": float(f1_macro),
		"per_class_accuracy": per_class_acc,
		"confusion_matrix": cm,
	}
	return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path, normalize: bool = True) -> None:
	# 混淆矩阵是把逐类预测结果汇总成二维表，归一化后更适合观察类别间的混淆模式。
	matrix = cm.astype(np.float64)
	if normalize:
		row_sum = matrix.sum(axis=1, keepdims=True)
		matrix = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum != 0)

	fig, ax = plt.subplots(figsize=(10, 8))
	im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	ax.set(
		xticks=np.arange(len(class_names)),
		yticks=np.arange(len(class_names)),
		xticklabels=class_names,
		yticklabels=class_names,
		ylabel="True label",
		xlabel="Predicted label",
		title="Normalized Confusion Matrix" if normalize else "Confusion Matrix",
	)

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	threshold = matrix.max() / 2.0 if matrix.size > 0 else 0.0
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			value = matrix[i, j]
			text = f"{value:.2f}" if normalize else str(int(value))
			ax.text(j, i, text, ha="center", va="center", color="white" if value > threshold else "black")

	fig.tight_layout()
	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(save_path, dpi=200)
	plt.close(fig)


def slice_batch_meta(meta: Mapping[str, Any], valid_mask: torch.Tensor) -> Dict[str, Any]:
	"""Slice collated metadata using a batch-level validity mask."""
	# 测试时也要按有效标签过滤元信息，否则 box 目标框会和过滤后的点云 batch 对不上。
	mask_cpu = valid_mask.detach().cpu()
	mask_list = mask_cpu.tolist()
	sliced: Dict[str, Any] = {}

	for key, value in meta.items():
		if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == len(mask_list):
			sliced[key] = value[mask_cpu]
		elif isinstance(value, list) and len(value) == len(mask_list):
			sliced[key] = [item for item, keep in zip(value, mask_list) if keep]
		elif isinstance(value, tuple) and len(value) == len(mask_list):
			sliced[key] = tuple(item for item, keep in zip(value, mask_list) if keep)
		else:
			sliced[key] = value

	return sliced


def compute_ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
	"""Compute area under the precision-recall curve using interpolation envelope."""
	# 这里采用插值包络线积分，和常见检测评估里的 AP 计算方式一致。
	mrec = np.concatenate(([0.0], recall, [1.0]))
	mpre = np.concatenate(([0.0], precision, [0.0]))

	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = max(mpre[i - 1], mpre[i])

	change_idx = np.where(mrec[1:] != mrec[:-1])[0]
	return float(np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mpre[change_idx + 1]))


def compute_box_ap_metrics(
	pred_classes: np.ndarray,
	pred_scores: np.ndarray,
	gt_classes: np.ndarray,
	ious: np.ndarray,
	num_classes: int,
	iou_thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
	"""
	Compute AP50 and AP@50:5:95 for single-box-per-sample detection.

	Each sample contributes one prediction (class + score + box IoU against its GT box).
	"""
	# 这里把“单样本单框”的问题转换成按类别统计的 AP；IoU 阈值从 0.50 到 0.95 逐步递增。
	if pred_classes.size == 0 or gt_classes.size == 0 or ious.size == 0:
		return {
			"AP50": float("nan"),
			"AP@50:5:95": float("nan"),
			"mean_iou_matched_cls": float("nan"),
			"thresholds": [],
			"ap_per_threshold": {},
		}

	if iou_thresholds is None:
		iou_thresholds = np.arange(0.50, 0.96, 0.05, dtype=np.float64)

	def _map_at_threshold(threshold: float) -> float:
		# 对每个类别分别算 AP：先按该类别的预测分数排序，再累计 TP / FP，最后积分得到 AP。
		per_class_ap: List[float] = []
		for cls_idx in range(num_classes):
			gt_count = int((gt_classes == cls_idx).sum())
			if gt_count <= 0:
				continue

			pred_mask = pred_classes == cls_idx
			if not np.any(pred_mask):
				per_class_ap.append(0.0)
				continue

			cls_scores = pred_scores[pred_mask]
			cls_pred_classes = pred_classes[pred_mask]
			cls_gt_classes = gt_classes[pred_mask]
			cls_ious = ious[pred_mask]

			tp = ((cls_pred_classes == cls_gt_classes) & (cls_ious >= threshold)).astype(np.float64)
			fp = 1.0 - tp

			order = np.argsort(-cls_scores)
			tp = tp[order]
			fp = fp[order]

			tp_cum = np.cumsum(tp)
			fp_cum = np.cumsum(fp)
			recall = tp_cum / max(gt_count, 1)
			precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
			per_class_ap.append(compute_ap_from_pr(recall, precision))

		if not per_class_ap:
			return float("nan")
		return float(np.mean(per_class_ap))

	ap_values: List[float] = []
	ap_per_threshold: Dict[str, float] = {}
	for thr in iou_thresholds:
		ap_thr = _map_at_threshold(float(thr))
		ap_values.append(ap_thr)
		ap_per_threshold[f"{thr:.2f}"] = ap_thr

	matched_cls_mask = pred_classes == gt_classes
	# mean_iou_matched_cls 只看“类别预测正确”的样本，用来分离分类错误和框回归误差。
	mean_iou_matched = float(np.mean(ious[matched_cls_mask])) if np.any(matched_cls_mask) else 0.0

	return {
		"AP50": float(ap_per_threshold.get("0.50", float("nan"))),
		"AP@50:5:95": float(np.mean(ap_values)) if ap_values else float("nan"),
		"mean_iou_matched_cls": mean_iou_matched,
		"thresholds": [float(t) for t in iou_thresholds],
		"ap_per_threshold": ap_per_threshold,
	}


def evaluate(
	model: nn.Module,
	loader,
	criterion: nn.Module,
	device: torch.device,
) -> Dict:
	# 测试阶段把分类分数、类别标签、box IoU 都逐样本收集起来，后面统一算总体指标。
	model.eval()
	total_loss = 0.0
	total_samples = 0
	top1_hits = 0
	top3_hits = 0

	all_preds: List[int] = []
	all_labels: List[int] = []
	all_pred_scores: List[float] = []
	all_box_ious: List[float] = []
	all_box_scores: List[float] = []
	all_box_pred_classes: List[int] = []
	all_box_gt_classes: List[int] = []
	box_eval_samples = 0
	box_eval_skipped = False

	with torch.no_grad():
		pbar = tqdm(loader, desc="Testing", leave=False)
		for points, labels, batch_meta in pbar:
			# 测试 batch 的结构与训练一致：点云、类别标签和元信息一起进入评估流程。
			points = points.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			valid_mask = labels >= 0
			if not valid_mask.any():
				continue

			points = points[valid_mask]
			labels = labels[valid_mask]

			model_outputs = model(prepare_model_inputs(points))
			logits, box_preds = split_cls_and_box_predictions(model_outputs)
			# 这里的 loss 只用于记录分类损失，不反向传播，也不更新参数。
			loss = criterion(logits, labels)

			batch_size = labels.size(0)
			total_samples += batch_size
			total_loss += float(loss.item()) * batch_size

			hits = compute_topk_hits(logits, labels, topk=(1, 3))
			top1_hits += hits[1]
			top3_hits += hits[3]

			probs = torch.softmax(logits, dim=1)
			pred_scores, preds = probs.max(dim=1)
			# 预测类别和置信度会同时用于分类报告和 box AP 的排序。
			all_preds.extend(preds.detach().cpu().tolist())
			all_labels.extend(labels.detach().cpu().tolist())
			all_pred_scores.extend(pred_scores.detach().cpu().tolist())

			if box_preds is not None and isinstance(batch_meta, Mapping):
				try:
					# box AP 的前提是模型和元信息都能构造出一一对应的预测框与真实框。
					meta_valid = slice_batch_meta(batch_meta, valid_mask)
					gt_boxes = build_spad_boxes_from_meta(meta_valid, device=device)
					pred_boxes = canonicalize_boxes_3d(box_preds, device=device, dtype=gt_boxes.dtype)

					if pred_boxes.shape != gt_boxes.shape:
						raise ValueError(
							f"Box shape mismatch: pred={tuple(pred_boxes.shape)} vs gt={tuple(gt_boxes.shape)}"
						)

					ious = box_iou_3d_aligned(pred_boxes, gt_boxes)
					# iou / score / class 会一起保存，后续按阈值和类别计算 AP50、AP@50:5:95。
					all_box_ious.extend(ious.detach().cpu().tolist())
					all_box_scores.extend(pred_scores.detach().cpu().tolist())
					all_box_pred_classes.extend(preds.detach().cpu().tolist())
					all_box_gt_classes.extend(labels.detach().cpu().tolist())
					box_eval_samples += int(labels.size(0))
				except Exception:
					box_eval_skipped = True
			elif box_preds is not None:
				box_eval_skipped = True

			avg_loss = total_loss / max(total_samples, 1)
			top1 = top1_hits / max(total_samples, 1)
			top3 = top3_hits / max(total_samples, 1)
			pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{top1:.4f}", top3=f"{top3:.4f}")

	return {
		# 这里返回的是整体验证/测试过程中逐样本拼接后的原始数据，后面的 compute_metrics 和 compute_box_ap_metrics 再做汇总。
		"loss": total_loss / max(total_samples, 1),
		"top1": top1_hits / max(total_samples, 1),
		"top3": top3_hits / max(total_samples, 1),
		"y_true": np.array(all_labels, dtype=np.int64),
		"y_pred": np.array(all_preds, dtype=np.int64),
		"pred_scores": np.array(all_pred_scores, dtype=np.float64),
		"box_ious": np.array(all_box_ious, dtype=np.float64),
		"box_scores": np.array(all_box_scores, dtype=np.float64),
		"box_pred_classes": np.array(all_box_pred_classes, dtype=np.int64),
		"box_gt_classes": np.array(all_box_gt_classes, dtype=np.int64),
		"box_eval_samples": int(box_eval_samples),
		"box_eval_skipped": bool(box_eval_skipped),
	}


def run_test(args: argparse.Namespace) -> Dict[str, str]:
	# 测试入口：加载 checkpoint、跑完整测试集、输出分类指标和 box AP 指标，并保存图表与 JSON。
	project_root = Path(__file__).resolve().parents[1]
	data_root = resolve_path(args.data_root, project_root)
	checkpoint_path = resolve_path(args.checkpoint, project_root)
	output_dir = resolve_path(args.output_dir, project_root)
	log_dir = resolve_path(args.log_dir, project_root)

	set_seed(args.seed)

	if args.device == "auto":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		if args.device == "cuda" and not torch.cuda.is_available():
			device = torch.device("cpu")
		else:
			device = torch.device(args.device)

	logger, log_file, timestamp = setup_logger(log_dir, args.model)

	_, _, test_loader, _, meta = create_spad_classification_dataloaders(
		data_root=str(data_root),
		batch_size=args.batch_size,
		num_points=(None if args.num_points <= 0 else args.num_points),
		train_ratio=args.train_ratio,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		num_workers=args.num_workers,
		seed=args.seed,
		augment_train=False,
		augment_eval=args.augment_eval,
		label_mode=args.label_mode,
	)

	model = build_model(args.model, num_classes=meta["num_classes"], project_root=project_root).to(device)
	# 测试时只需要模型参数，不需要优化器状态。
	checkpoint = torch.load(checkpoint_path, map_location=device)
	state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
	model.load_state_dict(state_dict)

	if "class_to_idx" in checkpoint:
		ckpt_class_to_idx = checkpoint["class_to_idx"]
		if ckpt_class_to_idx != meta["class_to_idx"]:
			logger.warning("class_to_idx in checkpoint differs from current dataset split mapping.")

	criterion = nn.CrossEntropyLoss()
	eval_out = evaluate(model, test_loader, criterion, device)

	y_true = eval_out["y_true"]
	y_pred = eval_out["y_pred"]
	# 分类部分的总体指标来自 y_true/y_pred；box 部分需要额外用 IoU 和类别匹配情况来算 AP。
	metrics = compute_metrics(y_true, y_pred, num_classes=meta["num_classes"])
	box_metrics: Optional[Dict[str, Any]] = None
	if eval_out["box_eval_samples"] > 0:
		# 只有模型真的输出了 box 且元信息能构造目标框时，才会计算 box AP。
		box_metrics = compute_box_ap_metrics(
			pred_classes=eval_out["box_pred_classes"],
			pred_scores=eval_out["box_scores"],
			gt_classes=eval_out["box_gt_classes"],
			ious=eval_out["box_ious"],
			num_classes=meta["num_classes"],
		)
	elif eval_out["box_eval_skipped"]:
		logger.warning(
			"Box AP metrics skipped. Ensure model outputs boxes with shape [B,6] or [B,3,2] and metadata provides target ranges."
		)

	idx_to_class = meta["idx_to_class"]
	class_names = [idx_to_class[i] for i in range(meta["num_classes"])]

	cm_path = output_dir / f"cm_{args.model}_{timestamp}.png"
	plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path, normalize=args.normalize_cm)

	report_text = classification_report(
		y_true,
		y_pred,
		target_names=class_names,
		labels=np.arange(meta["num_classes"]),
		zero_division=0,
	)

	logger.info("=== Test Summary ===")
	# 下面这些日志把分类指标、box 指标和输出文件路径一次性写全，方便定位结果。
	logger.info("checkpoint=%s", checkpoint_path)
	logger.info("data_root=%s", data_root)
	logger.info("num_test_samples=%d", meta["num_test_samples"])
	logger.info("loss=%.4f", eval_out["loss"])
	logger.info("top1=%.4f", eval_out["top1"])
	logger.info("top3=%.4f", eval_out["top3"])
	logger.info("balanced_accuracy=%.4f", metrics["balanced_accuracy"])
	logger.info("precision_macro=%.4f", metrics["precision_macro"])
	logger.info("recall_macro=%.4f", metrics["recall_macro"])
	logger.info("f1_macro=%.4f", metrics["f1_macro"])
	if box_metrics is not None:
		# AP50 和 AP@50:5:95 是 box 质量的核心指标，来自不同 IoU 阈值下的平均精度。
		logger.info("mean_iou_matched_cls=%.4f", box_metrics["mean_iou_matched_cls"])
		logger.info("AP50=%.4f", box_metrics["AP50"])
		logger.info("AP@50:5:95=%.4f", box_metrics["AP@50:5:95"])
	logger.info("confusion_matrix_path=%s", cm_path)
	logger.info("\n%s", report_text)

	output_dir.mkdir(parents=True, exist_ok=True)
	metrics_json_path = output_dir / f"metrics_{args.model}_{timestamp}.json"
	metrics_payload = {
		# JSON 里同时保存分类、box AP 和路径信息，便于后续画图或对比实验直接读取。
		"loss": eval_out["loss"],
		"top1": eval_out["top1"],
		"top3": eval_out["top3"],
		"accuracy": metrics["accuracy"],
		"balanced_accuracy": metrics["balanced_accuracy"],
		"precision_macro": metrics["precision_macro"],
		"recall_macro": metrics["recall_macro"],
		"f1_macro": metrics["f1_macro"],
		"per_class_accuracy": {class_names[k]: v for k, v in metrics["per_class_accuracy"].items()},
		"mean_iou_matched_cls": (None if box_metrics is None else box_metrics["mean_iou_matched_cls"]),
		"AP50": (None if box_metrics is None else box_metrics["AP50"]),
		"AP@50:5:95": (None if box_metrics is None else box_metrics["AP@50:5:95"]),
		"box_ap_thresholds": (None if box_metrics is None else box_metrics["thresholds"]),
		"box_ap_per_threshold": (None if box_metrics is None else box_metrics["ap_per_threshold"]),
		"num_box_eval_samples": eval_out["box_eval_samples"],
		"class_names": class_names,
		"checkpoint": str(checkpoint_path),
		"confusion_matrix_path": str(cm_path),
	}
	with open(metrics_json_path, "w", encoding="utf-8") as f:
		json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

	return {
		"log_file": str(log_file),
		"metrics_json": str(metrics_json_path),
		"confusion_matrix": str(cm_path),
	}


def build_parser() -> argparse.ArgumentParser:
	# 测试参数主要控制 checkpoint、数据源、batch 大小以及是否对 eval 集做增强。
	parser = argparse.ArgumentParser(description="SPAD 3D point cloud classification evaluation")
	parser.add_argument("--data-root", type=str, default=r"D:\PYproject\SPADdata", help="SPAD data root directory")
	parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
	parser.add_argument("--model", type=str, default="dgcnn", choices=["dgcnn", "pointnet2", "pointtransformer"], help="Backbone model")
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--num-points", type=int, default=1024, help="Fixed number of points per sample (deterministic sample/pad)")
	parser.add_argument("--train-ratio", type=float, default=0.7)
	parser.add_argument("--val-ratio", type=float, default=0.15)
	parser.add_argument("--test-ratio", type=float, default=0.15)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--label-mode", type=str, default="raw", choices=["generated", "raw"], help="Label source mode")
	parser.add_argument("--augment-eval", dest="augment_eval", action="store_true", help="Apply augmentation in eval dataset")
	parser.add_argument("--no-augment-eval", dest="augment_eval", action="store_false", help="Disable eval dataset augmentation")
	parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
	parser.add_argument("--log-dir", type=str, default="logs")
	parser.add_argument("--output-dir", type=str, default="logs")
	parser.add_argument("--normalize-cm", action="store_true", help="Use normalized confusion matrix")
	parser.set_defaults(augment_eval=True)
	return parser


def main(argv=None) -> None:
	parser = build_parser()
	args = parser.parse_args(argv)
	run_test(args)


if __name__ == "__main__":
	main()

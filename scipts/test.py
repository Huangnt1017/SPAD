import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
from scipts.train import build_model, compute_topk_hits, prepare_model_inputs, resolve_path, set_seed


def setup_logger(log_dir: Path, model_name: str) -> Tuple[logging.Logger, Path, str]:
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


def evaluate(
	model: nn.Module,
	loader,
	criterion: nn.Module,
	device: torch.device,
) -> Dict:
	model.eval()
	total_loss = 0.0
	total_samples = 0
	top1_hits = 0
	top3_hits = 0

	all_preds: List[int] = []
	all_labels: List[int] = []

	with torch.no_grad():
		pbar = tqdm(loader, desc="Testing", leave=False)
		for points, labels, _ in pbar:
			points = points.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			valid_mask = labels >= 0
			if not valid_mask.any():
				continue

			points = points[valid_mask]
			labels = labels[valid_mask]

			logits = model(prepare_model_inputs(points))
			loss = criterion(logits, labels)

			batch_size = labels.size(0)
			total_samples += batch_size
			total_loss += float(loss.item()) * batch_size

			hits = compute_topk_hits(logits, labels, topk=(1, 3))
			top1_hits += hits[1]
			top3_hits += hits[3]

			preds = logits.argmax(dim=1)
			all_preds.extend(preds.detach().cpu().tolist())
			all_labels.extend(labels.detach().cpu().tolist())

			avg_loss = total_loss / max(total_samples, 1)
			top1 = top1_hits / max(total_samples, 1)
			top3 = top3_hits / max(total_samples, 1)
			pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{top1:.4f}", top3=f"{top3:.4f}")

	return {
		"loss": total_loss / max(total_samples, 1),
		"top1": top1_hits / max(total_samples, 1),
		"top3": top3_hits / max(total_samples, 1),
		"y_true": np.array(all_labels, dtype=np.int64),
		"y_pred": np.array(all_preds, dtype=np.int64),
	}


def run_test(args: argparse.Namespace) -> Dict[str, str]:
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
	metrics = compute_metrics(y_true, y_pred, num_classes=meta["num_classes"])

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
	logger.info("confusion_matrix_path=%s", cm_path)
	logger.info("\n%s", report_text)

	output_dir.mkdir(parents=True, exist_ok=True)
	metrics_json_path = output_dir / f"metrics_{args.model}_{timestamp}.json"
	metrics_payload = {
		"loss": eval_out["loss"],
		"top1": eval_out["top1"],
		"top3": eval_out["top3"],
		"accuracy": metrics["accuracy"],
		"balanced_accuracy": metrics["balanced_accuracy"],
		"precision_macro": metrics["precision_macro"],
		"recall_macro": metrics["recall_macro"],
		"f1_macro": metrics["f1_macro"],
		"per_class_accuracy": {class_names[k]: v for k, v in metrics["per_class_accuracy"].items()},
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
	parser = argparse.ArgumentParser(description="SPAD 3D point cloud classification evaluation")
	parser.add_argument("--data-root", type=str, default=r"D:\PYproject\SPADdata", help="SPAD data root directory")
	parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
	parser.add_argument("--model", type=str, default="dgcnn", choices=["dgcnn", "pointnet2", "pointtransformer"], help="Backbone model")
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--num-points", type=int, default=0, help="Set >0 to enforce fixed N; 0 disables point-count check")
	parser.add_argument("--train-ratio", type=float, default=0.7)
	parser.add_argument("--val-ratio", type=float, default=0.15)
	parser.add_argument("--test-ratio", type=float, default=0.15)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--label-mode", type=str, default="generated", choices=["generated", "raw"], help="Label source mode")
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

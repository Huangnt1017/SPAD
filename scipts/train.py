import argparse
import importlib.util
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils.data import create_spad_classification_dataloaders
from utils.data_augment import augment_pytorch_batch


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def resolve_path(path_str: str, base_dir: Path) -> Path:
	path = Path(path_str)
	if path.is_absolute():
		return path
	return (base_dir / path).resolve()


def setup_logger(log_dir: Path, model_name: str) -> Tuple[logging.Logger, Path, str]:
	log_dir.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	log_file = log_dir / f"train_{model_name}_{timestamp}.log"

	logger_name = f"spad_train_{model_name}_{timestamp}"
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


def load_module_from_file(file_path: Path, module_name: str):
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Failed to load module from: {file_path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def build_model(model_name: str, num_classes: int, project_root: Path) -> nn.Module:
	baseline_dir = project_root / "baseline"
	name = model_name.lower()

	if name == "dgcnn":
		module = load_module_from_file(baseline_dir / "DGCNN.py", "baseline_dgcnn")
		return module.DGCNNCls(num_classes=num_classes)

	if name == "pointtransformer":
		module = load_module_from_file(baseline_dir / "pointTransformer.py", "baseline_point_transformer")
		return module.PointTransformerClassification(num_classes=num_classes)

	if name == "pointnet2":
		module = load_module_from_file(baseline_dir / "pointnet++.py", "baseline_pointnet2")
		return module.PointNet2ClassificationSSG(num_class=num_classes, normal_channel=False)

	raise ValueError(f"Unsupported model name: {model_name}")


def prepare_model_inputs(points_xyzi: torch.Tensor) -> torch.Tensor:
	"""Convert (B, N, 4) -> (B, 3, N), using xyz only for baseline classifiers."""
	xyz = points_xyzi[:, :, :3]
	return xyz.transpose(1, 2).contiguous()


def compute_topk_hits(logits: torch.Tensor, labels: torch.Tensor, topk: Iterable[int] = (1, 3)) -> Dict[int, int]:
	num_classes = logits.size(1)
	max_k = min(max(topk), num_classes)
	_, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
	pred = pred.t()
	correct = pred.eq(labels.view(1, -1).expand_as(pred))

	out: Dict[int, int] = {}
	for k in topk:
		kk = min(k, num_classes)
		out[k] = int(correct[:kk].reshape(-1).float().sum().item())
	return out


def run_epoch(
	loader,
	model: nn.Module,
	criterion: nn.Module,
	device: torch.device,
	epoch: int,
	phase: str,
	use_augment: bool = False,
	optimizer: Optional[optim.Optimizer] = None,
) -> Dict[str, float]:
	is_train = optimizer is not None
	model.train(is_train)

	total_loss = 0.0
	total_samples = 0
	correct_top1 = 0
	correct_top3 = 0

	pbar = tqdm(loader, desc=f"{phase} Epoch {epoch}", leave=False)
	context = torch.enable_grad() if is_train else torch.no_grad()

	with context:
		for points, labels, _ in pbar:
			points = points.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			valid_mask = labels >= 0
			if not valid_mask.any():
				continue

			points = points[valid_mask]
			labels = labels[valid_mask]

			if is_train and use_augment:
				points, _ = augment_pytorch_batch(points, label_class=None)

			inputs = prepare_model_inputs(points)
			logits = model(inputs)
			loss = criterion(logits, labels)

			if is_train:
				optimizer.zero_grad(set_to_none=True)
				loss.backward()
				optimizer.step()

			batch_size = labels.size(0)
			total_loss += float(loss.item()) * batch_size
			total_samples += batch_size

			topk_hits = compute_topk_hits(logits, labels, topk=(1, 3))
			correct_top1 += topk_hits[1]
			correct_top3 += topk_hits[3]

			avg_loss = total_loss / max(total_samples, 1)
			top1 = correct_top1 / max(total_samples, 1)
			top3 = correct_top3 / max(total_samples, 1)
			pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{top1:.4f}", top3=f"{top3:.4f}")

	metrics = {
		"loss": total_loss / max(total_samples, 1),
		"top1": correct_top1 / max(total_samples, 1),
		"top3": correct_top3 / max(total_samples, 1),
		"samples": float(total_samples),
	}
	return metrics


def save_checkpoint(
	path: Path,
	model: nn.Module,
	optimizer: optim.Optimizer,
	scheduler: CosineAnnealingLR,
	epoch: int,
	best_val_top1: float,
	class_to_idx: Dict[str, int],
	args: argparse.Namespace,
) -> None:
	payload = {
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict(),
		"best_val_top1": best_val_top1,
		"class_to_idx": class_to_idx,
		"args": vars(args),
	}
	torch.save(payload, path)


def run_training(args: argparse.Namespace) -> Dict[str, str]:
	project_root = Path(__file__).resolve().parents[1]
	data_root = resolve_path(args.data_root, project_root)
	save_dir = resolve_path(args.save_dir, project_root)
	log_dir = resolve_path(args.log_dir, project_root)

	if not data_root.exists():
		raise FileNotFoundError(f"Data root not found: {data_root}")

	set_seed(args.seed)

	if args.device == "auto":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		if args.device == "cuda" and not torch.cuda.is_available():
			device = torch.device("cpu")
		else:
			device = torch.device(args.device)

	logger, log_file, run_timestamp = setup_logger(log_dir, args.model)

	train_loader, val_loader, test_loader, unlabeled_loader, meta = create_spad_classification_dataloaders(
		data_root=str(data_root),
		batch_size=args.batch_size,
		num_points=args.num_points,
		train_ratio=args.train_ratio,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		num_workers=args.num_workers,
		seed=args.seed,
	)

	model = build_model(args.model, num_classes=meta["num_classes"], project_root=project_root).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

	save_dir.mkdir(parents=True, exist_ok=True)

	logger.info("=== Training Configuration ===")
	logger.info("run_timestamp=%s", run_timestamp)
	logger.info("data_root=%s", data_root)
	logger.info("device=%s", device)
	logger.info("model=%s", args.model)
	logger.info("num_classes=%d", meta["num_classes"])
	logger.info("num_labeled_samples=%d", meta["num_labeled_samples"])
	logger.info("num_unlabeled_samples=%d", meta["num_unlabeled_samples"])
	logger.info("split train/val/test = %d / %d / %d", meta["num_train_samples"], meta["num_val_samples"], meta["num_test_samples"])
	logger.info("unlabeled_loader_exists=%s", "yes" if unlabeled_loader is not None else "no")
	logger.info("args=%s", json.dumps(vars(args), ensure_ascii=False))

	start_epoch = 1
	best_val_top1 = 0.0

	if args.resume:
		resume_path = resolve_path(args.resume, project_root)
		checkpoint = torch.load(resume_path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])
		if "optimizer_state_dict" in checkpoint:
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		if "scheduler_state_dict" in checkpoint:
			scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
		start_epoch = int(checkpoint.get("epoch", 0)) + 1
		best_val_top1 = float(checkpoint.get("best_val_top1", 0.0))
		logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

	best_ckpt = save_dir / f"{args.model}_{run_timestamp}_best.pth"
	last_ckpt = save_dir / f"{args.model}_{run_timestamp}_last.pth"

	for epoch in range(start_epoch, args.epochs + 1):
		train_metrics = run_epoch(
			loader=train_loader,
			model=model,
			criterion=criterion,
			device=device,
			epoch=epoch,
			phase="Train",
			use_augment=args.use_augment,
			optimizer=optimizer,
		)

		val_metrics = run_epoch(
			loader=val_loader,
			model=model,
			criterion=criterion,
			device=device,
			epoch=epoch,
			phase="Val",
			use_augment=False,
			optimizer=None,
		)

		scheduler.step()

		logger.info(
			"Epoch [%d/%d] | train_loss=%.4f train_top1=%.4f train_top3=%.4f | "
			"val_loss=%.4f val_top1=%.4f val_top3=%.4f",
			epoch,
			args.epochs,
			train_metrics["loss"],
			train_metrics["top1"],
			train_metrics["top3"],
			val_metrics["loss"],
			val_metrics["top1"],
			val_metrics["top3"],
		)

		if val_metrics["top1"] >= best_val_top1:
			best_val_top1 = val_metrics["top1"]
			save_checkpoint(
				path=best_ckpt,
				model=model,
				optimizer=optimizer,
				scheduler=scheduler,
				epoch=epoch,
				best_val_top1=best_val_top1,
				class_to_idx=meta["class_to_idx"],
				args=args,
			)
			logger.info("Saved new best checkpoint to %s", best_ckpt)

	save_checkpoint(
		path=last_ckpt,
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		epoch=args.epochs,
		best_val_top1=best_val_top1,
		class_to_idx=meta["class_to_idx"],
		args=args,
	)
	logger.info("Saved last checkpoint to %s", last_ckpt)
	logger.info("Training finished. Best val top1=%.4f", best_val_top1)

	return {
		"log_file": str(log_file),
		"best_checkpoint": str(best_ckpt),
		"last_checkpoint": str(last_ckpt),
	}


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="SPAD 3D point cloud classification training")
	parser.add_argument("--data-root", type=str, default=r"D:\PYproject\SPADdata", help="SPAD data root directory")
	parser.add_argument("--model", type=str, default="dgcnn", choices=["dgcnn", "pointnet2", "pointtransformer"], help="Backbone model")
	parser.add_argument("--epochs", type=int, default=80)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--num-points", type=int, default=1024)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--min-lr", type=float, default=1e-5)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--train-ratio", type=float, default=0.7)
	parser.add_argument("--val-ratio", type=float, default=0.15)
	parser.add_argument("--test-ratio", type=float, default=0.15)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
	parser.add_argument("--log-dir", type=str, default="logs")
	parser.add_argument("--save-dir", type=str, default="checkpoints")
	parser.add_argument("--resume", type=str, default="", help="checkpoint path to resume")

	parser.add_argument("--use-augment", dest="use_augment", action="store_true", help="Enable batch augmentation")
	parser.add_argument("--no-augment", dest="use_augment", action="store_false", help="Disable batch augmentation")
	parser.set_defaults(use_augment=True)
	return parser


def main(argv=None) -> None:
	parser = build_parser()
	args = parser.parse_args(argv)
	run_training(args)


if __name__ == "__main__":
	main()



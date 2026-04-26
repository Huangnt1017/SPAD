import argparse
import importlib.util
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

"""
SPAD 训练入口模块。

模块目的：
- 组织数据加载、模型构建、损失计算、训练循环、验证与 checkpoint 保存。

主要导出内容：
- run_training: 执行完整训练流程。
- build_parser/main: 命令行入口。
"""

# 训练脚本只负责把数据、模型、损失和日志串起来，核心计算都保留在各自模块中。
from utils.data import create_spad_classification_dataloaders
from utils.loss import PointCloudMultiTaskLoss, build_spad_boxes_from_meta, split_cls_and_box_predictions


def set_seed(seed: int) -> None:
	"""设置随机种子。

	Args:
		seed: 随机种子值。

	Returns:
		None。
	"""
	# 固定随机源，保证数据划分、增强采样和模型初始化都尽量可复现。
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def resolve_path(path_str: str, base_dir: Path) -> Path:
	"""解析路径为绝对路径。

	Args:
		path_str: 命令行传入路径。
		base_dir: 相对路径的参照目录。

	Returns:
		解析后的绝对路径。
	"""
	# 允许命令行传相对路径，同时统一转成项目根目录下的绝对路径。
	path = Path(path_str)
	if path.is_absolute():
		return path
	return (base_dir / path).resolve()


def setup_logger(log_dir: Path, model_name: str) -> Tuple[logging.Logger, Path, str]:
	"""创建训练日志器。

	Args:
		log_dir: 日志目录。
		model_name: 模型名称，用于日志文件命名。

	Returns:
		(logger, log_file, timestamp)。
	"""
	# 同时写文件和控制台，便于训练过程中实时观察，也方便回看完整日志。
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
	# 通过文件路径动态加载 baseline 模型，避免文件名里出现 pointnet++.py 这类不方便直接 import 的情况。
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Failed to load module from: {file_path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def build_model(model_name: str, num_classes: int, project_root: Path) -> nn.Module:
	"""按名称构建分类+框回归模型。

	Args:
		model_name: 模型名称，支持 dgcnn/pointnet2/pointtransformer。
		num_classes: 分类类别数。
		project_root: 项目根目录。

	Returns:
		构建完成的 nn.Module。

	Raises:
		ValueError: 模型名不在支持列表中。
	"""
	# 这里决定训练主干网络；分类头输出的类别数由数据集实际类别数决定。
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
		return module.PointNet2ClassificationSSG(num_class=num_classes)

	raise ValueError(f"Unsupported model name: {model_name}")


def prepare_model_inputs(points_xyzi: torch.Tensor) -> torch.Tensor:
	"""准备模型输入。

	Args:
		points_xyzi: 形状为 (B, N, 4) 的点云张量。

	Returns:
		直接返回 (B, N, 4)，保持与模型前向契约一致。
	"""
	# 点云样本已经在数据集里整理成 (B, N, 4)，这里保持原样传入模型即可。
	return points_xyzi


def compute_topk_hits(logits: torch.Tensor, labels: torch.Tensor, topk: Iterable[int] = (1, 3)) -> Dict[int, int]:
	"""统计 top-k 命中数。

	Args:
		logits: 分类输出，形状 [B, C]。
		labels: 真实标签，形状 [B]。
		topk: 需要统计的 k 列表。

	Returns:
		{K: 命中数} 字典。
	"""
	# top-k 命中数是从模型 logits 里直接统计的，不依赖 box 分支。
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


def slice_batch_meta(meta: Mapping[str, Any], valid_mask: torch.Tensor) -> Dict[str, Any]:
	"""Slice collated metadata using a batch-level validity mask."""
	# DataLoader 会把 batch 内元信息拼起来，这里要按有效样本掩码同步裁剪，避免点云和标签对不齐。
	mask_cpu = valid_mask.detach().cpu()
	mask_list = mask_cpu.tolist()
	sliced: Dict[str, Any] = {}

	for key, value in meta.items():
		# 复杂条件说明：仅当该字段具备 batch 第一维且长度与掩码一致时才切片，
		# 否则保持原值，避免把全局配置/标量字段误当成逐样本字段处理。
		if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == len(mask_list):
			sliced[key] = value[mask_cpu]
		elif isinstance(value, list) and len(value) == len(mask_list):
			sliced[key] = [item for item, keep in zip(value, mask_list) if keep]
		elif isinstance(value, tuple) and len(value) == len(mask_list):
			sliced[key] = tuple(item for item, keep in zip(value, mask_list) if keep)
		else:
			sliced[key] = value

	return sliced


def run_epoch(
	loader,
	model: nn.Module,
	criterion: PointCloudMultiTaskLoss,
	device: torch.device,
	epoch: int,
	phase: str,
	optimizer: Optional[optim.Optimizer] = None,
) -> Dict[str, float]:
	"""执行单个 epoch 的训练或验证。

	Args:
		loader: 数据加载器，输出 (points, labels, meta)。
		model: 模型。
		criterion: 多任务损失对象。
		device: 运行设备。
		epoch: 当前 epoch 序号。
		phase: 阶段名（Train/Val）。
		optimizer: 训练时提供；验证时为 None。

	Returns:
		包含 loss/top1/top3 与 box 指标的聚合字典。
	"""
	is_train = optimizer is not None
	model.train(is_train)

	total_loss = 0.0
	total_samples = 0
	correct_top1 = 0
	correct_top3 = 0
	total_box_l1 = 0.0
	total_box_iou_loss = 0.0
	total_box_iou = 0.0
	box_metric_samples = 0

	pbar = tqdm(loader, desc=f"{phase} Epoch {epoch}", leave=False)
	context = torch.enable_grad() if is_train else torch.no_grad()

	with context:
		for points, labels, batch_meta in pbar:
			# 输入 batch 来自数据集：points 是点云，labels 是类别索引，batch_meta 保存 box 监督所需的辅助信息。
			points = points.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			# 有些样本可能带无效标签，先过滤掉，再让后面的分类和 box 监督都只看有效样本。
			valid_mask = labels >= 0
			if not valid_mask.any():
				continue

			# 过滤后，点云和标签的 batch 维度保持完全一致。
			points = points[valid_mask]
			labels = labels[valid_mask]

			# 模型前向只吃点云，输出里会同时包含分类 logits 和 3D box 预测。
			inputs = prepare_model_inputs(points)
			model_outputs = model(inputs)
			logits, box_preds = split_cls_and_box_predictions(model_outputs)

			box_targets = None
			if isinstance(batch_meta, Mapping):
				# box 监督来自样本元信息里的 target_x / target_y / target_z 范围，按有效样本同步裁剪后再构造目标框。
				meta_valid = slice_batch_meta(batch_meta, valid_mask)
				try:
					box_targets = build_spad_boxes_from_meta(meta_valid, device=device)
				except Exception:
					# // TODO(copilot) 2026-04-26: 改为细粒度异常分类并记录样本路径，便于定位脏元信息来源。
					box_targets = None

			# 多任务损失内部会分别计算分类损失、box L1 损失和 IoU 损失，并汇总成 total_loss。
			loss_dict = criterion(
				model_outputs=model_outputs,
				cls_targets=labels,
				box_targets=box_targets,
			)
			loss = loss_dict["total_loss"]

			if is_train:
				# 训练阶段才反向传播；验证阶段只做前向统计，不更新参数。
				optimizer.zero_grad(set_to_none=True)
				loss.backward()
				optimizer.step()

			batch_size = labels.size(0)
			# 所有指标都按样本数加权累计，最后再除以总样本数，得到 epoch 级平均值。
			total_loss += float(loss.item()) * batch_size
			total_samples += batch_size

			# top1 / top3 只来自分类 logits；这里只统计类别是否命中真实标签。
			topk_hits = compute_topk_hits(logits, labels, topk=(1, 3))
			correct_top1 += topk_hits[1]
			correct_top3 += topk_hits[3]

			if box_preds is not None and box_targets is not None:
				# box 指标只有在预测框和目标框都能成功构造时才累计，避免缺失元信息污染统计。
				total_box_l1 += float(loss_dict["box_l1_loss"].item()) * batch_size
				total_box_iou_loss += float(loss_dict["box_iou_loss"].item()) * batch_size
				total_box_iou += float(loss_dict["box_iou_mean"].item()) * batch_size
				box_metric_samples += batch_size

			avg_loss = total_loss / max(total_samples, 1)
			top1 = correct_top1 / max(total_samples, 1)
			top3 = correct_top3 / max(total_samples, 1)
			if box_metric_samples > 0:
				box_iou = total_box_iou / box_metric_samples
				pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{top1:.4f}", top3=f"{top3:.4f}", box_iou=f"{box_iou:.4f}")
			else:
				pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{top1:.4f}", top3=f"{top3:.4f}")

	metrics = {
		# loss/topk 是分类主指标；box_* 是 box 分支的监督损失和几何质量指标。
		"loss": total_loss / max(total_samples, 1),
		"top1": correct_top1 / max(total_samples, 1),
		"top3": correct_top3 / max(total_samples, 1),
		"box_l1": total_box_l1 / max(box_metric_samples, 1),
		"box_iou_loss": total_box_iou_loss / max(box_metric_samples, 1),
		"box_iou": total_box_iou / max(box_metric_samples, 1),
		"box_samples": float(box_metric_samples),
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
	"""保存训练状态到 checkpoint。

	Args:
		path: checkpoint 输出路径。
		model: 当前模型。
		optimizer: 当前优化器。
		scheduler: 学习率调度器。
		epoch: 当前 epoch。
		best_val_top1: 历史最佳验证 top1。
		class_to_idx: 类别映射。
		args: 训练参数。

	Returns:
		None。
	"""
	# checkpoint 里同时保存模型、优化器、学习率调度器和运行参数，方便后续恢复训练或做测试复现。
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
	"""执行完整训练流程。

	Args:
		args: 命令行参数命名空间。

	Returns:
		包含日志文件与 checkpoint 路径的字典。

	Raises:
		FileNotFoundError: 数据根目录不存在。
	"""
	# 训练入口：负责把路径解析、随机种子、数据加载、模型、损失和保存逻辑组装成完整流程。
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

	# 数据加载器返回 train/val/test 三个有监督集合，外加一份包含类别映射与样本数量的 meta。
	train_loader, val_loader, test_loader, unlabeled_loader, meta = create_spad_classification_dataloaders(
		data_root=str(data_root),
		batch_size=args.batch_size,
		num_points=(None if args.num_points <= 0 else args.num_points),
		train_ratio=args.train_ratio,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		num_workers=args.num_workers,
		seed=args.seed,
		augment_train=args.augment_train,
		augment_eval=args.augment_eval,
		label_mode=args.label_mode,
	)

	model = build_model(args.model, num_classes=meta["num_classes"], project_root=project_root).to(device)
	# 多任务损失的三个权重控制分类、box L1 和 box IoU 三部分对总损失的贡献。
	criterion = PointCloudMultiTaskLoss(
		cls_weight=args.cls_loss_weight,
		box_l1_weight=args.box_l1_weight,
		box_iou_weight=args.box_iou_weight,
		label_smoothing=args.label_smoothing,
	)
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
		# 恢复训练时把模型、优化器和 scheduler 状态一起读回，epoch 从 checkpoint 里接着往后走。
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
		# 每个 epoch 先训练，再验证；验证指标用于决定是否刷新 best checkpoint。
		train_metrics = run_epoch(
			loader=train_loader,
			model=model,
			criterion=criterion,
			device=device,
			epoch=epoch,
			phase="Train",
			optimizer=optimizer,
		)

		val_metrics = run_epoch(
			loader=val_loader,
			model=model,
			criterion=criterion,
			device=device,
			epoch=epoch,
			phase="Val",
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
		if train_metrics["box_samples"] > 0 or val_metrics["box_samples"] > 0:
			# box 指标单独打印，方便区分分类收敛和几何框收敛是否一致。
			logger.info(
				"Epoch [%d/%d] | train_box_iou=%.4f train_box_l1=%.4f train_box_iou_loss=%.4f | "
				"val_box_iou=%.4f val_box_l1=%.4f val_box_iou_loss=%.4f",
				epoch,
				args.epochs,
				train_metrics["box_iou"],
				train_metrics["box_l1"],
				train_metrics["box_iou_loss"],
				val_metrics["box_iou"],
				val_metrics["box_l1"],
				val_metrics["box_iou_loss"],
			)

		if val_metrics["top1"] >= best_val_top1:
			# 这里用验证集 top1 作为 best 依据，和常规分类训练保持一致。
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
	# last checkpoint 记录的是最终 epoch 的完整状态，便于后续继续训练或直接测试。
	logger.info("Saved last checkpoint to %s", last_ckpt)
	logger.info("Training finished. Best val top1=%.4f", best_val_top1)

	return {
		"log_file": str(log_file),
		"best_checkpoint": str(best_ckpt),
		"last_checkpoint": str(last_ckpt),
	}


def build_parser() -> argparse.ArgumentParser:
	"""构建训练命令行参数解析器。"""
	# 命令行参数覆盖数据路径、训练超参、损失权重和增强开关，便于不同实验复用同一脚本。
	parser = argparse.ArgumentParser(description="SPAD 3D point cloud classification training")
	parser.add_argument("--data-root", type=str, default=r"D:\PYproject\SPADdata\2025-04-30-dpc", help="SPAD data root directory")
	parser.add_argument("--model", type=str, default="dgcnn", choices=["dgcnn", "pointnet2", "pointtransformer"], help="Backbone model")
	parser.add_argument("--epochs", type=int, default=80)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--num-points", type=int, default=1024, help="Fixed number of points per sample (deterministic sample/pad)")
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--min-lr", type=float, default=1e-5)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--train-ratio", type=float, default=0.6)
	parser.add_argument("--val-ratio", type=float, default=0.2)
	parser.add_argument("--test-ratio", type=float, default=0.2)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", type=str, default="cuda", help="auto/cpu/cuda")
	parser.add_argument("--log-dir", type=str, default="logs")
	parser.add_argument("--save-dir", type=str, default="checkpoints")
	parser.add_argument("--resume", type=str, default="", help="checkpoint path to resume")
	parser.add_argument("--label-mode", type=str, default="raw", choices=["generated", "raw"], help="Label source mode")
	parser.add_argument("--cls-loss-weight", type=float, default=1.0, help="Classification loss weight")
	parser.add_argument("--box-l1-weight", type=float, default=1.0, help="3D box SmoothL1 loss weight")
	parser.add_argument("--box-iou-weight", type=float, default=1.0, help="3D box IoU loss weight")
	parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for classification loss")
	parser.add_argument("--augment-train", dest="augment_train", action="store_true", help="Apply augmentation in train dataset")
	parser.add_argument("--no-augment-train", dest="augment_train", action="store_false", help="Disable train dataset augmentation")
	parser.add_argument("--augment-eval", dest="augment_eval", action="store_true", help="Apply augmentation in val/test dataset")
	parser.add_argument("--no-augment-eval", dest="augment_eval", action="store_false", help="Disable val/test dataset augmentation")
	parser.set_defaults(augment_train=True, augment_eval=True)
	return parser


def main(argv=None) -> None:
	"""训练脚本入口函数。"""
	parser = build_parser()
	args = parser.parse_args(argv)
	run_training(args)


if __name__ == "__main__":
	main()



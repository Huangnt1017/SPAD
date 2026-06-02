import argparse
import importlib.util
import json
import logging
import random
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

# 训练脚本只负责把数据、模型、损失和日志串起来，核心计算都保留在各自模块中。
from utils.data import create_dataloaders
from utils.loss import PointCloudMultiTaskLoss, build_spad_boxes_from_meta, split_cls_and_box_predictions
from utils.checkpoint import save_checkpoint


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


def configure_torch_runtime(enable_tf32: bool) -> None:
	"""配置 CUDA 数值后端，默认允许 TF32 提升 Ampere/Ada GPU 训练吞吐。"""
	if not torch.cuda.is_available():
		return
	torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
	torch.backends.cudnn.allow_tf32 = bool(enable_tf32)


def has_spikingjelly_state(model: nn.Module) -> bool:
	"""判断模型中是否包含需要跨 batch 重置的 SpikingJelly 状态模块。"""
	return any(
		type(module).__module__.startswith("spikingjelly1.")
		and callable(getattr(module, "reset", None))
		for module in model.modules()
	)


def reset_spikingjelly_state(model: nn.Module) -> None:
	"""重置 SpikingJelly 膜电位状态，避免旧 batch 的计算图被下一 batch 持有。"""
	for module in model.modules():
		if type(module).__module__.startswith("spikingjelly1."):
			reset = getattr(module, "reset", None)
			if callable(reset):
				reset()


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


def setup_logger(
	log_dir: Path,
	model_name: str,
	timestamp: Optional[str] = None,
	log_file: Optional[Path] = None,
	append: bool = False,
) -> Tuple[logging.Logger, Path, str]:
	"""创建训练日志器。

	Args:
		log_dir: 日志目录。
		model_name: 模型名称，用于日志文件命名。
		timestamp: 可选。恢复训练时传入原 run 的时间戳，保持 checkpoint 文件名连续。
		log_file: 可选。恢复训练时传入原日志文件，继续追加写入。
		append: 是否以追加模式打开日志文件。

	Returns:
		(logger, log_file, timestamp)。
	"""
	# 同时写文件和控制台，便于训练过程中实时观察，也方便回看完整日志。
	log_dir.mkdir(parents=True, exist_ok=True)
	if timestamp is None:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	if log_file is None:
		log_file = log_dir / f"train_{model_name}_{timestamp}.log"
	else:
		log_file = Path(log_file)
		if not log_file.is_absolute():
			log_file = log_dir / log_file
	log_file.parent.mkdir(parents=True, exist_ok=True)

	logger_name = f"spad_train_{model_name}_{timestamp}"
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.INFO)
	logger.propagate = False
	for handler in list(logger.handlers):
		handler.close()
		logger.removeHandler(handler)

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

	file_handler = logging.FileHandler(log_file, mode="a" if append else "w", encoding="utf-8")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	return logger, log_file, timestamp


def load_module_from_file(file_path: Path, module_name: str):
	# 通过文件路径动态加载 baseline 模型，避免文件名里出现 PointNet++.py 这类不方便直接 import 的情况。
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Failed to load module from: {file_path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def build_model(model_name: str, num_classes: int, project_root: Path, args: Optional[argparse.Namespace] = None) -> nn.Module:
	"""按名称构建分类+框回归模型。

	Args:
		model_name: 模型名称，支持 dgcnn/pointnet/pointnet2/pointnet2msg/pointtransformer/pointtransv2/pointtransv3/pointmlp/pointmlpelite/spt/3detr/pointrwkv/pointbert/pointmae/upp/graph_residual/graph_residual_gcn。
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

	if name == "pointnet2":
		module = load_module_from_file(baseline_dir / "PointNet++.py", "baseline_pointnet2")
		return module.PointNet2ClassificationSSG(num_class=num_classes)

	if name == "pointnet":
		module = load_module_from_file(baseline_dir / "pointnet.py", "baseline_pointnet")
		return module.PointNetCls(num_classes=num_classes)

	if name == "pointnet2msg":
		module = load_module_from_file(baseline_dir / "PointNet++.py", "baseline_pointnet2")
		return module.PointNet2ClassificationMSG(num_class=num_classes)

	if name == "pointmlp":
		module = load_module_from_file(baseline_dir / "PointMLP.py", "baseline_pointmlp")
		return module.PointMLPClassification(num_classes=num_classes)

	if name == "pointmlpelite":
		module = load_module_from_file(baseline_dir / "PointMLP.py", "baseline_pointmlp")
		return module.PointMLPClassification(num_classes=num_classes, variant="pointmlpelite")

	if name == "spt":
		module = load_module_from_file(baseline_dir / "SPT.py", "baseline_spt")
		import types
		cfg = types.SimpleNamespace()
		cfg.num_point = int(getattr(args, "num_points", 1024)) if args is not None else 1024
		cfg.model = types.SimpleNamespace()
		cfg.model.nblocks = int(getattr(args, "spt_nblocks", 4)) if args is not None else 4
		cfg.model.nneighbor = int(getattr(args, "spt_nneighbor", 16)) if args is not None else 16
		cfg.model.blocks = [1] * (cfg.model.nblocks + 1)
		cfg.model.num_samples = int(getattr(args, "spt_num_samples", 512)) if args is not None else 512
		spike_mode = getattr(args, "spt_spike_mode", "lif") if args is not None else "lif"
		cfg.model.spike_mode = None if spike_mode in (None, "none", "None", "ann") else spike_mode
		cfg.model.timestep = int(getattr(args, "spt_timestep", 2)) if args is not None else 2
		cfg.model.use_encoder = bool(getattr(args, "spt_use_encoder", True)) if args is not None else True
		cfg.model.transformer_dim = int(getattr(args, "spt_transformer_dim", 512)) if args is not None else 512
		cfg.model.use_moe_lif = bool(getattr(args, "spt_use_moe_lif", True)) if args is not None else True
		cfg.input_dim = 4
		cfg.num_classes = num_classes
		return module.SPTNet(cfg)

	if name == "3detr":
		module = load_module_from_file(baseline_dir / "3DETR.py", "baseline_3detr")
		return module.ThreeDETRClassification(num_classes=num_classes)

	# === Point Transformer 系列 (Pointcept 复现) ===
	if name == "pointtransformer":
		module = load_module_from_file(baseline_dir / "PointTransformer.py", "baseline_point_transformer")
		return module.PointTransformerClassification(num_classes=num_classes)

	if name == "pointtransv2":
		module = load_module_from_file(baseline_dir / "PointTransV2.py", "baseline_point_trans_v2")
		return module.PointTransV2Classification(num_classes=num_classes)

	if name == "pointtransv3":
		module = load_module_from_file(baseline_dir / "PointTransV3.py", "baseline_point_trans_v3")
		return module.PointTransV3Classification(num_classes=num_classes)

	# === PointRWKV ===
	if name == "pointrwkv":
		module = load_module_from_file(baseline_dir / "PointRWKV.py", "baseline_point_rwkv")
		return module.PointRWKVClassification(num_classes=num_classes)

	# === Point-BERT ===
	if name == "pointbert":
		module = load_module_from_file(baseline_dir / "PointBERT.py", "baseline_point_bert")
		return module.PointBERTClassification(num_classes=num_classes)

	# === Point-MAE ===
	if name == "pointmae":
		module = load_module_from_file(baseline_dir / "PointMAE.py", "baseline_point_mae")
		return module.PointMAEClassification(num_classes=num_classes)

	# === UPP (ICCV 2025): Point-MAE 之上的 point-level prompting PEFT 框架 ===
	if name == "upp":
		module = load_module_from_file(baseline_dir / "UPP.py", "baseline_upp")
		return module.UPPClassification(num_classes=num_classes)

	# === Graph Residual (本课题自研, model/readme.md 任务 1) ===
	# 模型文件不在 baseline/ 而在 model/, 走 load_module_from_file 同样加载。
	if name == "graph_residual":
		model_module = load_module_from_file(
			project_root / "model" / "graph_residual.py", "model_graph_residual"
		)
		return model_module.GraphResidualMultiTaskNet(num_classes=num_classes)

	# === Graph Residual GCN (PyG SAGEConv 版, model/graph_res_GCN.py) ===
	if name == "graph_residual_gcn":
		model_module = load_module_from_file(
			project_root / "model" / "graph_res_GCN.py", "model_graph_res_gcn"
		)
		return model_module.GraphResidualMultiTaskNetGCN(num_classes=num_classes)

	raise ValueError(f"Unsupported model name: {model_name}")


def merge_resume_model_args(args: argparse.Namespace, checkpoint: Mapping[str, Any]) -> argparse.Namespace:
	"""恢复训练时优先使用 checkpoint 中保存的模型结构参数。"""
	ckpt_args = checkpoint.get("args") if isinstance(checkpoint, Mapping) else None
	if not isinstance(ckpt_args, Mapping):
		return args

	merged = vars(args).copy()
	for key, value in ckpt_args.items():
		if key == "model" or key == "num_points" or key.startswith("spt_"):
			merged[key] = value
	if merged.get("model") == "spt" and "spt_use_moe_lif" not in ckpt_args:
		state_dict = checkpoint.get("model_state_dict", {})
		if isinstance(state_dict, Mapping):
			merged["spt_use_moe_lif"] = any(
				".fc1.0.gate." in key or ".fc1.0.experts." in key
				for key in state_dict
			)
	return argparse.Namespace(**merged)


def prepare_model_inputs(points_xyzi: torch.Tensor) -> torch.Tensor:
	"""准备模型输入。

	Args:
		points_xyzi: 形状为 (B, N, 4) 的点云张量。

	Returns:
		直接返回 (B, N, 4)，保持与模型前向契约一致。
	"""
	# 兼容 (B, N, 4) 与 (B, 4, N)，统一成 (B, N, 4) 给模型。
	if points_xyzi.ndim != 3:
		raise ValueError(f"prepare_model_inputs expects 3D input, got shape {tuple(points_xyzi.shape)}")
	if points_xyzi.shape[-1] == 4:
		return points_xyzi
	if points_xyzi.shape[1] == 4:
		return points_xyzi.transpose(1, 2).contiguous()
	
	raise ValueError(f"prepare_model_inputs expects shape (B,N,4) or (B,4,N), got {tuple(points_xyzi.shape)}")


def compute_topk_hits(logits: torch.Tensor, labels: torch.Tensor, topk: Iterable[int] = (1, 3)) -> Dict[int, torch.Tensor]:
	"""统计 top-k 命中数 (返回 GPU 标量 tensor, 避免 hot loop 同步)。

	Args:
		logits: 分类输出，形状 [B, C]。
		labels: 真实标签，形状 [B]。
		topk: 需要统计的 k 列表。

	Returns:
		{K: 命中数 tensor (标量, dtype=long, 与 logits 同 device)} 字典。
		注意: 原版返回 ``int`` 会触发 ``.item()`` GPU→CPU 同步; 这里返回 GPU tensor,
		让上层累加器以 tensor 形式聚合, 仅在 epoch 末统一 sync 一次。数值完全一致。
	"""
	# top-k 命中数是从模型 logits 里直接统计的，不依赖 box 分支。
	num_classes = logits.size(1)
	max_k = min(max(topk), num_classes)
	_, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
	pred = pred.t()
	correct = pred.eq(labels.view(1, -1).expand_as(pred))

	out: Dict[int, torch.Tensor] = {}
	for k in topk:
		kk = min(k, num_classes)
		# 不再 .item(), 保留为 GPU 上的标量 long tensor; 累加 / 比较都合法。
		out[k] = correct[:kk].reshape(-1).sum()
	return out


def _clamp_unit(value: float) -> float:
	"""把度量值压到 [0, 1]，用于组合评分的统一尺度。"""
	if not np.isfinite(value):
		return 0.0
	return float(min(1.0, max(0.0, value)))


def depth_loss_to_score(depth_loss: float, depth_scale: float) -> float:
	"""将越小越好的 depth loss 映射成 [0, 1] 分数。"""
	if depth_scale <= 0:
		raise ValueError(f"best_score_depth_scale must be positive, got {depth_scale}")
	if not np.isfinite(depth_loss):
		return 0.0
	return float(1.0 / (1.0 + max(0.0, depth_loss) / depth_scale))


def compute_composite_score(
	metrics: Mapping[str, float],
	args: argparse.Namespace,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
	"""计算统一尺度的多任务组合分数。

	组合分数只用于 best checkpoint 选择，不反向传播。分类 Top-1、box IoU 与 depth loss
	先统一成 [0, 1] 质量分数，再按归一化权重加权，避免某个任务因数值范围天然更大而主导选模。
	"""
	has_box_metrics = float(metrics.get("box_samples", 0.0)) > 0
	components: Dict[str, float] = {
		"cls_top1": _clamp_unit(float(metrics.get("top1", 0.0))),
	}
	raw_weights: Dict[str, float] = {
		"cls_top1": max(0.0, float(args.best_score_cls_weight)),
	}

	if has_box_metrics:
		components["box_iou"] = _clamp_unit(float(metrics.get("box_iou", 0.0)))
		components["box_depth"] = depth_loss_to_score(
			float(metrics.get("box_depth", 0.0)),
			float(args.best_score_depth_scale),
		)
		raw_weights["box_iou"] = max(0.0, float(args.best_score_iou_weight))
		raw_weights["box_depth"] = max(0.0, float(args.best_score_depth_weight))

	weight_sum = sum(raw_weights.values())
	if weight_sum <= 0:
		raise ValueError("At least one active best-score weight must be positive.")

	weights = {key: value / weight_sum for key, value in raw_weights.items()}
	score = sum(weights[key] * components[key] for key in components)
	return float(score), components, weights


def score_config_from_args(args: argparse.Namespace) -> Dict[str, float]:
	"""提取组合评分配置，保存到 checkpoint 便于复现实验口径。"""
	return {
		"cls_weight": float(args.best_score_cls_weight),
		"iou_weight": float(args.best_score_iou_weight),
		"depth_weight": float(args.best_score_depth_weight),
		"depth_scale": float(args.best_score_depth_scale),
	}


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
	scaler: Optional[torch.amp.GradScaler] = None,
	use_amp: bool = False,
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
	amp_enabled = bool(use_amp and device.type == "cuda")

	# 性能优化: 累加器保留在 device 上 (GPU tensor), 避免每个 batch 多次 .item() 触发
	# CPU-GPU 同步。仅样本计数 (total_samples / box_metric_samples) 是从 Python int
	# (labels.size(0)) 直接累加的, 不涉及 GPU 同步。tqdm 进度条按 log_every 节流,
	# 一次 .cpu().tolist() 把所有需要展示的标量批量搬到 CPU。
	total_loss = torch.zeros((), device=device)
	total_box_depth = torch.zeros((), device=device)
	total_box_iou = torch.zeros((), device=device)
	correct_top1 = torch.zeros((), device=device, dtype=torch.long)
	correct_top3 = torch.zeros((), device=device, dtype=torch.long)
	total_samples = 0
	box_metric_samples = 0

	pbar = tqdm(loader, desc=f"{phase} Epoch {epoch}", leave=False)
	context = torch.enable_grad() if is_train else torch.no_grad()
	# 进度条每 ~20 次更新一次 (但首/末 batch 一定更新), 避免每 batch 都触发同步与字符串格式化。
	num_batches = len(loader) if hasattr(loader, "__len__") else None
	log_every = max(1, (num_batches // 20)) if num_batches else 1
	reset_snn_state = has_spikingjelly_state(model)

	with context:
		for batch_step, batch in enumerate(pbar):
			if len(batch) == 2:
				points, targets = batch
				batch_meta = None
			else:
				points, targets, batch_meta = batch
			# 输入 batch 来自数据集：points 是点云，labels 是类别索引，batch_meta 保存 box 监督所需的辅助信息。
			points = points.to(device, non_blocking=True)

			labels = None
			box_targets = None

			if isinstance(targets, Mapping) and "bbox_targets" in targets:
				cls_targets = targets.get("cls_targets")
				bbox_targets = targets.get("bbox_targets")
				mask = targets.get("mask")
				if cls_targets is None or bbox_targets is None or mask is None:
					raise ValueError("targets must include cls_targets, bbox_targets, and mask")

				cls_targets = cls_targets.to(device, non_blocking=True)
				bbox_targets = bbox_targets.to(device, non_blocking=True)
				mask = mask.to(device, non_blocking=True).bool()

				# 假设单样本单目标：选择每个样本的第一个有效框作为监督。
				valid_obj = mask
				has_obj = valid_obj.any(dim=1)
				if not has_obj.any():
					continue

				first_idx = valid_obj.float().argmax(dim=1)
				batch_idx = torch.arange(points.size(0), device=points.device)
				labels = cls_targets[batch_idx, first_idx].long()
				box_targets = bbox_targets[batch_idx, first_idx]

				points = points[has_obj]
				labels = labels[has_obj]
				box_targets = box_targets[has_obj]
			else:
				labels = targets.to(device, non_blocking=True)
				# 有些样本可能带无效标签，先过滤掉，再让后面的分类和 box 监督都只看有效样本。
				valid_mask = labels >= 0
				if not valid_mask.any():
					continue

				points = points[valid_mask]
				labels = labels[valid_mask]

				if isinstance(batch_meta, Mapping):
					meta_valid = slice_batch_meta(batch_meta, valid_mask)
					try:
						box_targets = build_spad_boxes_from_meta(meta_valid, device=device)
					except Exception:
						# // TODO(copilot) 2026-04-26: 改为细粒度异常分类并记录样本路径，便于定位脏元信息来源。
						box_targets = None

			# 模型前向只吃点云，输出里会同时包含分类 logits 和 3D box 预测。
			inputs = prepare_model_inputs(points)
			if reset_snn_state:
				reset_spikingjelly_state(model)
			with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
				model_outputs = model(inputs)
				logits, box_preds = split_cls_and_box_predictions(model_outputs)

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
				if amp_enabled:
					if scaler is None:
						raise ValueError("AMP training requires a GradScaler.")
					scaler.scale(loss).backward()
					scaler.step(optimizer)
					scaler.update()
				else:
					loss.backward()
					optimizer.step()

			batch_size = labels.size(0)
			# 所有指标都按样本数加权累计，最后再除以总样本数，得到 epoch 级平均值。
			# .detach() 切断与计算图的连接, 累加器只用其数值, 避免占用反传内存。
			total_loss += loss.detach() * batch_size
			total_samples += batch_size

			# top1 / top3 只来自分类 logits；这里只统计类别是否命中真实标签。
			# 现在 compute_topk_hits 返回 GPU 标量 tensor, 累加保持在 GPU 上 (无同步)。
			topk_hits = compute_topk_hits(logits, labels, topk=(1, 3))
			correct_top1 += topk_hits[1]
			correct_top3 += topk_hits[3]

			if box_preds is not None and box_targets is not None:
				# box 指标只有在预测框和目标框都能成功构造时才累计，避免缺失元信息污染统计。
				total_box_depth += loss_dict["box_depth_loss"].detach() * batch_size
				total_box_iou += loss_dict["box_iou_mean"].detach() * batch_size
				box_metric_samples += batch_size

			if reset_snn_state:
				reset_spikingjelly_state(model)

			# 进度条节流: 只在间隔点 / 末 batch 更新一次, 把所有标量一次性 stack→cpu→tolist
			# 来批量同步, 减少 GPU 等待。中间 batch 不显示瞬时值 (最终聚合数值不受影响)。
			is_last = (num_batches is not None) and (batch_step + 1 == num_batches)
			if (batch_step % log_every == 0) or is_last:
				# 单次同步: 把 4 个 GPU 标量 stack 到一起再 .cpu().tolist(), 一次 sync 完成。
				snap = torch.stack([
					total_loss,
					correct_top1.to(total_loss.dtype),
					correct_top3.to(total_loss.dtype),
					total_box_iou,
				]).cpu().tolist()
				s_loss, s_c1, s_c3, s_box_iou = snap
				avg_loss = s_loss / max(total_samples, 1)
				top1 = s_c1 / max(total_samples, 1)
				top3 = s_c3 / max(total_samples, 1)
				if box_metric_samples > 0:
					box_iou = s_box_iou / box_metric_samples
					pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{top1:.4f}", top3=f"{top3:.4f}", box_iou=f"{box_iou:.4f}")
				else:
					pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{top1:.4f}", top3=f"{top3:.4f}")

	# epoch 末单次同步: 把 6 个累加器 stack 到一起搬到 CPU, 减少多次 .item() 的开销。
	final_snap = torch.stack([
		total_loss,
		correct_top1.to(total_loss.dtype),
		correct_top3.to(total_loss.dtype),
		total_box_depth,
		total_box_iou,
	]).cpu().tolist()
	f_loss, f_c1, f_c3, f_bd, f_bi = final_snap
	metrics = {
		"loss": f_loss / max(total_samples, 1),
		"top1": f_c1 / max(total_samples, 1),
		"top3": f_c3 / max(total_samples, 1),
		"box_depth": f_bd / max(box_metric_samples, 1),
		"box_iou": f_bi / max(box_metric_samples, 1),
		"box_samples": float(box_metric_samples),
		"samples": float(total_samples),
	}
	return metrics

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
	configure_torch_runtime(args.tf32)

	if args.device == "auto":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		if args.device == "cuda" and not torch.cuda.is_available():
			device = torch.device("cpu")
		else:
			device = torch.device(args.device)

	resume_checkpoint = None
	if args.resume:
		resume_path = resolve_path(args.resume, project_root)
		resume_checkpoint = torch.load(resume_path, map_location="cpu")
		args = merge_resume_model_args(args, resume_checkpoint)

	resume_log_file = getattr(args, "resume_log_file", None)
	resume_run_timestamp = getattr(args, "resume_run_timestamp", None)
	logger, log_file, run_timestamp = setup_logger(
		log_dir=log_dir,
		model_name=args.model,
		timestamp=resume_run_timestamp or None,
		log_file=Path(resume_log_file) if resume_log_file else None,
		append=bool(args.resume and resume_log_file),
	)

	# 使用统一的多任务数据管线，训练集强制开启增强。
	if args.train_ratio + args.val_ratio + args.test_ratio <= 0:
		raise ValueError("train/val/test ratios must be positive.")

	train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
		data_root=str(data_root),
		batch_size=args.batch_size,
		num_points=args.num_points,
		train_ratio=args.train_ratio,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		num_workers=args.num_workers,
		seed=args.seed,
		augment_train=args.augment_train,
		augment_eval=args.augment_eval,
		label_mode=args.label_mode,
	)

	num_classes = len(class_to_idx)
	model = build_model(args.model, num_classes=num_classes, project_root=project_root, args=args).to(device)
	# 当前多任务损失包含 CrossEntropy 与 Soft-histogram 深度回归两项；默认用固定权重 λ_cls · L_cls + λ_depth · L_depth。
	criterion = PointCloudMultiTaskLoss(
		cls_weight=args.cls_loss_weight,
		box_weight=args.box_loss_weight,
		label_smoothing=args.label_smoothing,
		auto_balance=args.auto_balance,
	)
	# 将 loss 中的可学习参数 (Kendall 不确定性权重) 一并加入 optimizer
	optimizer = optim.AdamW(
		list(model.parameters()) + list(criterion.parameters()),
		lr=args.lr, weight_decay=args.weight_decay,
	)
	scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
	use_amp = bool(args.amp and device.type == "cuda")
	scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

	save_dir.mkdir(parents=True, exist_ok=True)

	logger.info("=== Training Configuration ===")
	logger.info("run_timestamp=%s", run_timestamp)
	if args.resume and resume_log_file:
		logger.info("append_log_file=%s", log_file)
	logger.info("data_root=%s", data_root)
	logger.info("device=%s", device)
	logger.info("model=%s", args.model)
	logger.info("num_classes=%d", num_classes)
	logger.info("split train/val/test = %d / %d / %d", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
	logger.info("label_mode=%s", args.label_mode)
	logger.info("augment_train=%s augment_eval=%s", args.augment_train, args.augment_eval)
	logger.info("amp=%s tf32=%s", use_amp, args.tf32)
	logger.info("loss_auto_balance=%s", args.auto_balance)
	logger.info(
		"best_score_weights cls=%.4f iou=%.4f depth=%.4f depth_scale=%.6f",
		args.best_score_cls_weight,
		args.best_score_iou_weight,
		args.best_score_depth_weight,
		args.best_score_depth_scale,
	)
	logger.info("args=%s", json.dumps(vars(args), ensure_ascii=False))

	start_epoch = 1
	best_val_top1 = 0.0
	best_val_score = float("-inf")
	best_val_metrics: Dict[str, float] = {}
	current_score_config = score_config_from_args(args)

	if args.resume:
		# 恢复训练时把模型、优化器和 scheduler 状态一起读回，epoch 从 checkpoint 里接着往后走。
		resume_path = resolve_path(args.resume, project_root)
		checkpoint = torch.load(resume_path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])
		if "criterion_state_dict" in checkpoint:
			criterion.load_state_dict(checkpoint["criterion_state_dict"])
		if "optimizer_state_dict" in checkpoint:
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		if "scheduler_state_dict" in checkpoint:
			scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
		start_epoch = int(checkpoint.get("epoch", 0)) + 1
		best_val_top1 = float(checkpoint.get("best_val_top1", 0.0))
		if "best_val_score" in checkpoint:
			best_val_score = float(checkpoint["best_val_score"])
		else:
			best_val_score = float("-inf")
		loaded_best_val_metrics = checkpoint.get("best_val_metrics", {})
		if isinstance(loaded_best_val_metrics, Mapping):
			best_val_metrics = {
				str(key): float(value)
				for key, value in loaded_best_val_metrics.items()
				if isinstance(value, (int, float))
			}
		logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)
		if not np.isfinite(best_val_score):
			logger.info("Checkpoint has no best_val_score; composite best selection restarts from this resume run.")

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
			scaler=scaler,
			use_amp=use_amp,
		)

		val_metrics = run_epoch(
			loader=val_loader,
			model=model,
			criterion=criterion,
			device=device,
			epoch=epoch,
			phase="Val",
			optimizer=None,
			scaler=None,
			use_amp=use_amp,
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
				"Epoch [%d/%d] | train_box_iou=%.4f train_box_depth=%.4f | "
				"val_box_iou=%.4f val_box_depth=%.4f",
				epoch,
				args.epochs,
				train_metrics["box_iou"],
				train_metrics["box_depth"],
				val_metrics["box_iou"],
				val_metrics["box_depth"],
			)

		val_score, score_components, score_weights = compute_composite_score(val_metrics, args)
		logger.info(
			"Epoch [%d/%d] | val_score=%.4f | components cls=%.4f iou=%.4f depth=%.4f | "
			"weights cls=%.3f iou=%.3f depth=%.3f",
			epoch,
			args.epochs,
			val_score,
			score_components.get("cls_top1", 0.0),
			score_components.get("box_iou", 0.0),
			score_components.get("box_depth", 0.0),
			score_weights.get("cls_top1", 0.0),
			score_weights.get("box_iou", 0.0),
			score_weights.get("box_depth", 0.0),
		)

		if val_score >= best_val_score:
			# 用统一尺度的组合评分选 best，避免分类、IoU 或 depth 任一任务因数值范围主导选模。
			best_val_score = val_score
			best_val_top1 = val_metrics["top1"]
			best_val_metrics = {
				key: float(value)
				for key, value in val_metrics.items()
				if isinstance(value, (int, float))
			}
			best_val_metrics["score"] = best_val_score
			save_checkpoint(
				path=best_ckpt,
				model=model,
				optimizer=optimizer,
				scheduler=scheduler,
				epoch=epoch,
				best_val_top1=best_val_top1,
				class_to_idx=class_to_idx,
				args=args,
				criterion=criterion,
				best_val_score=best_val_score,
				best_val_metrics=best_val_metrics,
				score_config=current_score_config,
			)
			logger.info("Saved new best checkpoint to %s (score=%.4f)", best_ckpt, best_val_score)

		# last checkpoint 每个 epoch 覆盖保存一次，确保中断后能从最近完整 epoch 继续。
		save_checkpoint(
			path=last_ckpt,
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			epoch=epoch,
			best_val_top1=best_val_top1,
			class_to_idx=class_to_idx,
			args=args,
			criterion=criterion,
			best_val_score=best_val_score,
			best_val_metrics=best_val_metrics,
			score_config=current_score_config,
		)
		logger.info("Saved last checkpoint to %s", last_ckpt)

	logger.info("Training finished. Best val score=%.4f best val top1=%.4f", best_val_score, best_val_top1)

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
	parser.add_argument(
		"--model",
		type=str,
		default="graph_residual_gcn",
		choices=[
			"dgcnn",
			"pointnet",
			"pointnet2",
			"pointnet2msg",
			"pointbert",
			"pointmae",
			"pointrwkv",
			"pointtransformer",
			"pointtransv2",
			"pointtransv3",
			"pointmlp", 
			"pointmlpelite",
			"spt",
			"upp",
			"graph_residual",
			"graph_residual_gcn",
			"3detr",
		],
		help="Backbone model",
	)
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--batch-size", type=int, default=32)
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
	parser.add_argument("--cls-loss-weight", type=float, default=1.0, help="Classification loss weight when auto-balance is disabled")
	parser.add_argument("--box-loss-weight", type=float, default=10.0, help="Box Soft-histogram depth loss weight when auto-balance is disabled")
	parser.add_argument("--auto-balance", dest="auto_balance", action="store_true", help="Use Kendall log-variance task balancing")
	parser.add_argument("--no-auto-balance", dest="auto_balance", action="store_false", help="Use fixed cls/box loss weights")
	parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing for classification loss")
	parser.add_argument("--best-score-cls-weight", type=float, default=1.0, help="Composite best-checkpoint weight for validation Top-1")
	parser.add_argument("--best-score-iou-weight", type=float, default=1.0, help="Composite best-checkpoint weight for validation box IoU")
	parser.add_argument("--best-score-depth-weight", type=float, default=1.0, help="Composite best-checkpoint weight for validation depth score")
	parser.add_argument("--best-score-depth-scale", type=float, default=0.01, help="Depth loss scale used by depth_score = 1 / (1 + depth_loss / scale)")
	parser.add_argument("--spt-timestep", type=int, default=2, help="SPT temporal steps T; default follows Hengshuang.yaml")
	parser.add_argument("--spt-nneighbor", type=int, default=16, help="SPT kNN neighborhood size")
	parser.add_argument("--spt-transformer-dim", type=int, default=512, help="SPT internal transformer channel width; default follows Hengshuang.yaml")
	parser.add_argument("--spt-nblocks", type=int, default=4, help="SPT transition-down stages; default follows Hengshuang.yaml")
	parser.add_argument("--spt-num-samples", type=int, default=512, help="SPT Q-SDE samples per timestep when encoder mode is enabled")
	parser.add_argument("--spt-spike-mode", type=str, default="lif", choices=["lif", "elif", "plif", "if", "none", "ann"], help="SPT neuron type; none/ann runs the ANN path")
	parser.add_argument("--spt-use-encoder", dest="spt_use_encoder", action="store_true", help="Enable SPT Q-SDE encoder path")
	parser.add_argument("--spt-no-encoder", dest="spt_use_encoder", action="store_false", help="Disable SPT Q-SDE encoder path")
	parser.add_argument("--spt-use-moe-lif", dest="spt_use_moe_lif", action="store_true", help="Use original SPT MoE-LIF input neuron in each transformer block")
	parser.add_argument("--spt-no-moe-lif", dest="spt_use_moe_lif", action="store_false", help="Use a single spike node instead of MoE-LIF for faster/lower-memory training")
	parser.add_argument("--amp", dest="amp", action="store_true", help="Enable CUDA automatic mixed precision")
	parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable CUDA automatic mixed precision")
	parser.add_argument("--tf32", dest="tf32", action="store_true", help="Allow TF32 matmul/cuDNN on Ampere/Ada GPUs")
	parser.add_argument("--no-tf32", dest="tf32", action="store_false", help="Disable TF32 matmul/cuDNN")
	parser.add_argument("--augment-train", dest="augment_train", action="store_true", help="Apply augmentation in train dataset")
	parser.add_argument("--no-augment-train", dest="augment_train", action="store_false", help="Disable train dataset augmentation")
	parser.add_argument("--augment-eval", dest="augment_eval", action="store_true", help="Apply augmentation in val/test dataset")
	parser.add_argument("--no-augment-eval", dest="augment_eval", action="store_false", help="Disable val/test dataset augmentation")
	parser.set_defaults(augment_train=True, augment_eval=True, amp=True, tf32=True, spt_use_encoder=True, spt_use_moe_lif=True, auto_balance=False)
	return parser
#CUDA AMP 会在部分算子里使用半精度，通常是 FP16，比如：Conv / Linear / MatMul / 部分归一化相关算子，同时用 GradScaler 避免 FP16 梯度下溢。它的目标是加速和省显存，不保证和纯 FP32 完全一致。
#TF32 是 NVIDIA Ampere/Ada GPU 上对 FP32 matmul/conv 的加速格式。它仍然用 FP32 存储和 FP32 输出，但乘法内部精度降低，mantissa 比标准 FP32 少。
def main(argv=None) -> None:
	"""训练脚本入口函数。"""
	parser = build_parser()
	args = parser.parse_args(argv)
	run_training(args)


if __name__ == "__main__":
	# 用法示例 (PowerShell, conda env 路径见 memory/reference_train_env.md):
	#   $env:PYTHONPATH = "D:\PYproject\SPAD"  # 不需要输入到命令行（如果是vscode打开的
	#   & "D:\anaconda3\envs\torchnew\python.exe" "D:\PYproject\SPAD\scripts\train.py" --model pointmlp --batch-size 32 --epochs 100
	#
	# 常用参数 (完整列表见 build_parser):
	#   --model <name>          模型, 支持: dgcnn / pointnet / pointnet2 / pointnet2msg /
	#                           pointtransformer / pointtransv2 / pointtransv3 / pointmlp / pointmlpelite /
	#                           pointbert / pointmae / pointrwkv / spt / upp / 3detr /
	#                           graph_residual / graph_residual_gcn
	#   --batch-size 32         batch 大小
	#   --epochs 100            训练轮数
	#   --num-points 1024       每样本固定点数 (随机采样/补齐到该数)
	#   --lr 1e-3 --min-lr 1e-5 余弦退火上下界
	#   --weight-decay 1e-4     AdamW 权重衰减
	#   --label-mode raw        raw=用文件夹/文件名标签; generated=用增强生成的标签
	#   --amp / --no-amp        CUDA 混合精度 (默认开)
	#   --tf32 / --no-tf32      Ampere/Ada GPU TF32 加速 (默认开)
	#   --augment-eval          验证/测试时开启增强 (默认关)
	#   --num-workers 0         DataLoader worker 数 (>0 时自动启用 persistent_workers)
	#
	# 输出:
	#   logs/train_<model>_<timestamp>.log     每 epoch 两行: 分类 + box 指标
	#   checkpoints/<model>_<timestamp>_best.pth  val 最优 ckpt
	#   checkpoints/<model>_<timestamp>_last.pth  最后一个 epoch ckpt
	main()

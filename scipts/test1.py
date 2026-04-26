from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from scipts.train import build_model, resolve_path, set_seed
from utils.data import discover_spad_classification_samples, load_point_cloud_auto
from utils.loss import DEFAULT_SPAD_BOX_BOUNDS, box_iou_3d_aligned, canonicalize_boxes_3d, decode_normalized_boxes_3d


def setup_logger(log_dir: Path) -> Tuple[logging.Logger, Path]:
	"""Create a logger for the single-sample test run."""
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


def randint_inclusive(low: int, high: int, generator: Optional[torch.Generator] = None) -> int:
	"""Sample an integer from [low, high]."""
	if low > high:
		raise ValueError(f"Invalid randint range: [{low}, {high}]")
	return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def augment_single_point_cloud(points: torch.Tensor, class_name: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, object]:
	"""Augment one point cloud and return augmented points plus box metadata."""
	if points.ndim != 2 or points.shape[-1] < 4:
		raise ValueError(f"points shape must be (N, >=4), got {tuple(points.shape)}")

	aug_points = points.clone()
	xyz = aug_points[:, :3]

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

	dx_range = (xy_limit[0] - tgt_x[0], xy_limit[1] - (tgt_x[1] - 1))
	dy_range = (xy_limit[0] - tgt_y[0], xy_limit[1] - (tgt_y[1] - 1))
	dz_target_range = (target_z_limit[0] - tgt_z[0], target_z_limit[1] - (tgt_z[1] - 1))
	dz_fog_range = (fog_z_limit[0] - fog_z[0], fog_z_limit[1] - (fog_z[1] - 1))

	sample_generator: Optional[torch.Generator] = None
	if seed is not None:
		sample_generator = torch.Generator(device="cpu")
		sample_generator.manual_seed(int(seed))

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

	dx = randint_inclusive(dx_range[0], dx_range[1], sample_generator)
	dy = randint_inclusive(dy_range[0], dy_range[1], sample_generator)
	valid = False
	dz_target = 0
	dz_fog = 0
	for _ in range(20):
		dz_target = randint_inclusive(dz_target_range[0], dz_target_range[1], sample_generator)
		dz_fog_max_by_gap = tgt_z[0] + dz_target - min_gap_bins - (fog_z[1] - 1)
		dz_fog_low = dz_fog_range[0]
		dz_fog_high = min(dz_fog_range[1], dz_fog_max_by_gap)
		if dz_fog_low <= dz_fog_high:
			dz_fog = randint_inclusive(dz_fog_low, dz_fog_high, sample_generator)
			valid = True
			break
	if not valid:
		raise RuntimeError("Unable to sample valid target/fog shifts under current constraints.")

	if target_mask.any():
		aug_points[target_mask, 0] += dx
		aug_points[target_mask, 1] += dy
		aug_points[target_mask, 2] += dz_target
		aug_points[target_mask, 0] = torch.clamp(aug_points[target_mask, 0], xy_limit[0], xy_limit[1])
		aug_points[target_mask, 1] = torch.clamp(aug_points[target_mask, 1], xy_limit[0], xy_limit[1])
		aug_points[target_mask, 2] = torch.clamp(aug_points[target_mask, 2], target_z_limit[0], target_z_limit[1])

	if fog_mask.any():
		aug_points[fog_mask, 2] += dz_fog
		aug_points[fog_mask, 2] = torch.clamp(aug_points[fog_mask, 2], fog_z_limit[0], fog_z_limit[1])

	target_x_new = [int(tgt_x[0] + dx), int(tgt_x[1] + dx)]
	target_y_new = [int(tgt_y[0] + dy), int(tgt_y[1] + dy)]
	target_z_new = [int(tgt_z[0] + dz_target), int((tgt_z[1] - 1) + dz_target)]
	fog_z_new = [int(fog_z[0] + dz_fog), int((fog_z[1] - 1) + dz_fog)]

	return {
		"aug_points": aug_points,
		"meta": {
			"label": class_name,
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


def find_sample(data_root: Path, sample_path: Optional[str], sample_index: int):
	"""Resolve a single sample to test."""
	labeled_samples, _ = discover_spad_classification_samples(str(data_root))
	if not labeled_samples:
		raise ValueError(f"No labeled samples found under {data_root}")

	if sample_path:
		resolved = Path(sample_path)
		if not resolved.is_absolute():
			resolved = (data_root / resolved).resolve()
		for sample in labeled_samples:
			if Path(sample["path"]).resolve() == resolved:
				return sample
		raise FileNotFoundError(f"Sample not found: {resolved}")

	if sample_index < 0 or sample_index >= len(labeled_samples):
		raise IndexError(f"sample_index out of range: {sample_index}")
	return labeled_samples[sample_index]


def run_single_test(args: argparse.Namespace) -> Dict[str, str]:
	"""Run a single-sample augmentation + inference pass."""
	project_root = Path(__file__).resolve().parents[1]
	data_root = resolve_path(args.data_root, project_root)
	checkpoint_path = resolve_path(args.checkpoint, project_root)
	log_dir = resolve_path(args.log_dir, project_root)
	set_seed(args.seed)

	if args.device == "auto":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	elif args.device == "cuda" and not torch.cuda.is_available():
		device = torch.device("cpu")
	else:
		device = torch.device(args.device)

	logger, log_file = setup_logger(log_dir)
	sample = find_sample(data_root, args.sample_path, args.sample_index)
	class_name = str(sample.get("label") or "")
	points_np = load_point_cloud_auto(str(sample["path"]))
	points = torch.from_numpy(points_np)
	aug_out = augment_single_point_cloud(points, class_name=class_name, seed=args.seed)
	aug_points = aug_out["aug_points"].to(device)
	aug_meta = aug_out["meta"]

	checkpoint = torch.load(checkpoint_path, map_location=device)
	class_to_idx = checkpoint.get("class_to_idx")
	if not isinstance(class_to_idx, dict):
		raise KeyError("Checkpoint does not contain class_to_idx mapping.")
	idx_to_class = {idx: name for name, idx in class_to_idx.items()}
	num_classes = len(class_to_idx)

	model = build_model(args.model, num_classes=num_classes, project_root=project_root).to(device)
	state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
	model.load_state_dict(state_dict)
	model.eval()

	with torch.no_grad():
		logits, box_pred = model(aug_points.unsqueeze(0))
		probs = torch.softmax(logits, dim=1)
		pred_score, pred_idx = probs.max(dim=1)
		pred_class = idx_to_class.get(int(pred_idx.item()), str(int(pred_idx.item())))

		raw_pred_box = canonicalize_boxes_3d(box_pred, device=device, dtype=logits.dtype).squeeze(0)
		if args.box_space == "normalized":
			decoded_pred_box = decode_normalized_boxes_3d(
				raw_pred_box.unsqueeze(0),
				bounds=DEFAULT_SPAD_BOX_BOUNDS,
				device=device,
				dtype=logits.dtype,
			).squeeze(0)
		else:
			decoded_pred_box = raw_pred_box

		gt_box = torch.tensor(
			[
				aug_meta["target_box"]["x_range"][0],
				aug_meta["target_box"]["x_range"][1],
				aug_meta["target_box"]["y_range"][0],
				aug_meta["target_box"]["y_range"][1],
				aug_meta["target_box"]["z_range"][0],
				aug_meta["target_box"]["z_range"][1],
			],
			dtype=decoded_pred_box.dtype,
			device=device,
		)
		iou = float(box_iou_3d_aligned(decoded_pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item())

	result = {
		"sample_path": str(sample["path"]),
		"class_folder": class_name,
		"label_index": int(class_to_idx.get(class_name, -1)),
		"box_space": args.box_space,
		"augmented_meta": aug_meta,
		"prediction": {
			"class": pred_class,
			"score": float(pred_score.item()),
			"raw_box": [float(v) for v in raw_pred_box.detach().cpu().tolist()],
			"decoded_box": [float(v) for v in decoded_pred_box.detach().cpu().tolist()],
			"iou_vs_gt": iou,
		},
	}

	logger.info("=== Single Sample Test ===")
	logger.info("checkpoint=%s", checkpoint_path)
	logger.info("data_root=%s", data_root)
	logger.info("sample_path=%s", sample["path"])
	logger.info("class_folder=%s", class_name)
	logger.info("box_space=%s", args.box_space)
	logger.info("augmented_target_box=%s", json.dumps(aug_meta["target_box"], ensure_ascii=False))
	logger.info("augmented_fog_box=%s", json.dumps(aug_meta["fog_box"], ensure_ascii=False))
	logger.info("prediction_class=%s", pred_class)
	logger.info("prediction_score=%.4f", float(pred_score.item()))
	logger.info("prediction_raw_box=%s", json.dumps([float(v) for v in raw_pred_box.detach().cpu().tolist()], ensure_ascii=False))
	logger.info("prediction_decoded_box=%s", json.dumps([float(v) for v in decoded_pred_box.detach().cpu().tolist()], ensure_ascii=False))
	logger.info("prediction_iou_vs_gt=%.4f", iou)
	logger.info("result_json=%s", json.dumps(result, ensure_ascii=False, indent=2))

	return {"log_file": str(log_file)}


def build_parser() -> argparse.ArgumentParser:
	"""Build CLI parser for the single-sample test."""
	parser = argparse.ArgumentParser(description="SPAD single-sample DGCNN test")
	parser.add_argument("--data-root", type=str, default=r"D:\PYproject\SPADdata\2025-04-30-dpc")
	parser.add_argument("--checkpoint", type=str, default=r"D:\PYproject\SPAD\checkpoints\dgcnn_20260426_183404_669391_best.pth")
	parser.add_argument("--model", type=str, default="dgcnn", choices=["dgcnn", "pointnet2", "pointtransformer"])
	parser.add_argument("--sample-path", type=str, default="", help="Optional point cloud file path relative to data-root or absolute path")
	parser.add_argument("--sample-index", type=int, default=0, help="Fallback sample index if sample-path is empty")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--log-dir", type=str, default="logs")
	parser.add_argument("--box-space", type=str, default="absolute", choices=["absolute", "normalized"], help="How to interpret model box outputs")
	return parser


def main(argv=None) -> None:
	"""CLI entry point."""
	parser = build_parser()
	args = parser.parse_args(argv)
	run_single_test(args)


if __name__ == "__main__":
	main()
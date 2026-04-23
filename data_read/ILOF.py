from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

EPS = 1e-12
C_LIGHT = 3e8


@dataclass
class LCLOFConfig:
	input_dir: Path
	output_subdir: str = "lclof_results"

	# 阶段1: Rough-Denoising (Eq.9-13)
	k1: int = 30
	psi: float = 1e-7

	# 阶段2: Over-Denoising (Eq.14-16)
	k2: int = 8
	w: float = 0.4

	# 阶段3: Completion (Eq.17-20 & Def.1)
	k3: int = 15
	delta: float = 0.6

	# SPAD 物理参数 (需根据实际硬件标定)
	alpha: float = 1.5
	K_n: float = 0.005
	Ta: float = 1e-9
	Tp: float = 1e-9

	recursive: bool = False


def load_xyz_point_set(file_path: Path) -> np.ndarray:
	"""
	安全加载逗号分隔点云，仅保留前3列(X,Y,Z)。
	兼容多行矩阵或单行扁平化格式 (XYZI 或 XYZ)。
	"""
	try:
		data = np.loadtxt(file_path, delimiter=",")
	except Exception:
		# 兼容特殊换行/空格混排
		with open(file_path, "r", encoding="utf-8") as f:
			txt = f.read().replace("\n", ",").strip()
			data = np.fromstring(txt, sep=",")

	if data.ndim == 1:
		if data.size == 0: return np.zeros((0, 3), dtype=np.float64)
		if data.size % 4 == 0:
			data = data.reshape(-1, 4)
		elif data.size % 3 == 0:
			data = data.reshape(-1, 3)
		else:
			raise ValueError(f"Invalid data length in {file_path}: {data.size}")

	if data.shape[1] < 3:
		raise ValueError(f"Expected at least 3 columns, got {data.shape}")
	return data[:, :3].astype(np.float64, copy=False)


def _spad_trigger_prob(dists: np.ndarray, cfg: LCLOFConfig) -> np.ndarray:
	"""Eq.9-12: 向量化计算SPAD离散触发概率 P_s(I_j)"""
	K_s = cfg.alpha / (dists**2 + EPS)
	I_j = np.maximum(1, np.ceil(2 * dists / (C_LIGHT * cfg.Ta)).astype(int))
	return (1 - np.exp(-(cfg.K_n + K_s))) * np.exp(-(I_j - 1) * cfg.K_n)


def rough_denoising(p_raw: np.ndarray, cfg: LCLOFConfig) -> Tuple[np.ndarray, np.ndarray]:
	"""
	阶段1: Rough-Denoising (Eq.9-13)
	目标: 滤除孤立噪声，保留目标主体结构 P'
	"""
	if len(p_raw) < cfg.k1 + 1:
		return p_raw, np.ones(len(p_raw), dtype=bool)

	tree = cKDTree(p_raw)
	dists, _ = tree.query(p_raw, k=cfg.k1 + 1)
	dists = dists[:, 1:]  # 排除自身

	scores = np.mean(_spad_trigger_prob(dists, cfg), axis=1)
	# 自适应保护: 若全被滤除，自动放宽至5%分位
	if np.all(scores > cfg.psi):
		cfg.psi = float(np.percentile(scores, 5))
		print(f"  ⚠️ psi过严，已自动放宽至 {cfg.psi:.2e}")

	keep_mask = scores <= cfg.psi
	return p_raw[keep_mask], keep_mask


def over_denoising(p_rough: np.ndarray, cfg: LCLOFConfig) -> Tuple[np.ndarray, np.ndarray]:
	"""
	阶段2: Over-Denoising (Eq.14-16)
	目标: Z轴分层 + 自适应半径密度过滤，生成无噪基底 P''
	"""
	if len(p_rough) == 0:
		return p_rough, np.array([], dtype=bool)

	z_min, z_max = p_rough[:, 2].min(), p_rough[:, 2].max()
	z_range = max(z_max - z_min, EPS)

	# 物理层厚: Tp为时间，转为单程距离。防内存溢出加安全钳位
	layer_thickness = max(cfg.Tp * C_LIGHT / 2, z_range / 50.0)
	h = max(1, int(np.ceil(z_range / layer_thickness)))

	sort_idx = np.argsort(p_rough[:, 2])
	layers = np.array_split(p_rough[sort_idx], h)
	keep_mask = np.zeros(len(p_rough), dtype=bool)
	cur = 0

	for layer in layers:
		if len(layer) == 0:
			continue
		z_flag = layer[:, 2].min()
		K_s_flag = cfg.alpha / (z_flag**2 + EPS)
		I_flag = max(1, int(np.ceil(2 * z_flag / (C_LIGHT * cfg.Ta))))
		P_s_flag = (1 - np.exp(-(cfg.K_n + K_s_flag))) * np.exp(-(I_flag - 1) * cfg.K_n)
		zeta = cfg.w / (P_s_flag + EPS)  # Eq.16

		# 防半径过大导致全图搜索
		zeta = min(zeta, z_range * 0.1)

		tree_layer = cKDTree(layer)
		neighbors = tree_layer.query_ball_point(layer, r=zeta)
		layer_keep = np.array([len(n) >= cfg.k2 for n in neighbors])

		end = cur + len(layer)
		keep_mask[sort_idx[cur:end]] = layer_keep
		cur = end

	return p_rough[keep_mask], keep_mask


def completion_ilof(
	p_rough: np.ndarray,
	p_incomplete: np.ndarray,
	keep_mask_stage2: np.ndarray,
	cfg: LCLOFConfig,
) -> np.ndarray:
	"""
	阶段3: Point Clouds Completion (Eq.17-20 & Def.1)
	核心约束: F与P''的近邻搜索 仅基于 P'' 建树 (Def.1)
	"""
	F = p_rough[~keep_mask_stage2]
	if len(p_incomplete) == 0 or len(F) == 0:
		return p_incomplete

	tree_inc = cKDTree(p_incomplete)
	k3 = min(cfg.k3, len(p_incomplete))
	if k3 < 2: return p_incomplete

	# 预计算 P'' 中点的 lrd (Eq.19)
	dists_inc, idx_inc = tree_inc.query(p_incomplete, k=k3 + 1)
	kdist_inc = dists_inc[:, -1]
	lrd_q = k3 / (kdist_inc[idx_inc[:, 1:]].sum(axis=1) + EPS)

	# 计算 F 中点的 ILOF (Eq.17-20)
	dists_F, idx_F = tree_inc.query(F, k=k3)
	kdist_neighbors = kdist_inc[idx_F]
	rdist = np.maximum(kdist_neighbors, dists_F)  # Eq.17
	lrd_p = k3 / (rdist.sum(axis=1) + EPS)        # Eq.18

	mean_lrd_q = lrd_q[idx_F].mean(axis=1)
	ilof_scores = np.abs(mean_lrd_q / (lrd_p + EPS) - 1)  # Eq.20

	mask_F = ilof_scores <= cfg.delta
	if mask_F.any():
		return np.vstack([p_incomplete, F[mask_F]])
	return p_incomplete


def save_stage_xyz(file_path: Path, points: np.ndarray) -> None:
	if points.shape[0] == 0:
		np.savetxt(file_path, np.zeros((0, 3)), delimiter=",", fmt="%.6f", header="x,y,z", comments="")
		return
	np.savetxt(file_path, points, delimiter=",", fmt="%.6f", header="x,y,z", comments="")


def run_batch(cfg: LCLOFConfig) -> Path:
	out_dir = cfg.input_dir / cfg.output_subdir
	out_dir.mkdir(parents=True, exist_ok=True)
	stage_dirs = {
		"stage0": out_dir / "1_raw",
		"stage1": out_dir / "2_rough_denoised",
		"stage2": out_dir / "3_incomplete",
		"stage3": out_dir / "4_completed",
	}
	for d in stage_dirs.values(): d.mkdir(parents=True, exist_ok=True)

	txt_files = sorted([p for p in cfg.input_dir.glob("**/*.txt" if cfg.recursive else "*.txt") if p.is_file()])
	txt_files = [p for p in txt_files if cfg.output_subdir not in p.parts]
	if not txt_files:
		raise FileNotFoundError(f"No .txt files found in {cfg.input_dir}")

	summary_rows = []
	print(f"📂 开始处理: {cfg.input_dir}")

	for idx, fp in enumerate(txt_files, 1):
		try:
			p_raw = load_xyz_point_set(fp)
			p_rough, mask1 = rough_denoising(p_raw, cfg)
			p_incomplete, mask2 = over_denoising(p_rough, cfg)
			p_final = completion_ilof(p_rough, p_incomplete, mask2, cfg)

			base = fp.stem
			save_stage_xyz(stage_dirs["stage0"] / f"{base}_raw.txt", p_raw)
			save_stage_xyz(stage_dirs["stage1"] / f"{base}_rough.txt", p_rough)
			save_stage_xyz(stage_dirs["stage2"] / f"{base}_incomplete.txt", p_incomplete)
			save_stage_xyz(stage_dirs["stage3"] / f"{base}_completed.txt", p_final)

			row = {
				"file": fp.name,
				"raw": len(p_raw),
				"rough": len(p_rough),
				"incomplete": len(p_incomplete),
				"completed": len(p_final),
				"k1": cfg.k1, "psi": cfg.psi,
				"k2": cfg.k2, "w": cfg.w,
				"k3": cfg.k3, "delta": cfg.delta,
			}
			summary_rows.append(row)
			print(f"[{idx}/{len(txt_files)}] {fp.name} | {row['raw']}→{row['rough']}→{row['incomplete']}→{row['completed']}")
		except Exception as e:
			print(f"❌ {fp.name} 失败: {str(e)}")

	csv_path = out_dir / "summary.csv"
	fieldnames = list(summary_rows[0].keys()) if summary_rows else []
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(summary_rows)

	print(f"🎉 批量完成。结果已保存至: {out_dir}")
	return csv_path


def parse_args() -> LCLOFConfig:
	parser = argparse.ArgumentParser(description="LCLOF: SPAD点云去噪三阶段方法 (Eq.9-20)")
	parser.add_argument("--input_dir", type=str, default=r"D:\PYproject\SPADdata\20250430")
	parser.add_argument("--output_subdir", type=str, default="lclof_results")

	# 阶段参数
	parser.add_argument("--k1", type=int, default=30, help="Rough-denoising 近邻数")
	parser.add_argument("--psi", type=float, default=1e-7, help="Rough-denoising 概率阈值")
	parser.add_argument("--k2", type=int, default=8, help="Over-denoising 密度阈值")
	parser.add_argument("--w", type=float, default=0.4, help="自适应边界框系数")
	parser.add_argument("--k3", type=int, default=15, help="Completion 近邻数")
	parser.add_argument("--delta", type=float, default=0.6, help="ILOF 过滤阈值")

	# SPAD物理参数
	parser.add_argument("--alpha", type=float, default=1.5, help="激光雷达方程系数")
	parser.add_argument("--K_n", type=float, default=0.005, help="平均噪声光电子数")
	parser.add_argument("--Ta", type=float, default=1e-9, help="时间门间隔(s)")
	parser.add_argument("--Tp", type=float, default=1e-9, help="脉冲宽度(s)")
	parser.add_argument("--recursive", action="store_true")

	args = parser.parse_args()
	return LCLOFConfig(
		input_dir=Path(args.input_dir),
		output_subdir=args.output_subdir,
		k1=args.k1, psi=args.psi,
		k2=args.k2, w=args.w,
		k3=args.k3, delta=args.delta,
		alpha=args.alpha, K_n=args.K_n,
		Ta=args.Ta, Tp=args.Tp,
		recursive=args.recursive,
	)


if __name__ == "__main__":
	cfg = parse_args()
	run_batch(cfg)

	# 在未指定范围时，结果不受i控制，导致倾向于密集区域，即雾区为主要保留对象
	
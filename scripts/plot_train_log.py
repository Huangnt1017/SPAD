"""
SPAD 训练日志 → 曲线图脚本。

输入: scripts/train.py 输出的 ``logs/train_<model>_<timestamp>.log``。
日志中每个 epoch 写两行 (与 scripts/train.py run_epoch 末尾的 logger.info 一一对应):
    ``... | Epoch [E/T] | train_loss=X train_top1=Y train_top3=Z |
                          val_loss=A val_top1=B val_top3=C``
    ``... | Epoch [E/T] | train_box_iou=X train_box_l1=Y train_box_iou_loss=Z |
                          val_box_iou=A val_box_l1=B val_box_iou_loss=C``
这两行 epoch 字段一致, 用 ``Epoch [E/T]`` 做 key 把它们合并到同一个数据点。

输出:
    - 单文件: 2×3 子图 (loss / top1 / top3 / box_iou / box_l1 / box_iou_loss),
      每个子图叠 train + val 两条曲线。文件名 ``curve_<model>_<timestamp>.png``。
    - 多文件 (--compare): 同样 2×3 布局, 但每个子图把多个 log 的 val 曲线叠到一起
      横向对比, 文件名 ``compare_<时间戳>.png``。

用法:
    # 单个 log → 单张曲线图 (默认存到 log 同目录)
    python scripts/plot_train_log.py --log "logs/train_pointnet_20260522_003326_448064.log"

    # 多个 log 横向对比 val 曲线 (传 glob 或多个 --log)
    python scripts/plot_train_log.py --log "logs/train_pointnet_*.log" "logs/train_dgcnn_*.log" --compare

    # 指定输出目录
    python scripts/plot_train_log.py --log "logs/train_*.log" --output-dir D:/figs
"""

from __future__ import annotations

# 必须在 import torch / matplotlib 之前设置: Windows conda 环境 mkl + matplotlib
# 都链了 libiomp5md.dll, 进程退出时 OMP Error #15 会让图来不及落盘。
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import glob
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")    # 非交互后端: 避免无头环境弹窗失败
import matplotlib.pyplot as plt


# ============================================================================
# 解析正则: 与 scripts/train.py:run_epoch 末尾的 logger.info 格式一一对应
# ============================================================================
# 主指标行:
#   ... | Epoch [E/T] | train_loss=X train_top1=Y train_top3=Z |
#                       val_loss=A val_top1=B val_top3=C
_CLS_LINE = re.compile(
	r"Epoch \[(?P<epoch>\d+)/(?P<total>\d+)\] \| "
	r"train_loss=(?P<train_loss>[-+0-9.eE]+) "
	r"train_top1=(?P<train_top1>[-+0-9.eE]+) "
	r"train_top3=(?P<train_top3>[-+0-9.eE]+) \| "
	r"val_loss=(?P<val_loss>[-+0-9.eE]+) "
	r"val_top1=(?P<val_top1>[-+0-9.eE]+) "
	r"val_top3=(?P<val_top3>[-+0-9.eE]+)"
)
# Box 指标行:
#   ... | Epoch [E/T] | train_box_iou=X train_box_l1=Y train_box_iou_loss=Z |
#                       val_box_iou=A val_box_l1=B val_box_iou_loss=C
_BOX_LINE = re.compile(
	r"Epoch \[(?P<epoch>\d+)/(?P<total>\d+)\] \| "
	r"train_box_iou=(?P<train_box_iou>[-+0-9.eE]+) "
	r"train_box_l1=(?P<train_box_l1>[-+0-9.eE]+) "
	r"train_box_iou_loss=(?P<train_box_iou_loss>[-+0-9.eE]+) \| "
	r"val_box_iou=(?P<val_box_iou>[-+0-9.eE]+) "
	r"val_box_l1=(?P<val_box_l1>[-+0-9.eE]+) "
	r"val_box_iou_loss=(?P<val_box_iou_loss>[-+0-9.eE]+)"
)
# 从文件名提取 (model, timestamp), 例如:
#   train_graph_residual_20260522_050314_945951.log
#   train_pointnet_20260522_003326_448064.log
_FNAME_PARSE = re.compile(r"^train_(?P<model>.+?)_(?P<timestamp>\d{8}_\d{6}_\d{6})\.log$")


def parse_log(log_path: Path) -> Dict[str, object]:
	"""把一个 train log 解析成 {model, timestamp, epochs, metrics: {key: [values...]}}。

	缺失的指标会缺失对应 key (例如旧 log 没有 box 行就只有 cls 指标)。
	所有指标按 epoch 升序排列, 与 ``epochs`` 列表一一对应。

	Args:
		log_path: 待解析的 train log 文件路径。

	Returns:
		dict:
		    - model: str    — 文件名解析出的模型名 (如 ``"pointnet"``); 解析失败回退到 stem
		    - timestamp: str — 训练启动时间戳 (如 ``"20260522_003326_448064"``); 没识别到为 ""
		    - epochs: list[int] — 已记录的 epoch 序号 (从 1 开始)
		    - metrics: dict[str, list[float]] — 每个指标按 epoch 排序的数值列表; 缺失指标该 key 不存在
	"""
	stem = log_path.stem
	m = _FNAME_PARSE.match(log_path.name)
	model = m.group("model") if m else stem
	timestamp = m.group("timestamp") if m else ""

	# 用 dict[epoch] = partial metrics 收集, 最后排序展平。
	# 一个 epoch 的指标可能分散在两行 (cls / box), 用 epoch 做 key 合并。
	per_epoch: Dict[int, Dict[str, float]] = {}

	with open(log_path, "r", encoding="utf-8", errors="replace") as f:
		for line in f:
			m_cls = _CLS_LINE.search(line)
			if m_cls:
				ep = int(m_cls.group("epoch"))
				bucket = per_epoch.setdefault(ep, {})
				for key in ("train_loss", "train_top1", "train_top3",
				            "val_loss", "val_top1", "val_top3"):
					bucket[key] = float(m_cls.group(key))
				continue
			m_box = _BOX_LINE.search(line)
			if m_box:
				ep = int(m_box.group("epoch"))
				bucket = per_epoch.setdefault(ep, {})
				for key in ("train_box_iou", "train_box_l1", "train_box_iou_loss",
				            "val_box_iou", "val_box_l1", "val_box_iou_loss"):
					bucket[key] = float(m_box.group(key))

	# 按 epoch 排序, 把 dict[epoch][key] 重整为 metrics[key][i]
	epochs_sorted = sorted(per_epoch.keys())
	all_keys: set = set()
	for d in per_epoch.values():
		all_keys.update(d.keys())

	metrics: Dict[str, List[float]] = {k: [] for k in all_keys}
	for ep in epochs_sorted:
		bucket = per_epoch[ep]
		for k in all_keys:
			# 个别 epoch 可能某行没写 (训练崩溃断行), 用 NaN 占位让 matplotlib 跳过
			metrics[k].append(bucket.get(k, float("nan")))

	return {
		"model": model,
		"timestamp": timestamp,
		"epochs": epochs_sorted,
		"metrics": metrics,
	}


# ============================================================================
# 子图布局: 2×3 (loss / top1 / top3 / box_iou / box_l1 / box_iou_loss)
# ============================================================================
# (panel 标题, train 指标 key, val 指标 key, y 轴标签, 单位/范围提示)
_PANEL_SPEC: List[Tuple[str, str, str, str]] = [
	("Loss",         "train_loss",         "val_loss",         "loss"),
	("Top-1 Acc",    "train_top1",         "val_top1",         "accuracy"),
	("Top-3 Acc",    "train_top3",         "val_top3",         "accuracy"),
	("Box IoU",      "train_box_iou",      "val_box_iou",      "iou"),
	("Box L1",       "train_box_l1",       "val_box_l1",       "L1 (normalized)"),
	("Box IoU Loss", "train_box_iou_loss", "val_box_iou_loss", "1 - iou"),
]


def plot_single(parsed: Dict[str, object], save_path: Path) -> None:
	"""单 log → 2×3 子图, 每图叠 train (实线) + val (虚线)。

	Args:
		parsed: ``parse_log`` 返回结构。
		save_path: 输出 PNG 路径; 上级目录会自动创建。
	"""
	epochs = parsed["epochs"]
	metrics = parsed["metrics"]
	model = parsed["model"]
	timestamp = parsed["timestamp"]

	fig, axes = plt.subplots(2, 3, figsize=(15, 8))
	axes = axes.flatten()

	for ax, (title, key_tr, key_va, ylabel) in zip(axes, _PANEL_SPEC):
		tr = metrics.get(key_tr)
		va = metrics.get(key_va)
		# 没记录到该指标时, 隐藏 panel 避免空轴
		if tr is None and va is None:
			ax.set_visible(False)
			continue
		if tr is not None:
			ax.plot(epochs, tr, label="train", color="tab:blue", linewidth=1.5)
		if va is not None:
			ax.plot(epochs, va, label="val", color="tab:orange", linewidth=1.5, linestyle="--")
		ax.set_title(title)
		ax.set_xlabel("epoch")
		ax.set_ylabel(ylabel)
		ax.grid(alpha=0.3)
		ax.legend(loc="best", fontsize=8)

	fig.suptitle(f"{model}  (run @ {timestamp})  —  {len(epochs)} epochs", fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, 0.96))

	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(save_path, dpi=140, bbox_inches="tight")
	plt.close(fig)


def plot_compare(parsed_list: List[Dict[str, object]], save_path: Path,
                 split: str = "val") -> None:
	"""多个 log 横向对比, 每子图叠多模型的同名指标 (默认 val)。

	Args:
		parsed_list: 解析后的 log 列表。
		save_path: 输出 PNG。
		split: "val" 或 "train"; 默认 "val" (主关心泛化性能)。
	"""
	fig, axes = plt.subplots(2, 3, figsize=(15, 8))
	axes = axes.flatten()

	# 颜色按 parsed_list 顺序分配, tab10 足够 (最多 10 个)
	colors = plt.get_cmap("tab10").colors

	for ax, (title, key_tr, key_va, ylabel) in zip(axes, _PANEL_SPEC):
		key = key_va if split == "val" else key_tr
		any_data = False
		for i, parsed in enumerate(parsed_list):
			vals = parsed["metrics"].get(key)
			if vals is None:
				continue
			ax.plot(parsed["epochs"], vals,
			        label=parsed["model"], color=colors[i % len(colors)], linewidth=1.5)
			any_data = True
		if not any_data:
			ax.set_visible(False)
			continue
		ax.set_title(f"{title} ({split})")
		ax.set_xlabel("epoch")
		ax.set_ylabel(ylabel)
		ax.grid(alpha=0.3)
		ax.legend(loc="best", fontsize=8)

	fig.suptitle(f"Comparison ({split}): {len(parsed_list)} runs", fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, 0.96))

	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(save_path, dpi=140, bbox_inches="tight")
	plt.close(fig)


# ============================================================================
# CLI
# ============================================================================

def _expand_logs(patterns: List[str]) -> List[Path]:
	"""把 --log 里的 glob / 普通路径展开成实际存在的 log 文件列表。"""
	out: List[Path] = []
	for pat in patterns:
		# glob 直接展开 (无通配符时 glob 也会返回原路径如果存在)
		matched = sorted(glob.glob(pat))
		if matched:
			for p in matched:
				path = Path(p)
				if path.is_file():
					out.append(path.resolve())
		else:
			# 当 glob 没匹配到时, 也可能是用户传了绝对路径; 直接当文件检查
			path = Path(pat)
			if path.is_file():
				out.append(path.resolve())
	# 去重保留顺序
	seen: set = set()
	uniq: List[Path] = []
	for p in out:
		if p not in seen:
			seen.add(p)
			uniq.append(p)
	return uniq


def build_parser() -> argparse.ArgumentParser:
	"""SPAD train log → curve PNG CLI。"""
	parser = argparse.ArgumentParser(description="Convert SPAD train log(s) into curve plots.")
	parser.add_argument(
		"--log", type=str, nargs="+", required=True,
		help="一个或多个 log 文件路径; 支持 glob (例如 logs/train_*.log)",
	)
	parser.add_argument(
		"--output-dir", type=str, default="",
		help="输出 PNG 目录; 留空时单文件存到 log 同目录, --compare 模式存到 logs/",
	)
	parser.add_argument(
		"--compare", action="store_true",
		help="把多个 log 横向对比绘制到同一张图 (默认每个 log 单独出图)",
	)
	parser.add_argument(
		"--split", type=str, default="val", choices=["train", "val"],
		help="--compare 模式下画 train 还是 val 曲线 (默认 val)",
	)
	return parser


def main(argv: Optional[List[str]] = None) -> None:
	logging.basicConfig(level=logging.INFO, format="%(message)s")
	logger = logging.getLogger("plot_train_log")

	parser = build_parser()
	args = parser.parse_args(argv)

	logs = _expand_logs(args.log)
	if not logs:
		logger.error("No log files matched: %s", args.log)
		sys.exit(1)
	logger.info("Matched %d log file(s):", len(logs))
	for p in logs:
		logger.info("  %s", p)

	parsed_list = [parse_log(p) for p in logs]
	# 过滤掉完全没 epoch 数据的 (训练第 1 个 epoch 都没结束就崩了的 log)
	parsed_list = [d for d in parsed_list if d["epochs"]]
	if not parsed_list:
		logger.error("None of the matched log files contained any Epoch records.")
		sys.exit(2)

	default_out_dir = Path(args.output_dir).resolve() if args.output_dir else None
	timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")

	if args.compare:
		out_dir = default_out_dir or Path("logs").resolve()
		save_path = out_dir / f"compare_{timestamp_now}.png"
		plot_compare(parsed_list, save_path, split=args.split)
		logger.info("Saved comparison plot to %s", save_path)
	else:
		for parsed, log_path in zip(parsed_list, logs):
			out_dir = default_out_dir or log_path.parent
			save_path = out_dir / f"curve_{parsed['model']}_{parsed['timestamp']}.png"
			plot_single(parsed, save_path)
			logger.info("Saved %s curve to %s", parsed["model"], save_path)


if __name__ == "__main__":
	# 用法示例 (PowerShell):
	#   $env:PYTHONPATH = "D:\PYproject\SPAD"
	#   # 单 log 出图 (PNG 默认存到 log 同目录, 文件名 curve_<model>_<timestamp>.png)
	#   & "D:\anaconda3\envs\pytorch\python.exe" "D:\PYproject\SPAD\scripts\plot_train_log.py" `
	#       --log "D:\PYproject\SPAD\logs\train_pointnet_20260522_003326_448064.log"
	#
	#   # 多 log 横向对比 (PNG 存到 logs/compare_<时间戳>.png, 默认画 val 曲线)
	#   & "D:\anaconda3\envs\pytorch\python.exe" "D:\PYproject\SPAD\scripts\plot_train_log.py" `
	#       --log "D:\PYproject\SPAD\logs\train_*.log" --compare
	#
	#   # 指定输出目录, 比较 train 曲线
	#   & "D:\anaconda3\envs\pytorch\python.exe" "D:\PYproject\SPAD\scripts\plot_train_log.py" `
	#       --log "logs/train_pointnet_*.log" "logs/train_graph_residual_*.log" `
	#       --compare --split train --output-dir D:/figs
	#
	# 输出:
	#   单 log 模式:       curve_<model>_<timestamp>.png      2×3 子图 (loss/top1/top3/box_iou/box_l1/box_iou_loss), 每图叠 train+val
	#   --compare 模式:    compare_<本次时间戳>.png            同样 2×3, 每图叠多个模型曲线 (val by default, --split train 切换)
	main()

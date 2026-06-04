"""
SPAD 训练日志 → 曲线图脚本。

输入: scripts/train.py 输出的 ``logs/CLS/train_<model>_<timestamp>.log``。
日志中每个 epoch 写两行 (与 scripts/train.py run_epoch 末尾的 logger.info 一一对应):
    ``... | Epoch [E/T] | train_loss=X train_top1=Y train_top3=Z |
                          val_loss=A val_top1=B val_top3=C``
    ``... | Epoch [E/T] | train_z_mae=X train_box_depth=Y |
                          val_z_mae=A val_box_depth=B``
这两行 epoch 字段一致, 用 ``Epoch [E/T]`` 做 key 把它们合并到同一个数据点。

输出:
    - 单 log 模式: loss / top1 / top3 / z_mae / box_depth 子图,
      每个子图叠 train + val 两条曲线。文件名 ``curve_<model>_<timestamp>.png``。
    - 多 log 对比模式: 每个子图把多个 log 的 val 曲线叠到一起
      横向对比, 文件名 ``compare_<时间戳>.png``。

使用方式: 直接在 if __name__ 中修改 log_patterns 列表即可运行。
"""

from __future__ import annotations

# Windows conda 环境 mkl + matplotlib 都链了 libiomp5md.dll,
# 进程退出时 OMP Error #15 会让图来不及落盘。
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import glob
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# ============================================================================
# 解析正则: 与 scripts/train.py:run_epoch 末尾的 logger.info 格式一一对应
# ============================================================================
_CLS_LINE = re.compile(
    r"Epoch \[(?P<epoch>\d+)/(?P<total>\d+)\] \| "
    r"train_loss=(?P<train_loss>[-+0-9.eE]+) "
    r"train_top1=(?P<train_top1>[-+0-9.eE]+) "
    r"train_top3=(?P<train_top3>[-+0-9.eE]+) \| "
    r"val_loss=(?P<val_loss>[-+0-9.eE]+) "
    r"val_top1=(?P<val_top1>[-+0-9.eE]+) "
    r"val_top3=(?P<val_top3>[-+0-9.eE]+)"
)
_BOX_LINE = re.compile(
    r"Epoch \[(?P<epoch>\d+)/(?P<total>\d+)\] \| "
    r"train_z_mae=(?P<train_z_mae>[-+0-9.eE]+) "
    r"train_box_depth=(?P<train_box_depth>[-+0-9.eE]+) \| "
    r"val_z_mae=(?P<val_z_mae>[-+0-9.eE]+) "
    r"val_box_depth=(?P<val_box_depth>[-+0-9.eE]+)"
)
# 从文件名提取 (model, timestamp), 例如:
#   train_graph_residual_20260522_050314_945951.log
_FNAME_PARSE = re.compile(r"^train_(?P<model>.+?)_(?P<timestamp>\d{8}_\d{6}_\d{6})\.log$")


def parse_log(log_path: Path) -> Dict[str, object]:
    """把一个 train log 解析成 {model, timestamp, epochs, metrics: {key: [values...]}}。

    Args:
        log_path: 待解析的 train log 文件路径。

    Returns:
        dict:
            - model: str
            - timestamp: str
            - epochs: list[int]
            - metrics: dict[str, list[float]]
    """
    stem = log_path.stem
    m = _FNAME_PARSE.match(log_path.name)
    model = m.group("model") if m else stem
    timestamp = m.group("timestamp") if m else ""

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
                for key in ("train_z_mae", "train_box_depth", "val_z_mae", "val_box_depth"):
                    bucket[key] = float(m_box.group(key))

    epochs_sorted = sorted(per_epoch.keys())
    all_keys: set = set()
    for d in per_epoch.values():
        all_keys.update(d.keys())

    metrics: Dict[str, List[float]] = {k: [] for k in all_keys}
    for ep in epochs_sorted:
        bucket = per_epoch[ep]
        for k in all_keys:
            metrics[k].append(bucket.get(k, float("nan")))

    return {
        "model": model,
        "timestamp": timestamp,
        "epochs": epochs_sorted,
        "metrics": metrics,
    }


# ============================================================================
# 子图布局: loss / top1 / top3 / z_mae / box_depth
# ============================================================================
_PANEL_SPEC: List[Tuple[str, str, str, str]] = [
    ("Loss",         "train_loss",         "val_loss",         "loss"),
    ("Top-1 Acc",    "train_top1",         "val_top1",         "accuracy"),
    ("Top-3 Acc",    "train_top3",         "val_top3",         "accuracy"),
    ("Z-MAE",        "train_z_mae",        "val_z_mae",        "depth error"),
    ("Box Depth",    "train_box_depth",    "val_box_depth",    "Soft-histogram"),
]


def plot_single(parsed: Dict[str, object], save_path: Path) -> None:
    """单 log -> 2x3 子图, 每图叠 train (实线) + val (虚线)。

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
    for ax in axes[len(_PANEL_SPEC):]:
        ax.set_visible(False)

    fig.suptitle(f"{model}  (run @ {timestamp})  —  {len(epochs)} epochs", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_compare(parsed_list: List[Dict[str, object]], save_path: Path,
                 split: str = "val") -> None:
    """多个 log 横向对比, 每子图叠多模型的同名指标 (默认 val)。

    Args:
        parsed_list: 解析后的 log 列表。
        save_path: 输出 PNG。
        split: "val" 或 "train"; 默认 "val"。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

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
    for ax in axes[len(_PANEL_SPEC):]:
        ax.set_visible(False)

    fig.suptitle(f"Comparison ({split}): {len(parsed_list)} runs", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _expand_logs(patterns: List[str]) -> List[Path]:
    """把 glob / 普通路径展开成实际存在的 log 文件列表。"""
    out: List[Path] = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        if matched:
            for p in matched:
                path = Path(p)
                if path.is_file():
                    out.append(path.resolve())
        else:
            path = Path(pat)
            if path.is_file():
                out.append(path.resolve())
    seen: set = set()
    uniq: List[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("plot_train_log")

    # ====================================================================
    # 单 log 出图: 每个 log 分别画 2x3 子图 (train+val 叠在一起)
    # 输出: curve_<model>_<timestamp>.png, 存到 log 同目录
    # ====================================================================
    log_patterns = [
        r"D:\PYproject\SPAD\logs\CLS\train_graph_residual_20260522_050314_945951.log",
    ]
    output_dir = ""  # 留空则存到 log 同目录; 指定路径如 r"D:\figs"

    logs = _expand_logs(log_patterns)
    if not logs:
        logger.error("No log files matched: %s", log_patterns)
        sys.exit(1)
    logger.info("Matched %d log file(s):", len(logs))
    for p in logs:
        logger.info("  %s", p)

    parsed_list = [parse_log(p) for p in logs]
    parsed_list = [d for d in parsed_list if d["epochs"]]
    if not parsed_list:
        logger.error("None of the matched log files contained any Epoch records.")
        sys.exit(2)

    out_dir = Path(output_dir).resolve() if output_dir else None
    for parsed, log_path in zip(parsed_list, logs):
        save_dir = out_dir or log_path.parent
        save_path = save_dir / f"curve_{parsed['model']}_{parsed['timestamp']}.png"
        plot_single(parsed, save_path)
        logger.info("Saved %s curve to %s", parsed["model"], save_path)

    # ====================================================================
    # 多 log 横向对比: 多个模型的 val 曲线叠到同一张图
    # 输出: compare_<时间戳>.png, 默认存到 logs/CLS/ 目录
    # ====================================================================
    # compare_patterns = [
    #     r"D:\PYproject\SPAD\logs\CLS\train_pointnet_20260522_003326_448064.log",
    #     r"D:\PYproject\SPAD\logs\CLS\train_graph_residual_20260522_050314_945951.log",
    # ]
    # compare_output_dir = ""  # 留空存到 logs/CLS/; 指定路径如 r"D:\figs"
    # compare_split = "val"    # "val" 或 "train"
    #
    # compare_logs = _expand_logs(compare_patterns)
    # if compare_logs:
    #     compare_parsed = [parse_log(p) for p in compare_logs]
    #     compare_parsed = [d for d in compare_parsed if d["epochs"]]
    #     if compare_parsed:
    #         cmp_dir = Path(compare_output_dir).resolve() if compare_output_dir else Path("logs/CLS").resolve()
    #         cmp_path = cmp_dir / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    #         plot_compare(compare_parsed, cmp_path, split=compare_split)
    #         logger.info("Saved comparison plot to %s", cmp_path)

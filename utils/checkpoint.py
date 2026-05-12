"""
SPAD 模型 checkpoint 存储与恢复工具。

提供训练状态持久化功能，将模型权重、优化器状态、
学习率调度器状态及其他训练元数据保存到磁盘，
支持训练中断后的精确续训。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_top1: float,
    class_to_idx: Dict[str, int],
    args: argparse.Namespace,
) -> None:
    """保存完整训练状态到 checkpoint 文件。

    保存内容包含模型参数、优化器和学习率调度器状态、
    分类映射和命令行参数，确保可以完整恢复训练。

    Args:
        path: checkpoint 保存路径（.pth）。
        model: 当前模型实例。
        optimizer: 当前优化器实例。
        scheduler: 学习率调度器实例。
        epoch: 当前 epoch 编号。
        best_val_top1: 历史最佳验证集 Top-1 准确率。
        class_to_idx: 类别到索引的映射字典。
        args: 训练命令行参数。

    Returns:
        None
    """
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_top1": best_val_top1,
        "class_to_idx": class_to_idx,
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """从 checkpoint 恢复训练状态。

    Args:
        path: checkpoint 文件路径。
        model: 待加载权重的模型实例。
        optimizer: 待恢复状态的优化器实例（可选）。
        scheduler: 待恢复状态的学习率调度器（可选）。
        device: 目标设备，默认使用模型当前设备。

    Returns:
        包含 epoch, best_val_top1, class_to_idx, args 的字典。
    """
    map_location = device if device is not None else next(model.parameters()).device
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_val_top1": checkpoint.get("best_val_top1", 0.0),
        "class_to_idx": checkpoint.get("class_to_idx", {}),
        "args": checkpoint.get("args", {}),
    }

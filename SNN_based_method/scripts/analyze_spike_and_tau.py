"""分析 SNN/PLIF checkpoint 的 tau、参数量和已记录的 spike 指标。

CLI example:
    python SNN_based_method/scripts/analyze_spike_and_tau.py --checkpoint checkpoints/SNN/train_xxx/best.pth --output SNN_based_method/artifacts/spike_tau_analysis.csv

Non-CLI example:
    python SNN_based_method/scripts/analyze_spike_and_tau.py

说明:
    PLIF 的可学习参数名通常以 ``.w`` 结尾, 对应 tau = 1 / sigmoid(w)。
    本脚本不读取训练数据, 只分析 checkpoint 中已有权重和 metrics, 适合生成论文中的
    tau 分布、参数规模和 spike_rate 表格。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from SNN_based_method.config.SNN_config import SNNConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ScriptConfig:
    """checkpoint 可解释性分析配置。"""

    checkpoints: list[Path]
    checkpoint_root: Path | None
    pattern: str
    output_path: Path
    build_model: bool = True


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(description="分析 SNN/PLIF checkpoint 的 tau 和 spike 指标")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="待分析 checkpoint 路径; 可重复传入",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=None, help="递归扫描 checkpoint 的根目录")
    parser.add_argument("--pattern", default="best.pth", help="配合 --checkpoint-root 使用的 glob, 如 best.pth 或 *.pth")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("SNN_based_method/artifacts/spike_tau_analysis.csv"),
        help="输出 CSV 路径; 同名 .json 会同步写出",
    )
    parser.add_argument("--no-build-model", action="store_true", help="不根据 config 构建模型, 只统计 state_dict 张量")
    return parser


def _resolve(path: str | Path) -> Path:
    """按项目根目录解析相对路径。"""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _find_checkpoints(config: ScriptConfig) -> list[Path]:
    """解析显式 checkpoint 与根目录扫描结果。"""
    paths = [_resolve(path) for path in config.checkpoints]
    if config.checkpoint_root is not None:
        root = _resolve(config.checkpoint_root)
        if not root.is_dir():
            raise NotADirectoryError(f"checkpoint root not found: {root}")
        paths.extend(sorted(root.rglob(config.pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        seen.add(path)
        unique.append(path)
    if not unique:
        raise ValueError("pass at least one --checkpoint or --checkpoint-root")
    return unique


def _checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """提取 checkpoint 中的模型 state_dict。"""
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("checkpoint model state must be a dict")
    return state


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """把嵌套字典展开为点分键。"""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value, ensure_ascii=False)
        else:
            flat[name] = value
    return flat


def _tensor_to_float_list(value: torch.Tensor) -> list[float]:
    """把任意形状张量转换为 float 列表。"""
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _tau_values_from_state(state: dict[str, torch.Tensor]) -> tuple[dict[str, list[float]], list[float]]:
    """从 state_dict 中提取 PLIF tau。"""
    by_key: dict[str, list[float]] = {}
    all_tau: list[float] = []
    for key, value in state.items():
        if not key.endswith(".w") or not isinstance(value, torch.Tensor):
            continue
        raw_values = torch.sigmoid(value.detach().float()).clamp_min(1.0e-8)
        tau_tensor = 1.0 / raw_values
        values = _tensor_to_float_list(tau_tensor)
        by_key[key] = values
        all_tau.extend(values)
    return by_key, all_tau


def _stats(values: list[float], prefix: str) -> dict[str, float | int | None]:
    """计算一组数值的基础统计量。"""
    if not values:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / max(len(values), 1)
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_mean": mean_value,
        f"{prefix}_std": math.sqrt(variance),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def _state_tensor_count(state: dict[str, torch.Tensor]) -> int:
    """统计 state_dict 中所有张量元素数量。"""
    return int(sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor)))


def _model_parameter_count(config_dict: dict[str, Any]) -> tuple[int | None, int | None]:
    """根据 checkpoint config 构建模型并统计参数量。"""
    try:
        if "split_ratios" in config_dict:
            config_dict = dict(config_dict)
            config_dict["split_ratios"] = tuple(config_dict["split_ratios"])
        cfg = SNNConfig(**config_dict)
        model = cfg.build_model()
    except Exception:
        return None, None
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def analyze_checkpoint(path: Path, *, build_model: bool = True) -> dict[str, Any]:
    """分析单个 checkpoint 并返回一行记录。"""
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dict: {path}")
    state = _checkpoint_state(checkpoint)
    config = checkpoint.get("config", {})
    metrics = checkpoint.get("metrics", {})
    tau_by_key, tau_values = _tau_values_from_state(state)

    row: dict[str, Any] = {
        "checkpoint": str(path),
        "run_dir": str(path.parent),
        "run_name": path.parent.name,
        "checkpoint_name": path.name,
        "epoch": checkpoint.get("epoch"),
        "state_tensor_count": _state_tensor_count(state),
        "tau_keys": json.dumps(sorted(tau_by_key), ensure_ascii=False),
    }
    row.update(_stats(tau_values, "tau"))

    if isinstance(config, dict):
        for key in (
            "model_backend",
            "spike_mode",
            "spike_tau",
            "spike_v_threshold",
            "C",
            "num_blocks",
            "chunk_size",
            "pages_per_group",
            "encoding_mode",
        ):
            row[f"config.{key}"] = config.get(key)
        if build_model:
            total_params, trainable_params = _model_parameter_count(config)
            row["parameter_count"] = total_params
            row["trainable_parameter_count"] = trainable_params

    if isinstance(metrics, dict):
        flat_metrics = _flatten(metrics)
        for key, value in flat_metrics.items():
            if (
                key in {"train_loss", "val_loss", "best_val_loss", "lr"}
                or key.startswith("val_metrics.")
                or "spike_rate" in key
            ):
                row[f"metrics.{key}"] = value
    return row


def _write_outputs(rows: list[dict[str, Any]], output_path: Path) -> None:
    """写出 CSV 和 JSON。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as file_obj:
        json.dump(rows, file_obj, indent=2, ensure_ascii=False)


def run_with_config(config: ScriptConfig) -> list[dict[str, Any]]:
    """执行 checkpoint 分析。"""
    checkpoints = _find_checkpoints(config)
    rows = [analyze_checkpoint(path, build_model=config.build_model) for path in checkpoints]
    output_path = _resolve(config.output_path)
    _write_outputs(rows, output_path)
    print(f"checkpoints={len(checkpoints)}")
    print(f"csv={output_path}")
    print(f"json={output_path.with_suffix('.json')}")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = ScriptConfig(
        checkpoints=[Path(path) for path in args.checkpoint],
        checkpoint_root=args.checkpoint_root,
        pattern=args.pattern,
        output_path=args.output,
        build_model=not bool(args.no_build_model),
    )
    run_with_config(config)
    return 0


def main_without_cli() -> None:
    """无命令行参数运行时的可编辑入口。"""
    # ===== Editable parameters =====
    checkpoint_root = Path("checkpoints/SNN")
    pattern = "best.pth"
    output_path = Path("SNN_based_method/artifacts/spike_tau_analysis.csv")

    # ===== Intermediate variables =====
    config = ScriptConfig(
        checkpoints=[],
        checkpoint_root=checkpoint_root,
        pattern=pattern,
        output_path=output_path,
    )
    run_with_config(config)


if __name__ == "__main__":
    # Usage examples:
    #   python SNN_based_method/scripts/analyze_spike_and_tau.py
    #       Run main_without_cli(), scanning checkpoints/SNN/**/best.pth.
    #   python SNN_based_method/scripts/analyze_spike_and_tau.py --checkpoint checkpoints/SNN/train_xxx/best.pth --output SNN_based_method/artifacts/tau.csv
    #       Analyze one checkpoint.
    #   python SNN_based_method/scripts/analyze_spike_and_tau.py --checkpoint-root checkpoints/SNN --pattern best.pth
    #       Analyze all best checkpoints under a root.
    #
    # Outputs:
    #   CSV/JSON rows with tau summary, parameter count, selected metrics and spike rates.
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()

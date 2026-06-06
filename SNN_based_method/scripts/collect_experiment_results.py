"""汇总 SPAD SNN 对比实验的配置、checkpoint 指标和测试 summary。

CLI example:
    python SNN_based_method/scripts/collect_experiment_results.py --checkpoint-root checkpoints/SNN --log-root logs/SNN --output SNN_based_method/artifacts/chapter_results.csv

Non-CLI example:
    python SNN_based_method/scripts/collect_experiment_results.py

说明:
    checkpoint run 目录应包含 config.json, 以及可选的 best.pth / last.pth。
    test.py / test1.py 生成的 summary.json 会作为 test_summary 行写入结果表。
    输出 CSV 适合直接导入论文表格或统计软件, 同时会生成同名 JSON 方便保留嵌套字段。
"""

from __future__ import annotations

import argparse
import csv
import json
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ScriptConfig:
    """结果汇总脚本运行配置。"""

    checkpoint_root: Path
    log_root: Path
    output_path: Path
    include_checkpoints: bool = True
    pattern: str = "*"


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(description="汇总 SPAD SNN 实验结果为 CSV/JSON")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/SNN"), help="checkpoint 根目录")
    parser.add_argument("--log-root", type=Path, default=Path("logs/SNN"), help="日志/test summary 根目录")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("SNN_based_method/artifacts/experiment_results.csv"),
        help="输出 CSV 路径; 同名 .json 会同步写出",
    )
    parser.add_argument("--pattern", default="*", help="checkpoint run 目录 glob 过滤, 例如 'chapter4_*'")
    parser.add_argument("--no-checkpoints", action="store_true", help="不加载 best.pth/last.pth, 只汇总 config 和 summary")
    return parser


def _resolve(path: str | Path) -> Path:
    """按项目根目录解析相对路径。"""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象。"""
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return data


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """把嵌套字典展开为点分键。"""
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten(value, flat_key))
        elif isinstance(value, (list, tuple)):
            flattened[flat_key] = json.dumps(value, ensure_ascii=False)
        else:
            flattened[flat_key] = value
    return flattened


def _safe_torch_load(path: Path) -> dict[str, Any]:
    """读取 checkpoint, 仅保留顶层元数据。"""
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dict: {path}")
    return checkpoint


def _checkpoint_row(run_dir: Path, checkpoint_name: str, config: dict[str, Any]) -> dict[str, Any] | None:
    """从单个 checkpoint 文件提取一行结果。"""
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.is_file():
        return None
    checkpoint = _safe_torch_load(checkpoint_path)
    metrics = checkpoint.get("metrics", {})
    row: dict[str, Any] = {
        "source": "checkpoint",
        "checkpoint_name": checkpoint_name,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
    }
    row.update({f"config.{key}": value for key, value in _flatten(config).items()})
    if isinstance(metrics, dict):
        row.update({f"metrics.{key}": value for key, value in _flatten(metrics).items()})
    return row


def _config_only_row(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """生成只包含配置的 run 行。"""
    row: dict[str, Any] = {
        "source": "checkpoint_config",
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
    }
    row.update({f"config.{key}": value for key, value in _flatten(config).items()})
    return row


def collect_checkpoint_rows(checkpoint_root: Path, pattern: str, include_checkpoints: bool) -> list[dict[str, Any]]:
    """扫描 checkpoint run 目录并汇总配置与 checkpoint 指标。"""
    rows: list[dict[str, Any]] = []
    if not checkpoint_root.is_dir():
        return rows

    for run_dir in sorted(path for path in checkpoint_root.glob(pattern) if path.is_dir()):
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            continue
        config = _read_json(config_path)
        if not include_checkpoints:
            rows.append(_config_only_row(run_dir, config))
            continue
        found_checkpoint = False
        for checkpoint_name in ("best.pth", "last.pth"):
            row = _checkpoint_row(run_dir, checkpoint_name, config)
            if row is not None:
                rows.append(row)
                found_checkpoint = True
        if not found_checkpoint:
            rows.append(_config_only_row(run_dir, config))
    return rows


def collect_summary_rows(log_root: Path) -> list[dict[str, Any]]:
    """扫描 test.py / test1.py 的 summary.json。"""
    rows: list[dict[str, Any]] = []
    if not log_root.is_dir():
        return rows

    for summary_path in sorted(log_root.rglob("summary.json")):
        summary = _read_json(summary_path)
        run_dir = summary_path.parent
        row: dict[str, Any] = {
            "source": "test_summary",
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "summary_path": str(summary_path),
        }
        row.update(_flatten(summary))
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """写出 CSV 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    """写出 JSON 文件。"""
    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(rows, file_obj, indent=2, ensure_ascii=False)


def run_with_config(config: ScriptConfig) -> list[dict[str, Any]]:
    """执行结果汇总并写出文件。"""
    checkpoint_root = _resolve(config.checkpoint_root)
    log_root = _resolve(config.log_root)
    output_path = _resolve(config.output_path)

    rows = collect_checkpoint_rows(checkpoint_root, config.pattern, config.include_checkpoints)
    rows.extend(collect_summary_rows(log_root))
    if not rows:
        raise RuntimeError(
            f"no experiment rows found: checkpoint_root={checkpoint_root}, log_root={log_root}"
        )

    _write_csv(rows, output_path)
    _write_json(rows, output_path)
    print(f"rows={len(rows)}")
    print(f"csv={output_path}")
    print(f"json={output_path.with_suffix('.json')}")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = ScriptConfig(
        checkpoint_root=args.checkpoint_root,
        log_root=args.log_root,
        output_path=args.output,
        include_checkpoints=not bool(args.no_checkpoints),
        pattern=args.pattern,
    )
    run_with_config(config)
    return 0


def main_without_cli() -> None:
    """无命令行参数运行时的可编辑入口。"""
    # ===== Editable parameters =====
    checkpoint_root = Path("checkpoints/SNN")
    log_root = Path("logs/SNN")
    output_path = Path("SNN_based_method/artifacts/experiment_results.csv")

    # ===== Intermediate variables =====
    config = ScriptConfig(
        checkpoint_root=checkpoint_root,
        log_root=log_root,
        output_path=output_path,
    )
    run_with_config(config)


if __name__ == "__main__":
    # Usage examples:
    #   python SNN_based_method/scripts/collect_experiment_results.py
    #       Run main_without_cli(), using editable parameters above.
    #   python SNN_based_method/scripts/collect_experiment_results.py --checkpoint-root checkpoints/SNN --log-root logs/SNN --output SNN_based_method/artifacts/chapter_results.csv
    #       Collect config/checkpoint/test-summary rows into CSV and JSON.
    #
    # Outputs:
    #   CSV table and same-name JSON file containing flattened experiment records.
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()

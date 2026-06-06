"""按 JSON 实验表生成或执行 SPAD SNN 对比实验命令。

CLI example:
    python SNN_based_method/scripts/run_experiment_grid.py --spec SNN_based_method/experiments/chapter_grid.json --dry-run

Non-CLI example:
    python SNN_based_method/scripts/run_experiment_grid.py

JSON 示例:
    {
      "python": "D:/Anaconda3/envs/torchnew/python.exe",
      "script": "SNN_based_method/scripts/train.py",
      "run_name_prefix": "chapter4",
      "base_args": {
        "epochs": 20,
        "batch-size": 8,
        "tf32": true,
        "cudnn-benchmark": true,
        "cuda-prefetch": true
      },
      "experiments": [
        {"name": "ann_gate", "args": {"model-backend": "ann_gate", "spike-mode": "lif"}},
        {"name": "snn_lif", "args": {"model-backend": "new", "spike-mode": "lif"}},
        {"name": "snn_plif", "args": {"model-backend": "new", "spike-mode": "plif", "spike-tau": 2.0}}
      ]
    }

说明:
    base_args 和每个 experiment.args 使用命令行长参数名, 可以带或不带前导 ``--``。
    bool True 会转换为开关参数, bool False 和 None 会被跳过。
    list/tuple 会转换为 ``--key value1 value2 ...``。
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ExperimentCommand:
    """一个展开后的实验命令。"""

    name: str
    argv: list[str]


@dataclass
class ScriptConfig:
    """实验矩阵脚本运行配置。"""

    spec_path: Path
    dry_run: bool = True
    execute: bool = False
    limit: int | None = None
    continue_on_error: bool = False


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(description="生成或执行 SPAD SNN 对比实验矩阵")
    parser.add_argument("--spec", type=Path, required=True, help="实验 JSON 配置文件")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令, 不执行")
    parser.add_argument("--execute", action="store_true", help="实际顺序执行所有命令")
    parser.add_argument("--limit", type=int, default=None, help="最多展开/执行前 N 个实验")
    parser.add_argument("--continue-on-error", action="store_true", help="单个实验失败后继续执行后续实验")
    return parser


def _resolve_path(path: str | Path) -> Path:
    """按项目根目录解析相对路径。"""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _load_spec(path: Path) -> dict[str, Any]:
    """读取并校验实验 JSON。"""
    resolved_path = _resolve_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"experiment spec not found: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as file_obj:
        spec = json.load(file_obj)
    if not isinstance(spec, dict):
        raise TypeError("experiment spec must be a JSON object")
    return spec


def _normalize_key(key: str) -> str:
    """把参数键转换为命令行长参数。"""
    normalized = str(key).strip()
    if not normalized:
        raise ValueError("empty argument key")
    if normalized.startswith("--"):
        return normalized
    return "--" + normalized.replace("_", "-")


def _append_arg(argv: list[str], key: str, value: Any) -> None:
    """把 JSON 参数项追加到命令行 argv。"""
    if value is None or value is False:
        return
    option = _normalize_key(key)
    if value is True:
        argv.append(option)
        return
    if isinstance(value, (list, tuple)):
        argv.append(option)
        argv.extend(str(item) for item in value)
        return
    argv.extend([option, str(value)])


def _args_to_argv(args: dict[str, Any]) -> list[str]:
    """把参数字典转换为 argv 片段。"""
    argv: list[str] = []
    for key, value in args.items():
        _append_arg(argv, key, value)
    return argv


def _matrix_experiments(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    """把 matrix 字段展开成实验列表。"""
    if not matrix:
        return []
    keys = list(matrix)
    values: list[list[Any]] = []
    for key in keys:
        raw_values = matrix[key]
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"matrix field '{key}' must be a non-empty list")
        values.append(raw_values)

    experiments: list[dict[str, Any]] = []
    for combo in itertools.product(*values):
        args = dict(zip(keys, combo))
        name = "_".join(f"{key.replace('-', '_')}={value}" for key, value in args.items())
        experiments.append({"name": name, "args": args})
    return experiments


def build_commands(spec: dict[str, Any]) -> list[ExperimentCommand]:
    """从 JSON spec 展开训练命令。"""
    python_exe = str(spec.get("python") or sys.executable)
    script_path = _resolve_path(spec.get("script", "SNN_based_method/scripts/train.py"))
    run_name_prefix = str(spec.get("run_name_prefix", "grid")).strip()
    base_args = spec.get("base_args", {})
    if not isinstance(base_args, dict):
        raise TypeError("base_args must be a JSON object")

    raw_experiments = list(spec.get("experiments", []))
    raw_experiments.extend(_matrix_experiments(spec.get("matrix", {})))
    if not raw_experiments:
        raise ValueError("spec must contain at least one experiment or matrix entry")

    commands: list[ExperimentCommand] = []
    for index, experiment in enumerate(raw_experiments, start=1):
        if not isinstance(experiment, dict):
            raise TypeError("each experiment must be a JSON object")
        name = str(experiment.get("name") or f"exp_{index:03d}")
        exp_args = experiment.get("args", {})
        if not isinstance(exp_args, dict):
            raise TypeError(f"experiment '{name}' args must be a JSON object")

        merged_args = dict(base_args)
        merged_args.update(exp_args)
        if "run-name" not in merged_args and "run_name" not in merged_args:
            safe_name = name.replace(" ", "_").replace("=", "-")
            merged_args["run-name"] = f"{run_name_prefix}_{safe_name}"

        argv = [python_exe, str(script_path)]
        argv.extend(_args_to_argv(merged_args))
        commands.append(ExperimentCommand(name=name, argv=argv))
    return commands


def run_with_config(config: ScriptConfig) -> int:
    """展开实验并按配置打印或执行。"""
    spec = _load_spec(config.spec_path)
    commands = build_commands(spec)
    if config.limit is not None:
        if config.limit <= 0:
            raise ValueError("limit must be positive")
        commands = commands[: config.limit]

    should_execute = bool(config.execute and not config.dry_run)
    for index, command in enumerate(commands, start=1):
        printable = subprocess.list2cmdline(command.argv)
        print(f"[{index:03d}/{len(commands):03d}] {command.name}")
        print(printable)
        if not should_execute:
            continue
        completed = subprocess.run(command.argv, cwd=PROJECT_ROOT)
        if completed.returncode != 0:
            if not config.continue_on_error:
                return completed.returncode
            print(f"[warning] experiment failed but continue_on_error=True: {command.name}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = ScriptConfig(
        spec_path=args.spec,
        dry_run=bool(args.dry_run or not args.execute),
        execute=bool(args.execute),
        limit=args.limit,
        continue_on_error=bool(args.continue_on_error),
    )
    return run_with_config(config)


def main_without_cli() -> None:
    """无命令行参数运行时的可编辑入口。"""
    # ===== Editable parameters =====
    spec_path = Path("SNN_based_method/experiments/chapter_grid.json")
    dry_run = True
    execute = False

    # ===== Intermediate variables =====
    config = ScriptConfig(
        spec_path=spec_path,
        dry_run=dry_run,
        execute=execute,
    )
    run_with_config(config)


if __name__ == "__main__":
    # Usage examples:
    #   python SNN_based_method/scripts/run_experiment_grid.py
    #       Run main_without_cli(), using editable parameters above.
    #   python SNN_based_method/scripts/run_experiment_grid.py --spec SNN_based_method/experiments/chapter_grid.json --dry-run
    #       Print all expanded commands.
    #   python SNN_based_method/scripts/run_experiment_grid.py --spec SNN_based_method/experiments/chapter_grid.json --execute
    #       Execute all commands sequentially.
    #
    # Outputs:
    #   Printed command list; when --execute is used, train.py writes logs/checkpoints normally.
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()

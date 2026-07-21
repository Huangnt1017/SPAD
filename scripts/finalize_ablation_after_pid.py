"""在既有训练/测试队列自然结束后完成最终汇总和 Markdown 同步。

CLI 示例：
    D:\\Anaconda3\\envs\\torchnew\\python.exe scripts\\finalize_ablation_after_pid.py --execute --wait-for-pid 75596

默认 dry-run；执行模式只等待 PID 自然退出，不发送 terminate/kill，也不启动 GPU 训练。
等待结束后依次运行严格 core 汇总、Markdown 同步和资产审计；任何 8/8 缺失都会失败，
不会把不完整结果伪装成最终结果。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_BOOT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_PROJECT_ROOT))

from scripts.ablation_registry import FIXED_PYTHON, PROJECT_ROOT


FINALIZE_ROOT = PROJECT_ROOT / "outputs" / "ABL" / "finalize"


def process_exists(pid: int) -> bool:
    """只读检查 PID；不控制目标进程。"""
    if pid <= 0:
        return False
    completed = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    return f'"{pid}"' in completed.stdout


def wait_for_pid(pid: int) -> None:
    """等待 PID 自然退出。"""
    if pid <= 0:
        return
    print(f"waiting_for_pid={pid}", flush=True)
    while process_exists(pid):
        time.sleep(30)
    print(f"wait_complete_pid={pid}", flush=True)


def run_command(command: List[str], log_path: Path) -> int:
    """运行一个 CPU 后处理命令并记录输出。"""
    with log_path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write("\n=== " + datetime.now().isoformat(timespec="seconds") + " ===\n")
        stream.write(subprocess.list2cmdline(command) + "\n")
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def finalize(*, execute: bool, wait_pid: int, run_tag: str) -> int:
    """严格执行最终汇总、文档同步和资产审计。"""
    if Path(sys.executable).resolve() != FIXED_PYTHON.resolve():
        raise RuntimeError(f"Expected fixed Python {FIXED_PYTHON}, got {sys.executable}")
    commands = [
        [sys.executable, str(PROJECT_ROOT / "scripts" / "summarize_ablation.py"), "--families", "core"],
        [sys.executable, str(PROJECT_ROOT / "scripts" / "update_ablation_docs.py")],
        [sys.executable, str(PROJECT_ROOT / "scripts" / "audit_ablation_assets.py"), "--families", "core"],
    ]
    for command in commands:
        print("DRY-RUN", subprocess.list2cmdline(command))
    if not execute:
        return 0

    output_dir = FINALIZE_ROOT / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "finalize_manifest.json"
    log_path = output_dir / "finalize_console.log"
    manifest: Dict[str, object] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "wait_pid": wait_pid,
        "commands": commands,
        "status": "waiting",
        "steps": [],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    wait_for_pid(wait_pid)
    manifest["status"] = "running"
    manifest["execution_started_at"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    steps = manifest["steps"]
    for command in commands:
        return_code = run_command(command, log_path)
        step = {
            "command": command,
            "return_code": return_code,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        steps.append(step)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if return_code != 0:
            manifest["status"] = "failed"
            manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            return return_code

    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI。"""
    parser = argparse.ArgumentParser(description="等待既有队列后严格完成消融文档")
    parser.add_argument("--execute", action="store_true", help="实际等待并执行；省略时 dry-run")
    parser.add_argument("--wait-for-pid", type=int, default=0, help="等待该 PID 自然退出")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=datetime.now().strftime("finalize_%Y%m%d_%H%M%S"),
        help="后处理日志目录名",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(argv)
    return finalize(execute=args.execute, wait_pid=args.wait_for_pid, run_tag=args.run_tag)


def main_without_cli() -> None:
    """无参数模式：安全 dry-run。"""
    finalize(execute=False, wait_pid=0, run_tag="dry_run")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()

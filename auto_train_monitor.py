#!/usr/bin/env python3
"""
监控训练日志完成状态，自动启动下一个训练任务。

监控逻辑：
- 每次检查日志最后一行是否包含 "Epoch [100/100]" 或 "Training completed"
- 若检测到完成标记，启动 PointTransformer 训练（B=32, epoch=100）
- 记录启动时间戳到 auto_train_next.log
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_FILE = PROJECT_ROOT / "logs" / "train_pointtransv2_20260528_005554_877505.log"
MONITOR_LOG = PROJECT_ROOT / "auto_train_next.log"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
STARTED_FLAG = PROJECT_ROOT / ".auto_train_started"


def check_training_complete() -> bool:
    """检查日志是否显示训练完成。"""
    if not LOG_FILE.exists():
        return False

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return False

            # 检查最后几行是否包含完成标记
            last_lines = "".join(lines[-5:])
            return "Epoch [100/100]" in last_lines or "Training completed" in last_lines
    except Exception as e:
        print(f"[ERROR] 读取日志失败: {e}", file=sys.stderr)
        return False


def launch_pointtransformer_training():
    """启动 PointTransformer 训练。"""
    # 检查是否已启动过
    if STARTED_FLAG.exists():
        print("[INFO] PointTransformer 训练已启动，跳过重复启动")
        return False

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model", "pointtransformer",
        "--batch_size", "32",
        "--epochs", "100",
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] 启动 PointTransformer 训练 (B=32, epoch=100)\n"

    try:
        print(log_msg, end="")
        with open(MONITOR_LOG, "a", encoding="utf-8") as f:
            f.write(log_msg)

        # 后台启动训练
        subprocess.Popen(cmd, cwd=PROJECT_ROOT)
        print(f"[{timestamp}] 训练进程已启动")

        # 创建启动标志
        STARTED_FLAG.touch()
        return True
    except Exception as e:
        err_msg = f"[{timestamp}] 启动失败: {e}\n"
        print(err_msg, file=sys.stderr)
        with open(MONITOR_LOG, "a", encoding="utf-8") as f:
            f.write(err_msg)
        return False


def main():
    """主监控循环。"""
    if check_training_complete():
        print("[INFO] 检测到前一个训练已完成，启动 PointTransformer 训练...")
        launch_pointtransformer_training()
    else:
        # 获取当前进度
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    if "Epoch" in last_line:
                        print(f"[INFO] 训练进行中: {last_line.strip()}")
                    else:
                        print("[INFO] 等待训练完成...")
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

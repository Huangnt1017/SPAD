# SPAD 仓库执行约束

## 固定 Python 环境

- 本仓库唯一默认 Python 解释器：

  ```text
  D:\Anaconda3\envs\torchnew\python.exe
  ```

- 在 `D:\PYproject\SPAD` 工作时，所有 Python 脚本、模块、`py_compile`、
  `unittest`、依赖查询和 CUDA 检查都必须使用上述解释器的绝对路径。
- 不要调用裸 `python`、`python.exe`、`pip`、`pytest` 或依赖当前 PATH。
  当前 PATH 可能命中 Windows Store 的 CPU-only Python，导致
  `torch.cuda.is_available()` 错误显示为 `False`。
- 不要求先执行 `conda activate`。PowerShell 推荐写法：

  ```powershell
  $SPAD_PYTHON = "D:\Anaconda3\envs\torchnew\python.exe"
  & $SPAD_PYTHON --version
  & $SPAD_PYTHON -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
  ```

- 调用子 Python 进程时优先复用 `sys.executable`，不要重新解析 PATH。

## 项目与数据路径

- 项目根目录：

  ```text
  D:\PYproject\SPAD
  ```

- 正式三页窗口点云目录：

  ```text
  D:\PYproject\SPADdata\20250430\2025-04-30-pc
  ```

- 从项目根目录运行脚本和测试。需要显式模块搜索路径时使用：

  ```powershell
  $env:PYTHONPATH = "D:\PYproject\SPAD"
  ```

## 常用验证命令

```powershell
$SPAD_PYTHON = "D:\Anaconda3\envs\torchnew\python.exe"

& $SPAD_PYTHON -m py_compile `
  downsampling\task_agnostic_xyzi.py `
  scripts\train_task_agnostic_samplenet_xyzi.py

& $SPAD_PYTHON -m unittest discover -s downsampling\tests -v
& $SPAD_PYTHON -m unittest discover -s tests -v

& $SPAD_PYTHON scripts\train_task_agnostic_samplenet_xyzi.py --help
```

## CUDA 基线

- 2026-07-15 已验证环境：PyTorch `2.7.1+cu128`、CUDA runtime `12.8`、
  `torch.cuda.is_available() == True`。
- 已识别 GPU：NVIDIA GeForce RTX 4070 SUPER，显存约 12 GB。
- CUDA 诊断先同时检查指定环境和驱动：

  ```powershell
  & "D:\Anaconda3\envs\torchnew\python.exe" -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
  nvidia-smi
  ```

## 训练安全

- 用户未明确要求时，不自动启动全量长期训练。
- 先运行 `--help`、`sanity`、纯前向或少量文件 smoke；报告输入形状、设备、
  峰值显存、输出形状和唯一索引检查。
- 正式训练、恢复和导出命令仍必须使用固定解释器绝对路径。

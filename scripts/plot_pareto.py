"""
模型帕累托图: 精度 / 参数量 / 显存 / 推理速度 四维对比。

对每个模型现场测量 (参数量 / 峰值显存 / 推理延迟), 精度指标 (Top-1 / 3D Box IoU)
从 test.py 已落盘的 metrics_*.json 中读取 (按模型名匹配最新一份, 缺失则标 NaN)。

输出:
- logs/CLS/pareto/pareto_metrics.csv      逐模型汇总表
- logs/CLS/pareto/pareto_<axis>.png       精度 vs (参数量 / 显存 / 延迟) 散点帕累托前沿

用法 (PowerShell):
    $env:PYTHONPATH = "D:\\PYproject\\SPAD"
    & "D:\\anaconda3\\envs\\torchnew\\python.exe" scripts/plot_pareto.py
    # 只测部分模型:
    & $python scripts/plot_pareto.py --models dgcnn graph_residual graph_residual_gcn
    # 跳过现场基准测试 (只读 CSV 重绘):
    & $python scripts/plot_pareto.py --from-csv logs/CLS/pareto/pareto_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_model

# 默认参与对比的模型 (与 train.py --model choices 对齐)
DEFAULT_MODELS: Tuple[str, ...] = (
    "pointnet",
    "pointnet2",
    "pointnet2msg",
    "dgcnn",
    "pointmlp",
    "pointmlpelite",
    "pointtransformer",
    "pointtransv2",
    "pointtransv3",
    "pointbert",
    "pointmae",
    "pointrwkv",
    "spt",
    "upp",
    "3detr",
    "graph_residual",
    "graph_residual_gcn",
)

# 自研模型 (绘图时高亮标红)
OURS: Tuple[str, ...] = ("graph_residual", "graph_residual_gcn")


def count_parameters(model: torch.nn.Module) -> int:
    """统计可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_points: int,
    warmup: int = 5,
    iters: int = 30,
) -> float:
    """测量单次前向推理延迟 (ms/batch, 取中位数)。

    Args:
        model: 已在目标设备上的模型 (eval 模式)。
        device: 运行设备。
        batch_size: 推理 batch 大小。
        num_points: 每样本点数。
        warmup: 预热次数 (不计时, 触发 cuDNN autotune / 图捕获)。
        iters: 计时迭代次数。

    Returns:
        单 batch 推理延迟中位数 (毫秒)。
    """
    model.eval()
    # (B, N, 4) 随机输入, 模型 forward 内部会转 channel-first
    dummy = torch.randn(batch_size, num_points, 4, device=device)

    for _ in range(warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings: List[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    return float(np.median(timings))


def measure_peak_memory_train(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_points: int,
) -> float:
    """测量一次训练步 (forward+backward) 的峰值显存 (MB)。

    用 logits.sum()+box.sum() 构造临时 loss 触发反向, 与各模型 __main__ 显存测试口径一致。
    CPU 设备返回 NaN。

    Args:
        model: 已在目标设备上的模型。
        device: 运行设备。
        batch_size: 训练 batch 大小。
        num_points: 每样本点数。

    Returns:
        峰值显存 (MB); CPU 时为 NaN。
    """
    if device.type != "cuda":
        return float("nan")

    import gc

    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dummy = out = loss = None
    try:
        dummy = torch.randn(batch_size, num_points, 4, device=device)
        out = model(dummy)
        if isinstance(out, dict):
            logits, box = out.get("logits"), out.get("box_pred")
        elif isinstance(out, (tuple, list)):
            logits, box = out[0], (out[1] if len(out) > 1 else None)
        else:
            logits, box = out, None

        loss = logits.float().sum()
        if box is not None and torch.is_tensor(box):
            loss = loss + box.float().sum()
        loss.backward()
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    finally:
        # 彻底释放: 清梯度 → 删中间张量 → gc → 双重 empty_cache,
        # 避免大模型 (PointRWKV/UPP/PointMAE) 残留显存累积导致后续模型溢出到主存而极慢。
        model.zero_grad(set_to_none=True)
        del dummy, out, loss
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return float(peak_mb)


def find_latest_metrics_json(model_name: str, metrics_dir: Path) -> Optional[Path]:
    """按模型名匹配最新一份 metrics_<model>_<timestamp>.json。

    Args:
        model_name: 模型名 (test.py 落盘时用的 resolved_model)。
        metrics_dir: 指标 JSON 所在目录。

    Returns:
        最新 JSON 路径; 无匹配时 None。
    """
    candidates = sorted(metrics_dir.glob(f"metrics_{model_name}_*.json"))
    return candidates[-1] if candidates else None


def read_accuracy_metrics(model_name: str, metrics_dir: Path) -> Dict[str, float]:
    """从已落盘的 test 指标 JSON 读取精度 (Top-1 / 3D Box IoU / AP50)。

    Args:
        model_name: 模型名。
        metrics_dir: 指标 JSON 目录。

    Returns:
        {'top1', 'box_iou', 'ap50'} 字典, 缺失项为 NaN。
    """
    nan_result = {"top1": float("nan"), "box_iou": float("nan"), "ap50": float("nan")}
    json_path = find_latest_metrics_json(model_name, metrics_dir)
    if json_path is None:
        return nan_result

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return nan_result

    def _as_float(value) -> float:
        return float(value) if isinstance(value, (int, float)) else float("nan")

    return {
        "top1": _as_float(payload.get("top1")),
        "box_iou": _as_float(payload.get("mean_iou_matched_cls")),
        "ap50": _as_float(payload.get("AP50")),
    }


def benchmark_model(
    model_name: str,
    device: torch.device,
    batch_size: int,
    num_points: int,
    metrics_dir: Path,
) -> Optional[Dict[str, float]]:
    """对单个模型测量参数量/显存/延迟并读取精度。

    Args:
        model_name: 模型名。
        device: 运行设备。
        batch_size: benchmark batch 大小。
        num_points: 每样本点数。
        metrics_dir: 精度 JSON 目录。

    Returns:
        汇总指标字典; 构建失败 (如缺依赖) 返回 None。
    """
    try:
        model = build_model(model_name, num_classes=26, project_root=PROJECT_ROOT, args=None)
        model = model.to(device)
    except Exception as exc:
        print(f"  [skip] {model_name}: 构建失败 ({type(exc).__name__}: {exc})")
        return None

    import gc

    row: Dict[str, float] = {"params_m": count_parameters(model) / 1e6}

    try:
        row["peak_mem_mb"] = measure_peak_memory_train(model, device, batch_size, num_points)
    except torch.cuda.OutOfMemoryError:
        row["peak_mem_mb"] = float("nan")
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [warn] {model_name}: 显存 OOM (B={batch_size})")
    except Exception as exc:
        row["peak_mem_mb"] = float("nan")
        print(f"  [warn] {model_name}: 显存测量失败 ({type(exc).__name__}: {exc})")

    try:
        row["latency_ms"] = measure_latency(model, device, batch_size, num_points)
    except Exception as exc:
        row["latency_ms"] = float("nan")
        print(f"  [warn] {model_name}: 延迟测量失败 ({type(exc).__name__})")

    row.update(read_accuracy_metrics(model_name, metrics_dir))

    # 彻底释放: 先搬回 CPU 切断显存引用 → 删 → gc → 双重 empty_cache。
    # 不这样做时, 大模型权重 + autograd 残留会累积, 后续模型被迫溢出到主存导致极慢。
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return row


def write_csv(rows: Dict[str, Dict[str, float]], csv_path: Path) -> None:
    """汇总指标写入 CSV。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "params_m", "peak_mem_mb", "latency_ms", "top1", "box_iou", "ap50"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model_name, row in rows.items():
            writer.writerow({"model": model_name, **{k: row.get(k, float("nan")) for k in fields[1:]}})
    print(f"[csv] 已写出 {csv_path}")


def read_csv(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """从 CSV 读回汇总指标 (用于 --from-csv 重绘)。"""
    rows: Dict[str, Dict[str, float]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for record in csv.DictReader(f):
            model_name = record.pop("model")
            rows[model_name] = {
                k: (float(v) if v not in ("", "nan", "None") else float("nan"))
                for k, v in record.items()
            }
    return rows


def _pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """计算帕累托前沿掩码: x 越小越好 (cost), y 越大越好 (quality)。

    Args:
        x: 成本轴 (参数量/显存/延迟), 越小越好。
        y: 质量轴 (精度), 越大越好。

    Returns:
        bool 掩码, True 表示该点在帕累托前沿上 (无其它点同时更省成本且更高精度)。
    """
    n = len(x)
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            on_front[i] = False
            continue
        for j in range(n):
            if i == j or not np.isfinite(x[j]) or not np.isfinite(y[j]):
                continue
            # j 支配 i: 成本不高于 i 且精度不低于 i, 且至少一项严格更优
            if x[j] <= x[i] and y[j] >= y[i] and (x[j] < x[i] or y[j] > y[i]):
                on_front[i] = False
                break
    return on_front


def plot_pareto(
    rows: Dict[str, Dict[str, float]],
    cost_key: str,
    cost_label: str,
    quality_key: str,
    quality_label: str,
    save_path: Path,
) -> None:
    """绘制单张 精度 vs 成本 帕累托散点图。

    Args:
        rows: 逐模型指标。
        cost_key: 成本轴字段名 (越小越好)。
        cost_label: 成本轴显示名。
        quality_key: 质量轴字段名 (越大越好)。
        quality_label: 质量轴显示名。
        save_path: 图片保存路径。
    """
    import matplotlib.pyplot as plt

    names = list(rows.keys())
    x = np.array([rows[n].get(cost_key, float("nan")) for n in names], dtype=np.float64)
    y = np.array([rows[n].get(quality_key, float("nan")) for n in names], dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        print(f"  [skip plot] {quality_label} vs {cost_label}: 无有效数据 (检查是否已跑 test.py 生成精度)")
        return

    on_front = _pareto_front(x, y)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # 帕累托前沿连线 (按成本升序)
    front_idx = np.where(on_front)[0]
    if len(front_idx) >= 2:
        order = front_idx[np.argsort(x[front_idx])]
        ax.plot(x[order], y[order], "--", color="gray", alpha=0.6, zorder=1, label="Pareto front")

    for i, name in enumerate(names):
        if not valid[i]:
            continue
        is_ours = name in OURS
        ax.scatter(
            x[i], y[i],
            s=160 if is_ours else 90,
            c="#d62728" if is_ours else "#1f77b4",
            marker="*" if is_ours else "o",
            edgecolors="black", linewidths=0.6,
            zorder=3, alpha=0.9,
        )
        ax.annotate(
            name, (x[i], y[i]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, fontweight="bold" if is_ours else "normal",
            color="#d62728" if is_ours else "black",
        )

    ax.set_xlabel(cost_label, fontsize=11)
    ax.set_ylabel(quality_label, fontsize=11)
    ax.set_title(f"{quality_label} vs {cost_label}  (↖ 越靠左上越优)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[plot] 已保存 {save_path}")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(description="模型精度/参数/显存/速度 帕累托图")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS), help="参与对比的模型名列表")
    parser.add_argument("--batch-size", type=int, default=8, help="benchmark batch 大小 (显存测量只需相对比较, 默认 8 避免大模型溢出主存)")
    parser.add_argument("--num-points", type=int, default=1024, help="每样本点数")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备: auto/cpu/cuda")
    parser.add_argument("--metrics-dir", type=str, default="logs/CLS", help="test.py 落盘的 metrics_*.json 目录")
    parser.add_argument("--output-dir", type=str, default="logs/CLS/pareto", help="CSV 与图片输出目录")
    parser.add_argument("--from-csv", type=str, default="", help="跳过基准测试, 直接从该 CSV 读数据重绘")
    parser.add_argument("--quality", type=str, default="top1", choices=["top1", "box_iou", "ap50"], help="质量轴指标")
    return parser


def resolve_device(device_arg: str) -> torch.device:
    """解析设备参数。"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用, 回退 CPU (显存列将为 NaN)")
        return torch.device("cpu")
    return torch.device(device_arg)


def main(argv=None) -> None:
    """脚本入口。"""
    args = build_parser().parse_args(argv)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    metrics_dir = (PROJECT_ROOT / args.metrics_dir).resolve()
    csv_path = output_dir / "pareto_metrics.csv"

    if args.from_csv:
        rows = read_csv((PROJECT_ROOT / args.from_csv).resolve())
        print(f"[csv] 从 {args.from_csv} 读入 {len(rows)} 个模型")
    else:
        device = resolve_device(args.device)
        print(f"[device] {device}")
        if device.type == "cuda":
            print(f"[gpu] {torch.cuda.get_device_name(0)}")

        rows: Dict[str, Dict[str, float]] = {}
        for model_name in args.models:
            print(f"[bench] {model_name} ...")
            row = benchmark_model(model_name, device, args.batch_size, args.num_points, metrics_dir)
            if row is not None:
                rows[model_name] = row
                print(
                    f"    params={row['params_m']:.3f}M  "
                    f"mem={row['peak_mem_mb']:.0f}MB  "
                    f"lat={row['latency_ms']:.2f}ms  "
                    f"top1={row['top1']:.4f}"
                )
        write_csv(rows, csv_path)

    quality_label = {
        "top1": "Top-1 Accuracy",
        "box_iou": "3D Box IoU (matched)",
        "ap50": "Box AP50",
    }[args.quality]

    plot_pareto(rows, "params_m", "Params (M)", args.quality, quality_label, output_dir / f"pareto_params_{args.quality}.png")
    plot_pareto(rows, "peak_mem_mb", "Peak Train Mem (MB)", args.quality, quality_label, output_dir / f"pareto_mem_{args.quality}.png")
    plot_pareto(rows, "latency_ms", "Inference Latency (ms/batch)", args.quality, quality_label, output_dir / f"pareto_latency_{args.quality}.png")

    print("\n完成。CSV + 三张帕累托图已输出到:", output_dir)


if __name__ == "__main__":
    main()

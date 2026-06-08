"""Profile SPAD SNN training GPU memory for P/chunk/batch combinations.

CLI example:
    python SNN_based_method/scripts/profile_snn_memory.py --output SNN_based_method/artifacts/snn_memory_profile.csv

Non-CLI example:
    python SNN_based_method/scripts/profile_snn_memory.py

参数说明:
    默认按当前 SNNConfig 的模型、loss、C、num_blocks、PLIF/CuPy 后端测量。
    每个组合都会重新构建模型并执行一次 forward + loss + backward + AdamW step。
    默认把 PyTorch CUDA allocator 限制到 11.5 GiB, 避免 Windows 共享 GPU 内存拖慢。
    输出 CSV 中 allocated/reserved 为 PyTorch 记录的峰值 CUDA 显存, 单位 GiB。

输入/输出:
    输入为随机 raw ToF 张量 ``[B, 4096, P]`` 和随机弱标签 ``[B, 2, 64, 64]``。
    输出 CSV/Markdown 表格保存到 ``SNN_based_method/artifacts``。
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from SNN_based_method.config.SNN_config import SNNConfig


DEFAULT_PAGES = (128, 384, 480, 640, 960, 1000, 1200, 2400)
DEFAULT_CHUNKS = (32, 64, 128)
DEFAULT_BATCHES = (2, 4, 8)
BASE_RESERVED_GIB_AT_P128 = {
    32: {2: 1.515625, 4: 3.029296875, 8: 5.962890625},
    64: {2: 1.556640625, 4: 3.103515625, 8: 6.201171875},
    128: {2: 1.7109375, 4: 3.4296875, 8: 6.75},
}


@dataclass
class ScriptConfig:
    """运行配置, 同时服务 CLI 和无参数调试入口。"""

    output: Path
    markdown_output: Path
    pages: tuple[int, ...] = DEFAULT_PAGES
    chunks: tuple[int, ...] = DEFAULT_CHUNKS
    batches: tuple[int, ...] = DEFAULT_BATCHES
    device: str = "cuda"
    amp: bool = False
    spike_backend: str = "cupy"
    warmup: bool = True
    stop_larger_batches_after_oom: bool = True
    memory_limit_gib: float = 11.5
    cupy_pool_limit_mib: int = 256
    resume_existing: bool = True
    predict_oom: bool = True


def _parse_int_list(value: str) -> tuple[int, ...]:
    """解析逗号分隔的正整数列表。"""
    items: list[int] = []
    for raw_item in value.split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        number = int(raw_item)
        if number <= 0:
            raise argparse.ArgumentTypeError("all values must be positive integers")
        items.append(number)
    if not items:
        raise argparse.ArgumentTypeError("list must not be empty")
    return tuple(items)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Profile SPAD SNN one-step training CUDA memory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("SNN_based_method/artifacts/snn_memory_profile.csv"),
        help="CSV 输出路径。",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("SNN_based_method/artifacts/snn_memory_profile.md"),
        help="Markdown 表格输出路径。",
    )
    parser.add_argument(
        "--pages",
        type=_parse_int_list,
        default=DEFAULT_PAGES,
        help="逗号分隔的 P 列表, 默认 128,384,480,640,960,1000,1200,2400。",
    )
    parser.add_argument(
        "--chunks",
        type=_parse_int_list,
        default=DEFAULT_CHUNKS,
        help="逗号分隔的 chunk_size 列表, 默认 32,64,128。",
    )
    parser.add_argument(
        "--batches",
        type=_parse_int_list,
        default=DEFAULT_BATCHES,
        help="逗号分隔的 batch_size 列表, 默认 2,4,8。",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="测量设备。显存测量需要 cuda。",
    )
    parser.add_argument(
        "--spike-backend",
        default="cupy",
        choices=["cupy", "torch", "auto"],
        help="SpikingJelly 神经元后端。",
    )
    parser.add_argument("--amp", action="store_true", help="使用 AMP autocast 测量。")
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="跳过第一个组合前的 CUDA warmup。",
    )
    parser.add_argument(
        "--no-oom-prune",
        dest="stop_larger_batches_after_oom",
        action="store_false",
        help="某个 P/chunk 下小 batch OOM 后仍继续测更大的 batch。",
    )
    parser.add_argument(
        "--memory-limit-gib",
        type=float,
        default=11.5,
        help="PyTorch CUDA allocator 上限, 单位 GiB; 默认 11.5, 0 表示不限制。",
    )
    parser.add_argument(
        "--cupy-pool-limit-mib",
        type=int,
        default=256,
        help="CuPy memory pool 上限, 单位 MiB; 默认 256, 0 表示不限制。",
    )
    parser.add_argument(
        "--no-resume-existing",
        dest="resume_existing",
        action="store_false",
        help="不复用已有 CSV 中已经测过的组合。",
    )
    parser.add_argument(
        "--no-predict-oom",
        dest="predict_oom",
        action="store_false",
        help="不使用已测 P=128 基线预判 OOM, 强制实际运行所有组合。",
    )
    return parser


def _clear_cuda(device: torch.device) -> None:
    """释放 Python 和 CUDA 缓存, 降低组合之间的互相影响。"""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            import cupy

            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def _configure_memory_limits(
    device: torch.device,
    *,
    memory_limit_gib: float,
    cupy_pool_limit_mib: int,
) -> None:
    """限制 CUDA allocator, 避免 Windows WDDM 进入共享 GPU 内存后长时间拖慢。"""
    if device.type != "cuda":
        return
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    total_bytes = torch.cuda.get_device_properties(int(device_index)).total_memory
    if memory_limit_gib > 0:
        limit_bytes = int(memory_limit_gib * 1024**3)
        fraction = min(max(limit_bytes / total_bytes, 0.01), 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device=int(device_index))
    if cupy_pool_limit_mib > 0:
        try:
            import cupy

            with cupy.cuda.Device(int(device_index)):
                cupy.get_default_memory_pool().set_limit(
                    size=int(cupy_pool_limit_mib * 1024**2)
                )
        except Exception:
            pass


def _is_memory_error(message: str) -> bool:
    """判断 CUDA/CuPy 报错是否属于显存不足。"""
    lowered = message.lower()
    memory_markers = (
        "out of memory",
        "memory allocation",
        "cublas_status_alloc_failed",
        "cuda_error_out_of_memory",
        "std::bad_alloc",
    )
    return any(marker in lowered for marker in memory_markers)


def _load_existing_rows(path: Path) -> list[dict[str, object]]:
    """读取已有 CSV 结果, 用于中断后续跑。"""
    if not path.is_file():
        return []

    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            try:
                parsed = {
                    "pages_per_group": int(row["pages_per_group"]),
                    "chunk_size": int(row["chunk_size"]),
                    "batch_size": int(row["batch_size"]),
                    "status": row.get("status", ""),
                    "max_allocated_gib": _parse_optional_float(row.get("max_allocated_gib")),
                    "max_reserved_gib": _parse_optional_float(row.get("max_reserved_gib")),
                    "elapsed_s": _parse_optional_float(row.get("elapsed_s")),
                    "error": row.get("error", ""),
                }
            except (KeyError, ValueError):
                continue
            rows.append(parsed)
    return rows


def _parse_optional_float(value: object) -> float | None:
    """把 CSV 中的可选浮点数转回 Python 值。"""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def _predict_reserved_gib(
    *,
    pages_per_group: int,
    chunk_size: int,
    batch_size: int,
) -> float | None:
    """基于当前实测 P=128 基线估算 max_memory_reserved。"""
    chunk_table = BASE_RESERVED_GIB_AT_P128.get(int(chunk_size))
    if chunk_table is None:
        return None
    base = chunk_table.get(int(batch_size))
    if base is None:
        return None
    return float(base) * float(pages_per_group) / 128.0


def _make_input(batch_size: int, pages_per_group: int, device: torch.device) -> torch.Tensor:
    """构造随机 raw ToF 输入 ``[B, 4096, P]``。"""
    return torch.randint(
        low=0,
        high=129,
        size=(batch_size, 4096, pages_per_group),
        device=device,
        dtype=torch.int16,
    )


def _make_label(batch_size: int, device: torch.device) -> torch.Tensor:
    """构造随机弱标签 ``[B, 2, 64, 64]``。"""
    depth = torch.randint(
        low=1,
        high=129,
        size=(batch_size, 1, 64, 64),
        device=device,
        dtype=torch.int16,
    ).float()
    intensity = torch.rand(batch_size, 1, 64, 64, device=device)
    return torch.cat([depth, intensity], dim=1)


def _build_profile_config(
    *,
    batch_size: int,
    pages_per_group: int,
    chunk_size: int,
    spike_backend: str,
    amp: bool,
) -> SNNConfig:
    """按当前默认设置构建单个显存测试配置。"""
    return SNNConfig(
        batch_size=batch_size,
        pages_per_group=pages_per_group,
        chunk_size=chunk_size,
        model_backend="new",
        C=16,
        num_blocks=1,
        spike_mode="plif",
        spike_tau=2.0,
        spike_v_threshold=0.8,
        spike_v_reset=0.0,
        spike_backend=spike_backend,
        return_sequence=True,
        amp=amp,
    )


def _warmup_cuda(device: torch.device) -> None:
    """执行一次轻量 CUDA warmup, 避免首个组合包含初始化开销。"""
    cfg = _build_profile_config(
        batch_size=1,
        pages_per_group=32,
        chunk_size=32,
        spike_backend="torch",
        amp=False,
    )
    model = cfg.build_model().to(device)
    criterion = cfg.build_loss().to(device)
    model.train()
    x = _make_input(1, 32, device)
    y = _make_label(1, device)
    result = model(x)
    loss, _ = criterion(result, y)
    loss.backward()
    del loss, result, x, y, criterion, model
    _clear_cuda(device)


def measure_one(
    *,
    batch_size: int,
    pages_per_group: int,
    chunk_size: int,
    device: torch.device,
    spike_backend: str,
    amp: bool,
) -> dict[str, object]:
    """测量单个组合的一步训练峰值显存。"""
    _clear_cuda(device)
    model = None
    criterion = None
    x = None
    y = None
    result = None
    loss = None
    optimizer = None
    cfg = _build_profile_config(
        batch_size=batch_size,
        pages_per_group=pages_per_group,
        chunk_size=chunk_size,
        spike_backend=spike_backend,
        amp=amp,
    )
    start = time.perf_counter()
    status = "ok"
    error = ""
    allocated_gib: float | None = None
    reserved_gib: float | None = None
    elapsed_s: float | None = None
    try:
        model = cfg.build_model().to(device)
        criterion = cfg.build_loss().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        model.train()
        x = _make_input(batch_size, pages_per_group, device)
        y = _make_label(batch_size, device)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            result = model(x)
            loss, _ = criterion(result, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
            reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
        elapsed_s = time.perf_counter() - start
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc).splitlines()[0]
        elapsed_s = time.perf_counter() - start
    except RuntimeError as exc:
        message = str(exc)
        if _is_memory_error(message):
            status = "oom_limit"
            error = message.splitlines()[0]
            elapsed_s = time.perf_counter() - start
        else:
            status = "error"
            error = message.splitlines()[0]
            elapsed_s = time.perf_counter() - start
    except Exception as exc:
        message = str(exc)
        if _is_memory_error(message):
            status = "oom_limit"
        else:
            status = "error"
        error = message.splitlines()[0] if message else type(exc).__name__
        elapsed_s = time.perf_counter() - start
    finally:
        del loss, result, x, y, optimizer, criterion, model
        _clear_cuda(device)

    return {
        "pages_per_group": pages_per_group,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "status": status,
        "max_allocated_gib": allocated_gib,
        "max_reserved_gib": reserved_gib,
        "elapsed_s": elapsed_s,
        "error": error,
    }


def _format_gib(value: object) -> str:
    """格式化 GiB 数值或 OOM/error。"""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """写出 CSV 结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pages_per_group",
        "chunk_size",
        "batch_size",
        "status",
        "max_allocated_gib",
        "max_reserved_gib",
        "elapsed_s",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, object]], device: torch.device) -> None:
    """写出按 chunk/B 展开的 Markdown 表格。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pages = sorted({int(row["pages_per_group"]) for row in rows})
    chunks = sorted({int(row["chunk_size"]) for row in rows})
    batches = sorted({int(row["batch_size"]) for row in rows})
    by_key = {
        (int(row["pages_per_group"]), int(row["chunk_size"]), int(row["batch_size"])): row
        for row in rows
    }
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    total_gib = (
        torch.cuda.get_device_properties(device).total_memory / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    lines = [
        "# SPAD SNN 显存 profiling",
        "",
        f"- device: {device_name}",
        f"- total_memory_gib: {total_gib:.2f}",
        "- model: `SPADSpikeNet`, `C=16`, `num_blocks=1`, `spike_mode=plif`, `return_sequence=True`",
        "- measurement: one random training step, `forward + SPADImagingLoss + backward + AdamW step`",
        "- table value: `max_memory_reserved / max_memory_allocated` in GiB; `OOM_LIMIT` means the 12GB-class cap was hit.",
        "",
    ]
    for chunk_size in chunks:
        lines.append(f"## chunk_size={chunk_size}")
        header = ["P"] + [f"B={batch_size}" for batch_size in batches]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for page_count in pages:
            cells = [str(page_count)]
            for batch_size in batches:
                row = by_key.get((page_count, chunk_size, batch_size))
                if row is None:
                    cells.append("")
                    continue
                if row["status"] == "ok":
                    reserved = _format_gib(row["max_reserved_gib"])
                    allocated = _format_gib(row["max_allocated_gib"])
                    cells.append(f"{reserved} / {allocated}")
                elif row["status"] == "oom_predicted" and row.get("max_reserved_gib") is not None:
                    cells.append(f"OOM_PREDICTED ({_format_gib(row['max_reserved_gib'])})")
                else:
                    cells.append(str(row["status"]).upper())
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_with_config(config: ScriptConfig) -> list[dict[str, object]]:
    """执行完整 sweep 并写出结果。"""
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot profile GPU memory.")
    device = torch.device(config.device)
    _configure_memory_limits(
        device,
        memory_limit_gib=config.memory_limit_gib,
        cupy_pool_limit_mib=config.cupy_pool_limit_mib,
    )
    if config.warmup and device.type == "cuda":
        _warmup_cuda(device)

    existing_rows = _load_existing_rows(config.output) if config.resume_existing else []
    existing_by_key = {
        (
            int(row["pages_per_group"]),
            int(row["chunk_size"]),
            int(row["batch_size"]),
        ): row
        for row in existing_rows
    }
    requested_keys = {
        (int(page_count), int(chunk_size), int(batch_size))
        for chunk_size in config.chunks
        for page_count in config.pages
        for batch_size in config.batches
    }
    rows: list[dict[str, object]] = [
        row
        for key, row in existing_by_key.items()
        if key not in requested_keys
    ]
    for chunk_size in config.chunks:
        for page_count in config.pages:
            oom_seen = False
            for batch_size in config.batches:
                key = (page_count, chunk_size, batch_size)
                predicted_reserved = _predict_reserved_gib(
                    pages_per_group=page_count,
                    chunk_size=chunk_size,
                    batch_size=batch_size,
                )
                if key in existing_by_key:
                    row = existing_by_key[key]
                    if str(row["status"]).startswith("oom"):
                        oom_seen = True
                elif (
                    config.predict_oom
                    and predicted_reserved is not None
                    and predicted_reserved > config.memory_limit_gib
                ):
                    row = {
                        "pages_per_group": page_count,
                        "chunk_size": chunk_size,
                        "batch_size": batch_size,
                        "status": "oom_predicted",
                        "max_allocated_gib": None,
                        "max_reserved_gib": predicted_reserved,
                        "elapsed_s": 0.0,
                        "error": (
                            f"Predicted reserved {predicted_reserved:.2f} GiB exceeds "
                            f"memory limit {config.memory_limit_gib:.2f} GiB."
                        ),
                    }
                    oom_seen = True
                elif config.stop_larger_batches_after_oom and oom_seen:
                    row = {
                        "pages_per_group": page_count,
                        "chunk_size": chunk_size,
                        "batch_size": batch_size,
                        "status": "oom_inferred",
                        "max_allocated_gib": None,
                        "max_reserved_gib": None,
                        "elapsed_s": 0.0,
                        "error": "Skipped because a smaller batch already OOMed for the same P/chunk.",
                    }
                else:
                    row = measure_one(
                        batch_size=batch_size,
                        pages_per_group=page_count,
                        chunk_size=chunk_size,
                        device=device,
                        spike_backend=config.spike_backend,
                        amp=config.amp,
                    )
                    if str(row["status"]).startswith("oom"):
                        oom_seen = True
                rows.append(row)
                reserved = row["max_reserved_gib"]
                allocated = row["max_allocated_gib"]
                print(
                    f"P={page_count:<4} chunk={chunk_size:<3} B={batch_size:<2} "
                    f"status={row['status']:<12} "
                    f"reserved={_format_gib(reserved):>5}GiB "
                    f"allocated={_format_gib(allocated):>5}GiB "
                    f"elapsed={float(row['elapsed_s'] or 0.0):.2f}s",
                    flush=True,
                )
                write_csv(config.output, rows)
                write_markdown(config.markdown_output, rows, device)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = ScriptConfig(
        output=args.output,
        markdown_output=args.markdown_output,
        pages=args.pages,
        chunks=args.chunks,
        batches=args.batches,
        device=args.device,
        amp=bool(args.amp),
        spike_backend=args.spike_backend,
        warmup=bool(args.warmup),
        stop_larger_batches_after_oom=bool(args.stop_larger_batches_after_oom),
        memory_limit_gib=float(args.memory_limit_gib),
        cupy_pool_limit_mib=int(args.cupy_pool_limit_mib),
        resume_existing=bool(args.resume_existing),
        predict_oom=bool(args.predict_oom),
    )
    run_with_config(config)
    return 0


def main_without_cli() -> None:
    """无命令行直接运行入口。"""
    # ===== Editable parameters =====
    output = Path("SNN_based_method/artifacts/snn_memory_profile.csv")
    markdown_output = Path("SNN_based_method/artifacts/snn_memory_profile.md")
    pages = DEFAULT_PAGES
    chunks = DEFAULT_CHUNKS
    batches = DEFAULT_BATCHES
    device = "cuda"
    amp = False
    spike_backend = "cupy"
    memory_limit_gib = 11.5
    cupy_pool_limit_mib = 256

    # ===== Intermediate variables =====
    config = ScriptConfig(
        output=output,
        markdown_output=markdown_output,
        pages=tuple(pages),
        chunks=tuple(chunks),
        batches=tuple(batches),
        device=device,
        amp=amp,
        spike_backend=spike_backend,
        memory_limit_gib=memory_limit_gib,
        cupy_pool_limit_mib=cupy_pool_limit_mib,
        predict_oom=True,
    )
    run_with_config(config)


if __name__ == "__main__":
    # Usage examples:
    #   python SNN_based_method/scripts/profile_snn_memory.py
    #       Run main_without_cli(), using editable parameters above.
    #
    #   python SNN_based_method/scripts/profile_snn_memory.py --pages 128,384 --chunks 32 --batches 2,4
    #       Profile selected combinations.
    #
    # Common parameters:
    #   --pages <list>             P values, comma separated.
    #   --chunks <list>            chunk_size values, comma separated.
    #   --batches <list>           batch_size values, comma separated.
    #   --spike-backend <backend>  cupy/torch/auto.
    #   --amp                      Profile with autocast.
    #   --memory-limit-gib <GiB>    Strict allocator cap, default 11.5.
    #   --no-predict-oom           Actually run combinations predicted to exceed the cap.
    #
    # Outputs:
    #   SNN_based_method/artifacts/snn_memory_profile.csv
    #   SNN_based_method/artifacts/snn_memory_profile.md
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()

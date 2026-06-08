"""SPAD SNN 模型的标准训练入口。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from SNN_based_method.config.SNN_config import SNNConfig
from SNN_based_method.utils.data import seed_everything
from SNN_based_method.utils.runtime import (
    add_config_arguments,
    build_run_name,
    config_from_checkpoint_and_args,
    load_checkpoint,
    make_checkpoint_run_dir,
    prepare_model_input,
    resolve_output_root,
    save_checkpoint,
)

# 数据根目录: 通过环境变量 SPAD_DATA_ROOT 覆盖, 默认指向服务器数据集目录。
#   本地 Windows: set SPAD_DATA_ROOT=D:\PYproject\SPADdata
#   服务器 Linux: 无需设置, 使用下方默认值
SPAD_DATA_ROOT = os.environ.get(
    "SPAD_DATA_ROOT",
    "/public/home/202210183047/datasets/0825",
)
DEFAULT_TRAIN_DATA_PATHS = [
    os.path.join(SPAD_DATA_ROOT, "0825"),
    os.path.join(SPAD_DATA_ROOT, "0826"),
]
DEFAULT_TRAIN_CSV_PATHS = [
    os.path.join(SPAD_DATA_ROOT, "0825", "0825-group.csv"),
    os.path.join(SPAD_DATA_ROOT, "0826", "0826-group.csv"),
]


def setup_logger(log_dir: str | Path, run_name: str) -> tuple[logging.Logger, Path]:
    """创建单文件训练日志, 同时输出到控制台。"""
    log_root = resolve_output_root(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"{run_name}.log"

    logger_name = f"spad_snn_train_{run_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    mode = "a" if log_file.exists() else "w"

    file_handler = logging.FileHandler(log_file, mode=mode, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger, log_file


def _format_values(values: dict[str, float]) -> str:
    """把指标字典格式化为稳定的一行日志。"""
    if not values:
        return "none"
    return " ".join(f"{key}={value:.6f}" for key, value in sorted(values.items()))


def _update_tensor_sums(
    total: dict[str, torch.Tensor],
    values: dict[str, object],
    device: torch.device,
) -> None:
    """在 device 上累加 loss 分项, 避免每个 batch 触发 CPU 同步。"""
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            scalar = value.detach()
            if scalar.device != device:
                scalar = scalar.to(device=device, non_blocking=True)
        else:
            scalar = torch.tensor(float(value), device=device)
        total[key] = total.get(key, torch.zeros((), device=device)) + scalar


def _tensor_average_to_float(total: dict[str, torch.Tensor], count: int) -> dict[str, float]:
    """epoch 末把 device 上的累加器一次性转成 Python float。"""
    if not total:
        return {}
    divisor = max(count, 1)
    keys = sorted(total)
    stacked = torch.stack([total[key] / divisor for key in keys])
    values = stacked.detach().cpu().tolist()
    return {key: float(value) for key, value in zip(keys, values)}


def _update_spike_stats_sums(
    total_sum: dict[str, torch.Tensor],
    total_count: dict[str, torch.Tensor],
    spike_stats: dict[str, object] | None,
    device: torch.device,
) -> None:
    """累加每个 batch 返回的脉冲放电统计。"""
    if not spike_stats:
        return
    stat_sum = spike_stats.get("sum")
    stat_count = spike_stats.get("count")
    if not isinstance(stat_sum, dict) or not isinstance(stat_count, dict):
        return

    for key, value in stat_sum.items():
        if isinstance(value, torch.Tensor):
            scalar = value.detach()
            if scalar.device != device:
                scalar = scalar.to(device=device, non_blocking=True)
        else:
            scalar = torch.tensor(float(value), device=device)
        total_sum[key] = total_sum.get(key, torch.zeros((), device=device)) + scalar

    for key, value in stat_count.items():
        if isinstance(value, torch.Tensor):
            scalar = value.detach()
            if scalar.device != device:
                scalar = scalar.to(device=device, non_blocking=True)
        else:
            scalar = torch.tensor(float(value), device=device)
        total_count[key] = total_count.get(key, torch.zeros((), device=device)) + scalar


def _spike_rates_to_float(
    total_sum: dict[str, torch.Tensor],
    total_count: dict[str, torch.Tensor],
) -> dict[str, float]:
    """把累计的脉冲统计转成 epoch 级平均放电率。"""
    if not total_sum or not total_count:
        return {}
    keys = sorted(set(total_sum) & set(total_count))
    if not keys:
        return {}
    rates = []
    for key in keys:
        rates.append((total_sum[key] / total_count[key].clamp_min(1.0)).detach())
    values = torch.stack(rates).cpu().tolist()
    return {key: float(value) for key, value in zip(keys, values)}


def _format_spike_rates(spike_rates: dict[str, float]) -> str:
    """把放电率字典格式化为稳定的一行日志。"""
    if not spike_rates:
        return "none"
    return " ".join(f"{key}={value:.6f}" for key, value in sorted(spike_rates.items()))


def _build_progress_postfix(
    loss_value: float,
    spike_stats: dict[str, object] | None,
) -> dict[str, str]:
    """构造 tqdm 进度条右侧展示内容。"""
    postfix = {"loss": f"{loss_value:.4f}"}
    if not spike_stats:
        return postfix
    stat_rate = spike_stats.get("rate")
    if not isinstance(stat_rate, dict):
        return postfix

    for key in ("all", "stem", "blocks", "gate"):
        value = stat_rate.get(key)
        if isinstance(value, torch.Tensor):
            postfix[f"spk_{key}"] = f"{float(value.detach().cpu().item()):.4f}"
    return postfix


def configure_torch_runtime(cfg: SNNConfig) -> None:
    """配置 CUDA 后端, 用稳定输入尺寸换取更高吞吐。"""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)


def _prepare_device_batch(
    batch: dict[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    """把一个 CPU batch 整理成训练直接消费的 device batch。"""
    labels = batch.get("label")
    device_labels = (
        labels.to(device, non_blocking=True)
        if isinstance(labels, torch.Tensor)
        else None
    )

    keepalive: torch.Tensor | None = None
    model_input = batch.get("model_input")
    if isinstance(model_input, torch.Tensor):
        device_input = model_input.to(device, non_blocking=True)
    else:
        frames = batch["frames"]
        if not isinstance(frames, torch.Tensor):
            raise TypeError("batch['frames'] must be a torch.Tensor")
        device_frames = frames.to(device, non_blocking=True)
        device_input = prepare_model_input(device_frames)
        keepalive = device_frames

    return {
        "model_input": device_input,
        "label": device_labels,
        "_keepalive": keepalive,
    }


class CudaBatchPrefetcher:
    """用独立 CUDA stream 预取下一批数据, 降低主计算 stream 等待。"""

    def __init__(
        self,
        data_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self._iterator = iter(data_loader)
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._next_batch: dict[str, torch.Tensor | None] | None = None
        self._preload()

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor | None]:
        if self._next_batch is None:
            raise StopIteration

        current_stream = torch.cuda.current_stream(self._device)
        current_stream.wait_stream(self._stream)
        batch = self._next_batch
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(current_stream)

        self._preload()
        return batch

    def _preload(self) -> None:
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._next_batch = None
            return

        with torch.cuda.stream(self._stream):
            self._next_batch = _prepare_device_batch(batch, self._device)


def _iter_device_batches(
    data_loader: DataLoader,
    device: torch.device,
    cfg: SNNConfig,
):
    """按配置返回已搬到目标设备的 batch 迭代器。"""
    if device.type == "cuda" and cfg.cuda_prefetch:
        return CudaBatchPrefetcher(data_loader, device)

    def generator():
        for batch in data_loader:
            yield _prepare_device_batch(batch, device)

    return generator()


def _sync_if_cuda(device: torch.device) -> None:
    """同步 CUDA, 让阶段耗时更接近真实 GPU 执行时间。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stamp(stage_times: dict[str, float], name: str, device: torch.device) -> None:
    """记录一个训练阶段的时间戳。"""
    _sync_if_cuda(device)
    stage_times[name] = time.perf_counter()


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: SNNConfig,
    epoch: int,
    trace_steps: int = 0,
) -> tuple[float, dict[str, float]]:
    """训练一个 epoch, 返回平均总 loss 与各子 loss。"""
    model.train()
    total_loss = torch.zeros((), device=device)
    loss_sums: dict[str, torch.Tensor] = {}
    spike_sum_sums: dict[str, torch.Tensor] = {}
    spike_count_sums: dict[str, torch.Tensor] = {}
    num_steps = 0
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum_steps = max(1, int(cfg.grad_accum_steps))
    progress_interval = max(1, int(cfg.progress_interval))

    device_batches = _iter_device_batches(data_loader, device, cfg)
    progress = tqdm(device_batches, total=len(data_loader), desc=f"train {epoch:03d}", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(progress):
        stage_times: dict[str, float] | None = {} if batch_index < trace_steps else None
        if stage_times is not None:
            _stamp(stage_times, "batch_ready", device)

        model_input = batch["model_input"]
        labels = batch.get("label")
        if not isinstance(model_input, torch.Tensor):
            raise TypeError("device batch must contain tensor model_input")
        if stage_times is not None:
            _stamp(stage_times, "input_ready", device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            result = model(model_input)
            if stage_times is not None:
                _stamp(stage_times, "forward", device)
            loss, loss_items = criterion(result, labels)
            loss_for_backward = loss / grad_accum_steps
            if stage_times is not None:
                _stamp(stage_times, "loss", device)

        should_step = (
            (batch_index + 1) % grad_accum_steps == 0
            or (batch_index + 1) == len(data_loader)
        )
        if use_amp:
            scaler.scale(loss_for_backward).backward()
            if stage_times is not None:
                _stamp(stage_times, "backward", device)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if stage_times is not None:
                    _stamp(stage_times, "clip_grad", device)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if stage_times is not None:
                    _stamp(stage_times, "optimizer_step", device)
        else:
            loss_for_backward.backward()
            if stage_times is not None:
                _stamp(stage_times, "backward", device)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if stage_times is not None:
                    _stamp(stage_times, "clip_grad", device)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if stage_times is not None:
                    _stamp(stage_times, "optimizer_step", device)

        detached_loss = loss.detach()
        if stage_times is not None:
            _stamp(stage_times, "batch_done", device)
        total_loss = total_loss + detached_loss
        _update_tensor_sums(loss_sums, loss_items, device)
        if cfg.log_spike_stats:
            _update_spike_stats_sums(
                spike_sum_sums,
                spike_count_sums,
                result.get("spike_stats") if isinstance(result, dict) else None,
                device,
            )
        num_steps += 1

        is_last = batch_index + 1 == len(data_loader)
        if batch_index % progress_interval == 0 or is_last:
            loss_value = float(detached_loss.cpu().item())
            progress.set_postfix(
                _build_progress_postfix(
                    loss_value,
                    result.get("spike_stats") if cfg.log_spike_stats and isinstance(result, dict) else None,
                )
            )

        if stage_times is not None:
            names = list(stage_times)
            durations = {
                f"{names[i - 1]}->{names[i]}": stage_times[names[i]] - stage_times[names[i - 1]]
                for i in range(1, len(names))
            }
            tqdm.write(
                f"[trace] epoch={epoch:03d} batch={batch_index:04d} "
                + " ".join(f"{key}={value:.3f}s" for key, value in durations.items())
            )

    avg_loss = float((total_loss / max(num_steps, 1)).detach().cpu().item())
    train_items = _tensor_average_to_float(loss_sums, num_steps)
    if cfg.log_spike_stats:
        for key, value in _spike_rates_to_float(spike_sum_sums, spike_count_sums).items():
            train_items[f"spike_rate_{key}"] = value
    return avg_loss, train_items


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    metrics,
    device: torch.device,
    cfg: SNNConfig,
    epoch: int,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """验证一个 epoch, 返回 loss、子 loss 和图像指标。"""
    model.eval()
    total_loss = torch.zeros((), device=device)
    loss_sums: dict[str, torch.Tensor] = {}
    metric_sums: dict[str, torch.Tensor] = {}
    spike_sum_sums: dict[str, torch.Tensor] = {}
    spike_count_sums: dict[str, torch.Tensor] = {}
    num_steps = 0
    progress_interval = max(1, int(cfg.progress_interval))

    device_batches = _iter_device_batches(data_loader, device, cfg)
    progress = tqdm(device_batches, total=len(data_loader), desc=f"val {epoch:03d}", leave=False)
    for batch_index, batch in enumerate(progress):
        model_input = batch["model_input"]
        labels = batch.get("label")
        if not isinstance(model_input, torch.Tensor):
            raise TypeError("device batch must contain tensor model_input")

        result = model(model_input)
        loss, loss_items = criterion(result, labels)

        if labels is not None:
            _update_tensor_sums(metric_sums, metrics.compute_tensors(result, labels), device)

        detached_loss = loss.detach()
        total_loss = total_loss + detached_loss
        _update_tensor_sums(loss_sums, loss_items, device)
        if cfg.log_spike_stats:
            _update_spike_stats_sums(
                spike_sum_sums,
                spike_count_sums,
                result.get("spike_stats") if isinstance(result, dict) else None,
                device,
            )
        num_steps += 1
        is_last = batch_index + 1 == len(data_loader)
        if batch_index % progress_interval == 0 or is_last:
            loss_value = float(detached_loss.cpu().item())
            progress.set_postfix(
                _build_progress_postfix(
                    loss_value,
                    result.get("spike_stats") if cfg.log_spike_stats and isinstance(result, dict) else None,
                )
            )

    val_items = _tensor_average_to_float(loss_sums, num_steps)
    if cfg.log_spike_stats:
        for key, value in _spike_rates_to_float(spike_sum_sums, spike_count_sums).items():
            val_items[f"spike_rate_{key}"] = value
    return (
        float((total_loss / max(num_steps, 1)).detach().cpu().item()),
        val_items,
        _tensor_average_to_float(metric_sums, num_steps),
    )


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="训练 SPAD SNN 成像模型")
    add_config_arguments(parser)
    parser.add_argument(
        "--resume-run-dir",
        default=None,
        help="指定训练 run 的 checkpoint 文件夹, 自动从其中的 last.pth 接续训练",
    )
    parser.add_argument("--save-every", type=int, default=None, help="每 N 个 epoch 额外保存一次 checkpoint")
    parser.add_argument("--trace-steps", type=int, default=0, help="打印前 N 个训练 batch 的阶段耗时")
    return parser


def resolve_resume_run_dir(path: str | Path) -> Path:
    """解析续训目录; 相对路径按项目根目录解释。"""
    resume_path = Path(path)
    if not resume_path.is_absolute():
        resume_path = resolve_output_root(resume_path)
    resume_path = resume_path.resolve()
    if resume_path.is_file() and resume_path.suffix.lower() == ".pth":
        resume_path = resume_path.parent
    if not resume_path.is_dir():
        raise NotADirectoryError(f"resume run directory not found: {resume_path}")
    return resume_path


def prepare_resume_args(args: argparse.Namespace) -> Path | None:
    """把 ``--resume-run-dir`` 转换为 checkpoint/config/run_name 参数。"""
    if not args.resume_run_dir:
        return None

    resume_run_dir = resolve_resume_run_dir(args.resume_run_dir)
    checkpoint_path = resume_run_dir / "last.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"last checkpoint not found: {checkpoint_path}")

    if args.checkpoint_path:
        explicit_checkpoint = Path(args.checkpoint_path)
        if not explicit_checkpoint.is_absolute():
            explicit_checkpoint = resolve_output_root(explicit_checkpoint)
        if explicit_checkpoint.resolve() != checkpoint_path.resolve():
            raise ValueError(
                "--resume-run-dir already selects last.pth; do not pass a different --checkpoint"
            )
    args.checkpoint_path = str(checkpoint_path)

    config_path = resume_run_dir / "config.json"
    if args.config is None and config_path.is_file():
        args.config = str(config_path)
    if args.run_name is None:
        args.run_name = resume_run_dir.name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(resume_run_dir.parent)
    return resume_run_dir


def _same_paths(left: list[str] | None, right: list[str]) -> bool:
    """比较路径集合, 用于识别默认训练目录。"""
    if left is None:
        return False
    left_resolved = {str(Path(path).resolve()).lower() for path in left}
    right_resolved = {str(Path(path).resolve()).lower() for path in right}
    return left_resolved == right_resolved


def apply_default_train_paths(cfg: SNNConfig) -> SNNConfig:
    """无参数运行时默认使用 0825/0826 训练数据及其 CSV 清单。"""
    updates: dict[str, object] = {}
    if not cfg.data_paths:
        updates["data_paths"] = DEFAULT_TRAIN_DATA_PATHS
        updates["csv_paths"] = DEFAULT_TRAIN_CSV_PATHS
    elif not cfg.csv_paths and _same_paths(cfg.data_paths, DEFAULT_TRAIN_DATA_PATHS):
        updates["csv_paths"] = DEFAULT_TRAIN_CSV_PATHS

    if not updates:
        return cfg
    return cfg.clone_with(**updates)


def _build_existing_label_stats(expected_paths: Sequence[Path]):
    """把已完整存在的 label 路径汇总为统一统计对象。"""
    from SNN_based_method.utils.generate_precomputed_labels import GenerateLabelStats

    stats = GenerateLabelStats(
        planned=len(expected_paths),
        skipped_existing=len(expected_paths),
    )
    stats.label_roots = {
        str(path.parents[1])
        for path in expected_paths
        if len(path.parents) >= 2
    }
    return stats


def _infer_prior_debug_dir_name(label_dir_name: str | Path) -> Path:
    """根据 label_prior 目录名推导默认 debug 目录名。"""
    label_dir = Path(str(label_dir_name).strip())
    return label_dir.with_name(f"{label_dir.name}_debug")


def _generate_train_precomputed_labels(cfg: SNNConfig):
    """按 ``precomputed_label_dir_name`` 选择对应的预生成 label 实现。"""
    normalized_label_dir_name = Path(str(cfg.precomputed_label_dir_name).strip()).name.lower()

    if normalized_label_dir_name == "label_prior":
        from SNN_based_method.scripts.label_generate_new import (
            NewLabelConfig,
            run_with_config as run_prior_label_generation,
        )

        label_config = NewLabelConfig(
            data_paths=[Path(path) for path in cfg.data_paths or []],
            csv_paths=[Path(path) for path in cfg.csv_paths or []],
            pages_per_group=cfg.pages_per_group,
            total_pages=cfg.total_pages,
            time_threshold=cfg.time_threshold,
            label_dir_name=cfg.precomputed_label_dir_name,
            debug_dir_name=_infer_prior_debug_dir_name(cfg.precomputed_label_dir_name),
            labels_per_class=cfg.precomputed_labels_per_class,
            recursive=cfg.recursive,
            skip_missing_csv_raw=cfg.skip_missing_csv_raw,
            overwrite=False,
            dry_run=False,
            progress_interval=cfg.progress_interval,
        )
        return run_prior_label_generation(label_config)

    from SNN_based_method.utils.generate_precomputed_labels import (
        GenerateLabelConfig,
        ensure_precomputed_labels,
    )

    label_config = GenerateLabelConfig(
        data_paths=[Path(path) for path in cfg.data_paths or []],
        csv_paths=[Path(path) for path in cfg.csv_paths or []],
        pages_per_group=cfg.pages_per_group,
        total_pages=cfg.total_pages,
        time_threshold=cfg.time_threshold,
        active_point=cfg.active_point,
        label_dir_name=cfg.precomputed_label_dir_name,
        labels_per_class=cfg.precomputed_labels_per_class,
        recursive=cfg.recursive,
        skip_missing_csv_raw=cfg.skip_missing_csv_raw,
        overwrite=False,
        dry_run=False,
        progress_interval=cfg.progress_interval,
    )
    return ensure_precomputed_labels(label_config)


def ensure_train_precomputed_labels(cfg: SNNConfig):
    """训练前检查当前 pages_per_group 的 label 池。

    当 ``require_precomputed_labels=True`` 时, 会先按目录名选择对应生成器补齐
    缺失 label, 然后再做严格复检; 复检仍缺失时才报错, 不回退到在线弱标签。
    """
    if not cfg.return_label or not cfg.use_precomputed_labels:
        return None
    if not cfg.data_paths or not cfg.csv_paths:
        return None

    from SNN_based_method.utils.generate_precomputed_labels import (
        discover_expected_label_paths,
    )

    expected_paths = discover_expected_label_paths(
        cfg.data_paths,
        cfg.csv_paths,
        pages_per_group=cfg.pages_per_group,
        label_dir_name=cfg.precomputed_label_dir_name,
        labels_per_class=cfg.precomputed_labels_per_class,
        recursive=cfg.recursive,
        skip_missing_csv_raw=cfg.skip_missing_csv_raw,
    )
    if expected_paths and all(path.is_file() for path in expected_paths):
        stats = _build_existing_label_stats(expected_paths)
        existing_key = "required_existing" if cfg.require_precomputed_labels else "skipped_existing"
        print(
            "[precomputed labels] "
            f"pages_per_group={cfg.pages_per_group} {existing_key}={stats.skipped_existing} "
            f"roots={sorted(stats.label_roots or set())}"
        )
        return stats

    stats = _generate_train_precomputed_labels(cfg)
    if cfg.require_precomputed_labels:
        expected_paths = discover_expected_label_paths(
            cfg.data_paths,
            cfg.csv_paths,
            pages_per_group=cfg.pages_per_group,
            label_dir_name=cfg.precomputed_label_dir_name,
            labels_per_class=cfg.precomputed_labels_per_class,
            recursive=cfg.recursive,
            skip_missing_csv_raw=cfg.skip_missing_csv_raw,
        )
        if not expected_paths:
            raise FileNotFoundError(
                "precomputed label generation found no expected label paths; "
                "please verify data_paths/csv_paths/target_class columns"
            )
        missing_paths = [path for path in expected_paths if not path.is_file()]
        if missing_paths:
            preview = "\n".join(str(path) for path in missing_paths[:10])
            raise FileNotFoundError(
                f"missing required precomputed labels after generation: "
                f"{len(missing_paths)}/{len(expected_paths)}\n{preview}"
            )
    print(
        "[precomputed labels] "
        f"pages_per_group={cfg.pages_per_group} planned={stats.planned} "
        f"generated={stats.generated} skipped_existing={stats.skipped_existing} "
        f"roots={sorted(stats.label_roots or set())}"
        + (
            f" rejected={stats.rejected}"
            if hasattr(stats, "rejected")
            else ""
        )
    )
    return stats


def main(argv: Sequence[str] | None = None) -> None:
    """执行标准训练流程。"""
    args = build_argparser().parse_args(argv)
    resume_run_dir = prepare_resume_args(args)
    cfg = config_from_checkpoint_and_args(args)
    cfg = apply_default_train_paths(cfg)
    if args.save_every is not None:
        cfg = cfg.clone_with(save_every=args.save_every)
    seed_everything(cfg.seed)
    configure_torch_runtime(cfg)

    label_stats = ensure_train_precomputed_labels(cfg)
    train_loader, val_loader, test_loader, dataset = cfg.build_dataloaders()
    device = cfg.resolved_device()
    model = cfg.build_model().to(device)
    criterion = cfg.build_loss().to(device)
    metrics = cfg.build_metrics()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
    )

    start_epoch = 1
    best_val_loss = float("inf")
    scheduler_loaded = False
    if cfg.checkpoint_path:
        checkpoint = load_checkpoint(
            cfg.checkpoint_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("metrics", {}).get("best_val_loss", best_val_loss))
        scheduler_loaded = bool(checkpoint.get("_scheduler_loaded", False))
        if hasattr(scheduler, "T_max"):
            scheduler.T_max = max(cfg.epochs, 1)
        if not scheduler_loaded:
            scheduler.last_epoch = start_epoch - 1

    run_name = build_run_name(cfg, "train")
    cfg = cfg.clone_with(run_name=run_name)
    checkpoint_run_dir = make_checkpoint_run_dir(cfg, "train", run_name=run_name)
    logger, log_file = setup_logger(cfg.log_dir, run_name)
    cfg.save(checkpoint_run_dir / "config.json")

    logger.info("=== Training Configuration ===")
    logger.info("run_name=%s", run_name)
    logger.info("log_file=%s", log_file)
    logger.info("checkpoint_run_dir=%s", checkpoint_run_dir)
    if resume_run_dir is not None:
        logger.info("resume_run_dir=%s", resume_run_dir)
    logger.info("data_paths=%s", cfg.data_paths)
    logger.info("csv_paths=%s", cfg.csv_paths)
    if label_stats is not None:
        logger.info(
            "precomputed_labels pages_per_group=%d planned=%d generated=%d "
            "skipped_existing=%d roots=%s",
            cfg.pages_per_group,
            label_stats.planned,
            label_stats.generated,
            label_stats.skipped_existing,
            sorted(label_stats.label_roots or set()),
        )
    logger.info(
        "split train/val/test=%d/%d/%d dataset_size=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
        len(dataset),
    )
    for line in cfg.summary().splitlines():
        logger.info(line)
    logger.info("config_json=%s", json.dumps(cfg.to_dict(), ensure_ascii=False))
    if cfg.checkpoint_path:
        logger.info(
            "resumed_from=%s start_epoch=%d scheduler_loaded=%s",
            cfg.checkpoint_path,
            start_epoch,
            scheduler_loaded,
        )
    if start_epoch > cfg.epochs:
        logger.warning(
            "start_epoch=%d is greater than target epochs=%d; "
            "--epochs 表示总目标 epoch, 需要调大才会继续训练",
            start_epoch,
            cfg.epochs,
        )

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss, train_items = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg,
            epoch,
            trace_steps=max(0, args.trace_steps),
        )
        val_loss, val_items, val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            metrics,
            device,
            cfg,
            epoch,
        )
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        epoch_record = {
            "event": "epoch",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_items": train_items,
            "val_items": val_items,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
            "lr": scheduler.get_last_lr()[0],
        }

        logger.info(
            "Epoch [%d/%d] | train_loss=%.6f val_loss=%.6f best_val_loss=%.6f lr=%.8f",
            epoch,
            cfg.epochs,
            train_loss,
            val_loss,
            best_val_loss,
            scheduler.get_last_lr()[0],
        )
        logger.info("Epoch [%d/%d] | train_items: %s", epoch, cfg.epochs, _format_values(train_items))
        logger.info("Epoch [%d/%d] | val_items: %s", epoch, cfg.epochs, _format_values(val_items))
        logger.info("Epoch [%d/%d] | val_metrics: %s", epoch, cfg.epochs, _format_values(val_metrics))

        save_checkpoint(
            checkpoint_run_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            cfg=cfg,
            metrics=epoch_record,
        )
        if is_best:
            save_checkpoint(
                checkpoint_run_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                cfg=cfg,
                metrics=epoch_record,
            )
            logger.info(
                "Saved new best checkpoint to %s (val_loss=%.6f)",
                checkpoint_run_dir / "best.pth",
                best_val_loss,
            )
        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(
                checkpoint_run_dir / f"epoch_{epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                cfg=cfg,
                metrics=epoch_record,
            )
            logger.info("Saved epoch checkpoint to %s", checkpoint_run_dir / f"epoch_{epoch:03d}.pth")

        logger.info("Updated last checkpoint to %s", checkpoint_run_dir / "last.pth")

    logger.info("Training finished. Best val loss=%.6f", best_val_loss)
    logger.info("log_file=%s", log_file)
    logger.info("checkpoint_run_dir=%s", checkpoint_run_dir)


def main_without_cli() -> None:
    """无参数运行入口: 完全使用 SNNConfig 默认值启动纯 PLIF 训练。"""
    main([])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        main_without_cli()
# 接续训练命令示例:
# & D:/Anaconda3/envs/torchnew/python.exe d:/PYproject/SPAD/SNN_based_method/scripts/train.py --resume-run-dir d:/PYproject/SPAD/checkpoints/SNN/train_20260604_113734

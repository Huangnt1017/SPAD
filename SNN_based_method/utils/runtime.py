"""SNN 训练、测试脚本共享的运行时工具。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from SNN_based_method.config.SNN_config import SNNConfig
from SNN_based_method.utils.data import time_first_to_model_input

_ARG_UNSET = object()


def _parse_optional_float(value: str) -> float | None:
    """解析可选浮点数; 允许用 none/null 表示 Python None。"""
    normalized = str(value).strip().lower()
    if normalized in {"none", "null"}:
        return None
    return float(value)


class _StoreOptionalFloat(argparse.Action):
    """支持显式把命令行参数设置为浮点数或 ``None``。"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, _parse_optional_float(values))
        setattr(namespace, f"{self.dest}__specified", True)


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """向 argparse 解析器添加 SNNConfig 的通用命令行覆盖参数。"""
    parser.add_argument("--config", default=None, help="SNNConfig JSON 配置文件路径")
    parser.add_argument("--data-paths", nargs="+", default=None, help="raw 文件或包含 raw 文件的目录")
    parser.add_argument("--csv-paths", nargs="+", default=None, help="CSV 样本清单路径, 需包含 file_path 列")
    parser.add_argument(
        "--skip-missing-csv-raw",
        action="store_true",
        help="CSV 中列出的 raw 文件缺失时跳过该行; 默认严格报错",
    )
    parser.add_argument("--pages-per-group", type=int, default=None, help="每个样本使用的 raw page 数 P")
    parser.add_argument("--total-pages", type=int, default=None, help="每个 raw 文件最多读取的 page 数")
    parser.add_argument("--time-threshold", type=int, default=None, help="有效 ToF 上限, 超过后视为无效")
    parser.add_argument(
        "--raw-load-mode",
        choices=["group", "file_cache"],
        default=None,
        help="raw 读取模式: group 只读当前分组; file_cache 读取并缓存整文件",
    )
    parser.add_argument("--split-ratios", nargs=3, type=float, default=None, metavar=("TRAIN", "VAL", "TEST"), help="train/val/test 划分比例, 三项之和必须为 1")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker 数")
    pin_memory_group = parser.add_mutually_exclusive_group()
    pin_memory_group.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=None,
        help="DataLoader 使用 pinned memory 加速 CPU 到 CUDA 拷贝",
    )
    pin_memory_group.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="关闭 DataLoader pinned memory",
    )
    persistent_workers_group = parser.add_mutually_exclusive_group()
    persistent_workers_group.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        default=None,
        help="num_workers > 0 时保持 DataLoader worker 常驻",
    )
    persistent_workers_group.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="每个 epoch 结束后关闭 DataLoader worker",
    )
    parser.add_argument("--prefetch-factor", type=int, default=None, help="每个 DataLoader worker 预取的 batch 数")
    precompute_group = parser.add_mutually_exclusive_group()
    precompute_group.add_argument(
        "--precompute-model-input",
        dest="precompute_model_input",
        action="store_true",
        default=None,
        help="在 DataLoader worker 中预先生成 model_input=[B,4096,P]",
    )
    precompute_group.add_argument(
        "--no-precompute-model-input",
        dest="precompute_model_input",
        action="store_false",
        help="在训练循环中从 frames 生成 model_input",
    )
    parser.add_argument("--epochs", type=int, default=None, help="训练 epoch 数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW 权重衰减")
    parser.add_argument("--grad-clip", type=float, default=None, help="梯度裁剪最大范数")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--device", default=None, help="运行设备: auto/cpu/cuda")
    parser.add_argument("--log-dir", default=None, help="训练日志、测试结果输出目录")
    parser.add_argument("--checkpoint-dir", default=None, help="训练 checkpoint 输出目录")
    parser.add_argument("--output-dir", default=None, help="旧版统一实验产物输出目录")
    parser.add_argument("--run-name", default=None, help="本次运行名称")
    parser.add_argument("--checkpoint", dest="checkpoint_path", default=None, help="checkpoint 路径")
    parser.add_argument(
        "--model-backend",
        choices=[
            "new",
            "activation",
            "activation_based",
            "legacy",
            "clock",
            "clock_driven",
            "rnn",
            "recurrent",
            "srnn",
            "lstm",
            "clstm",
            "convlstm",
            "gru",
            "cgru",
            "convgru",
        ],
        default=None,
        help="模型后端; new 为官方 activation_based SNN, rnn/lstm/gru 为显式时序递推版本",
    )
    parser.add_argument("--encoding-mode", choices=["sinusoidal", "lut"], default=None, help="ToF 编码方式")
    parser.add_argument("--embed-dim", type=int, default=None, help="LUT 编码维度")
    parser.add_argument("--lut-init", choices=["sinusoidal", "rbf", "random"], default=None, help="LUT 初始化方式")
    parser.add_argument("--C", type=int, default=None, help="模型主干通道数")
    parser.add_argument("--chunk-size", type=int, default=None, help="按时间维切块的 chunk 大小")
    parser.add_argument("--spike-mode", choices=["plif", "lif", "if"], default=None, help="脉冲神经元类型")
    parser.add_argument("--spike-tau", type=float, default=None, help="LIF/PLIF 膜时间常数; IF 模式下忽略")
    parser.add_argument("--spike-v-threshold", type=float, default=None, help="脉冲发放阈值")
    parser.add_argument(
        "--spike-v-reset",
        action=_StoreOptionalFloat,
        default=_ARG_UNSET,
        help="脉冲重置电位; 传 none 或 null 表示 soft reset",
    )
    parser.add_argument(
        "--spike-backend",
        choices=["auto", "cupy", "torch"],
        default=None,
        help="spikingjelly 神经元后端; auto 在 CUDA 可用时优先 cupy",
    )
    parser.add_argument("--num-blocks", type=int, default=None, help="SpikeBlock 数量")
    parser.add_argument("--refine-mid", type=int, default=None, help="深度/强度精修头的中间通道数")
    sequence_group = parser.add_mutually_exclusive_group()
    sequence_group.add_argument(
        "--return-sequence",
        dest="return_sequence",
        action="store_true",
        default=None,
        help="返回完整 gate/tof/valid 时间序列; 训练 var/sparse loss 时需要",
    )
    sequence_group.add_argument(
        "--no-return-sequence",
        dest="return_sequence",
        action="store_false",
        help="只返回最终图像输出; 推理时可降低显存和日志负担",
    )
    parser.add_argument("--w-gt", type=float, default=None, help="GT L1 loss 权重")
    parser.add_argument("--w-depth-reg", type=float, default=None, help="有效 depth 区域回归 loss 权重")
    parser.add_argument("--w-ssim", type=float, default=None, help="SSIM loss 权重")
    parser.add_argument("--w-var", type=float, default=None, help="gate 方差 loss 权重")
    parser.add_argument("--w-sparse", type=float, default=None, help="gate 稀疏 loss 权重")
    parser.add_argument("--w-smooth", type=float, default=None, help="强度引导平滑 loss 权重")
    parser.add_argument("--w-lut-smooth", type=float, default=None, help="LUT 相邻 bin 平滑正则权重")
    parser.add_argument("--w-lut-norm", type=float, default=None, help="LUT 范数一致性正则权重")
    parser.add_argument("--sigma-target", type=float, default=None, help="gate 方差 loss 的目标 sigma, 单位为 bin")
    parser.add_argument("--rho-target", type=float, default=None, help="gate 稀疏 loss 的目标平均激活率")
    parser.add_argument(
        "--sparse-mode",
        choices=["upper", "target", "band"],
        default=None,
        help="gate 稀疏正则模式: upper=旧单边阈值, target=贴近目标率, band=保持在区间内",
    )
    parser.add_argument("--rho-min", type=float, default=None, help="sparse_mode=band 时的平均 gate 激活率下限")
    parser.add_argument("--rho-max", type=float, default=None, help="sparse_mode=band 时的平均 gate 激活率上限")
    parser.add_argument("--beta-smooth", type=float, default=None, help="强度引导平滑 loss 的边缘衰减系数")
    parser.add_argument("--ssim-kernel-size", type=int, default=None, help="SSIM 高斯窗口大小")
    parser.add_argument(
        "--ssim-smooth-kernel-size",
        type=int,
        default=None,
        help="SSIM 前均值滤波窗口; 1 表示关闭, 必须为奇数",
    )
    parser.add_argument(
        "--depth-reg-mode",
        choices=["mse", "charbonnier", "l1"],
        default=None,
        help="depth 回归项类型: mse 对齐 PSNR, charbonnier 抗噪, l1 更稳健",
    )
    depth_reg_mask_group = parser.add_mutually_exclusive_group()
    depth_reg_mask_group.add_argument(
        "--depth-reg-use-mask",
        dest="depth_reg_use_mask",
        action="store_true",
        default=None,
        help="depth 回归项仅在 depth_gt > 0 区域计算",
    )
    depth_reg_mask_group.add_argument(
        "--no-depth-reg-mask",
        dest="depth_reg_use_mask",
        action="store_false",
        help="depth 回归项在全图计算",
    )
    parser.add_argument(
        "--depth-reg-charbonnier-eps",
        type=float,
        default=None,
        help="depth_reg_mode=charbonnier 时的平滑常数",
    )
    gt_mask_group = parser.add_mutually_exclusive_group()
    gt_mask_group.add_argument(
        "--gt-use-mask",
        dest="gt_use_mask",
        action="store_true",
        default=None,
        help="GT L1 仅在 depth_gt > 0 区域计算",
    )
    gt_mask_group.add_argument(
        "--no-gt-mask",
        dest="gt_use_mask",
        action="store_false",
        help="GT L1 在全图计算",
    )
    ssim_mask_group = parser.add_mutually_exclusive_group()
    ssim_mask_group.add_argument(
        "--ssim-use-mask",
        dest="ssim_use_mask",
        action="store_true",
        default=None,
        help="SSIM 仅在 depth_gt > 0 区域计算",
    )
    ssim_mask_group.add_argument(
        "--no-ssim-mask",
        dest="ssim_use_mask",
        action="store_false",
        help="SSIM 在全图计算",
    )
    parser.add_argument("--recursive", action="store_true", help="递归搜索数据目录")
    parser.add_argument("--no-label", action="store_true", help="关闭弱标签生成")
    precomputed_labels_group = parser.add_mutually_exclusive_group()
    precomputed_labels_group.add_argument(
        "--use-precomputed-labels",
        dest="use_precomputed_labels",
        action="store_true",
        default=None,
        help="训练时优先读取 CSV 对应的预生成 .npy 标签",
    )
    precomputed_labels_group.add_argument(
        "--no-precomputed-labels",
        dest="use_precomputed_labels",
        action="store_false",
        help="关闭预生成 .npy 标签读取, 改为在线生成弱标签",
    )
    require_precomputed_group = parser.add_mutually_exclusive_group()
    require_precomputed_group.add_argument(
        "--require-precomputed-labels",
        dest="require_precomputed_labels",
        action="store_true",
        default=None,
        help="预生成标签缺失时直接报错, 不回退到在线弱标签",
    )
    require_precomputed_group.add_argument(
        "--no-require-precomputed-labels",
        dest="require_precomputed_labels",
        action="store_false",
        help="预生成标签缺失时允许回退到在线弱标签",
    )
    parser.add_argument(
        "--precomputed-label-dir-name",
        default=None,
        help="预生成标签目录名; 相对路径按 CSV 所在目录解析, 默认 label",
    )
    parser.add_argument(
        "--precomputed-labels-per-class",
        type=int,
        default=None,
        help="每个 target_class 预生成并随机抽取的 label 数量, 默认 5",
    )
    parser.add_argument("--normalize-input", action="store_true", help="按 time_threshold 归一化输入")
    parser.add_argument(
        "--augment-train",
        action="store_true",
        help="训练集启用 raw group 级 ToF shift 增强",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=None,
        help="每个训练样本额外生成的增强样本份数, 默认使用配置值",
    )
    original_sample_group = parser.add_mutually_exclusive_group()
    original_sample_group.add_argument(
        "--keep-original-sample",
        dest="keep_original_sample",
        action="store_true",
        default=None,
        help="训练增强展开时保留 aug_index=0 的原始样本",
    )
    original_sample_group.add_argument(
        "--no-keep-original-sample",
        dest="keep_original_sample",
        action="store_false",
        help="训练增强展开时不保留 aug_index=0 的原始样本",
    )
    parser.add_argument("--tof-shift-max", type=int, default=None, help="ToF shift 最大整数偏移, 默认 15")
    parser.add_argument("--tof-shift-prob", type=float, default=None, help="每个训练样本执行 ToF shift 的概率")
    parser.add_argument("--page-dropout", action="store_true", help="训练集随机丢弃整页 raw page")
    parser.add_argument("--page-dropout-prob", type=float, default=None, help="PageDropout 中每页被置 0 的概率")
    parser.add_argument("--shuffle-pages", action="store_true", help="训练集打乱单个样本内部的 page 顺序")
    parser.add_argument("--page-shuffle", action="store_true", help="--shuffle-pages 的同义开关")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help="启用 CUDA autocast 混合精度",
    )
    amp_group.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="关闭 CUDA autocast 混合精度",
    )
    tf32_group = parser.add_mutually_exclusive_group()
    tf32_group.add_argument(
        "--tf32",
        dest="tf32",
        action="store_true",
        default=None,
        help="允许 Ampere/Ada GPU 使用 TF32 matmul/conv",
    )
    tf32_group.add_argument(
        "--no-tf32",
        dest="tf32",
        action="store_false",
        help="关闭 TF32",
    )
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--cudnn-benchmark",
        dest="cudnn_benchmark",
        action="store_true",
        default=None,
        help="固定输入尺寸时启用 cuDNN benchmark",
    )
    benchmark_group.add_argument(
        "--no-cudnn-benchmark",
        dest="cudnn_benchmark",
        action="store_false",
        help="关闭 cuDNN benchmark",
    )
    cuda_prefetch_group = parser.add_mutually_exclusive_group()
    cuda_prefetch_group.add_argument(
        "--cuda-prefetch",
        dest="cuda_prefetch",
        action="store_true",
        default=None,
        help="训练/验证时用独立 CUDA stream 预取下一批",
    )
    cuda_prefetch_group.add_argument(
        "--no-cuda-prefetch",
        dest="cuda_prefetch",
        action="store_false",
        help="关闭 CUDA stream batch 预取",
    )
    parser.add_argument("--progress-interval", type=int, default=None, help="进度条每 N 个 batch 同步一次 loss")
    spike_log_group = parser.add_mutually_exclusive_group()
    spike_log_group.add_argument(
        "--log-spike-stats",
        dest="log_spike_stats",
        action="store_true",
        default=None,
        help="统计并输出脉冲神经元放电率",
    )
    spike_log_group.add_argument(
        "--no-log-spike-stats",
        dest="log_spike_stats",
        action="store_false",
        help="关闭脉冲神经元放电率统计",
    )


def config_from_args(args: argparse.Namespace) -> SNNConfig:
    """加载配置文件, 并应用命令行参数覆盖。"""
    if args.config:
        cfg = SNNConfig.load(args.config)
    else:
        cfg = SNNConfig()

    updates: dict[str, Any] = {}
    for key in (
        "data_paths",
        "csv_paths",
        "pages_per_group",
        "total_pages",
        "time_threshold",
        "raw_load_mode",
        "split_ratios",
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "precompute_model_input",
        "epochs",
        "lr",
        "weight_decay",
        "grad_clip",
        "grad_accum_steps",
        "device",
        "log_dir",
        "checkpoint_dir",
        "output_dir",
        "run_name",
        "checkpoint_path",
        "model_backend",
        "encoding_mode",
        "embed_dim",
        "lut_init",
        "C",
        "chunk_size",
        "spike_mode",
        "spike_tau",
        "spike_v_threshold",
        "spike_v_reset",
        "spike_backend",
        "num_blocks",
        "refine_mid",
        "return_sequence",
        "w_gt",
        "w_depth_reg",
        "w_ssim",
        "w_var",
        "w_sparse",
        "w_smooth",
        "w_lut_smooth",
        "w_lut_norm",
        "sigma_target",
        "rho_target",
        "sparse_mode",
        "rho_min",
        "rho_max",
        "beta_smooth",
        "ssim_kernel_size",
        "ssim_smooth_kernel_size",
        "depth_reg_mode",
        "depth_reg_use_mask",
        "depth_reg_charbonnier_eps",
        "gt_use_mask",
        "ssim_use_mask",
        "use_precomputed_labels",
        "require_precomputed_labels",
        "precomputed_label_dir_name",
        "precomputed_labels_per_class",
        "num_aug",
        "keep_original_sample",
        "tof_shift_max",
        "tof_shift_prob",
        "page_dropout_prob",
        "amp",
        "tf32",
        "cudnn_benchmark",
        "cuda_prefetch",
        "progress_interval",
        "log_spike_stats",
    ):
        if hasattr(args, key):
            value = getattr(args, key)
            is_explicit_none = bool(getattr(args, f"{key}__specified", False))
            if (value is not None and value is not _ARG_UNSET) or is_explicit_none:
                updates[key] = value
    if getattr(args, "data_paths", None) is not None and getattr(args, "csv_paths", None) is None:
        updates["csv_paths"] = None

    if getattr(args, "recursive", False):
        updates["recursive"] = True
    if getattr(args, "skip_missing_csv_raw", False):
        updates["skip_missing_csv_raw"] = True
    if getattr(args, "no_label", False):
        updates["return_label"] = False
    if getattr(args, "normalize_input", False):
        updates["normalize_input"] = True
    if getattr(args, "augment_train", False):
        updates["augment_train"] = True
    if getattr(args, "page_dropout", False):
        updates["page_dropout"] = True
    if getattr(args, "shuffle_pages", False) or getattr(args, "page_shuffle", False):
        updates["shuffle_pages"] = True
    _apply_legacy_output_dir(updates)
    return cfg.clone_with(**updates)


def config_from_checkpoint_and_args(args: argparse.Namespace) -> SNNConfig:
    """从 --config 或 checkpoint 元数据加载配置, 再应用命令行覆盖。"""
    cfg: SNNConfig | None = None
    if args.config:
        cfg = SNNConfig.load(args.config)
    elif getattr(args, "checkpoint_path", None):
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
            checkpoint_config = checkpoint.get("config")
            if checkpoint_config is not None:
                if "split_ratios" in checkpoint_config:
                    checkpoint_config["split_ratios"] = tuple(checkpoint_config["split_ratios"])
                cfg = SNNConfig(**checkpoint_config)
        except Exception:
            cfg = None

    if cfg is None:
        cfg = SNNConfig()

    args_without_config = argparse.Namespace(**vars(args))
    args_without_config.config = None
    cfg = _apply_arg_overrides(cfg, args_without_config)
    if (
        getattr(args, "checkpoint_path", None)
        and not getattr(args, "config", None)
        and getattr(args, "run_name", None) is None
    ):
        cfg = cfg.clone_with(run_name=None)
    return cfg


def _apply_arg_overrides(cfg: SNNConfig, args: argparse.Namespace) -> SNNConfig:
    """把命令行参数覆盖到已有配置对象上。"""
    updates: dict[str, Any] = {}
    for key in (
        "data_paths",
        "csv_paths",
        "pages_per_group",
        "total_pages",
        "time_threshold",
        "raw_load_mode",
        "split_ratios",
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "precompute_model_input",
        "epochs",
        "lr",
        "weight_decay",
        "grad_clip",
        "grad_accum_steps",
        "device",
        "log_dir",
        "checkpoint_dir",
        "output_dir",
        "run_name",
        "checkpoint_path",
        "model_backend",
        "encoding_mode",
        "embed_dim",
        "lut_init",
        "C",
        "chunk_size",
        "spike_mode",
        "spike_tau",
        "spike_v_threshold",
        "spike_v_reset",
        "spike_backend",
        "num_blocks",
        "refine_mid",
        "return_sequence",
        "w_gt",
        "w_depth_reg",
        "w_ssim",
        "w_var",
        "w_sparse",
        "w_smooth",
        "w_lut_smooth",
        "w_lut_norm",
        "sigma_target",
        "rho_target",
        "sparse_mode",
        "rho_min",
        "rho_max",
        "beta_smooth",
        "ssim_kernel_size",
        "ssim_smooth_kernel_size",
        "depth_reg_mode",
        "depth_reg_use_mask",
        "depth_reg_charbonnier_eps",
        "gt_use_mask",
        "ssim_use_mask",
        "use_precomputed_labels",
        "require_precomputed_labels",
        "precomputed_label_dir_name",
        "precomputed_labels_per_class",
        "num_aug",
        "keep_original_sample",
        "tof_shift_max",
        "tof_shift_prob",
        "page_dropout_prob",
        "amp",
        "tf32",
        "cudnn_benchmark",
        "cuda_prefetch",
        "progress_interval",
        "log_spike_stats",
    ):
        if hasattr(args, key):
            value = getattr(args, key)
            is_explicit_none = bool(getattr(args, f"{key}__specified", False))
            if (value is not None and value is not _ARG_UNSET) or is_explicit_none:
                updates[key] = value
    if getattr(args, "data_paths", None) is not None and getattr(args, "csv_paths", None) is None:
        updates["csv_paths"] = None

    if getattr(args, "recursive", False):
        updates["recursive"] = True
    if getattr(args, "skip_missing_csv_raw", False):
        updates["skip_missing_csv_raw"] = True
    if getattr(args, "no_label", False):
        updates["return_label"] = False
    if getattr(args, "normalize_input", False):
        updates["normalize_input"] = True
    if getattr(args, "augment_train", False):
        updates["augment_train"] = True
    if getattr(args, "page_dropout", False):
        updates["page_dropout"] = True
    if getattr(args, "shuffle_pages", False) or getattr(args, "page_shuffle", False):
        updates["shuffle_pages"] = True
    _apply_legacy_output_dir(updates)
    return cfg.clone_with(**updates)


def _apply_legacy_output_dir(updates: dict[str, Any]) -> None:
    """兼容旧版 ``--output-dir``: 未显式指定时同时覆盖日志和 checkpoint 根目录。"""
    if "output_dir" not in updates:
        return
    if "log_dir" not in updates:
        updates["log_dir"] = updates["output_dir"]
    if "checkpoint_dir" not in updates:
        updates["checkpoint_dir"] = updates["output_dir"]


def resolve_output_root(path: str | Path) -> Path:
    """把输出根目录解析为绝对路径; 相对路径以项目根目录为基准。"""
    output_root = Path(path)
    if not output_root.is_absolute():
        output_root = _PROJECT_ROOT / output_root
    return output_root


def build_run_name(cfg: SNNConfig, prefix: str) -> str:
    """生成运行目录名; 未指定 ``run_name`` 时使用前缀和时间戳。"""
    run_name = cfg.run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{prefix}_{timestamp}"
    return run_name


def make_run_dir(cfg: SNNConfig, prefix: str) -> Path:
    """创建旧版统一运行目录; 新代码优先使用日志/checkpoint 专用目录。"""
    run_dir = resolve_output_root(cfg.output_dir) / build_run_name(cfg, prefix)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_log_run_dir(cfg: SNNConfig, prefix: str, *, run_name: str | None = None) -> Path:
    """创建日志、summary 和预测结果的运行目录。"""
    run_dir = resolve_output_root(cfg.log_dir) / (run_name or build_run_name(cfg, prefix))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_checkpoint_run_dir(cfg: SNNConfig, prefix: str, *, run_name: str | None = None) -> Path:
    """创建 checkpoint 的运行目录。"""
    run_dir = resolve_output_root(cfg.checkpoint_dir) / (run_name or build_run_name(cfg, prefix))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """向 JSONL 日志追加一条记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None = None,
    epoch: int,
    cfg: SNNConfig,
    metrics: dict[str, Any],
) -> None:
    """保存模型 checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """加载 checkpoint 到模型, 并按需恢复优化器和学习率调度器。"""
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    has_adapted_state = state_dict_needs_model_adaptation(state_dict, model)
    state_dict = adapt_state_dict_for_model(state_dict, model)
    model.load_state_dict(state_dict)
    optimizer_state = checkpoint.get("optimizer", checkpoint.get("optimizer_state_dict"))
    if optimizer is not None and optimizer_state is not None and not has_adapted_state:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = checkpoint.get("scheduler", checkpoint.get("scheduler_state_dict"))
    scheduler_loaded = False
    if scheduler is not None and scheduler_state is not None and not has_adapted_state:
        scheduler.load_state_dict(scheduler_state)
        scheduler_loaded = True
    checkpoint["_scheduler_loaded"] = scheduler_loaded
    return checkpoint


def state_dict_needs_model_adaptation(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> bool:
    """当 checkpoint 需要旧结构兼容适配时返回 True。"""
    model_state = model.state_dict()
    for key, value in state_dict.items():
        if key.startswith("refine.net."):
            return True
        target = model_state.get(key)
        if target is not None and value.shape != target.shape:
            return True
    return False


def _pad_conv_input_if_needed(
    value: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """旧卷积输入通道少于新卷积时, 复制旧通道并把新增通道置 0。"""
    can_pad_conv_input = (
        value.ndim == 4
        and target.ndim == 4
        and value.shape[0] == target.shape[0]
        and value.shape[2:] == target.shape[2:]
        and value.shape[1] < target.shape[1]
    )
    if not can_pad_conv_input:
        return value

    padded = torch.zeros_like(target)
    padded[:, : value.shape[1], :, :] = value
    return padded


def _map_legacy_refine_key(
    key: str,
    value: torch.Tensor,
    model_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """把旧版共享 refine.net.* 权重映射到 depth/intensity 两个精修头。"""
    if not key.startswith("refine.net."):
        return {}

    suffix = key.removeprefix("refine.net.")
    mapped: dict[str, torch.Tensor] = {}

    if suffix.startswith("0.") or suffix.startswith("1."):
        for branch in ("depth_net", "intensity_net"):
            new_key = f"refine.{branch}.{suffix}"
            target = model_state.get(new_key)
            if target is not None:
                mapped[new_key] = _pad_conv_input_if_needed(value, target)
        return mapped

    if suffix.startswith("3."):
        for branch, channel_index in (("depth_net", 0), ("intensity_net", 1)):
            new_key = f"refine.{branch}.{suffix}"
            target = model_state.get(new_key)
            if target is None:
                continue
            if (
                value.ndim >= 1
                and target.ndim >= 1
                and value.shape[0] >= channel_index + 1
                and target.shape[0] == 1
            ):
                branch_value = value[channel_index : channel_index + 1]
            else:
                branch_value = value
            mapped[new_key] = _pad_conv_input_if_needed(branch_value, target)
        return mapped

    return {}


def adapt_state_dict_for_model(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """在严格加载前适配少量向后兼容的形状变化。

    兼容两类历史变化:
    1. 共享 ``refine.net.*`` 精修头拆成 ``depth_net`` 和 ``intensity_net``。
    2. 置信度门控让第一层卷积输入从 2 通道扩展到 3 通道。
    """
    model_state = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        legacy_refine = _map_legacy_refine_key(key, value, model_state)
        if legacy_refine:
            adapted.update(legacy_refine)
            continue
        if key.startswith("refine.net."):
            continue

        target = model_state.get(key)
        if target is None:
            adapted[key] = value
            continue
        if value.shape == target.shape:
            adapted[key] = value
            continue

        padded = _pad_conv_input_if_needed(value, target)
        if padded.shape == target.shape:
            adapted[key] = padded
        else:
            adapted[key] = value
    return adapted


def reset_spiking_state(model: torch.nn.Module) -> None:
    """按官方 ``spikingjelly.activation_based`` API 重置脉冲神经元状态。"""
    try:
        from spikingjelly.activation_based import functional
    except ImportError as exc:
        raise RuntimeError(
            "当前 SNN 项目需要环境中安装官方 spikingjelly, "
            "且必须包含 spikingjelly.activation_based。"
        ) from exc

    functional.reset_net(model)


def prepare_model_input(frames: torch.Tensor) -> torch.Tensor:
    """把 DataLoader 输出的 ``[P, B, 1, 64, 64]`` 转成模型输入。"""
    return time_first_to_model_input(frames)


def reduce_loss_dict(losses: dict[str, Any]) -> dict[str, float]:
    """把 loss 字典中的张量转换为普通 float。"""
    reduced: dict[str, float] = {}
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            reduced[key] = float(value.detach().cpu().item())
        else:
            reduced[key] = float(value)
    return reduced


def update_average(total: dict[str, float], values: dict[str, float]) -> None:
    """把指标或 loss 累加到可变字典中。"""
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + float(value)


def divide_average(total: dict[str, float], count: int) -> dict[str, float]:
    """按计数求平均值。"""
    divisor = max(count, 1)
    return {key: value / divisor for key, value in total.items()}

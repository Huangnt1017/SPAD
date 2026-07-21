"""对 20250430 根目录 A--Z ``xyzi`` 点云运行 I-WRS 降采样。

CLI 示例：
    python scripts/run_iwrs_az_downsampling.py --dry-run
    python scripts/run_iwrs_az_downsampling.py --num-samples 1024 --gamma 0.5 --seed 42
    python scripts/run_iwrs_az_downsampling.py --preview-label A

无参运行：
    python scripts/run_iwrs_az_downsampling.py
    使用 ``main_without_cli`` 中的显式默认路径和参数处理 A.txt--Z.txt。

参数说明：
    --input-dir: 只读取该目录根部名称严格为 A.txt--Z.txt 的 26 个文件。
    --output-dir: 默认按 K、gamma 和 seed 自动生成到输入数据目录下的
        ``downsampling_results`` 子目录。
    --num-samples: 每个字母输出的原始点数，默认 1024。
    --gamma: 光子计数权重指数，默认 0.5。
    --seed: 基础随机种子；每个字母使用 ``seed + 字母序号``。
    --device: cpu/cuda/auto，默认 cpu。
    --overwrite: 允许覆盖已存在的同名输出文件。
    --preview-label: 处理完成后调用 ``data_read/raw2pointcloud.py`` 的
        ``plot_pc(..., mode='ds')`` 手动预览指定字母；默认不预览、不保存图片。
    --dry-run: 只校验参数、文件列表和输出路径，不创建目录或结果文件。

输入输出契约：
    输入：逗号分隔 ``N x 4`` 整数文本，列为 ``x,y,z,intensity``。
    输出：逗号分隔 ``K x 4`` 整数文本，保持 A.txt--Z.txt 文件名；
        每行均是对应输入文件中的原始行。程序只保存点云文本，不保存图片。
"""

from __future__ import annotations

import argparse
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downsampling import IntensityWeightedRandomSampler, assert_unique_indices


DEFAULT_INPUT_DIR = Path(r"D:\PYproject\SPADdata\20250430")
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_DIR / "downsampling_results"
ALPHABET = tuple(string.ascii_uppercase)


@dataclass(frozen=True)
class IWRSAZConfig:
    """A--Z I-WRS 批处理配置。"""

    input_dir: Path
    output_dir: Optional[Path] = None
    num_samples: int = 1024
    gamma: float = 0.5
    seed: int = 42
    device: str = "cpu"
    overwrite: bool = False
    preview_label: str = ""
    dry_run: bool = False


def _float_path_token(value: float) -> str:
    """把浮点参数转换为适合目录名的稳定短字符串。"""

    return format(value, ".8g").replace("-", "m").replace(".", "p")


def resolve_output_dir(config: IWRSAZConfig) -> Path:
    """解析显式输出目录，或根据方法参数构造默认目录。"""

    if config.output_dir is not None:
        return config.output_dir.expanduser().resolve()
    gamma_token = _float_path_token(config.gamma)
    directory_name = (
        f"20250430_i_wrs_k{config.num_samples}_"
        f"gamma{gamma_token}_seed{config.seed}"
    )
    return (DEFAULT_OUTPUT_ROOT / directory_name).resolve()


def resolve_device(device_name: str) -> torch.device:
    """解析运行设备，并在显式 CUDA 不可用时给出可操作错误。"""

    normalized = device_name.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"device must be one of auto/cpu/cuda, got: {device_name!r}"
        )
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was requested, but torch.cuda.is_available() is False"
        )
    return torch.device(normalized)


def discover_az_files(input_dir: Path) -> Dict[str, Path]:
    """发现目录根部严格命名为 A.txt--Z.txt 的 26 份输入。"""

    resolved_input = input_dir.expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"input directory does not exist: {resolved_input}")
    if not resolved_input.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {resolved_input}")

    files = {
        label: resolved_input / f"{label}.txt"
        for label in ALPHABET
    }
    missing = [label for label, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing A-Z input files for labels: " + ", ".join(missing)
        )
    return files


def validate_config(config: IWRSAZConfig) -> None:
    """校验批处理参数。"""

    if config.num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {config.num_samples}")
    if config.gamma <= 0:
        raise ValueError(f"gamma must be positive, got {config.gamma}")
    if config.preview_label and config.preview_label not in ALPHABET:
        raise ValueError(
            f"preview_label must be empty or A-Z, got {config.preview_label!r}"
        )


def load_xyzi_text(path: Path) -> np.ndarray:
    """复用项目现有整数点云格式读取单份 ``N x 4`` 文本。"""

    points = np.loadtxt(path, delimiter=",", dtype=np.int32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(
            f"{path} must contain an N x 4 xyzi array, got {points.shape}"
        )
    if points.shape[0] <= 0:
        raise ValueError(f"{path} contains no points")
    if np.any(points[:, 3] < 0):
        raise ValueError(f"{path} contains negative photon counts")
    return points


def save_xyzi_text(path: Path, points: np.ndarray) -> None:
    """以逗号分隔四列整数格式原子保存点云。"""

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        np.savetxt(temporary_path, points, delimiter=",", fmt="%d")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def preview_saved_result(path: Path) -> None:
    """显式请求时复用现有 ``plot_pc`` 弹窗预览，不保存图片。"""

    from data_read.raw2pointcloud import plot_pc, read_pc

    points = read_pc(str(path))
    plot_pc(points, mode="ds")


def run_with_config(config: IWRSAZConfig) -> List[Path]:
    """运行 A--Z I-WRS 批处理并返回输出文件列表。"""

    validate_config(config)
    input_files = discover_az_files(config.input_dir)
    output_dir = resolve_output_dir(config)
    device = resolve_device(config.device)

    print("=== I-WRS A-Z Downsampling ===")
    print(f"input_dir={config.input_dir.expanduser().resolve()}")
    print(f"output_dir={output_dir}")
    print(f"num_files={len(input_files)}")
    print(f"num_samples={config.num_samples}")
    print(f"gamma={config.gamma}")
    print(f"base_seed={config.seed}")
    print(f"device={device}")
    print(f"dry_run={config.dry_run}")
    print("save_images=False")

    output_files = [output_dir / f"{label}.txt" for label in ALPHABET]
    if config.dry_run:
        for label in ALPHABET:
            output_path = output_dir / f"{label}.txt"
            status = " [exists]" if output_path.exists() else ""
            print(
                f"[dry-run] {input_files[label].name} -> "
                f"{output_path}{status}"
            )
        return output_files

    existing_outputs = [path for path in output_files if path.exists()]
    if existing_outputs and not config.overwrite:
        examples = ", ".join(path.name for path in existing_outputs[:5])
        raise FileExistsError(
            f"output files already exist in {output_dir}: {examples}. "
            "Use --overwrite to replace them."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = IntensityWeightedRandomSampler(
        num_samples=config.num_samples,
        gamma=config.gamma,
    ).to(device)
    sampler.eval()

    written_files: List[Path] = []
    for label_index, label in enumerate(ALPHABET):
        input_path = input_files[label]
        output_path = output_dir / f"{label}.txt"
        input_points = load_xyzi_text(input_path)
        if input_points.shape[0] < config.num_samples:
            raise ValueError(
                f"{input_path} contains {input_points.shape[0]} points, fewer than "
                f"num_samples={config.num_samples}"
            )

        point_tensor = torch.from_numpy(
            input_points.astype(np.float32, copy=False)
        ).unsqueeze(0).to(device)
        label_seed = config.seed + label_index
        generator = torch.Generator(device=device).manual_seed(label_seed)

        with torch.inference_mode():
            output = sampler(point_tensor, generator=generator)
        assert_unique_indices(output.indices)
        selected_indices = output.indices.squeeze(0).cpu().numpy()
        sampled_points = input_points[selected_indices]

        if sampled_points.shape != (config.num_samples, 4):
            raise RuntimeError(
                f"unexpected sampled shape for {label}: {sampled_points.shape}"
            )
        save_xyzi_text(output_path, sampled_points)
        written_files.append(output_path)
        print(
            f"[{label}] input={input_points.shape[0]} -> "
            f"output={sampled_points.shape[0]}, seed={label_seed}, "
            f"saved={output_path}"
        )

    if config.preview_label:
        preview_path = output_dir / f"{config.preview_label}.txt"
        preview_saved_result(preview_path)

    print(f"Finished: saved {len(written_files)} point-cloud files; saved 0 images.")
    return written_files


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description=(
            "Apply intensity-weighted random sampling to A.txt-Z.txt and save "
            "only K x 4 point-cloud text results."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="包含根目录 A.txt-Z.txt 的输入目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；省略时按 K/gamma/seed 自动生成",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="每个字母输出的点数，默认 1024",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="I-WRS 强度权重指数，默认 0.5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="基础随机种子，默认 42",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="运行设备，默认 cpu",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖输出目录中的同名 A.txt-Z.txt",
    )
    parser.add_argument(
        "--preview-label",
        type=str,
        choices=ALPHABET,
        default="",
        help="可选：处理后用现有 plot_pc 弹窗预览某个字母，不保存图片",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验配置和文件列表，不创建或写入结果",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""

    args = build_parser().parse_args(argv)
    config = IWRSAZConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
        preview_label=args.preview_label,
        dry_run=args.dry_run,
    )
    run_with_config(config)
    return 0


def main_without_cli() -> None:
    """无参运行入口；修改下方显式参数即可在 IDE 中直接调试。"""

    # ===== 可编辑参数 =====
    input_dir = Path(r"D:\PYproject\SPADdata\20250430")
    output_dir: Optional[Path] = None
    num_samples = 1024
    gamma = 0.5
    seed = 42
    device = "cpu"
    overwrite = False
    preview_label = ""
    dry_run = False

    # ===== 统一配置 =====
    config = IWRSAZConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        num_samples=num_samples,
        gamma=gamma,
        seed=seed,
        device=device,
        overwrite=overwrite,
        preview_label=preview_label,
        dry_run=dry_run,
    )
    run_with_config(config)


if __name__ == "__main__":
    # CLI：
    #   python scripts/run_iwrs_az_downsampling.py --dry-run
    #   python scripts/run_iwrs_az_downsampling.py --gamma 0.5 --seed 42
    # 无参：
    #   python scripts/run_iwrs_az_downsampling.py
    # 输出：仅 A.txt--Z.txt 降采样点云；默认不预览且从不保存图片。
    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()

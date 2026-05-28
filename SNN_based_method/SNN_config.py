"""Unified configuration for SNN training and inference.

Main exports:
    SNNConfig: Dataclass that owns data, model, loss, training and testing
        parameters.
    SINUSOIDAL_DEFAULT, LUT_RBF_16, LUT_SIN_16, LUT_RBF_32: Common model
        presets.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _as_list(values: Sequence[str] | None) -> list[str]:
    """Convert an optional string sequence to a JSON-friendly list."""
    if values is None:
        return []
    return [str(value) for value in values]


@dataclass
class SNNConfig:
    """Global SPAD SNN configuration.

    The same object is used by model construction, loss/metric construction,
    raw data loading, checkpointing and command-line scripts.
    """

    # ---- Data ----
    data_paths: list[str] | None = None
    """Raw files or directories containing ``.raw`` files."""

    pages_per_group: int = 500
    """Number of raw pages in one training sample, i.e. ``P``."""

    total_pages: Optional[int] = None
    """Optional page count per raw file. ``None`` means use all complete groups."""

    time_threshold: int = 150
    """Values larger than this ToF bin are treated as invalid and set to 0."""

    recursive: bool = False
    """Search data directories recursively for ``.raw`` files."""

    return_label: bool = True
    """Generate weak labels ``[B, 2, 64, 64]`` from grouped raw data."""

    normalize_input: bool = False
    """Divide input ToF values by ``time_threshold`` in the Dataset."""

    shuffle_pages: bool = False
    """Randomly permute the P dimension inside each sample during loading."""

    active_point: int = 1
    """Minimum duplicate count used by the weak label point filter."""

    cache_size: int = 2
    """Number of raw files cached as grouped arrays by the Dataset."""

    split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1)
    """Train/val/test split ratios."""

    # ---- Dataloader ----
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: Optional[bool] = None
    drop_last: bool = False
    seed: int = 42

    # ---- Encoding ----
    encoding_mode: str = "sinusoidal"
    """``sinusoidal`` or ``lut``."""

    n_freq: int = 8
    embed_dim: int = 16
    lut_init: str = "sinusoidal"
    lut_max_norm: Optional[float] = None

    # ---- Network ----
    model_backend: str = "legacy"
    """``new`` uses SNN_new.py; ``legacy`` uses SNN.py."""

    C: int = 32
    chunk_size: int = 128
    spike_mode: str = "plif"
    num_blocks: int = 3
    refine_mid: int = 8

    # ---- Loss weights ----
    w_gt: float = 0.3
    w_ssim: float = 0.1
    w_var: float = 1.0
    w_sparse: float = 0.05
    w_smooth: float = 0.1
    w_lut_smooth: float = 0.01
    w_lut_norm: float = 0.005

    # ---- Loss hyperparameters ----
    sigma_target: float = 4.0
    rho_target: float = 0.15
    beta_smooth: float = 5.0
    ssim_kernel_size: int = 7
    depth_range: float = 150.0
    intensity_range: float = 1.0

    # ---- Optimizer / scheduler ----
    epochs: int = 20
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    amp: bool = False

    # ---- Runtime / artifacts ----
    device: str = "auto"
    output_dir: str = "SNN/artifacts"
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    save_every: int = 1

    @property
    def t_max(self) -> int:
        """Alias used by the model and loss code."""
        return self.time_threshold

    @property
    def C_enc(self) -> int:
        """Encoded channel count before the stem layer."""
        if self.encoding_mode == "lut":
            return self.embed_dim
        return 2 * self.n_freq + 1

    def resolved_device(self) -> torch.device:
        """Return the configured device with ``auto`` mapped to CUDA if available."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)

    def build_model(self) -> torch.nn.Module:
        """Build a ``SPADSpikeNet`` model from this config."""
        backend = self.model_backend.lower()
        if backend in {"new", "activation", "activation_based"}:
            try:
                from SNN.SNN_new import SPADSpikeNet
            except ModuleNotFoundError:
                from SNN_new import SPADSpikeNet
        elif backend in {"legacy", "clock", "clock_driven"}:
            try:
                from SNN.SNN import SPADSpikeNet
            except ModuleNotFoundError:
                from SNN import SPADSpikeNet
        else:
            raise ValueError("model_backend must be 'new' or 'legacy'")

        return SPADSpikeNet(
            C=self.C,
            chunk_size=self.chunk_size,
            spike_mode=self.spike_mode,
            t_max=self.time_threshold,
            n_freq=self.n_freq,
            num_blocks=self.num_blocks,
            encoding_mode=self.encoding_mode,
            embed_dim=self.embed_dim,
            lut_init=self.lut_init,
        )

    def build_loss(self) -> torch.nn.Module:
        """Build the standard SNN imaging loss."""
        try:
            from SNN.loss import SPADImagingLoss
        except ModuleNotFoundError:
            from loss import SPADImagingLoss

        return SPADImagingLoss(
            w_gt=self.w_gt,
            w_ssim=self.w_ssim,
            w_var=self.w_var,
            w_sparse=self.w_sparse,
            w_smooth=self.w_smooth,
            w_lut_smooth=self.w_lut_smooth,
            w_lut_norm=self.w_lut_norm,
            sigma_target=self.sigma_target,
            rho_target=self.rho_target,
            beta_smooth=self.beta_smooth,
            ssim_kernel_size=self.ssim_kernel_size,
            depth_range=self.depth_range,
            intensity_range=self.intensity_range,
        )

    def build_metrics(self):
        """Build image metrics used for validation and testing."""
        try:
            from SNN.loss import ImageMetrics
        except ModuleNotFoundError:
            from loss import ImageMetrics

        return ImageMetrics(
            depth_range=self.depth_range,
            intensity_range=self.intensity_range,
            ssim_kernel_size=self.ssim_kernel_size,
        )

    def build_dataloaders(self):
        """Build train/val/test DataLoaders from configured raw paths."""
        if not self.data_paths:
            raise ValueError("data_paths is empty; pass --data-paths or use a config JSON")

        try:
            from SNN.data import create_spad_dataloaders
        except ModuleNotFoundError:
            from data import create_spad_dataloaders

        return create_spad_dataloaders(
            self.data_paths,
            pages_per_group=self.pages_per_group,
            total_pages=self.total_pages,
            time_threshold=self.time_threshold,
            batch_size=self.batch_size,
            split_ratios=self.split_ratios,
            seed=self.seed,
            return_label=self.return_label,
            normalize=self.normalize_input,
            shuffle_pages=self.shuffle_pages,
            active_point=self.active_point,
            cache_size=self.cache_size,
            recursive=self.recursive,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def build_dataloader(self, *, shuffle: bool = False):
        """Build one DataLoader from all configured raw paths."""
        if not self.data_paths:
            raise ValueError("data_paths is empty; pass --data-paths or use a config JSON")

        try:
            from SNN.data import create_spad_dataloader
        except ModuleNotFoundError:
            from data import create_spad_dataloader

        return create_spad_dataloader(
            self.data_paths,
            pages_per_group=self.pages_per_group,
            total_pages=self.total_pages,
            time_threshold=self.time_threshold,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.seed,
            return_label=self.return_label,
            normalize=self.normalize_input,
            shuffle_pages=self.shuffle_pages,
            active_point=self.active_point,
            cache_size=self.cache_size,
            recursive=self.recursive,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Export a JSON-serializable dictionary."""
        data = asdict(self)
        data["data_paths"] = _as_list(self.data_paths)
        data["split_ratios"] = list(self.split_ratios)
        return data

    def save(self, path: str | Path) -> None:
        """Save this config as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(self.to_dict(), file_obj, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "SNNConfig":
        """Load a config from JSON."""
        with Path(path).open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        if "split_ratios" in data:
            data["split_ratios"] = tuple(data["split_ratios"])
        return cls(**data)

    def clone_with(self, **updates: Any) -> "SNNConfig":
        """Create a new config with selected fields overridden."""
        data = self.to_dict()
        data.update(updates)
        if "split_ratios" in data:
            data["split_ratios"] = tuple(data["split_ratios"])
        return SNNConfig(**data)

    def summary(self) -> str:
        """Return a concise human-readable config summary."""
        lines = ["SNNConfig:"]
        lines.append(f"  data_paths={self.data_paths}")
        lines.append(
            f"  pages_per_group={self.pages_per_group}, "
            f"time_threshold={self.time_threshold}, batch_size={self.batch_size}"
        )
        lines.append(
            f"  model_backend={self.model_backend}, encoding={self.encoding_mode}, "
            f"C_enc={self.C_enc}, C={self.C}, chunk_size={self.chunk_size}"
        )
        lines.append(
            f"  epochs={self.epochs}, lr={self.lr}, weight_decay={self.weight_decay}, "
            f"device={self.resolved_device()}"
        )
        return "\n".join(lines)


SINUSOIDAL_DEFAULT = SNNConfig()
"""Default sinusoidal encoding config."""

LUT_RBF_16 = SNNConfig(encoding_mode="lut", embed_dim=16, lut_init="rbf")
"""LUT encoding with 16 dimensions and RBF initialization."""

LUT_SIN_16 = SNNConfig(encoding_mode="lut", embed_dim=16, lut_init="sinusoidal")
"""LUT encoding with 16 dimensions and sinusoidal initialization."""

LUT_RBF_32 = SNNConfig(encoding_mode="lut", embed_dim=32, lut_init="rbf")
"""LUT encoding with 32 dimensions and RBF initialization."""


if __name__ == "__main__":
    cfg = SNNConfig()
    print(cfg.summary())
    model = cfg.build_model()
    criterion = cfg.build_loss()
    metrics = cfg.build_metrics()
    print(f"model={model.__class__.__name__}")
    print(f"criterion={criterion.__class__.__name__}")
    print(f"metrics={metrics.__class__.__name__}")

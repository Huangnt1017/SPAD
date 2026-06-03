"""SPAD SNN raw group 级数据增强。

这些增强作用在 ``group_tof [4096, P]`` 上，发生在 Dataset 把 ToF
限制到 ``[1, T_max]`` 之前。0 始终表示无效触发，不会被 shift 变成
有效 photon。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def clip_tof_to_valid_range(group_tof: np.ndarray, t_max: int) -> np.ndarray:
    """把 ToF 限制到物理有效范围; 小于 1 或大于 ``t_max`` 的值置 0。"""
    if t_max <= 0:
        raise ValueError("t_max must be a positive integer")
    clipped = group_tof.astype(np.int32, copy=True)
    clipped[(clipped < 1) | (clipped > int(t_max))] = 0
    return clipped.astype(np.uint16, copy=False)


@dataclass
class RandomToFShift:
    """对所有非零 ToF 做同一个随机整数偏移, 并同步输入和 depth 标签。

    该增强模拟系统时间零点或整体深度的轻微漂移。它必须在 T_max 过滤前
    执行, 因此原始大于 T_max 的值在负向 shift 后可能重新落入有效范围。
    """

    max_shift: int = 15
    prob: float = 1.0
    t_max: int = 128

    def __post_init__(self) -> None:
        if self.max_shift < 0:
            raise ValueError("max_shift must be non-negative")
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError("prob must be in [0, 1]")
        if self.t_max <= 0:
            raise ValueError("t_max must be positive")

    def sample_delta(self) -> int:
        """采样本次样本使用的整数 ToF 偏移。"""
        if self.max_shift <= 0 or np.random.random() >= self.prob:
            return 0
        return int(np.random.randint(-self.max_shift, self.max_shift + 1))

    def apply_delta(self, group_tof: np.ndarray, delta: int) -> np.ndarray:
        """对非零 ToF 应用指定偏移, 再执行有效范围裁剪。"""
        shifted = group_tof.astype(np.int32, copy=True)
        valid_before_shift = shifted > 0
        shifted[valid_before_shift] += int(delta)
        shifted[(shifted < 1) | (shifted > self.t_max)] = 0
        return shifted.astype(np.uint16, copy=False)


@dataclass
class RandomPageDropout:
    """随机丢弃整页 raw page, 只改变输入 photon 密度, 不改变标签。"""

    page_drop_prob: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.page_drop_prob <= 1.0:
            raise ValueError("page_drop_prob must be in [0, 1]")

    def __call__(self, group_tof: np.ndarray) -> np.ndarray:
        if self.page_drop_prob <= 0.0:
            return group_tof
        page_count = group_tof.shape[1]
        drop_mask = np.random.random(page_count) < self.page_drop_prob
        if not np.any(drop_mask):
            return group_tof
        if np.all(drop_mask):
            drop_mask[np.random.randint(0, page_count)] = False

        augmented = group_tof.copy()
        augmented[:, drop_mask] = 0
        return augmented


class SpadRawTrainAugmentation:
    """组合训练用 raw group 增强。

    返回值为 ``(input_group, label_group)``。ToF shift 同步作用于输入和
    label group; PageDropout 只作用于 input group。
    """

    def __init__(
        self,
        *,
        t_max: int = 128,
        tof_shift_max: int = 15,
        tof_shift_prob: float = 1.0,
        page_dropout: bool = False,
        page_dropout_prob: float = 0.1,
    ) -> None:
        self.t_max = int(t_max)
        self.tof_shift = RandomToFShift(
            max_shift=int(tof_shift_max),
            prob=float(tof_shift_prob),
            t_max=self.t_max,
        )
        self.page_dropout = (
            RandomPageDropout(page_drop_prob=float(page_dropout_prob))
            if page_dropout
            else None
        )

    def __call__(self, group_tof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        input_group = group_tof
        label_group = group_tof

        delta = self.tof_shift.sample_delta()
        if delta != 0:
            input_group = self.tof_shift.apply_delta(input_group, delta)
            label_group = self.tof_shift.apply_delta(label_group, delta)
        else:
            input_group = clip_tof_to_valid_range(input_group, self.t_max)
            label_group = clip_tof_to_valid_range(label_group, self.t_max)

        if self.page_dropout is not None:
            input_group = self.page_dropout(input_group)

        return input_group, label_group

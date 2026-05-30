"""CSI 数据增强模块.

所有增强函数接收单个窗口 [window_size, n_subcarriers] 并返回增强后的窗口.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def augment_window(window: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """对单个窗口应用配置的增强策略.

    Args:
        window: [T, C] 幅度矩阵
        cfg: 增强配置

    Returns:
        增强后的窗口
    """
    aug = window.copy()

    # 时间偏移
    if cfg.get("time_shift", False):
        aug = time_shift(aug, max_shift=cfg.get("time_shift_range", 10))

    # 时间拉伸
    if cfg.get("time_stretch", False):
        aug = time_stretch(aug, stretch_range=cfg.get("time_stretch_range", [0.8, 1.2]))

    # 高斯噪声
    noise_scale = cfg.get("gaussian_noise", 0.0)
    if noise_scale > 0:
        aug = add_gaussian_noise(aug, scale=noise_scale)

    # 幅度缩放
    amp_scale = cfg.get("amplitude_scale", 0.0)
    if amp_scale > 0:
        aug = amplitude_scale(aug, scale_range=amp_scale)

    # 子载波 Mask
    sc_mask_ratio = cfg.get("subcarrier_mask_ratio", 0.0)
    if sc_mask_ratio > 0:
        aug = subcarrier_mask(aug, mask_ratio=sc_mask_ratio)

    # 时间步 Mask
    ts_mask_ratio = cfg.get("timestep_mask_ratio", 0.0)
    if ts_mask_ratio > 0:
        aug = timestep_mask(aug, mask_ratio=ts_mask_ratio)

    return aug


def time_shift(window: np.ndarray, max_shift: int = 10) -> np.ndarray:
    """随机时间偏移：非循环移位窗口，空出的部分用边缘帧填充.

    Args:
        window: [T, C]
        max_shift: 最大偏移量
    """
    if max_shift <= 0:
        return window
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return window

    aug = np.empty_like(window)
    if shift > 0:
        aug[:shift] = window[0]
        aug[shift:] = window[:-shift]
    else:
        shift_abs = abs(shift)
        aug[-shift_abs:] = window[-1]
        aug[:-shift_abs] = window[shift_abs:]
    return aug


def time_stretch(window: np.ndarray, stretch_range: list[float] = [0.8, 1.2]) -> np.ndarray:
    """随机时间拉伸/压缩：线性插值.

    Args:
        window: [T, C]
        stretch_range: [min, max] 缩放因子
    """
    lo, hi = stretch_range
    factor = np.random.uniform(lo, hi)
    t, c = window.shape
    new_t = int(t * factor)

    # 对每列（子载波）独立插值
    stretched = np.zeros((new_t, c), dtype=window.dtype)
    x_old = np.arange(t)
    x_new = np.linspace(0, t - 1, new_t)
    for i in range(c):
        stretched[:, i] = np.interp(x_new, x_old, window[:, i])

    # 裁剪或填充回原始长度
    if new_t >= t:
        # 裁剪中间段
        start = (new_t - t) // 2
        return stretched[start:start + t]
    else:
        # 两端填充（重复边缘值）
        pad_before = (t - new_t) // 2
        pad_after = t - new_t - pad_before
        return np.pad(stretched, ((pad_before, pad_after), (0, 0)), mode="edge")


def add_gaussian_noise(window: np.ndarray, scale: float = 0.01) -> np.ndarray:
    """添加高斯噪声.

    Args:
        window: [T, C]
        scale: 噪声标准差 = scale * window.std()
    """
    std = window.std()
    noise = np.random.normal(0, scale * std, size=window.shape)
    return window + noise.astype(window.dtype)


def amplitude_scale(window: np.ndarray, scale_range: float = 0.1) -> np.ndarray:
    """随机幅度缩放.

    Args:
        window: [T, C]
        scale_range: 缩放范围 [1-scale, 1+scale]
    """
    factor = np.random.uniform(1 - scale_range, 1 + scale_range)
    return window * factor


def subcarrier_mask(window: np.ndarray, mask_ratio: float = 0.05) -> np.ndarray:
    """随机 Mask 部分子载波（置为 0）.

    Args:
        window: [T, C]
        mask_ratio: Mask 比例
    """
    _, c = window.shape
    n_mask = max(1, int(c * mask_ratio))
    mask_idx = np.random.choice(c, size=n_mask, replace=False)
    aug = window.copy()
    aug[:, mask_idx] = 0
    return aug


def timestep_mask(window: np.ndarray, mask_ratio: float = 0.05) -> np.ndarray:
    """随机 Mask 部分时间步（置为 0）.

    Args:
        window: [T, C]
        mask_ratio: Mask 比例
    """
    t, _ = window.shape
    n_mask = max(1, int(t * mask_ratio))
    mask_idx = np.random.choice(t, size=n_mask, replace=False)
    aug = window.copy()
    aug[mask_idx, :] = 0
    return aug


def mixup(
    x1: np.ndarray,
    x2: np.ndarray,
    y1: int,
    y2: int,
    alpha: float = 0.4,
) -> tuple[np.ndarray, float]:
    """Mixup 增强.

    Args:
        x1, x2: [T, C] 窗口
        y1, y2: 标签（整数）
        alpha: Beta 分布参数

    Returns:
        (mixed_window, mixed_label)
    """
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y


class MixupCollator:
    """用于 DataLoader 的 Mixup Collator.

    用法:
        from torch.utils.data import DataLoader
        collator = MixupCollator(alpha=0.4, num_classes=5)
        loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
    """

    def __init__(self, alpha: float = 0.4, num_classes: int = 5):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """batch: list of (window, label_int)."""
        xs = torch.stack([item[0] for item in batch], dim=0)  # [B, T, C]
        ys = torch.stack([item[1] for item in batch], dim=0).long()

        # 只对同一批次内随机配对做 mixup
        lam = float(np.random.beta(self.alpha, self.alpha))
        batch_size = xs.shape[0]
        indices = torch.randperm(batch_size)

        mixed_x = lam * xs + (1 - lam) * xs[indices]

        # 标签转为 one-hot 后混合
        y_onehot = torch.zeros((batch_size, self.num_classes), dtype=torch.float32)
        y_onehot[torch.arange(batch_size), ys] = 1.0
        y_onehot_mix = lam * y_onehot + (1 - lam) * y_onehot[indices]

        return mixed_x, y_onehot_mix

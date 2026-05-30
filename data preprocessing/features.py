"""手工特征提取模块（供传统机器学习使用）.

对每个滑动窗口提取统计特征，生成固定长度的特征向量.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_temporal_stats(
    window: np.ndarray,
    stats: list[str] | None = None,
) -> dict[str, float]:
    """提取时域统计特征.

    Args:
        window: [window_size, n_subcarriers]
        stats: 需要的统计量列表

    Returns:
        特征名字典
    """
    if stats is None:
        stats = ["mean", "std", "max", "min", "ptp", "energy", "skewness", "kurtosis"]

    feats = {}
    # 按子载波计算，然后取平均（也可对每个子载波单独输出，维度会很大）
    # 这里选择：先对每个子载波计算统计量，再取子载波间的平均/标准差
    # 这样可以得到与子载波数无关的固定维度特征

    per_sc = {}
    for s in stats:
        if s == "mean":
            per_sc[s] = np.mean(window, axis=0)
        elif s == "std":
            per_sc[s] = np.std(window, axis=0)
        elif s == "max":
            per_sc[s] = np.max(window, axis=0)
        elif s == "min":
            per_sc[s] = np.min(window, axis=0)
        elif s == "ptp":
            per_sc[s] = np.ptp(window, axis=0)
        elif s == "energy":
            per_sc[s] = np.sum(window ** 2, axis=0)
        elif s == "skewness":
            per_sc[s] = _skewness(window, axis=0)
        elif s == "kurtosis":
            per_sc[s] = _kurtosis(window, axis=0)

    # 对每个统计量，取子载波间的均值和标准差
    for s, vals in per_sc.items():
        feats[f"{s}_mean"] = float(np.mean(vals))
        feats[f"{s}_std"] = float(np.std(vals))

    return feats


def extract_diff_stats(
    window: np.ndarray,
    stats: list[str] | None = None,
) -> dict[str, float]:
    """提取相邻帧差分统计特征.

    Args:
        window: [T, C]
        stats: ["mean", "std", "energy"]
    """
    if stats is None:
        stats = ["mean", "std", "energy"]

    diff = np.diff(window, axis=0)  # [T-1, C]
    feats = {}

    if "mean" in stats:
        feats["diff_mean"] = float(np.mean(np.abs(diff)))
    if "std" in stats:
        feats["diff_std"] = float(np.std(diff))
    if "energy" in stats:
        feats["diff_energy"] = float(np.sum(diff ** 2))

    return feats


def extract_rssi_stats(rssi_window: np.ndarray) -> dict[str, float]:
    """提取 RSSI 统计特征.

    Args:
        rssi_window: [window_size] RSSI 序列
    """
    return {
        "rssi_mean": float(np.mean(rssi_window)),
        "rssi_std": float(np.std(rssi_window)),
        "rssi_ptp": float(np.ptp(rssi_window)),
        "rssi_min": float(np.min(rssi_window)),
        "rssi_max": float(np.max(rssi_window)),
    }


def extract_inter_carrier_corr(window: np.ndarray) -> dict[str, float]:
    """提取子载波间相关性特征.

    Args:
        window: [T, C]
    """
    # 计算子载波间的相关系数矩阵
    corr_matrix = np.nan_to_num(np.corrcoef(window.T), nan=0.0, posinf=0.0, neginf=0.0)  # [C, C]
    # 取上三角（去掉对角线）
    triu_idx = np.triu_indices(corr_matrix.shape[0], k=1)
    corr_values = corr_matrix[triu_idx]

    return {
        "corr_mean": float(np.mean(np.abs(corr_values))),
        "corr_std": float(np.std(corr_values)),
        "corr_min": float(np.min(corr_values)),
        "corr_max": float(np.max(corr_values)),
    }


def extract_all_features(
    window: np.ndarray,
    rssi_window: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    """提取完整手工特征向量.

    Args:
        window: [T, C] 幅度窗口
        rssi_window: [T] RSSI 序列，可选
        cfg: 特征配置

    Returns:
        一维特征向量
    """
    if cfg is None:
        cfg = {}

    feat_dict = {}

    temporal_cfg = cfg.get("temporal_stats", ["mean", "std", "max", "min", "ptp", "energy"])
    if temporal_cfg:
        temporal_stats = (
            temporal_cfg
            if isinstance(temporal_cfg, list)
            else cfg.get("temporal_stats_list", ["mean", "std", "max", "min", "ptp", "energy"])
        )
        feat_dict.update(extract_temporal_stats(window, stats=temporal_stats))

    diff_cfg = cfg.get("diff_stats", ["mean", "std", "energy"])
    if diff_cfg:
        diff_stats = diff_cfg if isinstance(diff_cfg, list) else cfg.get("diff_stats_list", ["mean", "std", "energy"])
        feat_dict.update(extract_diff_stats(window, stats=diff_stats))

    rssi_cfg = cfg.get("rssi_stats", True)
    if rssi_cfg and rssi_window is not None:
        feat_dict.update(extract_rssi_stats(rssi_window))

    # 子载波间相关性
    if cfg.get("inter_carrier_corr", False):
        feat_dict.update(extract_inter_carrier_corr(window))

    return np.array(list(feat_dict.values()), dtype=np.float32)


def extract_features_for_windows(
    windows: list[np.ndarray],
    rssi_windows: list[np.ndarray] | None = None,
    cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    """批量提取手工特征.

    Args:
        windows: list of [T, C]
        rssi_windows: list of [T]，可选
        cfg: 配置

    Returns:
        [n_windows, n_features]
    """
    features = []
    for i, win in enumerate(windows):
        rssi = rssi_windows[i] if rssi_windows is not None else None
        feat = extract_all_features(win, rssi, cfg)
        features.append(feat)
    return np.stack(features, axis=0)


def _skewness(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """计算偏度（无 scipy 依赖）."""
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True) + 1e-8
    z = (x - mean) / std
    return np.mean(z ** 3, axis=axis)


def _kurtosis(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """计算峰度（无 scipy 依赖）."""
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True) + 1e-8
    z = (x - mean) / std
    return np.mean(z ** 4, axis=axis) - 3.0

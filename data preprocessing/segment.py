"""CSI 动作自动切分模块.

从"静止 → 动作 → 静止"的完整采集片段中，自动检测动作起止点.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_frame_variance(amp_matrix: np.ndarray) -> np.ndarray:
    """计算每帧的全子载波幅度方差.

    Args:
        amp_matrix: [n_frames, n_subcarriers]

    Returns:
        [n_frames] 每帧方差
    """
    return np.var(amp_matrix, axis=1)


def compute_diff_energy(amp_matrix: np.ndarray) -> np.ndarray:
    """计算相邻帧差分能量（每帧一个值，首帧为 0）.

    Args:
        amp_matrix: [n_frames, n_subcarriers]

    Returns:
        [n_frames] 差分能量
    """
    diff = np.diff(amp_matrix, axis=0)  # [n_frames-1, n_subcarriers]
    energy = np.sum(diff ** 2, axis=1)
    # 首帧补 0
    return np.concatenate([[0.0], energy])


def smooth_signal(signal: np.ndarray, window_size: int = 10) -> np.ndarray:
    """滑动平均平滑."""
    if window_size <= 1:
        return signal
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode="same")


def detect_segments(
    metric: np.ndarray,
    baseline_window: int = 50,
    threshold_ratio: float = 3.0,
    min_segment_len: int = 30,
    padding: int = 15,
) -> list[tuple[int, int]]:
    """基于方差/能量阈值检测动作片段.

    Args:
        metric: [n_frames] 每帧的活动指标（方差或差分能量）
        baseline_window: 用于计算基线的前 N 帧
        threshold_ratio: 超过基线多少倍视为动作
        min_segment_len: 最小动作片段长度（帧）
        padding: 动作前后保留的静止帧数

    Returns:
        动作片段起止索引列表 [(start, end), ...]
    """
    n = len(metric)
    if n < baseline_window + min_segment_len:
        return []

    # 用前 baseline_window 帧的均值作为基线
    baseline = np.mean(metric[:baseline_window])
    threshold = baseline * threshold_ratio

    # 二值化：超过阈值视为动作
    active = metric > threshold

    # 找连续 True 段
    segments = []
    in_segment = False
    seg_start = 0

    for i in range(n):
        if active[i] and not in_segment:
            seg_start = i
            in_segment = True
        elif not active[i] and in_segment:
            if i - seg_start >= min_segment_len:
                segments.append((seg_start, i))
            in_segment = False

    # 处理末尾
    if in_segment and n - seg_start >= min_segment_len:
        segments.append((seg_start, n))

    # 添加 padding
    padded_segments = []
    for s, e in segments:
        ps = max(0, s - padding)
        pe = min(n, e + padding)
        padded_segments.append((ps, pe))

    return padded_segments


def segment_action(
    amp_matrix: np.ndarray,
    labels: np.ndarray,
    method: str = "variance",
    **kwargs: Any,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """对完整片段做动作切分，返回切分后的子片段.

    Args:
        amp_matrix: [n_frames, n_subcarriers]
        labels: [n_frames] 原始标签数组
        method: "variance" | "diff_energy"
        **kwargs: 传给 detect_segments 的参数

    Returns:
        列表，每个元素为 (sub_amp, sub_labels, start_idx, end_idx)
    """
    if method == "variance":
        metric = compute_frame_variance(amp_matrix)
    elif method == "diff_energy":
        metric = compute_diff_energy(amp_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    metric = smooth_signal(metric, window_size=kwargs.get("smooth_window", 10))

    segments = detect_segments(
        metric,
        baseline_window=kwargs.get("baseline_window", 50),
        threshold_ratio=kwargs.get("threshold_ratio", 3.0),
        min_segment_len=kwargs.get("min_segment_len", 30),
        padding=kwargs.get("padding", 15),
    )

    results = []
    for s, e in segments:
        results.append((amp_matrix[s:e], labels[s:e], s, e))

    return results


def auto_segment_pipeline(
    amp_matrix: np.ndarray,
    labels: np.ndarray,
    cfg: dict[str, Any],
    default_label: str = "idle",
) -> tuple[np.ndarray, np.ndarray]:
    """自动切分流水线：输入完整片段，输出切分后的幅度矩阵和标签.

    策略:
        - 检测到的高活动片段保持原标签
        - 其余低活动片段标记为 default_label（如 idle）

    Args:
        amp_matrix: [n_frames, n_subcarriers]
        labels: [n_frames] 字符串数组
        cfg: segment 配置
        default_label: 低活动区域的默认标签

    Returns:
        (amp_matrix, new_labels) 长度不变，标签被重新分配
    """
    if not cfg.get("enabled", False):
        return amp_matrix, labels

    segments = segment_action(amp_matrix, labels, **cfg)
    new_labels = np.full_like(labels, default_label)

    for sub_amp, sub_labels, s, e in segments:
        # 动作片段内取 majority label
        from collections import Counter
        majority = Counter(sub_labels).most_common(1)[0][0]
        new_labels[s:e] = majority

    return amp_matrix, new_labels


if __name__ == "__main__":
    # 简单测试
    np.random.seed(0)
    n_frames = 500
    n_sc = 64

    # 模拟数据：前150帧静止，中间200帧动作，后150帧静止
    amp = np.random.normal(50, 2, size=(n_frames, n_sc)).astype(np.float32)
    amp[150:350] += np.random.normal(0, 8, size=(200, n_sc))  # 动作段方差大

    labels = np.array(["idle"] * n_frames)
    labels[150:350] = "walking"

    segments = segment_action(amp, labels, method="variance", baseline_window=50, threshold_ratio=3.0)
    print(f"Detected {len(segments)} segments:")
    for s, e in segments:
        print(f"  Frame {s} ~ {e} ({e-s} frames)")

"""CSI 数据预处理工具函数."""

import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_csv_raw(csv_path: str | Path) -> tuple[list[dict], list[str]]:
    """加载原始 CSI CSV 文件，返回行数据列表和字段名列表."""
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)
    return rows, fieldnames


def parse_iq_bytes(raw: str) -> np.ndarray:
    """解析空格分隔的 I/Q 字节字符串为 int8 数组.

    对串口粘包导致的超大值做容错处理：
    - 先用 int16 解析避免 OverflowError
    - 检查所有值是否在 int8 范围内，超出则标记为无效

    Args:
        raw: 如 "0 0 -40 -52 -38 -41 ..."

    Returns:
        shape为[N]的int8数组；若数据异常则返回空数组
    """
    if not raw or not raw.strip():
        return np.array([], dtype=np.int8)
    tokens = raw.strip().split()
    try:
        # 先用 int16 解析，避免单个超大值导致 crash
        values = np.array([int(t) for t in tokens], dtype=np.int16)
    except ValueError:
        return np.array([], dtype=np.int8)

    # 检查是否都在 int8 范围内；如有超出，说明串口粘包，标记为无效
    if np.any(values < -128) or np.any(values > 127):
        return np.array([], dtype=np.int8)

    return values.astype(np.int8)


def iq_to_amplitude(iq_array: np.ndarray) -> np.ndarray:
    """将交错 I/Q 字节转为幅度.

    Args:
        iq_array: shape [128] 的 int8 数组 (I0,Q0,I1,Q1,...)

    Returns:
        shape [64] 的 float32 幅度数组
    """
    if iq_array.size < 2:
        return np.array([], dtype=np.float32)
    # 确保偶数个元素
    if iq_array.size % 2 == 1:
        iq_array = iq_array[:-1]
    i_vals = iq_array[0::2].astype(np.float32)
    q_vals = iq_array[1::2].astype(np.float32)
    return np.sqrt(i_vals * i_vals + q_vals * q_vals)


def iq_to_phase(iq_array: np.ndarray) -> np.ndarray:
    """将交错 I/Q 字节转为相位（弧度）.

    Args:
        iq_array: shape [128] 的 int8 数组

    Returns:
        shape [64] 的 float32 相位数组
    """
    if iq_array.size < 2:
        return np.array([], dtype=np.float32)
    if iq_array.size % 2 == 1:
        iq_array = iq_array[:-1]
    i_vals = iq_array[0::2].astype(np.float32)
    q_vals = iq_array[1::2].astype(np.float32)
    return np.arctan2(q_vals, i_vals)


def find_zero_subcarriers(amp_matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """找出在所有帧中幅度接近 0 的子载波索引.

    Args:
        amp_matrix: shape [n_frames, n_subcarriers]
        eps: 判定为 0 的阈值

    Returns:
        需要保留的子载波索引数组（去掉全零后的）
    """
    max_per_sc = np.max(amp_matrix, axis=0)
    keep = max_per_sc > eps
    return np.where(keep)[0]


def save_norm_stats(stats: dict[str, Any], path: str | Path) -> None:
    """保存归一化统计量到 pickle 文件."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(stats, f)


def load_norm_stats(path: str | Path) -> dict[str, Any]:
    """从 pickle 文件加载归一化统计量."""
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_norm_stats(amp_matrix: np.ndarray, method: str = "zscore") -> dict[str, Any]:
    """计算归一化统计量.

    Args:
        amp_matrix: shape [n_frames, n_subcarriers]
        method: "minmax" | "zscore"

    Returns:
        包含统计量的字典
    """
    if method == "minmax":
        return {
            "method": "minmax",
            "min": float(np.min(amp_matrix)),
            "max": float(np.max(amp_matrix)),
        }
    elif method == "zscore":
        return {
            "method": "zscore",
            "mean": float(np.mean(amp_matrix)),
            "std": float(np.std(amp_matrix)) + 1e-8,
        }
    else:
        return {"method": "none"}


def apply_norm(amp_matrix: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """应用归一化.

    Args:
        amp_matrix: shape [n_frames, n_subcarriers] 或 [n_windows, window_size, n_subcarriers]
        stats: 由 compute_norm_stats 生成的统计量

    Returns:
        归一化后的数组
    """
    method = stats.get("method", "none")
    if method == "none":
        return amp_matrix
    elif method == "minmax":
        min_val = stats["min"]
        max_val = stats["max"]
        rng = max_val - min_val + 1e-8
        return (amp_matrix - min_val) / rng
    elif method == "zscore":
        mean = stats["mean"]
        std = stats["std"]
        return (amp_matrix - mean) / std
    else:
        raise ValueError(f"Unknown normalize method: {method}")


def apply_energy_norm(amp_matrix: np.ndarray) -> np.ndarray:
    """每帧能量归一化: 每帧除以该帧的 L2 范数.

    Args:
        amp_matrix: shape [..., n_subcarriers]

    Returns:
        归一化后的数组
    """
    energy = np.sqrt(np.sum(amp_matrix ** 2, axis=-1, keepdims=True)) + 1e-8
    return amp_matrix / energy

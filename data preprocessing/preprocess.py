"""CSI 数据核心预处理模块.

功能:
    1. 加载原始 CSV
    2. 过滤异常帧
    3. I/Q 转幅度/相位
    4. 去零子载波
    5. 滑动窗口切分
    6. 归一化
    7. 多文件批量处理
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils import (
    apply_energy_norm,
    apply_norm,
    compute_norm_stats,
    find_zero_subcarriers,
    iq_to_amplitude,
    iq_to_phase,
    load_csv_raw,
    load_norm_stats,
    save_norm_stats,
)


class CSIRawFrame:
    """单个 CSI 帧的原始数据容器.
    
    任何字段解析失败都会标记为无效帧（self.valid = False），
    供过滤逻辑丢弃。这能 robust 地处理串口粘包/截断导致的坏行。
    """

    def __init__(self, row: dict[str, str]):
        self.valid = True
        self.parse_error = None
        
        try:
            self.pc_time = row.get("pc_time_iso", "")
            self.label = row.get("label", "unknown")
            self.node_id = int(row.get("node_id", 0))
            self.seq = int(row.get("seq", 0))
            self.local_time_us = int(row.get("local_time_us", 0))
            self.rx_timestamp_us = int(row.get("rx_timestamp_us", 0)) if row.get("rx_timestamp_us") else None
            self.src_mac = row.get("src_mac", "")
            self.dst_mac = row.get("dst_mac", "")
            self.first_word_invalid = int(row.get("first_word_invalid", 0))
            self.rx_seq = int(row.get("rx_seq", 0)) if row.get("rx_seq") else None
            self.rssi = int(row.get("rssi", 0))
            self.channel = int(row.get("channel", 0))
            self.secondary_channel = int(row.get("secondary_channel", 0))
            self.sig_mode = int(row.get("sig_mode", 0))
            self.mcs = int(row.get("mcs", 0))
            self.cwb = int(row.get("cwb", 0))
            self.noise_floor = int(row.get("noise_floor", 0))
            self.csi_len = int(row.get("csi_len", 0))
            self.iq_raw = row.get("csi_raw_bytes", "")
        except (ValueError, TypeError) as e:
            self.valid = False
            self.parse_error = str(e)
            self.iq_raw = ""
            self.csi_len = -1
            self.iq_array = np.array([], dtype=np.int8)
            self.n_iq_pairs = -1
            return
        
        # 解析 I/Q
        from utils import parse_iq_bytes
        self.iq_array = parse_iq_bytes(self.iq_raw)
        self.n_iq_pairs = self.iq_array.size // 2


def filter_frames(
    frames: list[CSIRawFrame],
    filter_first_word_invalid: bool = True,
    expected_csi_len: int = 128,
    filter_by_csi_len: bool = True,
    expected_iq_pairs: int = 64,
    filter_by_iq_count: bool = True,
    rssi_range: tuple[float, float] | None = None,
) -> list[CSIRawFrame]:
    """过滤异常帧.

    Args:
        frames: 原始帧列表
        filter_first_word_invalid: 丢弃 first_word_invalid == 1
        expected_csi_len: 期望 CSI 字节长度
        filter_by_csi_len: 是否按 csi_len 过滤
        expected_iq_pairs: 期望 I/Q 对数
        filter_by_iq_count: 是否按 I/Q 对数过滤
        rssi_range: (min, max) RSSI 范围，None 表示不过滤

    Returns:
        过滤后的帧列表
    """
    filtered = []
    drop_reasons = {
        "parse_error": 0,
        "first_word_invalid": 0,
        "csi_len_mismatch": 0,
        "iq_count_mismatch": 0,
        "rssi_out_of_range": 0,
    }

    for f in frames:
        # 1. 解析失败的帧（串口粘包/截断导致）
        if not f.valid:
            drop_reasons["parse_error"] += 1
            continue

        if filter_first_word_invalid and f.first_word_invalid == 1:
            drop_reasons["first_word_invalid"] += 1
            continue

        if filter_by_csi_len and f.csi_len != expected_csi_len:
            drop_reasons["csi_len_mismatch"] += 1
            continue

        if filter_by_iq_count and f.n_iq_pairs != expected_iq_pairs:
            drop_reasons["iq_count_mismatch"] += 1
            continue

        if rssi_range is not None:
            rmin, rmax = rssi_range
            if not (rmin <= f.rssi <= rmax):
                drop_reasons["rssi_out_of_range"] += 1
                continue

        filtered.append(f)

    total_dropped = sum(drop_reasons.values())
    if total_dropped > 0:
        print(f"[Filter] Dropped {total_dropped}/{len(frames)} frames:")
        for reason, count in drop_reasons.items():
            if count > 0:
                print(f"         - {reason}: {count}")

    return filtered


def frames_to_amplitude_matrix(frames: list[CSIRawFrame]) -> np.ndarray:
    """将帧列表转为幅度矩阵 [n_frames, n_subcarriers]."""
    amps = []
    for f in frames:
        amp = iq_to_amplitude(f.iq_array)
        if amp.size > 0:
            amps.append(amp)
    return np.stack(amps, axis=0) if amps else np.array([], dtype=np.float32).reshape(0, 0)


def frames_to_feature_matrix(frames: list[CSIRawFrame], output_type: str = "amplitude") -> np.ndarray:
    """将帧列表转为模型输入矩阵.

    output_type:
        amplitude -> [T, C]
        phase -> [T, C]
        both -> [T, 2*C]，前半为幅度，后半为相位
    """
    features = []
    for f in frames:
        amp = iq_to_amplitude(f.iq_array)
        if amp.size == 0:
            continue

        if output_type == "amplitude":
            feat = amp
        elif output_type == "phase":
            feat = iq_to_phase(f.iq_array)
        elif output_type == "both":
            phase = iq_to_phase(f.iq_array)
            feat = np.concatenate([amp, phase], axis=0)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

        features.append(feat.astype(np.float32, copy=False))

    return np.stack(features, axis=0) if features else np.array([], dtype=np.float32).reshape(0, 0)


def frames_to_metadata(frames: list[CSIRawFrame]) -> dict[str, np.ndarray]:
    """提取元数据数组."""
    return {
        "rssi": np.array([f.rssi for f in frames], dtype=np.float32),
        "local_time_us": np.array([f.local_time_us for f in frames], dtype=np.int64),
        "label": np.array([f.label for f in frames]),
        "node_id": np.array([f.node_id for f in frames], dtype=np.int32),
        "channel": np.array([f.channel for f in frames], dtype=np.int32),
        "seq": np.array([f.seq for f in frames], dtype=np.int32),
    }


def extract_windows(
    amp_matrix: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, np.ndarray] | None = None,
    window_size: int = 128,
    stride: int = 64,
    label_strategy: str = "majority",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """滑动窗口切分.

    Args:
        amp_matrix: [n_frames, n_subcarriers]
        labels: [n_frames] 字符串标签数组
        window_size: 窗口帧数
        stride: 步长
        label_strategy: "majority" | "center" | "last"

    Returns:
        windows: [n_windows, window_size, n_features]
        window_labels: [n_windows] 字符串标签
        window_meta: 每个窗口的起止帧和可选 RSSI/时间序列
    """
    n_frames, n_sc = amp_matrix.shape
    if n_frames < window_size:
        return np.array([]).reshape(0, window_size, n_sc), np.array([]), []

    windows = []
    window_labels = []
    window_meta = []

    for start in range(0, n_frames - window_size + 1, stride):
        end = start + window_size
        win = amp_matrix[start:end]
        win_labels = labels[start:end]

        if label_strategy == "majority":
            from collections import Counter
            label = Counter(win_labels).most_common(1)[0][0]
        elif label_strategy == "center":
            label = win_labels[window_size // 2]
        elif label_strategy == "last":
            label = win_labels[-1]
        else:
            raise ValueError(f"Unknown label_strategy: {label_strategy}")

        windows.append(win)
        window_labels.append(label)
        meta_item: dict[str, Any] = {"start_frame": start, "end_frame": end}
        if metadata is not None:
            if "rssi" in metadata:
                meta_item["rssi"] = metadata["rssi"][start:end]
            if "local_time_us" in metadata:
                meta_item["local_time_us"] = metadata["local_time_us"][start:end]
            if "seq" in metadata:
                meta_item["seq"] = metadata["seq"][start:end]
        window_meta.append(meta_item)

    return np.stack(windows, axis=0), np.array(window_labels), window_meta


def _build_keep_idx(feature_matrix: np.ndarray, output_type: str) -> np.ndarray:
    """根据训练数据计算稳定的保留特征索引."""
    if output_type != "both":
        return find_zero_subcarriers(feature_matrix)

    half = feature_matrix.shape[1] // 2
    amp_keep = find_zero_subcarriers(feature_matrix[:, :half])
    return np.concatenate([amp_keep, amp_keep + half]).astype(np.int64)


def _apply_keep_idx(feature_matrix: np.ndarray, keep_idx: np.ndarray) -> np.ndarray:
    if np.max(keep_idx, initial=-1) >= feature_matrix.shape[1]:
        raise ValueError(
            f"keep_idx expects at least {int(np.max(keep_idx)) + 1} features, "
            f"but matrix has {feature_matrix.shape[1]}"
        )
    return feature_matrix[:, keep_idx]


def preprocess_single_file(
    csv_path: str | Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
    """预处理单个 CSV 文件.

    Args:
        csv_path: CSV 文件路径
        config: 配置字典

    Returns:
        (amp_matrix, labels, metadata) 或 None（如果无有效数据）
    """
    print(f"\n[Preprocess] Loading {csv_path}")
    rows, _ = load_csv_raw(csv_path)
    if not rows:
        print(f"  Warning: empty file {csv_path}")
        return None

    # 1. 包装为帧对象
    frames = [CSIRawFrame(row) for row in rows]

    # 2. 过滤
    pre_cfg = config["preprocess"]
    filter_cfg = pre_cfg["filter"]
    rssi_range = filter_cfg.get("rssi_range")
    if rssi_range is not None and rssi_range[0] is None:
        rssi_range = None

    frames = filter_frames(
        frames,
        filter_first_word_invalid=filter_cfg.get("first_word_invalid", True),
        expected_csi_len=filter_cfg.get("expected_csi_len", 128),
        filter_by_csi_len=filter_cfg.get("filter_by_csi_len", True),
        expected_iq_pairs=filter_cfg.get("expected_iq_pairs", 64),
        filter_by_iq_count=filter_cfg.get("filter_by_iq_count", True),
        rssi_range=rssi_range,
    )

    if not frames:
        print(f"  Warning: no valid frames after filtering in {csv_path}")
        return None

    # 3. 转 I/Q 特征矩阵
    output_type = pre_cfg.get("output_type", "amplitude")
    amp_matrix = frames_to_feature_matrix(frames, output_type=output_type)
    metadata = frames_to_metadata(frames)

    print(f"  Valid frames: {amp_matrix.shape[0]}, features: {amp_matrix.shape[1]}")

    return amp_matrix, metadata["label"], metadata


def preprocess_pipeline(
    csv_paths: list[str],
    config: dict[str, Any],
    norm_stats: dict[str, Any] | None = None,
    fit_norm: bool = True,
) -> dict[str, list[np.ndarray] | list[str] | dict[str, Any]]:
    """完整预处理流水线：加载多个文件 → 过滤 → 归一化 → 切窗口.

    Args:
        csv_paths: CSV 文件路径列表
        config: 配置字典
        norm_stats: 预计算的归一化统计量（验证/测试时使用）
        fit_norm: 是否从当前数据计算归一化统计量

    Returns:
        字典:
        {
            "windows": list[np.ndarray],      # 每个元素 [window_size, n_subcarriers]
            "labels": list[str],
            "metadata": list[dict],
            "norm_stats": dict,                # 归一化统计量
            "file_names": list[str],
        }
    """
    all_amp_matrices = []
    all_labels = []
    all_metadata = []
    file_names = []

    # 1. 逐个文件预处理
    for path in csv_paths:
        result = preprocess_single_file(path, config)
        if result is None:
            continue
        amp_matrix, labels, meta = result
        all_amp_matrices.append(amp_matrix)
        all_labels.append(labels)
        all_metadata.append(meta)
        file_names.append(str(path))

    if not all_amp_matrices:
        raise ValueError("No valid data found in all provided CSV files.")

    # 2. 根据训练集拟合稳定的子载波索引，再应用到所有 split
    pre_cfg = config["preprocess"]
    norm_method = pre_cfg.get("normalize", "zscore")
    output_type = pre_cfg.get("output_type", "amplitude")
    remove_zero = pre_cfg.get("remove_zero_subcarriers", True)

    if norm_stats is None:
        norm_stats = {}

    if remove_zero:
        if fit_norm:
            keep_idx = _build_keep_idx(np.concatenate(all_amp_matrices, axis=0), output_type)
            norm_stats["keep_idx"] = keep_idx
            print(f"\n[Subcarrier] Kept {len(keep_idx)} / {all_amp_matrices[0].shape[1]} features")
        elif "keep_idx" not in norm_stats:
            raise ValueError("norm_stats['keep_idx'] must be provided when fit_norm=False")

        keep_idx = np.asarray(norm_stats["keep_idx"], dtype=np.int64)
        all_amp_matrices = [_apply_keep_idx(m, keep_idx) for m in all_amp_matrices]
    else:
        norm_stats["keep_idx"] = None

    # 3. 可选自动切分/重标注
    seg_cfg = config.get("segment", {})
    if seg_cfg.get("enabled", False):
        from segment import auto_segment_pipeline

        for i, (amp_matrix, labels) in enumerate(zip(all_amp_matrices, all_labels)):
            all_amp_matrices[i], all_labels[i] = auto_segment_pipeline(
                amp_matrix, labels, seg_cfg, default_label=seg_cfg.get("default_label", "idle")
            )

    if fit_norm:
        concatenated = np.concatenate(all_amp_matrices, axis=0)
        if norm_method == "energy":
            # 能量归一化不需要全局统计
            norm_stats["method"] = "energy"
        else:
            norm_stats.update(compute_norm_stats(concatenated, method=norm_method))
        norm_stats["output_type"] = output_type
        print(f"\n[Norm] Computed {norm_method} stats: {norm_stats}")
    elif norm_stats is None and norm_method != "none":
        raise ValueError("norm_stats must be provided when fit_norm=False")

    # 4. 应用归一化 + 滑动窗口
    win_cfg = config["window"]
    window_size = win_cfg["size"]
    stride = win_cfg["stride"]
    label_strategy = win_cfg.get("label_strategy", "majority")
    min_windows = win_cfg.get("min_windows", 1)

    all_windows = []
    all_window_labels = []
    all_window_meta = []

    for amp_matrix, labels, meta, fname in zip(
        all_amp_matrices, all_labels, all_metadata, file_names
    ):
        # 归一化
        if norm_method == "energy":
            amp_matrix = apply_energy_norm(amp_matrix)
        elif norm_method != "none":
            amp_matrix = apply_norm(amp_matrix, norm_stats)

        # 滑动窗口
        windows, win_labels, win_meta = extract_windows(
            amp_matrix, labels, metadata=meta, window_size=window_size, stride=stride,
            label_strategy=label_strategy
        )

        if windows.size == 0:
            print(f"  Warning: no windows extracted from {fname}")
            continue
        if windows.shape[0] < min_windows:
            print(f"  Warning: only {windows.shape[0]} windows from {fname}; min_windows={min_windows}, skipped")
            continue

        for i in range(windows.shape[0]):
            all_windows.append(windows[i])
            all_window_labels.append(win_labels[i])
            start = int(win_meta[i]["start_frame"])
            end = int(win_meta[i]["end_frame"])
            all_window_meta.append({
                "file": fname,
                "window_idx": i,
                "start_frame": start,
                "end_frame": end,
                "node_id": int(meta["node_id"][0]) if meta["node_id"].size > 0 else 0,
                "rssi": win_meta[i].get("rssi"),
                "local_time_us": win_meta[i].get("local_time_us"),
                "seq": win_meta[i].get("seq"),
            })

    if not all_windows:
        raise ValueError("No windows extracted. Check window size, min_windows, and input file lengths.")

    print(f"\n[Result] Total windows: {len(all_windows)}, labels: {set(all_window_labels)}")

    return {
        "windows": all_windows,
        "labels": all_window_labels,
        "metadata": all_window_meta,
        "norm_stats": norm_stats,
        "file_names": file_names,
    }


def save_processed(
    output_dir: str | Path,
    data: dict[str, Any],
    split_name: str = "all",
) -> None:
    """保存预处理后的数据为 .npz 文件."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    windows = np.stack(data["windows"], axis=0)  # [N, T, C]
    labels = np.array(data["labels"])
    norm_stats = data["norm_stats"]

    save_path = out / f"{split_name}_windows.npz"
    np.savez(
        save_path,
        windows=windows,
        labels=labels,
        norm_stats=np.array(str(norm_stats)),  # 简单序列化
    )
    print(f"[Save] Saved {split_name}: {windows.shape} to {save_path}")

    # 同时保存 norm_stats 为 pickle
    save_norm_stats(norm_stats, out / "norm_stats.pkl")

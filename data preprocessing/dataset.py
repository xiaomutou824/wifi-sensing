"""PyTorch Dataset 封装，支持数据增强和类别采样."""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from augmentation import augment_window


class CSIDataset(Dataset):
    """CSI 动作识别数据集.

    Args:
        windows: 窗口列表，每个为 np.ndarray [window_size, n_subcarriers]
        labels: 标签列表
        label_map: 标签到整数的映射字典，如 {"idle": 0, "walking": 1, ...}
        augment_cfg: 数据增强配置字典，None 表示不增强
        is_training: 是否为训练模式（控制增强是否启用）
    """

    def __init__(
        self,
        windows: list[np.ndarray],
        labels: list[str],
        label_map: dict[str, int],
        augment_cfg: dict[str, Any] | None = None,
        is_training: bool = False,
    ):
        self.windows = windows
        self.labels_str = labels
        self.label_map = label_map
        self.augment_cfg = augment_cfg
        self.is_training = is_training

        # 转为整数标签
        self.labels = np.array([label_map[l] for l in labels], dtype=np.int64)

        # 统计
        self._print_stats()

    def _print_stats(self) -> None:
        counter = Counter(self.labels_str)
        print(f"[Dataset] Samples: {len(self.windows)}")
        for label, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            print(f"          - {label}: {count} ({count/len(self.windows)*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[idx].copy()  # [T, C]
        label = self.labels[idx]

        # 数据增强（仅训练时）
        if self.is_training and self.augment_cfg is not None:
            if self.augment_cfg.get("enabled", False):
                window = augment_window(window, self.augment_cfg)

        # 转为 torch tensor
        # 对于 1D-CNN/LSTM: [T, C]
        # 对于 2D-CNN: 可在 collate_fn 中增加维度
        x = torch.from_numpy(window).float()
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def build_weighted_sampler(labels: np.ndarray, class_weights: dict[str, float] | None = None) -> WeightedRandomSampler | None:
    """构建加权采样器，用于类别不平衡.

    Args:
        labels: 整数标签数组
        class_weights: 类别权重字典，如 {0: 1.0, 1: 1.0, 4: 3.0}

    Returns:
        WeightedRandomSampler 或 None
    """
    if class_weights is None:
        return None

    # 每个样本的权重
    sample_weights = np.array([class_weights.get(int(l), 1.0) for l in labels])
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(labels),
        replacement=True,
    )
    return sampler


def load_processed_npz(npz_path: str | Path) -> tuple[list[np.ndarray], list[str]]:
    """从 .npz 文件加载预处理后的数据."""
    data = np.load(npz_path, allow_pickle=True)
    windows = [data["windows"][i] for i in range(data["windows"].shape[0])]
    labels = data["labels"].tolist()
    return windows, labels


def create_datasets(
    train_data: dict[str, Any],
    val_data: dict[str, Any] | None,
    test_data: dict[str, Any] | None,
    augment_cfg: dict[str, Any] | None,
) -> tuple[CSIDataset, CSIDataset | None, CSIDataset | None, dict[str, int]]:
    """创建训练/验证/测试 Dataset.

    Args:
        train_data: preprocess_pipeline 输出的训练集数据
        val_data: 验证集数据
        test_data: 测试集数据
        augment_cfg: 增强配置

    Returns:
        (train_dataset, val_dataset, test_dataset, label_map)
    """
    # 统一标签映射（基于训练集）
    all_labels = sorted(set(train_data["labels"]))
    label_map = {label: idx for idx, label in enumerate(all_labels)}
    print(f"[Dataset] Label map: {label_map}")

    train_ds = CSIDataset(
        train_data["windows"],
        train_data["labels"],
        label_map,
        augment_cfg=augment_cfg,
        is_training=True,
    )

    val_ds = None
    if val_data is not None:
        val_ds = CSIDataset(
            val_data["windows"],
            val_data["labels"],
            label_map,
            augment_cfg=None,
            is_training=False,
        )

    test_ds = None
    if test_data is not None:
        test_ds = CSIDataset(
            test_data["windows"],
            test_data["labels"],
            label_map,
            augment_cfg=None,
            is_training=False,
        )

    return train_ds, val_ds, test_ds, label_map


def collate_fn_2d(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """用于 2D-CNN 的 collate_fn: [T, C] -> [1, T, C]."""
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)   # [B, T, C]
    xs = xs.unsqueeze(1)          # [B, 1, T, C]
    ys = torch.stack(ys, dim=0)
    return xs, ys


def collate_fn_tcn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """用于 TCN 的 collate_fn: [T, C] -> [C, T]."""
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)   # [B, T, C]
    xs = xs.permute(0, 2, 1)      # [B, C, T]
    ys = torch.stack(ys, dim=0)
    return xs, ys

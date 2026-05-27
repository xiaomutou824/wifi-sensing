"""完整的训练数据准备示例.

演示如何：
    1. 加载多个 CSV 文件
    2. 按 session（日期）划分训练/验证/测试集
    3. 预处理（过滤、转幅度、归一化、滑动窗口）
    4. 创建 PyTorch Dataset 和 DataLoader
    5. 保存为 .npz 供后续训练直接使用
"""

from __future__ import annotations

import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader

from dataset import CSIDataset, build_weighted_sampler, collate_fn_2d, collate_fn_tcn, create_datasets
from preprocess import preprocess_pipeline, save_processed


def group_files_by_session(csv_files: list[str]) -> dict[str, list[str]]:
    """按采集日期（文件名中的 YYYYMMDD）分组.

    例如:
        node1_idle_20260526_122147.csv -> session "20260526"
    """
    sessions: dict[str, list[str]] = defaultdict(list)
    for f in csv_files:
        # 从文件名提取日期，如 20260526
        m = re.search(r"(\d{8})", Path(f).name)
        if m:
            sessions[m.group(1)].append(f)
        else:
            sessions["unknown"].append(f)
    return dict(sessions)


def split_by_session(
    sessions: dict[str, list[str]],
    ratios: dict[str, float],
) -> tuple[list[str], list[str], list[str]]:
    """按 session 划分为训练/验证/测试集.

    策略：按 session 日期排序，按比例分配.
    例如 3 个 session，ratio 0.6/0.2/0.2 -> session1 训练, session2 验证, session3 测试
    """
    sorted_dates = sorted(sessions.keys())
    n = len(sorted_dates)

    train_cutoff = int(n * ratios["train"])
    val_cutoff = int(n * (ratios["train"] + ratios["val"]))

    train_dates = sorted_dates[:train_cutoff]
    val_dates = sorted_dates[train_cutoff:val_cutoff]
    test_dates = sorted_dates[val_cutoff:]

    train_files = []
    for d in train_dates:
        train_files.extend(sessions[d])

    val_files = []
    for d in val_dates:
        val_files.extend(sessions[d])

    test_files = []
    for d in test_dates:
        test_files.extend(sessions[d])

    return train_files, val_files, test_files


def main():
    # 1. 加载配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 搜索 CSV 文件
    csv_pattern = config["data"]["csv_paths"][0]
    csv_files = sorted(glob.glob(csv_pattern))
    print(f"Found {len(csv_files)} CSV files")

    if len(csv_files) < 3:
        print("Warning: Too few files. Using all for training demo.")
        train_files = csv_files
        val_files = csv_files[-1:] if len(csv_files) > 1 else []
        test_files = []
    else:
        sessions = group_files_by_session(csv_files)
        print(f"Sessions: {list(sessions.keys())}")
        train_files, val_files, test_files = split_by_session(
            sessions, config["data"]["split_ratio"]
        )

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

    # 3. 预处理训练集（计算归一化统计量）
    print("\n[1/3] Processing training set...")
    train_data = preprocess_pipeline(train_files, config, fit_norm=True)

    # 4. 预处理验证/测试集（使用训练集的统计量）
    print("\n[2/3] Processing validation set...")
    val_data = None
    if val_files:
        val_data = preprocess_pipeline(val_files, config, fit_norm=False, norm_stats=train_data["norm_stats"])

    print("\n[3/3] Processing test set...")
    test_data = None
    if test_files:
        test_data = preprocess_pipeline(test_files, config, fit_norm=False, norm_stats=train_data["norm_stats"])

    # 5. 保存为 .npz
    output_dir = config["data"]["output_dir"]
    save_processed(output_dir, train_data, split_name="train")
    if val_data:
        save_processed(output_dir, val_data, split_name="val")
    if test_data:
        save_processed(output_dir, test_data, split_name="test")

    # 6. 创建 PyTorch Dataset
    print("\n[4/4] Creating PyTorch Datasets...")
    augment_cfg = config.get("augment", None)
    train_ds, val_ds, test_ds, label_map = create_datasets(
        train_data, val_data, test_data, augment_cfg=augment_cfg
    )

    # 7. 创建 DataLoader
    batch_size = 32
    num_workers = 0  # Windows/macOS 建议设为 0

    # 类别权重采样（处理不平衡）
    class_weights = config.get("augment", {}).get("class_weights", None)
    sampler = None
    if class_weights and train_ds is not None:
        # 将字符串权重转为整数索引权重
        int_weights = {train_ds.label_map.get(k, 0): v for k, v in class_weights.items()}
        sampler = build_weighted_sampler(train_ds.labels, int_weights)
        print(f"Using weighted sampler with weights: {int_weights}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    # 8. 测试一个 batch
    print(f"\n{'='*50}")
    print("Testing DataLoader...")
    for batch_x, batch_y in train_loader:
        print(f"  Batch X shape: {batch_x.shape}")  # [B, T, C]
        print(f"  Batch Y shape: {batch_y.shape}")  # [B]
        print(f"  Batch Y labels: {batch_y[:10].tolist()}")
        break

    print(f"\n{'='*50}")
    print("All done! Files saved to:")
    print(f"  - {output_dir}/train_windows.npz")
    if val_data:
        print(f"  - {output_dir}/val_windows.npz")
    if test_data:
        print(f"  - {output_dir}/test_windows.npz")
    print(f"  - {output_dir}/norm_stats.pkl")
    print(f"\nLabel map: {label_map}")
    print("You can now use these files to train your model.")


if __name__ == "__main__":
    main()

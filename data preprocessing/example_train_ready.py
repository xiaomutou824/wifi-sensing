"""完整的训练数据准备示例.

演示如何：
    1. 加载多个 CSV 文件
    2. 按 label 分层 / session / 文件划分训练、验证、测试集
    3. 预处理（过滤、转幅度、归一化、滑动窗口）
    4. 创建 PyTorch Dataset 和 DataLoader
    5. 保存为 .npz 供后续训练直接使用
"""

from __future__ import annotations

import glob
import re
from collections import defaultdict
from pathlib import Path

import yaml

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


def infer_label_from_filename(csv_file: str) -> str:
    """从 node1_<label>_YYYYMMDD_HHMMSS.csv 形式的文件名中提取 label."""
    stem = Path(csv_file).stem
    m = re.match(r"node\d+_(.+)_\d{8}_\d{6}$", stem)
    if m:
        return m.group(1)

    # Fallback: remove timestamp suffix and optional node prefix.
    stem = re.sub(r"_\d{8}_\d{6}$", "", stem)
    stem = re.sub(r"^node\d+_", "", stem)
    return stem or "unknown"


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

    if n == 1:
        return list(sessions[sorted_dates[0]]), [], []

    train_cutoff = max(1, int(n * ratios["train"]))
    val_cutoff = int(n * (ratios["train"] + ratios["val"]))
    val_cutoff = max(train_cutoff, val_cutoff)

    if n >= 3 and val_cutoff == train_cutoff:
        val_cutoff = train_cutoff + 1

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


def split_by_file_ratio(
    csv_files: list[str],
    ratios: dict[str, float],
) -> tuple[list[str], list[str], list[str]]:
    """按文件划分，保证同一个文件只出现在一个 split。"""
    n = len(csv_files)
    if n == 0:
        return [], [], []
    if n == 1:
        return csv_files, [], []

    train_cutoff = max(1, int(n * ratios["train"]))
    val_cutoff = int(n * (ratios["train"] + ratios["val"]))
    if n >= 3 and val_cutoff == train_cutoff:
        val_cutoff = train_cutoff + 1

    return csv_files[:train_cutoff], csv_files[train_cutoff:val_cutoff], csv_files[val_cutoff:]


def split_by_label_stratified(
    csv_files: list[str],
    ratios: dict[str, float],
) -> tuple[list[str], list[str], list[str]]:
    """按文件名里的动作 label 分层划分.

    每个类别内部先按文件名排序，再按比例切 train/val/test，最后合并。
    这样可以尽量保证每个 split 都包含各个动作类别。
    """
    by_label: dict[str, list[str]] = defaultdict(list)
    for f in sorted(csv_files):
        by_label[infer_label_from_filename(f)].append(f)

    train_files: list[str] = []
    val_files: list[str] = []
    test_files: list[str] = []

    print("Label-stratified split:")
    for label, files in sorted(by_label.items()):
        n = len(files)
        if n == 1:
            label_train, label_val, label_test = files, [], []
        elif n == 2:
            label_train, label_val, label_test = files[:1], files[1:], []
        else:
            train_cutoff = max(1, int(n * ratios["train"]))
            val_count = max(1, int(n * ratios["val"]))
            if train_cutoff + val_count >= n:
                train_cutoff = max(1, n - 2)
                val_count = 1
            val_cutoff = train_cutoff + val_count
            label_train = files[:train_cutoff]
            label_val = files[train_cutoff:val_cutoff]
            label_test = files[val_cutoff:]

        train_files.extend(label_train)
        val_files.extend(label_val)
        test_files.extend(label_test)
        print(
            f"  {label}: total={n}, "
            f"train={len(label_train)}, val={len(label_val)}, test={len(label_test)}"
        )

    return sorted(train_files), sorted(val_files), sorted(test_files)


def split_files(
    csv_files: list[str],
    config: dict,
) -> tuple[list[str], list[str], list[str]]:
    """按配置划分 CSV 文件，兼容旧的 split_by_session 配置."""
    ratios = config["data"]["split_ratio"]
    split_strategy = config["data"].get("split_strategy")

    if split_strategy is None:
        split_strategy = "session" if config["data"].get("split_by_session", True) else "file"

    if split_strategy == "session":
        sessions = group_files_by_session(csv_files)
        print(f"Sessions: {list(sessions.keys())}")
        return split_by_session(sessions, ratios)
    if split_strategy == "file":
        return split_by_file_ratio(csv_files, ratios)
    if split_strategy == "stratified_label":
        return split_by_label_stratified(csv_files, ratios)

    raise ValueError(
        "Unknown split_strategy: "
        f"{split_strategy!r}. Expected one of: stratified_label, session, file."
    )


def main():
    # 1. 加载配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 搜索 CSV 文件
    csv_pattern = config["data"]["csv_paths"][0]
    csv_files = sorted(glob.glob(csv_pattern))
    print(f"Found {len(csv_files)} CSV files")

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for pattern: {csv_pattern}")

    train_files, val_files, test_files = split_files(csv_files, config)

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")
    if not val_files:
        print("Warning: no validation split. Add more sessions/files for reliable evaluation.")

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

    try:
        from torch.utils.data import DataLoader

        from augmentation import MixupCollator
        from dataset import build_weighted_sampler, create_datasets
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print("\n[4/4] PyTorch is not installed; skipped Dataset/DataLoader demo.")
        print("Preprocessed .npz files were saved successfully.")
        return

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
    augment_cfg = config.get("augment", {})
    class_weights = augment_cfg.get("class_weights", None) if augment_cfg.get("oversample", False) else None
    sampler = None
    if class_weights and train_ds is not None:
        # 将字符串权重转为整数索引权重
        int_weights = {
            train_ds.label_map[k]: v
            for k, v in class_weights.items()
            if k in train_ds.label_map
        }
        sampler = build_weighted_sampler(train_ds.labels, int_weights)
        print(f"Using weighted sampler with weights: {int_weights}")

    collate_fn = None
    if augment_cfg.get("enabled", False) and augment_cfg.get("mixup", False):
        collate_fn = MixupCollator(
            alpha=augment_cfg.get("mixup_alpha", 0.4),
            num_classes=len(label_map),
        )
        print("Using MixupCollator; train labels are soft one-hot vectors.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        drop_last=(len(train_ds) >= batch_size),
        collate_fn=collate_fn,
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

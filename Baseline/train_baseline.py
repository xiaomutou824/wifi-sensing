#!/usr/bin/env python3
"""CSI 动作识别 Baseline 训练脚本.

使用传统机器学习（XGBoost / Random Forest）快速验证数据可用性。
复用 data preprocessing/ 中的预处理和特征提取代码。

用法:
    cd Baseline
    pip install -r requirements.txt
    python train_baseline.py
"""

from __future__ import annotations

import glob
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# 把 data preprocessing/ 加入路径，复用其代码
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PREPROC_DIR = _SCRIPT_DIR.parent / "data preprocessing"
sys.path.insert(0, str(_PREPROC_DIR))

from features import extract_all_features
from preprocess import preprocess_pipeline
from utils import load_norm_stats, save_norm_stats


def group_files_by_session(csv_files: list[str]) -> dict[str, list[str]]:
    """按文件名中的日期分组."""
    sessions: dict[str, list[str]] = defaultdict(list)
    for f in csv_files:
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
    """按 session 日期排序后按比例分配."""
    sorted_dates = sorted(sessions.keys())
    n = len(sorted_dates)
    train_cutoff = max(1, int(n * ratios["train"]))
    val_cutoff = max(train_cutoff + 1, int(n * (ratios["train"] + ratios["val"])))

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


def extract_features_from_pipeline_result(
    result: dict,
    feature_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """从 preprocess_pipeline 的结果中提取手工特征向量.

    Args:
        result: preprocess_pipeline 输出
        feature_cfg: 特征配置

    Returns:
        X: [n_samples, n_features]
        y: [n_samples] 整数标签
    """
    windows = result["windows"]          # list of [T, C]
    labels_str = result["labels"]        # list of str
    meta_list = result.get("metadata", [])

    # 构建标签映射（基于所有数据）
    all_labels = sorted(set(labels_str))
    label_map = {label: idx for idx, label in enumerate(all_labels)}

    # 提取特征
    features = []
    valid_labels = []

    for i, win in enumerate(tqdm(windows, desc="Extracting features")):
        # RSSI 窗口：从原始帧的 metadata 中重建（简化：用幅度代替）
        # 如果未来需要真实 RSSI，需要把原始 metadata 传进来
        feat = extract_all_features(win, rssi_window=None, cfg=feature_cfg)
        features.append(feat)
        valid_labels.append(label_map[labels_str[i]])

    X = np.stack(features, axis=0)
    y = np.array(valid_labels, dtype=np.int64)
    return X, y, label_map


def train_model(
    model_name: str,
    model_cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: dict[str, float] | None,
    label_map: dict[str, int],
) -> object:
    """训练单个模型."""
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")

    # 类别权重（sklearn 格式）
    weight_dict = None
    if class_weights:
        # 将字符串标签映射到整数权重
        weight_dict = {}
        for label_str, w in class_weights.items():
            idx = label_map.get(label_str)
            if idx is not None:
                weight_dict[idx] = w
        print(f"Class weights: {weight_dict}")

    if model_name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=model_cfg.get("n_estimators", 200),
            max_depth=model_cfg.get("max_depth", 6),
            learning_rate=model_cfg.get("learning_rate", 0.1),
            subsample=model_cfg.get("subsample", 0.8),
            colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
            objective=model_cfg.get("objective", "multi:softprob"),
            eval_metric=model_cfg.get("eval_metric", "mlogloss"),
            random_state=model_cfg.get("random_state", 42),
            n_jobs=model_cfg.get("n_jobs", 4),
            use_label_encoder=False,
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 200),
            max_depth=model_cfg.get("max_depth", 12),
            min_samples_split=model_cfg.get("min_samples_split", 5),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 2),
            random_state=model_cfg.get("random_state", 42),
            n_jobs=model_cfg.get("n_jobs", 4),
            class_weight=weight_dict if weight_dict else "balanced",
        )
    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=model_cfg.get("n_estimators", 200),
            max_depth=model_cfg.get("max_depth", 6),
            learning_rate=model_cfg.get("learning_rate", 0.1),
            subsample=model_cfg.get("subsample", 0.8),
            colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
            random_state=model_cfg.get("random_state", 42),
            n_jobs=model_cfg.get("n_jobs", 4),
            class_weight=weight_dict if weight_dict else "balanced",
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    label_map: dict[str, int],
    split_name: str = "",
) -> dict[str, float]:
    """评估模型."""
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y, y_pred, average="macro", zero_division=0)

    inv_map = {v: k for k, v in label_map.items()}
    target_names = [inv_map[i] for i in range(len(inv_map))]

    print(f"\n[{split_name}] Results:")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  Macro F1:      {macro_f1:.4f}")
    print(f"  Macro Prec:    {macro_precision:.4f}")
    print(f"  Macro Recall:  {macro_recall:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=target_names, digits=4, zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "y_true": y,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict[str, int],
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    """绘制混淆矩阵."""
    inv_map = {v: k for k, v in label_map.items()}
    labels = [inv_map[i] for i in range(len(inv_map))]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 原始计数
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"{title} (Count)")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")

    # 归一化
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f"{title} (Normalized)")
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()


def plot_feature_importance(
    model: object,
    model_name: str,
    label_map: dict[str, int],
    save_path: str | None = None,
) -> None:
    """绘制特征重要性（仅限树模型）."""
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not support feature_importances_, skipping plot.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:30]  # Top 30

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [f"feat_{i}" for i in indices])
    plt.xlabel("Importance")
    plt.title(f"{model_name} - Top 30 Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved feature importance to {save_path}")
    plt.show()


def run_loso_cv(
    csv_files: list[str],
    preproc_config: dict,
    feature_cfg: dict,
    model_name: str,
    model_cfg: dict,
    class_weights: dict[str, float] | None,
    output_dir: Path,
) -> None:
    """Leave-One-Session-Out 交叉验证."""
    sessions = group_files_by_session(csv_files)
    sorted_dates = sorted(sessions.keys())
    print(f"LOSO CV: {len(sorted_dates)} sessions -> {sorted_dates}")

    all_results = []
    best_f1 = -1.0
    best_model = None
    best_label_map = None

    for i, test_date in enumerate(sorted_dates):
        train_dates = [d for d in sorted_dates if d != test_date]
        train_files = []
        for d in train_dates:
            train_files.extend(sessions[d])
        test_files = sessions[test_date]

        print(f"\n{'='*60}")
        print(f"Fold {i+1}/{len(sorted_dates)}: test session = {test_date}")
        print(f"  Train: {len(train_dates)} sessions, {len(train_files)} files")
        print(f"  Test:  {test_date}, {len(test_files)} files")

        # 预处理
        train_data = preprocess_pipeline(train_files, preproc_config, fit_norm=True)
        test_data = preprocess_pipeline(test_files, preproc_config, fit_norm=False,
                                         norm_stats=train_data["norm_stats"])

        # 提取特征
        X_train, y_train, label_map = extract_features_from_pipeline_result(train_data, feature_cfg)
        X_test, y_test, _ = extract_features_from_pipeline_result(test_data, feature_cfg)

        if X_train.size == 0 or X_test.size == 0:
            print("  Skip: empty data")
            continue

        # 训练
        model = train_model(model_name, model_cfg, X_train, y_train, class_weights, label_map)

        # 评估
        result = evaluate_model(model, X_test, y_test, label_map, split_name=f"Test-{test_date}")
        all_results.append(result)

        if result["macro_f1"] > best_f1:
            best_f1 = result["macro_f1"]
            best_model = model
            best_label_map = label_map

    # 汇总
    if not all_results:
        print("No valid results from LOSO CV.")
        return

    avg_acc = np.mean([r["accuracy"] for r in all_results])
    avg_f1 = np.mean([r["macro_f1"] for r in all_results])
    print(f"\n{'='*60}")
    print(f"LOSO CV Summary ({model_name}):")
    print(f"  Average Accuracy: {avg_acc:.4f}")
    print(f"  Average Macro F1: {avg_f1:.4f}")

    # 保存最佳模型
    if best_model is not None:
        model_path = output_dir / f"best_{model_name}_loso.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": best_model, "label_map": best_label_map}, f)
        print(f"Saved best model to {model_path}")


def run_simple_split(
    csv_files: list[str],
    preproc_config: dict,
    feature_cfg: dict,
    model_cfgs: dict[str, dict],
    class_weights: dict[str, float] | None,
    output_dir: Path,
    ratios: dict[str, float],
) -> None:
    """简单划分：按文件/session 分成 train/val/test."""
    sessions = group_files_by_session(csv_files)
    train_files, val_files, test_files = split_by_session(sessions, ratios)

    print(f"Train: {len(train_files)} files")
    print(f"Val:   {len(val_files)} files")
    print(f"Test:  {len(test_files)} files")

    if not train_files:
        raise ValueError("No training files found.")

    # 预处理
    print("\n[Preprocess] Training set...")
    train_data = preprocess_pipeline(train_files, preproc_config, fit_norm=True)

    val_data = None
    if val_files:
        print("[Preprocess] Validation set...")
        val_data = preprocess_pipeline(val_files, preproc_config, fit_norm=False,
                                       norm_stats=train_data["norm_stats"])

    test_data = None
    if test_files:
        print("[Preprocess] Test set...")
        test_data = preprocess_pipeline(test_files, preproc_config, fit_norm=False,
                                        norm_stats=train_data["norm_stats"])

    # 提取特征
    X_train, y_train, label_map = extract_features_from_pipeline_result(train_data, feature_cfg)
    X_val, y_val, _ = (extract_features_from_pipeline_result(val_data, feature_cfg)
                       if val_data else (None, None, None))
    X_test, y_test, _ = (extract_features_from_pipeline_result(test_data, feature_cfg)
                         if test_data else (None, None, None))

    print(f"\nFeature shape: {X_train.shape}")

    # 训练所有启用的模型
    for model_name, model_cfg in model_cfgs.items():
        if not model_cfg.get("enabled", False):
            continue

        model = train_model(model_name, model_cfg, X_train, y_train, class_weights, label_map)

        # 验证集
        if X_val is not None and X_val.size > 0:
            evaluate_model(model, X_val, y_val, label_map, split_name="Validation")

        # 测试集
        if X_test is not None and X_test.size > 0:
            result = evaluate_model(model, X_test, y_test, label_map, split_name="Test")
            # 混淆矩阵
            cm_path = output_dir / f"cm_{model_name}.png"
            plot_confusion_matrix(result["y_true"], result["y_pred"], label_map,
                                  title=f"{model_name} Test", save_path=str(cm_path))
        else:
            # 没有测试集时，在训练集上评估（仅供调试）
            result = evaluate_model(model, X_train, y_train, label_map, split_name="Train")

        # 特征重要性
        fi_path = output_dir / f"feature_importance_{model_name}.png"
        plot_feature_importance(model, model_name, label_map, save_path=str(fi_path))

        # 保存模型
        model_path = output_dir / f"model_{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "label_map": label_map}, f)
        print(f"Saved model to {model_path}")


def main():
    # 加载配置
    config_path = _SCRIPT_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 搜索 CSV
    csv_pattern = config["data"]["csv_paths"][0]
    if not os.path.isabs(csv_pattern):
        csv_pattern = str(_SCRIPT_DIR / csv_pattern)
    csv_files = sorted(glob.glob(csv_pattern))
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files[:10]:
        print(f"  - {Path(f).name}")
    if len(csv_files) > 10:
        print(f"  ... and {len(csv_files)-10} more")

    if not csv_files:
        print("ERROR: No CSV files found. Please check csv_paths in config.yaml.")
        sys.exit(1)

    # 输出目录
    output_dir = Path(config["data"]["output_dir"])
    if not os.path.isabs(output_dir):
        output_dir = _SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 预处理配置（复用 data preprocessing 的格式）
    preproc_config = {
        "preprocess": config["preprocess"],
        "window": config["window"],
    }
    feature_cfg = config["features"]
    model_cfgs = config["models"]
    class_weights = config["training"].get("class_weights", None)

    # 判断使用 LOSO 还是简单划分
    use_loso = config["training"].get("loso_cv", False)

    if use_loso:
        # 对每个启用的模型做 LOSO
        for model_name, model_cfg in model_cfgs.items():
            if not model_cfg.get("enabled", False):
                continue
            run_loso_cv(
                csv_files, preproc_config, feature_cfg,
                model_name, model_cfg, class_weights, output_dir,
            )
    else:
        run_simple_split(
            csv_files, preproc_config, feature_cfg,
            model_cfgs, class_weights, output_dir,
            config["data"]["split_ratio"],
        )

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()

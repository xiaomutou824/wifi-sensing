#!/usr/bin/env python3
"""使用训练好的 Baseline 模型对单个 CSV 文件做预测示例."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PREPROC_DIR = _SCRIPT_DIR.parent / "data preprocessing"
sys.path.insert(0, str(_PREPROC_DIR))

from features import extract_all_features
from preprocess import preprocess_single_file


def load_model(model_path: str | Path) -> tuple[object, dict[str, int]]:
    """加载保存的模型."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["label_map"]


def predict_csv(
    csv_path: str | Path,
    model_path: str | Path,
    preproc_config: dict | None = None,
    feature_cfg: dict | None = None,
) -> dict:
    """对单个 CSV 文件做预测，返回每窗口的预测结果.

    Args:
        csv_path: 待预测的 CSV 文件
        model_path: 保存的模型 .pkl 文件
        preproc_config: 预处理配置（默认使用 config.yaml 中的）
        feature_cfg: 特征配置（默认使用 config.yaml 中的）

    Returns:
        {
            "window_predictions": list[str],  # 每个窗口的预测标签
            "window_probs": np.ndarray,       # [n_windows, n_classes]
            "majority_vote": str,             # 整段多数投票结果
            "confidence": float,              # 多数投票的置信度
        }
    """
    # 加载模型
    model, label_map = load_model(model_path)
    inv_label_map = {v: k for k, v in label_map.items()}

    # 默认配置
    if preproc_config is None:
        import yaml
        with open(_SCRIPT_DIR / "config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        preproc_config = {"preprocess": config["preprocess"], "window": config["window"]}
        feature_cfg = config["features"]

    # 预处理
    result = preprocess_single_file(csv_path, preproc_config)
    if result is None:
        raise ValueError(f"Failed to preprocess {csv_path}")

    amp_matrix, labels, meta = result

    # 提取特征
    features = []
    for win in result["windows"]:
        feat = extract_all_features(win, rssi_window=None, cfg=feature_cfg)
        features.append(feat)

    if not features:
        raise ValueError("No valid windows extracted.")

    X = np.stack(features, axis=0)

    # 预测
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)  # [n_windows, n_classes]
    else:
        preds = model.predict(X)
        n_classes = len(label_map)
        probs = np.zeros((len(preds), n_classes))
        probs[np.arange(len(preds)), preds] = 1.0

    pred_labels_idx = np.argmax(probs, axis=1)
    pred_labels = [inv_label_map[idx] for idx in pred_labels_idx]

    # 多数投票
    from collections import Counter
    vote = Counter(pred_labels).most_common(1)[0]
    majority_label = vote[0]
    confidence = vote[1] / len(pred_labels)

    return {
        "window_predictions": pred_labels,
        "window_probs": probs,
        "majority_vote": majority_label,
        "confidence": confidence,
    }


def main():
    import sys
    import yaml

    if len(sys.argv) < 3:
        print("Usage: python example_predict.py <csv_file> <model.pkl>")
        print("Example:")
        print("  python example_predict.py ../CSI collection/data/node1_walking_xxx.csv output/model_xgboost.pkl")
        sys.exit(1)

    csv_path = sys.argv[1]
    model_path = sys.argv[2]

    # 加载配置
    with open(_SCRIPT_DIR / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    preproc_config = {"preprocess": config["preprocess"], "window": config["window"]}
    feature_cfg = config["features"]

    result = predict_csv(csv_path, model_path, preproc_config, feature_cfg)

    print(f"\nPrediction Result:")
    print(f"  File: {csv_path}")
    print(f"  Total windows: {len(result['window_predictions'])}")
    print(f"  Window predictions: {result['window_predictions'][:20]}...")
    print(f"  Majority vote: {result['majority_vote']} (confidence: {result['confidence']:.2%})")

    # 各类占比
    from collections import Counter
    dist = Counter(result["window_predictions"])
    print(f"\nDistribution:")
    for label, count in dist.most_common():
        print(f"  {label}: {count} ({count/len(result['window_predictions'])*100:.1f}%)")


if __name__ == "__main__":
    main()

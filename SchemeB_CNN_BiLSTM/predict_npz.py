#!/usr/bin/env python3
"""Run CNN-BiLSTM inference on a processed NPZ split."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import CNNBiLSTM

ROOT_DIR = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = ROOT_DIR / "data preprocessing"
sys.path.insert(0, str(PREPROCESS_DIR))

from dataset import load_processed_npz  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CSI actions from a processed .npz")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--npz", required=True, help="Processed windows .npz")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(ckpt: dict, device: torch.device) -> CNNBiLSTM:
    cfg = ckpt["config"]["model"]
    label_map = ckpt["label_map"]
    model = CNNBiLSTM(
        n_features=int(ckpt["n_features"]),
        n_classes=len(label_map),
        cnn_channels=tuple(cfg.get("cnn_channels", [32, 64, 128])),
        lstm_hidden=int(cfg.get("lstm_hidden", 64)),
        lstm_layers=int(cfg.get("lstm_layers", 2)),
        lstm_dropout=float(cfg.get("lstm_dropout", 0.3)),
        classifier_dropout=float(cfg.get("classifier_dropout", 0.5)),
        use_attention=bool(cfg.get("use_attention", False)),
        attention_heads=int(cfg.get("attention_heads", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    label_map = ckpt["label_map"]
    idx_to_label = {idx: label for label, idx in label_map.items()}

    windows, labels = load_processed_npz(args.npz)
    x = torch.from_numpy(np.stack(windows, axis=0)).float()
    loader = DataLoader(TensorDataset(x), batch_size=args.batch_size, shuffle=False)

    model = build_model(ckpt, device)
    rows = []
    for (batch_x,) in loader:
        logits = model(batch_x.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_idx = probs.argmax(axis=1)
        for idx, prob in zip(pred_idx, probs):
            rows.append((idx_to_label[int(idx)], float(prob[int(idx)]), prob.tolist()))

    correct = None
    if labels:
        correct = sum(pred == true for (pred, _, _), true in zip(rows, labels))
        print(f"Accuracy: {correct / len(labels):.4f} ({correct}/{len(labels)})")

    for i, (pred, confidence, _) in enumerate(rows[:20]):
        suffix = f", true={labels[i]}" if labels else ""
        print(f"{i:04d}: pred={pred}, confidence={confidence:.4f}{suffix}")
    if len(rows) > 20:
        print(f"... {len(rows) - 20} more windows")

    if args.output_csv:
        class_names = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "true_label", "pred_label", "confidence", *[f"prob_{c}" for c in class_names]])
            for i, (pred, confidence, probs) in enumerate(rows):
                true_label = labels[i] if labels else ""
                writer.writerow([i, true_label, pred, confidence, *probs])
        print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()

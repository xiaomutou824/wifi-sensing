#!/usr/bin/env python3
"""Train the Scheme B CNN-BiLSTM model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CNNBiLSTM, count_parameters

ROOT_DIR = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = ROOT_DIR / "data preprocessing"
sys.path.insert(0, str(PREPROCESS_DIR))

from augmentation import MixupCollator  # noqa: E402
from dataset import CSIDataset, build_weighted_sampler, load_processed_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN-BiLSTM for CSI action recognition")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override train.learning_rate")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_path(path: str | None, base_dir: Path) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def log(message: str = "") -> None:
    print(message, flush=True)


def load_split(split_path: Path | None) -> tuple[Any, list[str]] | None:
    if split_path is None or not split_path.exists():
        return None
    log(f"[Load] {split_path}")
    windows, labels = load_processed_split(split_path)
    log(f"[Load] samples={len(labels)}, shape={getattr(windows, 'shape', 'list')}")
    return windows, labels


def configured_data_path(data_cfg: dict[str, Any], split: str) -> str | None:
    return data_cfg.get(f"{split}_path") or data_cfg.get(f"{split}_npz")


def build_label_map(train_labels: list[str], configured_labels: list[str] | None) -> dict[str, int]:
    observed = set(train_labels)
    if configured_labels:
        missing = observed.difference(configured_labels)
        if missing:
            raise ValueError(f"Labels in train set are missing from config data.labels: {sorted(missing)}")
        return {label: idx for idx, label in enumerate(configured_labels)}
    return {label: idx for idx, label in enumerate(sorted(observed))}


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    hard_criterion: nn.Module,
) -> torch.Tensor:
    if targets.ndim == 2:
        return soft_cross_entropy(logits, targets.to(logits.device))
    return hard_criterion(logits, targets.to(logits.device))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    log_interval: int = 50,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    preds: list[int] = []
    targets_all: list[int] = []

    iterator = tqdm(loader, desc="train" if is_train else "eval", leave=False)
    for batch_idx, (x, y) in enumerate(iterator, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(x)
                loss = compute_loss(logits, y, criterion)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        losses.append(float(loss.detach().cpu()))
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        preds.extend(pred.tolist())
        if y.ndim == 2:
            targets_all.extend(y.argmax(dim=1).detach().cpu().numpy().tolist())
        else:
            targets_all.extend(y.detach().cpu().numpy().tolist())

        iterator.set_postfix(loss=np.mean(losses), acc=accuracy_score(targets_all, preds))
        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == len(loader)):
            log(
                f"[{'train' if is_train else 'eval'}] "
                f"batch {batch_idx}/{len(loader)} "
                f"loss={np.mean(losses):.4f} acc={accuracy_score(targets_all, preds):.4f}"
            )

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(targets_all, preds)) if targets_all else 0.0,
        "macro_f1": float(f1_score(targets_all, preds, average="macro", zero_division=0)) if targets_all else 0.0,
    }


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    for x, y in loader:
        logits = model(x.to(device))
        preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        labels.extend(y.cpu().numpy().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(preds, dtype=np.int64)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metric: float,
    label_map: dict[str, int],
    config: dict[str, Any],
    n_features: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": metric,
            "label_map": label_map,
            "config": config,
            "n_features": n_features,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = Path(__file__).resolve().parent / config_path
    config_path = config_path.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["learning_rate"] = args.lr

    base_dir = config_path.parent
    output_dir = resolve_path(cfg["data"]["output_dir"], base_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["train"].get("seed", 42)))
    device = select_device(str(cfg["train"].get("device", "auto")))

    train_split = load_split(resolve_path(configured_data_path(cfg["data"], "train"), base_dir))
    if train_split is None:
        raise FileNotFoundError(f"Train split not found: {configured_data_path(cfg['data'], 'train')}")
    val_split = load_split(resolve_path(configured_data_path(cfg["data"], "val"), base_dir))
    test_split = load_split(resolve_path(configured_data_path(cfg["data"], "test"), base_dir))

    train_windows, train_labels = train_split
    label_map = build_label_map(train_labels, cfg["data"].get("labels"))
    class_names = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
    n_classes = len(label_map)
    n_features = int(train_windows[0].shape[1])

    augment_cfg = cfg.get("augment", {})
    train_ds = CSIDataset(train_windows, train_labels, label_map, augment_cfg=augment_cfg, is_training=True)
    val_ds = CSIDataset(*val_split, label_map=label_map, augment_cfg=None, is_training=False) if val_split else None
    test_ds = CSIDataset(*test_split, label_map=label_map, augment_cfg=None, is_training=False) if test_split else None

    class_weights_cfg = augment_cfg.get("class_weights", {}) if augment_cfg.get("oversample", False) else {}
    sampler_weights = {label_map[k]: v for k, v in class_weights_cfg.items() if k in label_map}
    sampler = build_weighted_sampler(train_ds.labels, sampler_weights) if sampler_weights else None

    collate_fn = None
    if augment_cfg.get("enabled", False) and augment_cfg.get("mixup", False):
        collate_fn = MixupCollator(alpha=augment_cfg.get("mixup_alpha", 0.4), num_classes=n_classes)

    batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=len(train_ds) >= batch_size,
        collate_fn=collate_fn,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        if val_ds is not None
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        if test_ds is not None
        else None
    )

    model_cfg = cfg["model"]
    model = CNNBiLSTM(
        n_features=n_features,
        n_classes=n_classes,
        cnn_channels=tuple(model_cfg.get("cnn_channels", [32, 64, 128])),
        lstm_hidden=int(model_cfg.get("lstm_hidden", 64)),
        lstm_layers=int(model_cfg.get("lstm_layers", 2)),
        lstm_dropout=float(model_cfg.get("lstm_dropout", 0.3)),
        classifier_dropout=float(model_cfg.get("classifier_dropout", 0.5)),
        use_attention=bool(model_cfg.get("use_attention", False)),
        attention_heads=int(model_cfg.get("attention_heads", 4)),
    ).to(device)

    ce_weights = torch.ones(n_classes, dtype=torch.float32)
    for label, weight in augment_cfg.get("class_weights", {}).items():
        if label in label_map:
            ce_weights[label_map[label]] = float(weight)
    criterion = nn.CrossEntropyLoss(weight=ce_weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["train"]["epochs"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("use_amp", True)) and device.type == "cuda")

    log(f"Device: {device}")
    log(f"Input shape: [N, T, {n_features}], classes: {class_names}")
    log(f"Trainable parameters: {count_parameters(model):,}")

    history: list[dict[str, Any]] = []
    best_metric = -1.0
    bad_epochs = 0
    patience = int(cfg["train"].get("patience", 15))

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            use_amp=bool(cfg["train"].get("use_amp", True)),
            log_interval=int(cfg["train"].get("log_interval", 50)),
        )

        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                criterion,
                device,
                log_interval=int(cfg["train"].get("log_interval", 50)),
            )
            monitor = val_metrics["macro_f1"]
        else:
            val_metrics = {}
            monitor = train_metrics["macro_f1"]

        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        log(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} f1={train_metrics['macro_f1']:.4f} | "
            f"val acc={val_metrics.get('accuracy', 0.0):.4f} f1={val_metrics.get('macro_f1', 0.0):.4f}"
        )

        if monitor > best_metric:
            best_metric = monitor
            bad_epochs = 0
            save_checkpoint(
                output_dir / "best_model.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                label_map,
                cfg,
                n_features,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"Early stopping at epoch {epoch}; best macro_f1={best_metric:.4f}")
                break

    save_checkpoint(
        output_dir / "last_model.pt",
        model,
        optimizer,
        scheduler,
        history[-1]["epoch"],
        best_metric,
        label_map,
        cfg,
        n_features,
    )

    eval_loader = test_loader or val_loader
    eval_name = "test" if test_loader is not None else "val"
    report = None
    if eval_loader is not None:
        best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
        y_true, y_pred = collect_predictions(model, eval_loader, device)
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(n_classes)),
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
        save_confusion_matrix(y_true, y_pred, class_names, output_dir / "confusion_matrix.png")
        log(f"\n{eval_name} classification report:")
        log(classification_report(y_true, y_pred, labels=list(range(n_classes)), target_names=class_names, zero_division=0))

    (output_dir / "metrics.json").write_text(
        json.dumps({"history": history, "best_metric": best_metric, "eval": report}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    log(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()

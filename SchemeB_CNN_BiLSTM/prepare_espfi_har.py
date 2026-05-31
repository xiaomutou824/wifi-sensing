#!/usr/bin/env python3
"""Prepare ESP-Fi HAR .mat files for the CNN-BiLSTM trainer."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio


LABEL_MAP = {
    1: "run",
    2: "fall",
    3: "walk",
    4: "turn",
    5: "jump",
    6: "squat",
    7: "arm_wave",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ESP-Fi HAR .mat files to train/val/test NPZ windows.")
    parser.add_argument(
        "--raw-dir",
        default="../datasets/ESP-Fi-HAR/raw",
        help="ESP-Fi HAR raw directory containing EnvironmentNo.* folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="../datasets/ESP-Fi-HAR/processed_cnn_bilstm",
        help="Directory for generated NPZ splits.",
    )
    parser.add_argument("--window-size", type=int, default=256, help="Sliding window length in CSI frames.")
    parser.add_argument("--stride", type=int, default=128, help="Sliding window stride in CSI frames.")
    parser.add_argument(
        "--split-mode",
        choices=["subject", "environment", "sample"],
        default="subject",
        help="Split strategy. subject is recommended to reduce identity leakage.",
    )
    parser.add_argument("--train-subjects", default="1,2,3,4,5,6", help="Subject IDs for train split.")
    parser.add_argument("--val-subjects", default="7", help="Subject IDs for val split.")
    parser.add_argument("--test-subjects", default="8", help="Subject IDs for test split.")
    parser.add_argument("--train-envs", default="1,2,3", help="Environment IDs for train split when split-mode=environment.")
    parser.add_argument("--val-envs", default="4", help="Environment IDs for val split when split-mode=environment.")
    parser.add_argument("--test-envs", default="", help="Environment IDs for test split when split-mode=environment.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio when split-mode=sample.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio when split-mode=sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample split.")
    parser.add_argument(
        "--normalize",
        choices=["window_zscore", "sample_zscore", "none"],
        default="window_zscore",
        help="Normalization method applied before saving windows.",
    )
    parser.add_argument(
        "--include-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include one final window ending at the sample tail if stride does not land there.",
    )
    parser.add_argument(
        "--output-format",
        choices=["npy", "npz"],
        default="npy",
        help="npy writes memory-mapped *_windows.npy + *_labels.npy files and is recommended for servers.",
    )
    parser.add_argument(
        "--save-meta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-window metadata JSON. Disabled by default to avoid large memory use.",
    )
    return parser.parse_args()


def log(message: str = "") -> None:
    print(message, flush=True)


def parse_id_list(text: str) -> set[int]:
    if not text.strip():
        return set()
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def parse_meta(path: Path) -> tuple[int, int, int, int]:
    try:
        env_id, subject_id, activity_id, trial_id = map(int, path.stem.split("-"))
    except ValueError as exc:
        raise ValueError(f"Unexpected ESP-Fi HAR filename: {path.name}") from exc
    if activity_id not in LABEL_MAP:
        raise ValueError(f"Unknown activity id {activity_id} in {path}")
    return env_id, subject_id, activity_id, trial_id


def load_csiamp(path: Path) -> np.ndarray:
    data = sio.loadmat(path)
    if "CSIamp" not in data:
        raise KeyError(f"{path} does not contain CSIamp")
    csi = np.asarray(data["CSIamp"], dtype=np.float32)
    if csi.ndim != 2:
        raise ValueError(f"Expected CSIamp [T, C], got {csi.shape} in {path}")
    return csi


def zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def window_starts(n_frames: int, window_size: int, stride: int, include_tail: bool) -> list[int]:
    if n_frames < window_size:
        return []
    starts = list(range(0, n_frames - window_size + 1, stride))
    tail_start = n_frames - window_size
    if include_tail and starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def choose_split(
    env_id: int,
    subject_id: int,
    train_subjects: set[int],
    val_subjects: set[int],
    test_subjects: set[int],
    train_envs: set[int],
    val_envs: set[int],
    test_envs: set[int],
    split_mode: str,
) -> str | None:
    if split_mode == "subject":
        if subject_id in train_subjects:
            return "train"
        if subject_id in val_subjects:
            return "val"
        if subject_id in test_subjects:
            return "test"
        return None
    if split_mode == "environment":
        if env_id in train_envs:
            return "train"
        if env_id in val_envs:
            return "val"
        if env_id in test_envs:
            return "test"
        return None
    return None


def save_npz(path: Path, windows: list[np.ndarray], labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not windows:
        np.savez_compressed(path, windows=np.empty((0, 0, 0), dtype=np.float32), labels=np.asarray([], dtype=object))
        return
    np.savez_compressed(
        path,
        windows=np.stack(windows).astype(np.float32),
        labels=np.asarray(labels, dtype=object),
    )


def count_split_windows(
    split_files: dict[str, list[Path]],
    window_size: int,
    stride: int,
    include_tail: bool,
) -> tuple[dict[str, int], int]:
    counts = {"train": 0, "val": 0, "test": 0}
    n_features = 0
    for split, files in split_files.items():
        log(f"[Pass 1/2] Counting {split}: {len(files)} files")
        for idx, path in enumerate(sorted(files), start=1):
            csi = load_csiamp(path)
            if n_features == 0:
                n_features = int(csi.shape[1])
            elif csi.shape[1] != n_features:
                raise ValueError(f"Feature mismatch in {path}: expected {n_features}, got {csi.shape[1]}")
            counts[split] += len(window_starts(csi.shape[0], window_size, stride, include_tail))
            if idx % 100 == 0 or idx == len(files):
                log(f"  {split}: counted {idx}/{len(files)} files, windows={counts[split]}")
    if n_features == 0:
        raise ValueError("No usable CSIamp matrix found.")
    return counts, n_features


def open_split_memmaps(
    output_dir: Path,
    counts: dict[str, int],
    window_size: int,
    n_features: int,
) -> dict[str, np.ndarray]:
    arrays = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, count in counts.items():
        arrays[split] = np.lib.format.open_memmap(
            output_dir / f"{split}_windows.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, window_size, n_features),
        )
    return arrays


def write_npy_splits(
    split_files: dict[str, list[Path]],
    raw_dir: Path,
    output_dir: Path,
    counts: dict[str, int],
    n_features: int,
    args: argparse.Namespace,
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, int | str]]]]:
    windows_out = open_split_memmaps(output_dir, counts, args.window_size, n_features)
    labels_out = {split: [] for split in ["train", "val", "test"]}
    meta_out: dict[str, list[dict[str, int | str]]] = {split: [] for split in ["train", "val", "test"]}

    for split, files in split_files.items():
        write_idx = 0
        log(f"[Pass 2/2] Writing {split}: {len(files)} files -> {counts[split]} windows")
        for file_idx, path in enumerate(sorted(files), start=1):
            env_id, subject_id, activity_id, trial_id = parse_meta(path)
            label = LABEL_MAP[activity_id]
            csi = load_csiamp(path)
            if args.normalize == "sample_zscore":
                csi = zscore(csi).astype(np.float32, copy=False)

            starts = window_starts(csi.shape[0], args.window_size, args.stride, args.include_tail)
            for start in starts:
                window = csi[start:start + args.window_size].astype(np.float32, copy=True)
                if args.normalize == "window_zscore":
                    window = zscore(window).astype(np.float32, copy=False)
                windows_out[split][write_idx] = window
                labels_out[split].append(label)
                if args.save_meta:
                    meta_out[split].append(
                        {
                            "source": str(path.relative_to(raw_dir)),
                            "env_id": env_id,
                            "subject_id": subject_id,
                            "activity_id": activity_id,
                            "trial_id": trial_id,
                            "start": start,
                        }
                    )
                write_idx += 1

            if file_idx % 100 == 0 or file_idx == len(files):
                log(f"  {split}: wrote {file_idx}/{len(files)} files, windows={write_idx}/{counts[split]}")

        if write_idx != counts[split]:
            raise RuntimeError(f"{split} wrote {write_idx} windows but counted {counts[split]}")
        windows_out[split].flush()
        np.save(output_dir / f"{split}_labels.npy", np.asarray(labels_out[split], dtype="<U32"))

    return labels_out, meta_out


def sample_split_files(files: list[Path], val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[Path]]:
    grouped: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path in files:
        env_id, subject_id, _, _ = parse_meta(path)
        grouped[(env_id, subject_id)].append(path)

    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n_total = len(keys)
    n_test = round(n_total * test_ratio)
    n_val = round(n_total * val_ratio)
    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test:n_test + n_val])

    split_files = {"train": [], "val": [], "test": []}
    for key, paths in grouped.items():
        if key in test_keys:
            split_files["test"].extend(paths)
        elif key in val_keys:
            split_files["val"].extend(paths)
        else:
            split_files["train"].extend(paths)
    return split_files


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mat_files = sorted(raw_dir.glob("EnvironmentNo.*/mat/*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {raw_dir}")

    train_subjects = parse_id_list(args.train_subjects)
    val_subjects = parse_id_list(args.val_subjects)
    test_subjects = parse_id_list(args.test_subjects)
    train_envs = parse_id_list(args.train_envs)
    val_envs = parse_id_list(args.val_envs)
    test_envs = parse_id_list(args.test_envs)

    split_files: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    if args.split_mode == "sample":
        split_files = sample_split_files(mat_files, args.val_ratio, args.test_ratio, args.seed)
    else:
        for path in mat_files:
            env_id, subject_id, _, _ = parse_meta(path)
            split = choose_split(
                env_id,
                subject_id,
                train_subjects,
                val_subjects,
                test_subjects,
                train_envs,
                val_envs,
                test_envs,
                args.split_mode,
            )
            if split is not None:
                split_files[split].append(path)

    counts, n_features = count_split_windows(split_files, args.window_size, args.stride, args.include_tail)
    if args.output_format == "npy":
        split_labels, split_meta = write_npy_splits(split_files, raw_dir, output_dir, counts, n_features, args)
    else:
        log("[Warning] npz output keeps all windows in RAM. Use --output-format npy for large remote runs.")
        split_windows: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
        split_labels = {"train": [], "val": [], "test": []}
        split_meta = {"train": [], "val": [], "test": []}
        for split, files in split_files.items():
            log(f"[NPZ] Building {split}: {len(files)} files")
            for file_idx, path in enumerate(sorted(files), start=1):
                env_id, subject_id, activity_id, trial_id = parse_meta(path)
                label = LABEL_MAP[activity_id]
                csi = load_csiamp(path)
                if args.normalize == "sample_zscore":
                    csi = zscore(csi)

                starts = window_starts(csi.shape[0], args.window_size, args.stride, args.include_tail)
                for start in starts:
                    window = csi[start:start + args.window_size].astype(np.float32, copy=True)
                    if args.normalize == "window_zscore":
                        window = zscore(window).astype(np.float32)
                    split_windows[split].append(window)
                    split_labels[split].append(label)
                    if args.save_meta:
                        split_meta[split].append(
                            {
                                "source": str(path.relative_to(raw_dir)),
                                "env_id": env_id,
                                "subject_id": subject_id,
                                "activity_id": activity_id,
                                "trial_id": trial_id,
                                "start": start,
                            }
                        )
                if file_idx % 100 == 0 or file_idx == len(files):
                    log(f"  {split}: loaded {file_idx}/{len(files)} files, windows={len(split_windows[split])}")

        for split in ["train", "val", "test"]:
            save_npz(output_dir / f"{split}_windows.npz", split_windows[split], split_labels[split])

    summary = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "split_mode": args.split_mode,
        "window_size": args.window_size,
        "stride": args.stride,
        "include_tail": args.include_tail,
        "normalize": args.normalize,
        "output_format": args.output_format,
        "n_features": n_features,
        "labels": LABEL_MAP,
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        summary["splits"][split] = {
            "source_files": len(split_files[split]),
            "windows": counts[split],
            "label_counts": dict(sorted(Counter(split_labels[split]).items())),
        }
        log(f"{split}: {len(split_files[split])} files, {counts[split]} windows")
        for label, count in sorted(Counter(split_labels[split]).items()):
            log(f"  - {label}: {count}")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_meta:
        (output_dir / "window_meta.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
    log(f"\nSaved ESP-Fi HAR splits to: {output_dir}")


if __name__ == "__main__":
    main()

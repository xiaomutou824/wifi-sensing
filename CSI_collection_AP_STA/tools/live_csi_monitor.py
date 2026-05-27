#!/usr/bin/env python3
"""Live CSI viewer and recorder for ESP32-S3 serial output."""

import argparse
import csv
import datetime as dt
import pathlib
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import serial


HEADER = [
    "type", "node_id", "seq", "local_time_us", "rx_timestamp_us",
    "src_mac", "dst_mac", "first_word_invalid", "rx_seq", "payload_len",
    "rssi", "channel", "secondary_channel", "rate", "sig_mode", "mcs",
    "cwb", "stbc", "sgi", "noise_floor", "ant", "sig_len", "rx_state",
    "csi_len",
]

RSSI_INDEX = HEADER.index("rssi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show ESP32-S3 CSI data live and save it to CSV."
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM3 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--label", default="unlabeled", help="Activity label")
    parser.add_argument("--node", default=None, help="Optional expected node id")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument("--window", type=int, default=120, help="Frames shown in live plots")
    parser.add_argument("--update-every", type=int, default=10, help="Refresh plot every N frames")
    parser.add_argument("--no-save", action="store_true", help="View only, do not save CSV")
    parser.add_argument("--no-plot", action="store_true", help="Save only, do not show live plot")
    return parser.parse_args()


def parse_iq_bytes(raw_values: list[str]) -> np.ndarray:
    try:
        values = np.asarray([float(v) for v in raw_values if v], dtype=np.float32)
    except ValueError:
        return np.array([], dtype=np.float32)
    if values.size < 2:
        return np.array([], dtype=np.float32)
    if values.size % 2 == 1:
        values = values[:-1]
    return np.sqrt(values[0::2] * values[0::2] + values[1::2] * values[1::2])


def make_output_path(args: argparse.Namespace) -> pathlib.Path:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    node_part = f"node{args.node}_" if args.node else ""
    return out_dir / f"{node_part}{args.label}_live_{stamp}.csv"


class LivePlot:
    def __init__(self, window: int):
        self.amp_frames: deque[np.ndarray] = deque(maxlen=window)
        self.mean_amp: deque[float] = deque(maxlen=window)
        self.diff_energy: deque[float] = deque(maxlen=window)
        self.rssi: deque[float] = deque(maxlen=window)
        self.prev_amp: np.ndarray | None = None

        plt.ion()
        self.fig, (self.ax_heatmap, self.ax_curve, self.ax_rssi) = plt.subplots(
            3, 1, figsize=(11, 8), height_ratios=[2.4, 1.2, 1.0]
        )
        self.fig.canvas.manager.set_window_title("ESP32-S3 CSI Live Monitor")

    def append(self, amp: np.ndarray, rssi: float) -> None:
        if self.prev_amp is None:
            diff = 0.0
        else:
            n = min(self.prev_amp.size, amp.size)
            diff = float(np.mean(np.abs(amp[:n] - self.prev_amp[:n])))
        self.prev_amp = amp
        self.amp_frames.append(amp)
        self.mean_amp.append(float(np.mean(amp)))
        self.diff_energy.append(diff)
        self.rssi.append(rssi)

    def draw(self, rows_saved: int) -> None:
        if not self.amp_frames:
            return
        min_len = min(frame.size for frame in self.amp_frames)
        matrix = np.vstack([frame[:min_len] for frame in self.amp_frames])

        self.ax_heatmap.clear()
        self.ax_curve.clear()
        self.ax_rssi.clear()

        self.ax_heatmap.imshow(matrix.T, aspect="auto", origin="lower", interpolation="nearest")
        self.ax_heatmap.set_ylabel("Subcarrier")
        self.ax_heatmap.set_title(f"CSI amplitude heatmap | saved frames: {rows_saved}")

        x = np.arange(len(self.mean_amp))
        self.ax_curve.plot(x, list(self.mean_amp), label="Mean amplitude", color="#1f77b4")
        self.ax_curve.plot(x, list(self.diff_energy), label="Diff energy", color="#d62728")
        self.ax_curve.set_ylabel("Amplitude")
        self.ax_curve.legend(loc="upper right")
        self.ax_curve.grid(True, alpha=0.25)

        self.ax_rssi.plot(x, list(self.rssi), color="#2ca02c")
        self.ax_rssi.set_xlabel("Recent frame index")
        self.ax_rssi.set_ylabel("RSSI")
        self.ax_rssi.grid(True, alpha=0.25)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main() -> int:
    args = parse_args()
    out_path = make_output_path(args)

    csv_file = None
    writer = None
    if not args.no_save:
        csv_file = out_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["pc_time_iso", "label", *HEADER, "csi_raw_bytes"])
        print(f"Saving CSV to {out_path}")

    live_plot = None if args.no_plot else LivePlot(args.window)

    print(f"Opening {args.port} at {args.baud} baud")
    print("Keep another terminal pinging the ESP32-S3 IP to generate CSI packets.")
    print("Press Ctrl+C to stop.")

    rows = 0
    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser:
            while True:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("CSI_DATA,"):
                    continue

                parts = line.split(",")
                if len(parts) < len(HEADER):
                    continue

                if args.node is not None and parts[1] != str(args.node):
                    continue

                meta = parts[: len(HEADER)]
                csi_bytes = parts[len(HEADER):]
                amp = parse_iq_bytes(csi_bytes)
                if amp.size == 0:
                    continue

                rows += 1
                rssi = float(meta[RSSI_INDEX])

                if writer is not None:
                    writer.writerow([
                        dt.datetime.now().isoformat(timespec="microseconds"),
                        args.label,
                        *meta,
                        " ".join(csi_bytes),
                    ])
                    if rows % 100 == 0 and csv_file is not None:
                        csv_file.flush()

                if live_plot is not None:
                    live_plot.append(amp, rssi)
                    if rows % args.update_every == 0:
                        live_plot.draw(rows)

                if rows % 100 == 0:
                    print(f"frames: {rows}, rssi: {rssi:.0f}, mean_amp: {np.mean(amp):.2f}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if csv_file is not None:
            csv_file.close()

    if not args.no_save:
        print(f"Saved {rows} frames to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

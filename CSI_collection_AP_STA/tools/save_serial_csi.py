#!/usr/bin/env python3
"""Save ESP32-S3 CSI CSV rows from a serial port."""

import argparse
import csv
import datetime as dt
import pathlib
import sys

import serial


HEADER = [
    "type", "node_id", "seq", "local_time_us", "rx_timestamp_us",
    "src_mac", "dst_mac", "first_word_invalid", "rx_seq", "payload_len",
    "rssi", "channel", "secondary_channel", "rate", "sig_mode", "mcs",
    "cwb", "stbc", "sgi", "noise_floor", "ant", "sig_len", "rx_state",
    "csi_len",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read CSI_DATA lines from ESP32-S3 serial output and save CSV."
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM3 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--label", default="unlabeled", help="Activity label")
    parser.add_argument("--node", default=None, help="Optional expected node id")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    node_part = f"node{args.node}_" if args.node else ""
    out_path = out_dir / f"{node_part}{args.label}_{stamp}.csv"

    print(f"Opening {args.port} at {args.baud} baud")
    print(f"Saving CSI rows to {out_path}")
    print("Press Ctrl+C to stop.")

    with serial.Serial(args.port, args.baud, timeout=1) as ser, out_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["pc_time_iso", "label", *HEADER, "csi_raw_bytes"])

        rows = 0
        while True:
            try:
                raw = ser.readline()
            except serial.SerialException as exc:
                print(f"Serial error: {exc}", file=sys.stderr)
                return 2

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
            csi_bytes = parts[len(HEADER) :]
            writer.writerow([
                dt.datetime.now().isoformat(timespec="microseconds"),
                args.label,
                *meta,
                " ".join(csi_bytes),
            ])
            rows += 1

            if rows % 100 == 0:
                f.flush()
                print(f"saved {rows} CSI rows")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")

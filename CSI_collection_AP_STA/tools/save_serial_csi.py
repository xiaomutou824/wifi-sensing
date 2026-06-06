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
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--scene", required=True, type=int, help="Scene ID, e.g. 1=bedroom, 2=meetingroom")
    parser.add_argument("--participant", required=True, type=int, help="Participant ID")
    parser.add_argument("--action", required=True, type=int, help="Action class ID, e.g. 1=idle")
    parser.add_argument("--node", required=True, type=int, help="Node ID, 1-3")
    parser.add_argument("--repeat", required=True, type=int, help="Repeat trial ID")
    parser.add_argument(
        "--label",
        default=None,
        help="Optional human-readable activity label written inside the CSV. Defaults to action ID.",
    )
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000,
        help="Stop automatically after saving this many CSI rows. Use 0 to run until Ctrl+C.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.node <= 3:
        print(f"Invalid node ID {args.node}; expected 1, 2, or 3.", file=sys.stderr)
        return 2

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / (
        f"{args.scene}-{args.participant}-{args.action}-{args.node}-{args.repeat}.csv"
    )
    if out_path.exists():
        print(f"Output file already exists: {out_path}", file=sys.stderr)
        return 2

    label = args.label if args.label is not None else str(args.action)

    print(f"Opening {args.port} at {args.baud} baud")
    print(f"Saving CSI rows to {out_path}")
    if args.max_rows > 0:
        print(f"Will stop automatically after {args.max_rows} CSI rows.")
    else:
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

            if parts[1] != str(args.node):
                continue

            meta = parts[: len(HEADER)]
            csi_bytes = parts[len(HEADER) :]
            writer.writerow([
                dt.datetime.now().isoformat(timespec="microseconds"),
                label,
                *meta,
                " ".join(csi_bytes),
            ])
            rows += 1

            if rows % 100 == 0:
                f.flush()
                print(f"saved {rows} CSI rows")

            if args.max_rows > 0 and rows >= args.max_rows:
                f.flush()
                print(f"Reached {args.max_rows} CSI rows. Saved to {out_path}")
                break

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")

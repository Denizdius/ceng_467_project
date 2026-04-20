#!/usr/bin/env python3
"""
Log GPU memory (MB), utilization (%), and power (mW) using NVML (pynvml).

Run while training in another terminal, or in the background:
    nohup python scripts/gpu_metrics_logger.py --output gpu_metrics.csv &

Activate environment:
    source ~/Documents/ee_563/.venv/bin/activate
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pynvml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Poll GPU metrics and append rows to a CSV file.")
    p.add_argument("--output", type=str, default="gpu_metrics.csv", help="CSV path to append.")
    p.add_argument("--gpu-index", type=int, default=0, help="CUDA device index to monitor.")
    p.add_argument("--interval-seconds", type=float, default=1.0, help="Sampling period.")
    p.add_argument(
        "--include-power-watts",
        action="store_true",
        help="Add an extra column power_w (NVML still reports mW internally).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
        if args.gpu_index < 0 or args.gpu_index >= count:
            print(f"[fatal] gpu-index {args.gpu_index} out of range (device count={count}).", file=sys.stderr)
            sys.exit(1)
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        print(f"[fatal] NVML init failed: {e}", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["timestamp_utc", "gpu_index", "gpu_name", "memory_used_mb", "gpu_util_percent", "power_mw"]
    if args.include_power_watts:
        fieldnames.append("power_w")

    write_header = not Path(args.output).exists()
    print(f"[gpu_metrics_logger] device {args.gpu_index}: {name} -> {args.output} every {args.interval_seconds}s")

    try:
        while True:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_mb = mem.used // (1024 * 1024)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            # Power in milliwatts (per NVML API).
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            ts = datetime.now(timezone.utc).isoformat()

            row = {
                "timestamp_utc": ts,
                "gpu_index": args.gpu_index,
                "gpu_name": name,
                "memory_used_mb": mem_mb,
                "gpu_util_percent": util.gpu,
                "power_mw": power_mw,
            }
            if args.include_power_watts:
                row["power_w"] = round(power_mw / 1000.0, 3)

            with open(args.output, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                    write_header = False
                w.writerow(row)

            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("\n[gpu_metrics_logger] stopped.")
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GPU metrics summary, plots, and energy report for a single eval run.
Called automatically at the end of every lm_eval runner script.

Usage:
  python3 scripts/eval_report.py \\
    --csv  outputs/eval_logs/tag/gpu_metrics_20260514_120000.csv \\
    --out-dir  outputs/eval_logs/tag \\
    --run-label "MMLU vLLM QLoRA base" \\
    --wall-seconds 1234
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU metrics report for one eval run.")
    p.add_argument("--csv", type=Path, required=True, help="Path to gpu_metrics CSV written by gpu_metrics_logger.py.")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory to write report files.")
    p.add_argument("--run-label", type=str, default="eval run", help="Human-readable run name for titles.")
    p.add_argument("--wall-seconds", type=float, default=None, help="Total wall-clock seconds for the run.")
    return p.parse_args()


def _read_csv(path: Path) -> tuple[list[datetime], list[float], list[float]]:
    times: list[datetime] = []
    mem_mib: list[float] = []
    power_w: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            times.append(t)
            try:
                mem_mib.append(float(row.get("memory_used_mb", "nan")))
            except ValueError:
                mem_mib.append(float("nan"))
            raw_w = row.get("power_w", "")
            raw_mw = row.get("power_mw", "")
            try:
                if raw_w:
                    power_w.append(float(raw_w))
                elif raw_mw:
                    power_w.append(float(raw_mw) / 1000.0)
                else:
                    power_w.append(float("nan"))
            except ValueError:
                power_w.append(float("nan"))
    return times, mem_mib, power_w


def _elapsed_minutes(times: list[datetime]) -> list[float]:
    if not times:
        return []
    t0 = times[0].astimezone(timezone.utc)
    return [(t.astimezone(timezone.utc) - t0).total_seconds() / 60.0 for t in times]


def _clean(values: list[float]) -> list[float]:
    return [v for v in values if v == v]  # remove NaN


def main() -> None:
    args = _parse_args()

    if not args.csv.is_file():
        print(f"[eval_report] CSV not found: {args.csv} — skipping report.")
        return

    times, mem_mib, power_w = _read_csv(args.csv)
    if not times:
        print("[eval_report] CSV is empty — skipping report.")
        return

    elapsed = _elapsed_minutes(times)
    wall_s = args.wall_seconds if args.wall_seconds is not None else (elapsed[-1] * 60 if elapsed else 0.0)

    mem_c = _clean(mem_mib)
    pwr_c = _clean(power_w)

    peak_vram = max(mem_c) if mem_c else 0.0
    avg_vram  = sum(mem_c) / len(mem_c) if mem_c else 0.0
    peak_pwr  = max(pwr_c) if pwr_c else 0.0
    avg_pwr   = sum(pwr_c) / len(pwr_c) if pwr_c else 0.0
    energy_wh = avg_pwr * wall_s / 3600.0
    energy_j  = avg_pwr * wall_s

    sep = "=" * 56
    print(sep)
    print(f"  GPU report: {args.run_label}")
    print(sep)
    print(f"  Wall time    : {wall_s:.0f} s  ({wall_s/60:.1f} min)")
    print(f"  VRAM peak    : {peak_vram:.0f} MiB")
    print(f"  VRAM avg     : {avg_vram:.0f} MiB")
    print(f"  Power peak   : {peak_pwr:.1f} W")
    print(f"  Power avg    : {avg_pwr:.1f} W")
    print(f"  Total energy : {energy_wh:.4f} Wh  ({energy_j:.1f} J)")
    print(f"  CSV samples  : {len(times)}")
    print(sep)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "run_label": args.run_label,
        "wall_seconds": wall_s,
        "vram_mib_peak": peak_vram,
        "vram_mib_avg": avg_vram,
        "power_w_peak": peak_pwr,
        "power_w_avg": avg_pwr,
        "energy_wh": energy_wh,
        "energy_j": energy_j,
        "csv_samples": len(times),
    }
    stats_path = args.out_dir / "gpu_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[eval_report] Stats -> {stats_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[eval_report] matplotlib not installed — skipping plots (pip install matplotlib).")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    ax1.plot(elapsed, mem_mib, color="tab:blue", linewidth=1.0)
    ax1.axhline(avg_vram, linestyle="--", color="tab:blue", alpha=0.6,
                label=f"avg {avg_vram:.0f} MiB  /  peak {peak_vram:.0f} MiB")
    ax1.set_ylim(0, 12000)
    ax1.set_ylabel("VRAM (MiB)")
    ax1.set_title(f"{args.run_label} — VRAM over time")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(elapsed, power_w, color="tab:orange", linewidth=1.0)
    ax2.axhline(avg_pwr, linestyle="--", color="tab:orange", alpha=0.6,
                label=f"avg {avg_pwr:.1f} W  /  peak {peak_pwr:.1f} W")
    ax2.set_ylabel("Power (W)")
    ax2.set_xlabel("Elapsed time (min)")
    ax2.set_title(f"{args.run_label} — GPU power  [total energy: {energy_wh:.4f} Wh]")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig_path = args.out_dir / "gpu_metrics.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[eval_report] Plot  -> {fig_path}")


if __name__ == "__main__":
    main()

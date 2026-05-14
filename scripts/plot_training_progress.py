#!/usr/bin/env python3
"""
Build comparison plots from DEITA training runs:
  - trainer_state/checkpoint-*/trainer_state.json (latest checkpoint)
  - metrics/gpu_metrics.csv
  - metrics/run_meta.json (wall clock; written by run_train_*.sh)

Example:
  python scripts/plot_training_progress.py \\
    --run-a outputs/baseline2_deita_seq1024 --label-a "Qwen3-4B LoRA" \\
    --run-b outputs/baseline3_deita_seq1024 --label-b "Qwen3-8B QLoRA" \\
    --out-dir reports/deita_seq1024_compare
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass
class RunBundle:
    root: Path
    label: str
    trainer_state: Path
    gpu_csv: Path | None
    meta: dict[str, Any]


def _latest_trainer_state(run_root: Path) -> Path:
    candidates = sorted(
        run_root.glob("trainer_state/checkpoint-*/trainer_state.json"),
        key=_checkpoint_step,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No trainer_state.json under {run_root / 'trainer_state'}")
    return candidates[0]


def _checkpoint_step(p: Path) -> int:
    m = re.search(r"checkpoint-(\d+)", str(p))
    return int(m.group(1)) if m else -1


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_meta(run_root: Path) -> dict[str, Any]:
    meta_path = run_root / "metrics" / "run_meta.json"
    if meta_path.is_file():
        return _load_json(meta_path)
    return {}


def _extract_step_logs(state: dict[str, Any]) -> tuple[list[int], list[float], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    lrs: list[float] = []
    for row in state.get("log_history", []):
        if not isinstance(row, dict):
            continue
        if "step" not in row or "loss" not in row:
            continue
        steps.append(int(row["step"]))
        losses.append(float(row["loss"]))
        lrs.append(float(row.get("learning_rate", 0.0)))
    return steps, losses, lrs


def _step_to_time_seconds(steps: Iterable[int], wall_seconds: float | None) -> list[float] | None:
    if wall_seconds is None or wall_seconds <= 0:
        return None
    s = list(steps)
    if not s:
        return None
    mx = max(s)
    mn = min(s)
    span = max(mx - mn, 1)
    return [wall_seconds * (float(x) - mn) / span for x in s]


def _read_gpu_csv(path: Path) -> tuple[list[datetime], list[float], list[float]]:
    times: list[datetime] = []
    mem_mb: list[float] = []
    power_w: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ts_raw = row.get("timestamp_utc") or ""
            try:
                t = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            times.append(t)
            try:
                mem_mb.append(float(row.get("memory_used_mb", "nan")))
            except ValueError:
                mem_mb.append(float("nan"))
            if "power_w" in row and row["power_w"] not in ("", None):
                try:
                    power_w.append(float(row["power_w"]))
                except ValueError:
                    power_w.append(float("nan"))
            else:
                try:
                    power_w.append(float(row.get("power_mw", "nan")) / 1000.0)
                except ValueError:
                    power_w.append(float("nan"))
    return times, mem_mb, power_w


def _elapsed_minutes(times: list[datetime]) -> list[float]:
    if not times:
        return []
    t0 = times[0].astimezone(timezone.utc)
    return [(t.astimezone(timezone.utc) - t0).total_seconds() / 60.0 for t in times]


def _bundle(run_root: Path, label: str) -> RunBundle:
    run_root = run_root.resolve()
    ts = _latest_trainer_state(run_root)
    gpu = run_root / "metrics" / "gpu_metrics.csv"
    return RunBundle(
        root=run_root,
        label=label,
        trainer_state=ts,
        gpu_csv=gpu if gpu.is_file() else None,
        meta=_load_run_meta(run_root),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Plot training loss/LR and GPU metrics from saved run artifacts.")
    p.add_argument("--run-a", type=Path, required=True)
    p.add_argument("--label-a", type=str, default="Run A")
    p.add_argument("--run-b", type=Path, required=True)
    p.add_argument("--label-b", type=str, default="Run B")
    p.add_argument("--out-dir", type=Path, default=Path("reports/training_compare"))
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("matplotlib is required: pip install matplotlib") from e

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    a = _bundle(args.run_a, args.label_a)
    b = _bundle(args.run_b, args.label_b)

    state_a = _load_json(a.trainer_state)
    state_b = _load_json(b.trainer_state)
    sa, la, lra = _extract_step_logs(state_a)
    sb, lb, lrb = _extract_step_logs(state_b)

    wall_a = float(a.meta["wall_seconds"]) if a.meta.get("wall_seconds") is not None else None
    wall_b = float(b.meta["wall_seconds"]) if b.meta.get("wall_seconds") is not None else None
    ta = _step_to_time_seconds(sa, wall_a)
    tb = _step_to_time_seconds(sb, wall_b)

    # --- Training curves (steps + wall time) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax00.plot(sa, la, label=a.label, linewidth=1.2)
    ax00.plot(sb, lb, label=b.label, linewidth=1.2)
    ax00.set_xlabel("Step")
    ax00.set_ylabel("Loss")
    ax00.set_title("Training loss vs step")
    ax00.legend()
    ax00.grid(True, alpha=0.3)

    if ta is not None and tb is not None:
        ax01.plot([x / 60.0 for x in ta], la, label=a.label, linewidth=1.2)
        ax01.plot([x / 60.0 for x in tb], lb, label=b.label, linewidth=1.2)
        ax01.set_xlabel("Approx. wall time (min)")
        ax01.set_title("Training loss vs time (linear step→time)")
        ax01.set_ylabel("Loss")
        ax01.legend()
        ax01.grid(True, alpha=0.3)
    else:
        ax01.text(0.1, 0.5, "Missing metrics/run_meta.json wall_seconds", transform=ax01.transAxes)
        ax01.set_axis_off()

    ax10.plot(sa, lra, label=a.label, linewidth=1.2)
    ax10.plot(sb, lrb, label=b.label, linewidth=1.2)
    ax10.set_xlabel("Step")
    ax10.set_ylabel("Learning rate")
    ax10.set_title("Learning rate vs step")
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    if ta is not None and tb is not None:
        ax11.plot([x / 60.0 for x in ta], lra, label=a.label, linewidth=1.2)
        ax11.plot([x / 60.0 for x in tb], lrb, label=b.label, linewidth=1.2)
        ax11.set_xlabel("Approx. wall time (min)")
        ax11.set_title("Learning rate vs time (linear step→time)")
        ax11.set_ylabel("Learning rate")
        ax11.legend()
        ax11.grid(True, alpha=0.3)
    else:
        ax11.text(0.1, 0.5, "Missing metrics/run_meta.json wall_seconds", transform=ax11.transAxes)
        ax11.set_axis_off()

    fig.suptitle("DEITA-6k training: loss and learning rate")
    fig_path = out / "training_loss_lr_compare.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    # --- GPU VRAM + power ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    def _gpu_series(bundle: RunBundle) -> tuple[list[datetime], list[float], list[float]]:
        if bundle.gpu_csv is None or not bundle.gpu_csv.is_file():
            return [], [], []
        return _read_gpu_csv(bundle.gpu_csv)

    for col, bundle in enumerate((a, b)):
        times, mem, pwr = _gpu_series(bundle)
        if not times:
            axes2[0, col].text(0.1, 0.5, f"No GPU CSV for\n{bundle.root}", transform=axes2[0, col].transAxes)
            axes2[0, col].set_axis_off()
            axes2[1, col].set_axis_off()
            continue
        em = _elapsed_minutes(times)
        axes2[0, col].plot(em, mem, color="tab:blue", linewidth=1.0)
        axes2[0, col].set_title(f"{bundle.label}: VRAM")
        axes2[0, col].set_xlabel("Elapsed (min)")
        axes2[0, col].set_ylabel("Memory used (MiB)")
        axes2[0, col].grid(True, alpha=0.3)

        axes2[1, col].plot(em, pwr, color="tab:orange", linewidth=1.0)
        axes2[1, col].set_title(f"{bundle.label}: GPU power")
        axes2[1, col].set_xlabel("Elapsed (min)")
        axes2[1, col].set_ylabel("Power (W)")
        axes2[1, col].grid(True, alpha=0.3)

    fig2.suptitle("GPU metrics during training (NVML polling)")
    fig2_path = out / "gpu_vram_power_compare.png"
    fig2.savefig(fig2_path, dpi=160)
    plt.close(fig2)

    # --- Summary JSON ---
    summary = {
        "run_a": {
            "label": a.label,
            "root": str(a.root),
            "trainer_state": str(a.trainer_state),
            "wall_seconds": wall_a,
            "global_step": state_a.get("global_step"),
            "max_steps": state_a.get("max_steps"),
        },
        "run_b": {
            "label": b.label,
            "root": str(b.root),
            "trainer_state": str(b.trainer_state),
            "wall_seconds": wall_b,
            "global_step": state_b.get("global_step"),
            "max_steps": state_b.get("max_steps"),
        },
    }

    def _gpu_stats(bundle: RunBundle) -> dict[str, Any]:
        if not bundle.gpu_csv or not bundle.gpu_csv.is_file():
            return {"gpu_csv": None}
        t, mem, pwr = _read_gpu_csv(bundle.gpu_csv)
        mem_clean = [x for x in mem if x == x]
        pwr_clean = [x for x in pwr if x == x]
        return {
            "gpu_csv": str(bundle.gpu_csv),
            "samples": len(t),
            "vram_mib_max": max(mem_clean) if mem_clean else None,
            "vram_mib_mean": sum(mem_clean) / len(mem_clean) if mem_clean else None,
            "power_w_max": max(pwr_clean) if pwr_clean else None,
            "power_w_mean": sum(pwr_clean) / len(pwr_clean) if pwr_clean else None,
        }

    summary["run_a"]["gpu"] = _gpu_stats(a)
    summary["run_b"]["gpu"] = _gpu_stats(b)
    summary["figures"] = {"training": str(fig_path), "gpu": str(fig2_path)}

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"Run A ({a.label}): wall_seconds={wall_a}",
        f"Run B ({b.label}): wall_seconds={wall_b}",
        f"Figures: {fig_path} , {fig2_path}",
        f"Summary: {out / 'summary.json'}",
    ]
    txt = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()

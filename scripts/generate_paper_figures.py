#!/usr/bin/env python3
"""
Generate all figures for the LNCS paper draft.
Outputs go to project_paper/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "project_paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (colour-blind safe) ────────────────────────────────────
C4B_BASE  = "#4878CF"
C4B_LORA  = "#6ACC65"
C8B_BASE  = "#D65F5F"
C8B_QLORA = "#B47CC7"

HATCHES = ["", "///", "", "///"]
COLORS  = [C4B_BASE, C4B_LORA, C8B_BASE, C8B_QLORA]
LABELS  = ["4B Base", "4B LoRA SFT", "8B Base (4-bit)", "8B QLoRA SFT"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})

# ══════════════════════════════════════════════════════════════════════════
# 1. Training loss curves
# ══════════════════════════════════════════════════════════════════════════
def load_loss(checkpoint_path):
    with open(checkpoint_path) as f:
        d = json.load(f)
    logs = [x for x in d["log_history"] if "loss" in x]
    return [x["step"] for x in logs], [x["loss"] for x in logs]

steps_4b, loss_4b = load_loss(
    ROOT / "outputs/baseline2_deita_seq1024/trainer_state/checkpoint-750/trainer_state.json"
)
steps_8b, loss_8b = load_loss(
    ROOT / "outputs/baseline3_deita_seq1024/trainer_state/checkpoint-750/trainer_state.json"
)

fig, ax = plt.subplots(figsize=(5.5, 2.8))
ax.plot(steps_4b, loss_4b, color=C4B_LORA, linewidth=1.3, label="4B LoRA (final 0.836)")
ax.plot(steps_8b, loss_8b, color=C8B_QLORA, linewidth=1.3, label="8B QLoRA (final 0.796)")
ax.set_xlabel("Training step")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("Training loss — one epoch on DEITA-6k (seq 1024, eff. batch 8)")
ax.legend(loc="upper right")
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.set_xlim(1, 750)
fig.tight_layout()
fig.savefig(FIG_DIR / "training_loss.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "training_loss.png", bbox_inches="tight")
plt.close(fig)
print("Saved training_loss")

# ══════════════════════════════════════════════════════════════════════════
# 2. Training resource comparison  (renamed: Train time, not Wall time)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
labels_tr = ["4B LoRA", "8B QLoRA"]
colors_tr = [C4B_LORA, C8B_QLORA]

axes[0].bar(labels_tr, [55.6, 98.3], color=colors_tr, edgecolor="black", linewidth=0.5)
axes[0].set_ylabel("Train time (min)")
axes[0].set_title("Train time")
for bar, val in zip(axes[0].patches, [55.6, 98.3]):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                 f"{val}", ha="center", va="bottom", fontsize=8)

axes[1].bar(labels_tr, [10080, 9214], color=colors_tr, edgecolor="black", linewidth=0.5)
axes[1].set_ylabel("VRAM peak (MiB)")
axes[1].set_title("Peak VRAM")
axes[1].set_ylim(8000, 11000)
for bar, val in zip(axes[1].patches, [10080, 9214]):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                 f"{val:,}", ha="center", va="bottom", fontsize=8)

axes[2].bar(labels_tr, [114.6, 217.7], color=colors_tr, edgecolor="black", linewidth=0.5)
axes[2].set_ylabel("GPU energy (Wh)")
axes[2].set_title("Total GPU energy")
for bar, val in zip(axes[2].patches, [114.6, 217.7]):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{val}", ha="center", va="bottom", fontsize=8)

for ax_ in axes:
    ax_.grid(axis="y", linewidth=0.4, alpha=0.5)
fig.suptitle("Training resources (RTX 4080 Laptop GPU, 12 GB)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "training_resources.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "training_resources.png", bbox_inches="tight")
plt.close(fig)
print("Saved training_resources")

# ══════════════════════════════════════════════════════════════════════════
# 3. SFT delta chart
# ══════════════════════════════════════════════════════════════════════════
tasks_short = ["MMLU", "ARC-C", "GSM8K*", "TruthfulQA", "WinoGrande", "GPQA"]
delta_4b = [-0.07, -1.88, None, +2.10, +0.24, -1.01]
delta_8b = [+2.24, +0.34, +4.09, +2.31, +0.87, +5.56]

fig, ax = plt.subplots(figsize=(5.5, 3.0))
x = np.arange(len(tasks_short))
w = 0.35
for i, (d4, d8) in enumerate(zip(delta_4b, delta_8b)):
    if d4 is not None:
        ax.bar(x[i] - w/2, d4, w*0.9, color=C4B_LORA, edgecolor="black",
               linewidth=0.4, hatch="///", label="4B LoRA SFT" if i == 0 else "")
    else:
        ax.bar(x[i] - w/2, 0, w*0.9, color="#cccccc", edgecolor="black", linewidth=0.4)
        ax.text(x[i] - w/2, 0.1, "N/A", ha="center", va="bottom", fontsize=6, color="#555")
    ax.bar(x[i] + w/2, d8, w*0.9, color=C8B_QLORA, edgecolor="black",
           linewidth=0.4, label="8B QLoRA SFT" if i == 0 else "")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(tasks_short, fontsize=8)
ax.set_ylabel("Accuracy delta (pp) vs. base model")
ax.set_title("SFT gain/loss vs. base\n(* GSM8K strict-match; 4B N/A = format issue)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "sft_delta.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "sft_delta.png", bbox_inches="tight")
plt.close(fig)
print("Saved sft_delta")

# ══════════════════════════════════════════════════════════════════════════
# 4. Inference efficiency: 2-panel (throughput + energy/token)
# ══════════════════════════════════════════════════════════════════════════
# TokenPowerBench data
duration_s   = [143.2, 314.7, 353.2, 610.1]
output_toks  = [180555, 311573, 188624, 276531]
throughput   = [t/d for t, d in zip(output_toks, duration_s)]  # tok/s
energy_mj    = [92.31, 123.57, 257.16, 291.89]                 # mJ/token

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

x = np.arange(4)
bars1 = ax1.bar(x, throughput, color=COLORS, edgecolor="black", linewidth=0.5,
                hatch=HATCHES)
ax1.set_xticks(x)
ax1.set_xticklabels(["4B\nBase", "4B\nLoRA", "8B\nBase", "8B\nQLoRA"], fontsize=8)
ax1.set_ylabel("Throughput (tok/s)")
ax1.set_title("Decoding throughput")
for bar, val in zip(bars1, throughput):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
             f"{val:.0f}", ha="center", va="bottom", fontsize=7.5)
ax1.grid(axis="y", linewidth=0.4, alpha=0.5)

bars2 = ax2.bar(x, energy_mj, color=COLORS, edgecolor="black", linewidth=0.5,
                hatch=HATCHES)
ax2.set_xticks(x)
ax2.set_xticklabels(["4B\nBase", "4B\nLoRA", "8B\nBase", "8B\nQLoRA"], fontsize=8)
ax2.set_ylabel("GPU energy per token (mJ)")
ax2.set_title("Energy per output token")
for bar, val in zip(bars2, energy_mj):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f"{val:.0f}", ha="center", va="bottom", fontsize=7.5)
ax2.grid(axis="y", linewidth=0.4, alpha=0.5)

handles = [mpatches.Patch(facecolor=c, edgecolor="black", hatch=h, label=l)
           for c, h, l in zip(COLORS, HATCHES, LABELS)]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
fig.suptitle("Inference efficiency — TokenPowerBench (vLLM, batch 128, Alpaca)",
             fontsize=9)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(FIG_DIR / "inference_efficiency.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "inference_efficiency.png", bbox_inches="tight")
plt.close(fig)
print("Saved inference_efficiency")

# ══════════════════════════════════════════════════════════════════════════
# 5. Evaluation resource cost figure (grouped bars by benchmark)
# ══════════════════════════════════════════════════════════════════════════

def load_wall_seconds(run_tag):
    path = ROOT / "outputs/eval_logs" / run_tag / "gpu_stats.json"
    with open(path) as f:
        return float(json.load(f)["wall_seconds"])

# [4B base, 4B LoRA SFT, 8B base, 8B QLoRA SFT]
model_suffixes = ["lora_base", "lora_sft", "qlora_base", "qlora_sft"]
bench_tasks = {
    "MMLU": "mmlu_vllm",
    "ARC-C": "arc_challenge_vllm",
    "GSM8K": "gsm8k_vllm",
    "TruthfulQA": "truthfulqa_mc2_vllm",
    "WinoGrande": "winogrande_vllm",
    "GPQA": "gpqa_diamond_zeroshot_vllm",
}

benchmarks = list(bench_tasks.keys())
eval_times = {}
for bname, prefix in bench_tasks.items():
    eval_times[bname] = [
        load_wall_seconds(f"{prefix}_{suffix}") for suffix in model_suffixes
    ]

models = ["4B Base", "4B LoRA SFT", "8B Base (4-bit)", "8B QLoRA SFT"]
colors = [C4B_BASE, C4B_LORA, C8B_BASE, C8B_QLORA]
hatches = ["", "///", "", "///"]

fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(benchmarks))
w = 0.18
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * w

for mi, (model, color, hatch, offset) in enumerate(zip(models, colors, hatches, offsets)):
    for i, bm in enumerate(benchmarks):
        val = eval_times[bm][mi] / 60.0
        bar = ax.bar(x[i] + offset, val, w * 0.92, color=color,
                     edgecolor="black", linewidth=0.4, hatch=hatch,
                     label=model if i == 0 else "")
        ax.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() + 0.25,
                f"{val:.1f}", ha="center", va="bottom", fontsize=5.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(benchmarks)
ax.set_ylabel("Evaluation time (min)")
ax.set_title("Per-benchmark evaluation time — vLLM backend, RTX 4080 Laptop GPU")
ax.legend(loc="upper left", ncol=2, framealpha=0.9, fontsize=7.5)
ax.set_ylim(0, 32)
ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "eval_resource.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "eval_resource.png", bbox_inches="tight")
plt.close(fig)
print("Saved eval_resource")

print(f"\nAll figures saved to {FIG_DIR}")

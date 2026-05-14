# Supervised Fine-Tuning of LLMs on Consumer GPUs

This repository contains the project report and reproducibility notes for a comparative study of parameter-efficient fine-tuning strategies for large language models on a single consumer GPU.

The project compares:

- **Qwen3-4B + LoRA**: a 16-bit base model fine-tuned with standard LoRA adapters.
- **Qwen3-8B + QLoRA**: a 4-bit quantized base model fine-tuned with LoRA adapters on top of frozen quantized weights.

Both models are instruction-tuned for one epoch on **DEITA-6k** and evaluated on **MMLU** and **ARC Challenge** using `lm-evaluation-harness`.

## Project Goals

- Test whether useful instruction-following and benchmark improvements can be obtained from a small high-quality dataset.
- Compare a smaller high-precision LoRA setup against a larger quantized QLoRA setup under consumer-GPU constraints.
- Measure not only benchmark accuracy, but also practical resource costs such as VRAM use, wall-clock time, and energy consumption.

## Methods

### Training Dataset

- **Dataset**: `hkust-nlp/deita-6k-v0`
- **Size**: 6,000 instruction-response conversations
- **Format**: multi-turn chat conversations using `human` / `gpt` roles
- **Maximum sequence length**: 1,024 tokens
- **Preprocessing**: chat template applied and EOS token appended

### Baselines

| Baseline | Model | Precision | Adaptation | Trainable Parameters |
| --- | --- | --- | --- | --- |
| 1 | `unsloth/Qwen3-4B-Base` | BF16 / 16-bit | LoRA | ~41M |
| 2 | `unsloth/Qwen3-8B-Base-unsloth-bnb-4bit` | 4-bit | QLoRA | ~43.6M |

Shared fine-tuning settings:

| Hyperparameter | Value |
| --- | --- |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Batch size per device | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Epochs | 1 |
| Maximum sequence length | 1,024 |
| Learning rate | `2e-4` |
| Warmup | 10 steps |

### Evaluation

The project evaluates base and fine-tuned checkpoints with `lm-evaluation-harness`:

- **MMLU**: 5-shot generative evaluation across 57 subjects
- **ARC Challenge**: 10-shot science-question evaluation
- **Backends**: planned comparison between `vllm` and HuggingFace `hf`

## Hardware

All training and evaluation were conducted on:

- **GPU**: NVIDIA GeForce RTX 4080 Laptop GPU
- **VRAM**: 12 GB
- **Environment**: WSL2 Ubuntu

The experiments are designed around a realistic consumer-GPU memory budget rather than datacenter hardware.

## Current Results Summary

The report includes:

- Training loss curves for both baselines
- Learning-rate schedules
- GPU VRAM and power measurements
- Initial benchmark comparison on MMLU and ARC Challenge
- Discussion of backend reliability and repeated-evaluation plans

Key observations so far:

- Both LoRA and QLoRA training runs fit within the 12 GB GPU limit.
- The 8B QLoRA run uses less VRAM than the 4B BF16 LoRA run because the frozen backbone is stored in 4-bit form.
- The 8B QLoRA run takes longer and consumes more energy, but provides a larger model capacity under the same hardware constraint.
- Benchmark gains from one epoch on DEITA-6k are expected to be modest, so repeated evaluations are planned to distinguish real effects from run-to-run variance.

## Repository Structure

```text
.
├── README.md
├── report/
│   ├── progress_report.tex
│   └── references.bib
└── .gitignore
```

The full experimental report is written in LaTeX at `report/progress_report.tex`.

## Building the Report

From the repository root, compile the report with:

```bash
cd report
pdflatex progress_report.tex
bibtex progress_report
pdflatex progress_report.tex
pdflatex progress_report.tex
```

The generated PDF will be `report/progress_report.pdf`.

## Planned Work

- Complete the HuggingFace-backend evaluation matrix for all checkpoints.
- Repeat evaluations at least three times per model/task/backend configuration.
- Report mean and standard deviation for each benchmark metric.
- Add evaluation energy and runtime measurements next to accuracy results.
- Finalize dependency documentation for full reproducibility.

## Citation

This project builds on LoRA, QLoRA, DEITA, vLLM, Unsloth, MMLU, and ARC Challenge. Full references are listed in `report/references.bib`.

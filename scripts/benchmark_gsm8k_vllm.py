#!/usr/bin/env python3
"""
GSM8K benchmark using vLLM:
- base model (no LoRA)
- base model + LoRA adapters (via LoRARequest)

Activate environment:
    source ~/Documents/ee_563/.venv/bin/activate

Examples:
  # Base model
  python scripts/benchmark_gsm8k_vllm.py \
    --base_model unsloth/Qwen3-8B-unsloth-bnb-4bit \
    --split test

  # Base + LoRA adapter
  python scripts/benchmark_gsm8k_vllm.py \
    --base_model unsloth/Qwen3-8B-unsloth-bnb-4bit \
    --lora_dir outputs/baseline3/lora_adapters \
    --split test
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


_ANS_RE = re.compile(r"####\s*([^\n\r]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark GSM8K with vLLM (optionally with LoRA adapters).")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--lora_dir", type=str, default=None)
    p.add_argument("--lora_name", type=str, default="adapter")
    p.add_argument("--lora_id", type=int, default=1)
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--max_examples", type=int, default=None, help="If set, limit to first N examples. Default: full split.")
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for vLLM.generate() calls. Default: auto-tune heuristic.",
    )
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--out_jsonl", type=str, default="gsm8k_preds_vllm.jsonl")
    return p.parse_args()


def extract_gsm8k_answer(text: str) -> str | None:
    m = _ANS_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def normalize_answer(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def build_prompt(question: str) -> str:
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"Instruction:\n{question}\n\n"
        "Response: Let's think step by step.\n"
    )


def main() -> None:
    args = parse_args()

    ds = load_dataset("gsm8k", "main", split=args.split)
    if args.max_examples is None:
        n = len(ds)
    else:
        n = min(args.max_examples, len(ds))
        ds = ds.select(range(n))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    llm = LLM(
        model=args.base_model,
        enable_lora=bool(args.lora_dir),
        max_lora_rank=64,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    lora_req = None
    if args.lora_dir:
        lora_req = LoRARequest(args.lora_name, args.lora_id, args.lora_dir)

    out_path = Path(args.out_jsonl)
    correct = 0
    total = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        # Heuristic batch size selection:
        # - Larger batches improve throughput, but increase concurrent KV / activations / scheduling overhead.
        # - Keep conservative defaults on a 12GB consumer GPU and let the user override explicitly.
        if args.batch_size is not None:
            batch_size = max(1, args.batch_size)
        else:
            # Start from a safe-ish default.
            batch_size = 32
            if args.max_model_len > 2048:
                batch_size = 16
            if args.max_tokens > 256:
                batch_size = min(batch_size, 16)
            if args.gpu_memory_utilization <= 0.82:
                batch_size = min(batch_size, 16)
            # If we detect a very small GPU, be extra conservative.
            try:
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if total_gb <= 12.5:
                    batch_size = min(batch_size, 32)
                if total_gb <= 8.5:
                    batch_size = min(batch_size, 16)
            except Exception:
                pass
        print(f"[config] batch_size={batch_size} (set with --batch_size to override)")
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = ds.select(range(start, end))

            prompts = [build_prompt(ex["question"]) for ex in batch]
            gts = [extract_gsm8k_answer(ex["answer"]) for ex in batch]

            outs = llm.generate(prompts, sampling, lora_request=lora_req)

            for j, out in enumerate(outs):
                i = start + j
                completion = out.outputs[0].text if out.outputs else ""
                pred = extract_gsm8k_answer(completion)
                if pred is None:
                    pred = completion.strip().split()[-1] if completion.strip() else None

                gt = gts[j]
                is_correct = False
                if gt is not None and pred is not None:
                    is_correct = normalize_answer(pred) == normalize_answer(gt)

                total += 1
                correct += int(is_correct)

                rec = {
                    "i": i,
                    "question": batch[j]["question"],
                    "gt": gt,
                    "pred": pred,
                    "correct": is_correct,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if total % 20 == 0 or total == end:
                acc = correct / total if total else 0.0
                print(f"[{total}/{n}] acc={acc:.3f} ({correct}/{total})")

    dt = time.time() - t0
    acc = correct / total if total else 0.0
    print(f"[done] split={args.split} n={total} acc={acc:.4f} time_s={dt:.1f} out={out_path.resolve()}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
GSM8K benchmark using Transformers (base) vs Transformers+PEFT (LoRA).

Activate environment:
    source ~/Documents/ee_563/.venv/bin/activate

Examples:
  # Base model only (no LoRA)
  python scripts/benchmark_gsm8k_transformers.py \
    --base_model unsloth/Qwen3-8B-unsloth-bnb-4bit \
    --split test

  # Base + LoRA adapter
  python scripts/benchmark_gsm8k_transformers.py \
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
from transformers import AutoModelForCausalLM, AutoTokenizer


_ANS_RE = re.compile(r"####\s*([^\n\r]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark GSM8K with Transformers (optionally with LoRA adapters).")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--lora_dir", type=str, default=None, help="If provided, load LoRA adapters via PEFT.")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="If set, limit to first N examples. Default: run the full split.",
    )
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy decoding.")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--out_jsonl", type=str, default="gsm8k_preds_transformers.jsonl")
    return p.parse_args()


def extract_gsm8k_answer(text: str) -> str | None:
    """
    GSM8K ground truth answers usually contain a final line like: '#### 42'.
    We compare normalized strings for exact match.
    """
    m = _ANS_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def normalize_answer(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def build_prompt(question: str) -> str:
    # Keep prompt simple and consistent for benchmarking.
    # You can switch to tokenizer.apply_chat_template if you prefer chat prompts, but keep it fixed across runs.
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"Instruction:\n{question}\n\n"
        "Response: Let's think step by step.\n"
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    ds = load_dataset("gsm8k", "main", split=args.split)
    if args.max_examples is None:
        n = len(ds)
    else:
        n = min(args.max_examples, len(ds))
        ds = ds.select(range(n))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if args.lora_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_dir)

    model.eval()

    out_path = Path(args.out_jsonl)
    correct = 0
    total = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            question = ex["question"]
            gt_full = ex["answer"]
            gt = extract_gsm8k_answer(gt_full)

            prompt = build_prompt(question)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-6) if args.temperature > 0 else 1.0,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
            # Try to find a numeric final answer from the model output.
            pred = extract_gsm8k_answer(decoded)
            if pred is None:
                # fallback: last number-ish token (very rough)
                pred = decoded.strip().split()[-1] if decoded.strip() else None

            is_correct = False
            if gt is not None and pred is not None:
                is_correct = normalize_answer(pred) == normalize_answer(gt)

            total += 1
            correct += int(is_correct)

            rec = {
                "i": i,
                "question": question,
                "gt": gt,
                "pred": pred,
                "correct": is_correct,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (i + 1) % 20 == 0 or i == 0:
                acc = correct / total if total else 0.0
                print(f"[{i+1}/{n}] acc={acc:.3f} ({correct}/{total})")

    dt = time.time() - t0
    acc = correct / total if total else 0.0
    print(f"[done] split={args.split} n={total} acc={acc:.4f} time_s={dt:.1f} out={out_path.resolve()}")


if __name__ == "__main__":
    main()


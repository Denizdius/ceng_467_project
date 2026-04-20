#!/usr/bin/env python3
"""
Inference with Transformers + PEFT LoRA adapters.

Activate environment:
    source ~/Documents/ee_563/.venv/bin/activate

Example:
    python scripts/infer_transformers_lora.py \
      --base_model unsloth/Qwen3-8B-unsloth-bnb-4bit \
      --lora_dir outputs/baseline3/lora_adapters \
      --prompt "Solve: 2x + 3 = 11. Show steps."
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a small Transformers generate() with a LoRA adapter.")
    p.add_argument("--base_model", type=str, required=True, help="Base HF model id or local path.")
    p.add_argument("--lora_dir", type=str, required=True, help="Path to LoRA adapter directory (PEFT).")
    p.add_argument("--prompt", type=str, default="Solve: 17 * 19. Show steps.", help="User prompt.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    return p.parse_args()


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    # Prefer chat template when available (Qwen-style tokenizers support this).
    try:
        chat = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_prompt


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.lora_dir)
    model.eval()

    prompt = build_chat_prompt(tokenizer, args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()


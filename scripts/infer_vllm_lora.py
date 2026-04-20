#!/usr/bin/env python3
"""
Offline inference with vLLM using LoRA adapters.

Activate environment:
    source ~/Documents/ee_563/.venv/bin/activate

Example:
    python scripts/infer_vllm_lora.py \
      --base_model unsloth/Qwen3-8B-unsloth-bnb-4bit \
      --lora_dir outputs/baseline3/lora_adapters \
      --prompt "Solve: 2x + 3 = 11. Show steps."

Notes:
- This script loads the *base model* with vLLM and applies the adapter via LoRARequest.
- Keep `gpu_memory_utilization` conservative on a 12GB RTX 4080 Laptop GPU.
"""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM generation with a LoRA adapter directory.")
    p.add_argument("--base_model", type=str, required=True, help="Base HF model id or local path.")
    p.add_argument("--lora_dir", type=str, required=True, help="Path to LoRA adapter directory (PEFT).")
    p.add_argument("--lora_name", type=str, default="adapter", help="Name for the LoRARequest.")
    p.add_argument("--lora_id", type=int, default=1, help="Integer id for the LoRARequest.")
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    p.add_argument("--prompt", type=str, default="Solve: 17 * 19. Show steps.")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    return p.parse_args()


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    try:
        chat = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_prompt


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    prompt = build_chat_prompt(tokenizer, args.prompt)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    lora = LoRARequest(args.lora_name, args.lora_id, args.lora_dir)
    outputs = llm.generate([prompt], sampling, lora_request=lora)

    for o in outputs:
        print(o.outputs[0].text)


if __name__ == "__main__":
    main()


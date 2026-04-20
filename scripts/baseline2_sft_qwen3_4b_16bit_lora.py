#!/usr/bin/env python3
"""
Baseline 2: Qwen3-4B Base, 16-bit LoRA (no 4-bit base load), DEITA-6k SFT, save LoRA adapters (PEFT).

Activate environment (dependencies already installed):
    source ~/Documents/ee_563/.venv/bin/activate

Example:
    python scripts/baseline2_sft_qwen3_4b_16bit_lora.py --output_dir outputs/baseline2_deita
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset

from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "unsloth/Qwen3-4B-Base"


def repair_special_tokens(model, tokenizer) -> None:
    """Fix placeholder eos/pad strings so TRL 0.24+ vocab checks pass."""
    eos_ids = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_ids, (list, tuple)):
        eos_ids = eos_ids[0] if len(eos_ids) else None

    cur = getattr(tokenizer, "eos_token", None)
    cur_tid = tokenizer.convert_tokens_to_ids(cur) if cur else None
    if eos_ids is not None and (cur is None or cur_tid is None):
        tid = int(eos_ids)
        toks = tokenizer.convert_ids_to_tokens(tid)
        if isinstance(toks, list):
            toks = toks[0]
        tokenizer.eos_token = toks
        tokenizer.eos_token_id = tid

    pad = getattr(tokenizer, "pad_token", None)
    pad_tid = tokenizer.convert_tokens_to_ids(pad) if pad else None
    if pad is None or pad_tid is None:
        tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, "pad_token_id", None) is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    if getattr(model, "config", None) is not None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT Qwen3-4B Base (16-bit LoRA) on DEITA-6k, save LoRA adapters.")
    p.add_argument("--output_dir", type=str, default="outputs/baseline2_deita", help="Root directory for checkpoints.")
    p.add_argument("--max_samples", type=int, default=None, help="If set, only use the first N training examples (DEITA is 6k).")
    p.add_argument("--max_steps", type=int, default=None, help="If set, cap training by steps; otherwise use num_train_epochs.")
    p.add_argument("--num_train_epochs", type=float, default=1.0, help="Default 1 epoch over DEITA-6k.")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max_seq_length", type=int, default=None, help="Force this context length (default: try 2048 then 1024 on OOM).")
    p.add_argument("--per_device_train_batch_size", type=int, default=None, help="Override batch size (default: auto-tune up to use VRAM).")
    p.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Override grad accumulation (default: auto-tune).")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    return p.parse_args()


def build_deita_dataset(tokenizer, max_samples: int | None):
    ds = load_dataset("hkust-nlp/deita-6k-v0", split="train")
    if max_samples is not None:
        n = min(max_samples, len(ds))
        ds = ds.select(range(n))

    eos = tokenizer.eos_token

    cols = set(ds.column_names)
    if "text" in cols:
        def ensure_eos(examples):
            texts = examples["text"]
            out = []
            for t in texts:
                if t is None:
                    t = ""
                if eos and not t.endswith(eos):
                    t = t + eos
                out.append(t)
            return {"text": out}

        return ds.map(ensure_eos, batched=True)

    if "conversations" not in cols:
        raise KeyError(
            f"Expected DEITA-6k to have either 'text' or 'conversations', found columns={sorted(cols)}"
        )

    def conversations_to_text(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            messages = []
            if isinstance(convo, list):
                for turn in convo:
                    if not isinstance(turn, dict):
                        continue
                    role = (turn.get("role") or turn.get("from") or "user").lower()
                    content = turn.get("content") or turn.get("value") or ""
                    if role in {"human", "user"}:
                        role = "user"
                    elif role in {"assistant", "gpt", "bot"}:
                        role = "assistant"
                    elif role == "system":
                        role = "system"
                    else:
                        role = "user"
                    messages.append({"role": role, "content": content})

            text = None
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                parts = []
                for m in messages:
                    parts.append(f"{m['role'].capitalize()}: {m['content']}")
                text = "\n".join(parts)

            if eos and text and not text.endswith(eos):
                text = text + eos
            texts.append(text or "")
        return {"text": texts}

    return ds.map(conversations_to_text, batched=True)


def run_pipeline(args: argparse.Namespace, max_seq_length: int) -> None:
    out = Path(args.output_dir)
    lora_dir = out / "lora_adapters"

    print(f"[config] model={MODEL_NAME} load_in_4bit=False max_seq_length={max_seq_length}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
        fast_inference=False,
    )
    repair_special_tokens(model, tokenizer)

    train_dataset = build_deita_dataset(tokenizer, args.max_samples)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    if per_device_train_batch_size is None:
        per_device_train_batch_size = 2
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = 4

    train_kwargs = dict(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=str(out / "trainer_state"),
        logging_steps=1,
        report_to="none",
        # TRL >= 0.24: sequence length and text field live on SFTConfig (not SFTTrainer).
        dataset_text_field="text",
        max_length=max_seq_length,
        eos_token=None,
        pad_token=None,
    )
    if args.max_steps is not None:
        train_kwargs["max_steps"] = args.max_steps
    else:
        train_kwargs["num_train_epochs"] = args.num_train_epochs
        train_kwargs["max_steps"] = -1

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(**train_kwargs),
    )

    trainer.train()

    os.makedirs(lora_dir, exist_ok=True)
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"[save] LoRA adapters (PEFT): {lora_dir.resolve()}")
    print(f"[next] vLLM LoRA inference: python scripts/infer_vllm_lora.py --base_model {MODEL_NAME} --lora_dir {lora_dir.resolve()}")
    print(f"[next] Transformers LoRA inference: python scripts/infer_transformers_lora.py --base_model {MODEL_NAME} --lora_dir {lora_dir.resolve()}")

    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.max_seq_length is not None:
        try:
            run_pipeline(args, args.max_seq_length)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            print("[fatal] OOM at forced max_seq_length; try lowering --max_seq_length or --max_samples.", file=sys.stderr)
            raise
        return

    candidates: list[tuple[int, int, int]] = [
        (2048, 2, 4),
        (2048, 1, 8),
        (1024, 2, 4),
        (1024, 1, 8),
    ]
    if args.per_device_train_batch_size is not None or args.gradient_accumulation_steps is not None:
        candidates = [
            (2048, args.per_device_train_batch_size or 1, args.gradient_accumulation_steps or 8),
            (1024, args.per_device_train_batch_size or 1, args.gradient_accumulation_steps or 8),
        ]

    last_err: Exception | None = None
    for max_seq_length, bs, gas in candidates:
        try:
            args.per_device_train_batch_size = bs
            args.gradient_accumulation_steps = gas
            print(f"[autotune] trying max_seq_length={max_seq_length} batch_size={bs} grad_accum={gas}")
            run_pipeline(args, max_seq_length)
            if max_seq_length == 1024:
                print("[info] Completed run using max_seq_length=1024 (fallback).")
            return
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            torch.cuda.empty_cache()
            gc.collect()
            print(f"[warn] CUDA OOM for seq={max_seq_length} bs={bs} gas={gas} — trying a smaller config.", file=sys.stderr)
            continue
    assert last_err is not None
    raise last_err


if __name__ == "__main__":
    main()

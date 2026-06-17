#!/usr/bin/env python3
"""
Side-by-side qualitative comparison for the four project checkpoints.

Same system instruction for every model; two user questions only.
Uses plain System:/User:/Assistant: prompts (Qwen3-Base has no chat template).
Goal: compare base vs SFT (LoRA / QLoRA) — Q2 is multi-part and should
show clearer structure from DEITA-6k SFT models.

Models compared:
  - Qwen3-8B 4-bit base
  - Qwen3-8B QLoRA SFT (DEITA-6k)
  - Qwen3-4B base
  - Qwen3-4B LoRA SFT (DEITA-6k)

Example:
  python scripts/compare_qualitative_outputs.py
  python scripts/compare_qualitative_outputs.py --max_tokens 384
"""

from __future__ import annotations

import argparse
import gc
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


ROOT = Path(__file__).resolve().parents[1]

SYSTEM_INSTRUCTION = "You are a helpful assistant."

# Q1: short, low-structure — bases and SFT should look similar.
QUESTION_1 = (
    "What are three common causes of climate change? "
    "Answer briefly as three bullet points."
)

# Q2: multi-part with explicit structure — expect LoRA/QLoRA SFT to follow better.
QUESTION_2 = (
    "I want to understand how a CPU cache works. "
    "First give a short analogy for a beginner, then explain L1, L2, and L3 in plain language, "
    "and finally suggest one hands-on way to observe cache effects. "
    "Use clear section headings."
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    base_model: str
    lora_dir: Path | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qualitative vLLM comparison across project checkpoints.")
    p.add_argument(
        "--lora-4b",
        type=Path,
        default=ROOT / "outputs/baseline2_deita_seq1024/lora_adapters",
        help="4B LoRA adapter directory.",
    )
    p.add_argument(
        "--lora-8b",
        type=Path,
        default=ROOT / "outputs/baseline3_deita_seq1024/lora_adapters",
        help="8B QLoRA adapter directory.",
    )
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_lora_rank", type=int, default=64)
    p.add_argument("--max_tokens", type=int, default=800)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/qualitative_compare",
        help="Directory for JSON + text report.",
    )
    p.add_argument("--question-1", type=str, default=QUESTION_1)
    p.add_argument("--question-2", type=str, default=QUESTION_2)
    p.add_argument("--system-instruction", type=str, default=SYSTEM_INSTRUCTION)
    return p.parse_args()


def model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    lora_4b = args.lora_4b.resolve()
    lora_8b = args.lora_8b.resolve()
    for label, path in (("4B LoRA SFT", lora_4b), ("8B QLoRA SFT", lora_8b)):
        if not path.exists():
            raise FileNotFoundError(f"{label} adapter not found: {path}")

    return [
        ModelSpec(
            label="8B 4-bit base",
            base_model="unsloth/Qwen3-8B-Base-unsloth-bnb-4bit",
            lora_dir=None,
        ),
        ModelSpec(
            label="8B QLoRA SFT",
            base_model="unsloth/Qwen3-8B-Base-unsloth-bnb-4bit",
            lora_dir=lora_8b,
        ),
        ModelSpec(
            label="4B base",
            base_model="unsloth/Qwen3-4B-Base",
            lora_dir=None,
        ),
        ModelSpec(
            label="4B LoRA SFT",
            base_model="unsloth/Qwen3-4B-Base",
            lora_dir=lora_4b,
        ),
    ]


def build_prompt(system_instruction: str, user_question: str) -> str:
    """Plain role-tagged prompt for Qwen3-Base (no chat template).

    Matches the DEITA SFT fallback in baseline2/baseline3 when
    tokenizer.chat_template is unset.
    """
    lines = [
        f"System: {system_instruction}",
        f"User: {user_question}",
        "Assistant:",
    ]
    return "\n".join(lines)


def render_text_report(payload: dict) -> str:
    lines: list[str] = []
    lines.append("Qualitative model comparison")
    lines.append("=" * 72)
    lines.append(f"Generated: {payload['meta']['timestamp_utc']}")
    lines.append("")
    lines.append("System instruction (all models):")
    lines.append(textwrap.fill(payload["meta"]["system_instruction"], width=72))
    lines.append("")

    for qid in ("q1", "q2"):
        qtext = payload["meta"]["questions"][qid]
        lines.append(f"{'#' * 72}")
        lines.append(f"QUESTION {qid.upper()}")
        lines.append(f"{'#' * 72}")
        lines.append(textwrap.fill(qtext, width=72))
        lines.append("")

        block = [r for r in payload["results"] if r["question_id"] == qid]
        for row in block:
            lines.append(f"[{row['model']}]")
            lines.append("-" * 40)
            lines.append(row["output"].strip())
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_model(
    spec: ModelSpec,
    questions: list[tuple[str, str]],
    args: argparse.Namespace,
) -> list[dict]:
    """Load one checkpoint, generate both questions, unload."""
    print(f"\n=== Loading: {spec.label} ===")
    llm_kwargs = dict(
        model=spec.base_model,
        enable_lora=spec.lora_dir is not None,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    if spec.lora_dir is not None:
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
    llm = LLM(**llm_kwargs)

    # Stop the model from looping back into a new User/System turn.
    # Critical for base models which have no EOS-at-turn-end behaviour.
    stop_sequences = ["\nUser:", "\nSystem:", "<|endoftext|>", "<|im_end|>"]
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=stop_sequences,
    )

    lora_req = None
    if spec.lora_dir is not None:
        lora_req = LoRARequest("adapter", 1, str(spec.lora_dir))

    rows: list[dict] = []
    for question_id, question in questions:
        prompt = build_prompt(args.system_instruction, question)
        outputs = llm.generate([prompt], sampling, lora_request=lora_req)
        text = outputs[0].outputs[0].text
        rows.append(
            {
                "model": spec.label,
                "base_model": spec.base_model,
                "lora_dir": str(spec.lora_dir) if spec.lora_dir else None,
                "system_instruction": args.system_instruction,
                "question_id": question_id,
                "question": question,
                "output": text,
            }
        )
        print(f"  done: {spec.label} | {question_id}")

    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return rows


def main() -> None:
    args = parse_args()
    specs = model_specs(args)

    questions = [
        ("q1", args.question_1),
        ("q2", args.question_2),
    ]

    all_rows: list[dict] = []
    for spec in specs:
        all_rows.extend(run_model(spec, questions, args))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"comparison_{ts}.json"
    txt_path = args.out_dir / f"comparison_{ts}.txt"

    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "system_instruction": args.system_instruction,
            "generation": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_model_len": args.max_model_len,
            },
            "questions": {
                "q1": args.question_1,
                "q2": args.question_2,
            },
            "prompt_format": "System:/User:/Assistant: (Qwen3-Base has no chat template)",
            "notes": (
                "Same system instruction for all models. Q2 is structured/multi-part; "
                "compare whether LoRA and QLoRA SFT answers are more organized than bases."
            ),
        },
        "results": all_rows,
    }

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    txt_path.write_text(render_text_report(payload), encoding="utf-8")

    print(f"\nWrote JSON: {json_path}")
    print(f"Wrote text: {txt_path}")


if __name__ == "__main__":
    main()

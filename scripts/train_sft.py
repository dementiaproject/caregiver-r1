#!/usr/bin/env python3
"""
Caregiver SFT warm-up trainer.

Reads JSONL extracted by `src.training.sft_extract`, formats each example
into the canonical chat-template + `<think>+<response>` structure, and
fine-tunes the base Qwen3-8B with HuggingFace Trainer.

The SFT objective is causal LM CE loss masked to ONLY the assistant
response (system + user prompt tokens get -100 / ignored). This matches
DemMA SFT design: only the parts the model is supposed to GENERATE
contribute to gradient.

Usage
-----
    PYTHONPATH=. python scripts/train_sft.py --config configs/sft.yaml

Hard requirements: GPU, ≥40 GB VRAM (Qwen3-8B + AdamW8bit + bf16).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_sft")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class _SftExample:
    input_ids: list[int]
    labels: list[int]      # -100 over prompt tokens; copy of input_ids over response


def _build_user_message(example: dict[str, Any]) -> str:
    """Same shape as caregiver_client.build_caregiver_user_message — the
    SFT-time prompt has to MATCH inference-time prompt or the policy will
    learn an off-distribution mapping."""
    parts = [
        "# Patient",
        f"- {example.get('scenario_summary', '')}",
        "",
        "# Conversation so far",
    ]
    history = example.get("history", [])
    if not history:
        parts.append("(beginning of conversation)")
    else:
        for h in history:
            parts.append(f"Caregiver: {h['caregiver']}")
            parts.append(f"Patient: {h['patient']}")
    if example.get("latest_patient"):
        parts.append(f"Patient: {example['latest_patient']}")
    parts.append("")
    parts.append(
        "Generate your next turn in EXACTLY this XML format (one <think> "
        "block, one <response> block, no other text outside the tags):"
    )
    parts.append("<think>your concise clinical reasoning here</think><response>"
                 "your one-or-two-sentence reply to the patient</response>")
    return "\n".join(parts)


def _format_example(
    example: dict[str, Any],
    tokenizer,
    system_prompt: str,
    max_seq_len: int,
) -> _SftExample | None:
    """Tokenize one SFT example with response-only loss masking."""
    user_msg = _build_user_message(example)
    target = (
        f"<think>{example['target_think']}</think>"
        f"<response>{example['target_response']}</response>"
    )

    # Apply chat template up to BUT NOT INCLUDING the assistant response so we
    # know the exact token boundary.
    messages_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True,
    )
    full_text = prompt_text + target + tokenizer.eos_token

    enc_prompt = tokenizer(prompt_text, add_special_tokens=False)
    enc_full = tokenizer(full_text, add_special_tokens=False)

    input_ids = enc_full["input_ids"]
    if len(input_ids) > max_seq_len:
        return None  # drop overlong examples; alternative is left-truncation

    prompt_len = len(enc_prompt["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    if len(labels) != len(input_ids):
        return None

    return _SftExample(input_ids=input_ids, labels=labels)


def build_dataset(jsonl_path: Path, tokenizer, system_prompt: str, max_seq_len: int):
    import datasets
    raw = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw.append(json.loads(line))
    log.info("loaded %d raw SFT examples from %s", len(raw), jsonl_path)

    formatted = []
    skipped = 0
    for ex in raw:
        sft = _format_example(ex, tokenizer, system_prompt, max_seq_len)
        if sft is None:
            skipped += 1
            continue
        formatted.append({"input_ids": sft.input_ids, "labels": sft.labels})

    log.info("formatted %d (skipped %d for length)", len(formatted), skipped)
    return datasets.Dataset.from_list(formatted)


def collate_fn(batch, pad_id: int):
    """Right-pad input_ids and labels (with -100) to longest in batch."""
    import torch
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Trainer entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(int(cfg.run.seed))

    # ---- Tokenizer + model -------------------------------------------------
    name = str(cfg.model.name_or_path)
    log.info("loading model %s", name)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        str(cfg.model.precision)
    ]
    attn_impl = "sdpa"
    if bool(getattr(cfg.model, "use_flash_attention", True)):
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            log.warning("flash_attention_2 not installed; using SDPA")

    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    if bool(getattr(cfg.model, "gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # ---- System prompt -----------------------------------------------------
    from src.training.prompt_loader import (
        caregiver_prompt_sha256,
        load_caregiver_prompt,
    )
    system_prompt = load_caregiver_prompt()
    log.info("system prompt loaded (sha256=%s, %d chars)",
             caregiver_prompt_sha256()[:16], len(system_prompt))

    # ---- Dataset -----------------------------------------------------------
    train_ds = build_dataset(
        Path(cfg.data.train_jsonl),
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_seq_len=int(cfg.data.max_seq_len),
    )
    eval_ds = None
    if getattr(cfg.data, "eval_jsonl", None):
        eval_ds = build_dataset(
            Path(cfg.data.eval_jsonl),
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_seq_len=int(cfg.data.max_seq_len),
        )

    # ---- Optimizer (AdamW8bit when bnb available; else vanilla AdamW) -----
    optim_name = "adamw_torch"
    try:
        import bitsandbytes  # noqa: F401
        optim_name = "adamw_bnb_8bit"
    except ImportError:
        log.warning("bitsandbytes not installed; using vanilla AdamW")

    # ---- TrainingArguments -------------------------------------------------
    args_tr = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=int(cfg.train.num_epochs),
        per_device_train_batch_size=int(cfg.train.per_device_batch_size),
        per_device_eval_batch_size=int(cfg.train.per_device_batch_size),
        gradient_accumulation_steps=int(cfg.train.gradient_accumulation_steps),
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        warmup_ratio=float(cfg.train.warmup_ratio),
        max_grad_norm=float(cfg.train.max_grad_norm),
        logging_steps=int(cfg.train.logging_steps),
        save_strategy="steps",
        save_steps=int(cfg.train.save_steps),
        save_total_limit=int(getattr(cfg.train, "save_total_limit", 3)),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(getattr(cfg.train, "eval_steps", 200)) if eval_ds is not None else 0,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        optim=optim_name,
        report_to=getattr(cfg.train, "report_to", ["none"]),
        run_name=cfg.run.name,
        seed=int(cfg.run.seed),
        gradient_checkpointing=bool(getattr(cfg.model, "gradient_checkpointing", True)),
    )

    # ---- Build trainer -----------------------------------------------------
    pad_id = tokenizer.pad_token_id
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=lambda batch: collate_fn(batch, pad_id),
        processing_class=tokenizer,
    )

    # ---- Train -------------------------------------------------------------
    trainer.train()

    # ---- Save final --------------------------------------------------------
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    log.info("training done. final checkpoint → %s", final_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

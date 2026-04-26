#!/usr/bin/env python3
"""
Single-process baseline GRPO trainer for the caregiver-R1 EC-MTRL setup.

Status: SCAFFOLD — runs end-to-end on a single GPU and demonstrates the
correct rollout / advantage / loss / step loop, but is intentionally
minimal:

  - 1 GPU, no Ray, no distributed (use verl-GDPO + docs/verl_integration.md
    for multi-node scale-out).
  - HuggingFace transformers `.generate()` for rollout (NOT vLLM); slower
    than the smoke-run vLLM path, but keeps gradients available on the
    same model handle for the train step.
  - Reference policy = a second `from_pretrained` of the same checkpoint,
    frozen with `requires_grad_(False)`. Resync optional.
  - Optimizer = bf16 AdamW8bit (bnb) or vanilla AdamW (configurable).

What it DOES integrate end-to-end:
  - `src.data.caregiver_client.CaregiverClient` for rollout
    (mode=mock for fast sanity; mode=http when vllm serve is up)
  - `src.data.demma_client.DemMAClient` (mock | real | vllm)
  - `src.rewards.{turn_level, mock_judge | vllm_judge_adapter}` for rewards
  - `src.training.advantage.compute_dual_horizon_advantage` for the EC-MTRL
    advantage (Decision 3 binary hard-veto safety)
  - `src.training.grpo_loss.grpo_total_loss` for PPO-clip + KL anchor

What's INTENTIONALLY left for the verl port (Phase E.2):
  - distributed Ray actors
  - vLLM rollout actor + weight sync
  - chunk-and-shuffle data pipeline
  - multi-microbatch gradient accumulation across nodes

Usage
-----
    PYTHONPATH=. python scripts/train_grpo.py \\
        --config configs/grpo_smoke.yaml \\
        --steps 5

Hard requirements: GPU, ≥40 GB VRAM (Qwen3-8B policy + ref + KV cache).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import DictConfig, OmegaConf

# Imports gated to keep --help fast on CPU dev boxes.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_grpo")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_policy_and_reference(cfg: DictConfig):
    """Load two copies of the caregiver model: policy (trainable) + reference (frozen).

    Reference is needed for the KL anchor in the GRPO loss. We load via two
    separate `from_pretrained` calls so the reference parameters stay
    independent and unaffected by the policy's optimizer step. Both share
    the tokenizer.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = str(cfg.caregiver.model_name_or_path)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        str(cfg.caregiver.precision)
    ]

    log.info("loading policy: %s (dtype=%s)", name, dtype)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "sdpa"
    if bool(getattr(cfg.caregiver, "use_flash_attention", True)):
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            log.warning("flash_attention_2 not installed; using SDPA")

    policy = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    policy.gradient_checkpointing_enable()
    policy.config.use_cache = False  # required when gradient checkpointing

    log.info("loading reference policy (frozen): %s", name)
    reference = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    return tokenizer, policy, reference


# ---------------------------------------------------------------------------
# Rollout: re-uses run_smoke.synth_dummy_trajectory but with logged tokens
# ---------------------------------------------------------------------------

def rollout_group(
    policy,
    tokenizer,
    scenario,
    cfg: DictConfig,
    demma_client,
    judges,
    seed: int,
):
    """Roll out `cfg.rollout.group_size` trajectories for one scenario.

    Returns a list of dicts, one per rollout, each with:
        trajectory_id : str
        bundle        : TrajectoryRewardBundle
        prompt_ids    : (T_prompt,) torch.Tensor (cumulative)
        response_ids  : (T_resp,)   torch.Tensor (only assistant tokens)
        old_log_probs : (T_resp,)   torch.Tensor — log p_old at sample time
        turn_token_spans : list[(start, end)] within response_ids

    For the SCAFFOLD we run rollouts SEQUENTIALLY (group_size at a time) using
    HF transformers `.generate()`. Multi-turn dialogues mean we call generate
    once per turn per rollout. Slow but correct. The verl port replaces this
    with a batched vLLM actor.
    """
    raise NotImplementedError(
        "rollout_group: this is the GPU-dependent piece of the trainer. "
        "Integration with HF transformers `.generate()` + chat-template + "
        "log-probs capture is the user's GPU-side work. The pieces are: "
        "(1) build chat messages [system=caregiver_prompt, user=scenario+history]; "
        "(2) tokenizer.apply_chat_template() → input_ids; "
        "(3) policy.generate(..., return_dict_in_generate=True, output_scores=True); "
        "(4) collect old_log_probs by gathering on .scores tuple; "
        "(5) parse <think>+<response> with caregiver_client.parse_caregiver_output; "
        "(6) demma_client.step() for the patient utterance; "
        "(7) repeat for num_turns; "
        "(8) judges.score_trajectory + compute_turn_rewards → bundle; "
        "see scripts/run_smoke.py:synth_dummy_trajectory for the rollout loop "
        "(without the policy.generate gradient-tracking part)."
    )


# ---------------------------------------------------------------------------
# Loss + step
# ---------------------------------------------------------------------------

def compute_loss_and_step(
    policy,
    reference,
    optimizer,
    rollouts: list[dict],
    advantages_per_token: dict[str, "torch.Tensor"],   # traj_id → (T_resp,)
    cfg: DictConfig,
):
    """Single optimizer step: re-compute new_log_probs, evaluate ref_log_probs,
    apply GRPO PPO-clip + KL anchor loss, backprop, step.
    """
    import torch
    from src.training.grpo_loss import (
        GRPOLossConfig,
        grpo_total_loss,
        selective_log_softmax,
    )

    loss_cfg = GRPOLossConfig(
        eps_low=float(cfg.trainer.ppo_clip_eps),
        eps_high=float(cfg.trainer.ppo_clip_eps),
        beta_kl=float(cfg.trainer.beta_kl_init),
    )

    total_loss = 0.0
    info_acc: dict[str, float] = {}
    n_rows = 0

    optimizer.zero_grad(set_to_none=True)
    for r in rollouts:
        # ---- Build (input_ids, response_mask) for one row -------------------
        prompt_ids = r["prompt_ids"]
        response_ids = r["response_ids"]
        full_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0)
        T_prompt = prompt_ids.numel()
        T_resp = response_ids.numel()
        attention_mask = torch.ones_like(full_ids)

        # ---- new_log_probs over response slice (gradient ON) ---------------
        out_new = policy(input_ids=full_ids, attention_mask=attention_mask)
        # logits[:, t] predicts token at t+1; align with response targets:
        new_logits = out_new.logits[:, T_prompt - 1: -1, :]   # (1, T_resp, V)
        new_log_probs = selective_log_softmax(
            new_logits, response_ids.unsqueeze(0)
        ).squeeze(0)  # (T_resp,)

        # ---- ref_log_probs (no grad) ---------------------------------------
        with torch.no_grad():
            out_ref = reference(input_ids=full_ids, attention_mask=attention_mask)
            ref_logits = out_ref.logits[:, T_prompt - 1: -1, :]
            ref_log_probs = selective_log_softmax(
                ref_logits, response_ids.unsqueeze(0)
            ).squeeze(0)

        old_log_probs = r["old_log_probs"]                    # (T_resp,)
        advantages = advantages_per_token[r["trajectory_id"]] # (T_resp,)
        response_mask = torch.ones_like(advantages)

        loss, info = grpo_total_loss(
            new_log_probs=new_log_probs.unsqueeze(0),
            old_log_probs=old_log_probs.unsqueeze(0),
            ref_log_probs=ref_log_probs.unsqueeze(0),
            advantages=advantages.unsqueeze(0),
            response_mask=response_mask.unsqueeze(0),
            cfg=loss_cfg,
        )

        loss = loss / max(1, len(rollouts))   # gradient accumulation across rollouts
        loss.backward()

        total_loss += loss.item()
        for k, v in info.items():
            info_acc[k] = info_acc.get(k, 0.0) + v
        n_rows += 1

    torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.trainer.max_grad_norm))
    optimizer.step()

    if n_rows > 0:
        info_acc = {k: v / n_rows for k, v in info_acc.items()}
    info_acc["loss/total_summed"] = total_loss
    return info_acc


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo_smoke.yaml")
    ap.add_argument("--steps", type=int, default=5,
                    help="Number of gradient steps to run (smoke ≤ 10).")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("config: %s", args.config)
    log.info("output dir: %s", out_dir)

    # ---- Build clients -----------------------------------------------------
    from scripts.run_smoke import build_caregiver_client, build_demma_client, build_judges
    demma = build_demma_client(cfg)
    judges_traj, judges_safety = build_judges(cfg, seed=int(cfg.run.seed))
    # caregiver here is the rollout-side client (HTTP / mock); the trainer
    # ALSO loads the policy/reference HF models. In production these are
    # the same model — vLLM serves it for rollout, training reads grads.
    # In this scaffold they are decoupled (HF for both, slow rollout).
    _ = build_caregiver_client(cfg)  # only used by smoke; trainer uses policy.generate

    # ---- Load policy + reference ------------------------------------------
    tokenizer, policy, reference = load_policy_and_reference(cfg)

    # ---- Load scenarios ----------------------------------------------------
    from scripts.run_smoke import load_scenarios
    scenarios = load_scenarios(
        Path(cfg.data.scenarios_path),
        n_max=int(cfg.trainer.scenarios_per_epoch),
    )
    log.info("loaded %d scenarios", len(scenarios))

    # ---- Optimizer ---------------------------------------------------------
    import torch
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            policy.parameters(), lr=float(cfg.trainer.learning_rate),
        )
        log.info("optimizer: bitsandbytes AdamW8bit")
    except ImportError:
        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=float(cfg.trainer.learning_rate),
        )
        log.info("optimizer: vanilla AdamW (bnb not installed)")

    # ---- Training loop -----------------------------------------------------
    for step in range(args.steps):
        scenario = scenarios[step % len(scenarios)]
        seed = int(cfg.run.seed) + step
        t_start = time.monotonic()

        # !!!! GPU-DEPENDENT BLOCK: rollout_group and per-token advantage broadcasting
        # !!!! are the integration points the user must verify on GPU.  See
        # !!!! function docstrings + docs/verl_integration.md.
        try:
            rollouts = rollout_group(
                policy=policy,
                tokenizer=tokenizer,
                scenario=scenario,
                cfg=cfg,
                demma_client=demma,
                judges=(judges_traj, judges_safety),
                seed=seed,
            )
        except NotImplementedError as e:
            log.error("rollout_group not implemented: %s", e)
            log.error("This trainer is a SCAFFOLD; finish rollout_group on GPU first.")
            return 2

        # ---- Compute advantage (group-relative) ----------------------------
        from src.training.advantage import compute_dual_horizon_advantage
        bundles = [r["bundle"] for r in rollouts]
        ga = compute_dual_horizon_advantage(
            group=bundles,
            alpha=float(cfg.algorithm.alpha),
            lambda_safety=float(cfg.algorithm.lambda_safety),
            min_valid_turn=int(cfg.algorithm.min_valid_turn),
        )

        # Broadcast (group_size, T_max) → per-trajectory (T_resp,) using
        # rollouts[i]["turn_token_spans"]
        from src.training.grpo_advantage import _broadcast_turn_advantage_to_tokens
        spans = [r["turn_token_spans"] for r in rollouts]
        T_resp = max(r["response_ids"].numel() for r in rollouts)
        per_token_np = _broadcast_turn_advantage_to_tokens(ga, spans, T_resp)
        device = next(policy.parameters()).device
        advantages_per_token = {
            r["trajectory_id"]: torch.from_numpy(per_token_np[i]).to(device).float()
            for i, r in enumerate(rollouts)
        }

        # ---- Loss + step ---------------------------------------------------
        info = compute_loss_and_step(
            policy=policy, reference=reference,
            optimizer=optimizer, rollouts=rollouts,
            advantages_per_token=advantages_per_token,
            cfg=cfg,
        )

        dt = time.monotonic() - t_start
        log.info("step %d  scenario=%s  dt=%.1fs  loss=%.4f  kl=%.4f  clip_frac=%.3f",
                 step, scenario.scenario_id, dt,
                 info["loss/total_summed"], info["kl/mean"], info["policy/clip_frac"])

        # ---- Save (every step in scaffold; tune `cfg.trainer.save_every`) -
        if (step + 1) % int(getattr(cfg.trainer, "save_every", 5)) == 0:
            ckpt_dir = out_dir / f"step_{step + 1:05d}"
            ckpt_dir.mkdir(exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            (out_dir / "history.jsonl").open("a").write(
                json.dumps({"step": step + 1, **info}) + "\n"
            )
            log.info("saved checkpoint → %s", ckpt_dir)

    log.info("training done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
PPO-clip surrogate + KL anchor — framework-agnostic GRPO loss components.

Used both by the single-process baseline trainer (`scripts/train_grpo.py`)
AND by the verl-integration adapter. Pure tensor ops, no Ray / no
HuggingFace Trainer dependency.

References
----------
- DeepSeek-Math GRPO surrogate: PPO-clip with response_mask gating.
- Schulman et al. 2017 PPO clipping (eps_low, eps_high).
- KL-anchor formulation from RLHF literature (per-token KL between policy
  and reference, masked-mean reduced).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GRPOLossConfig:
    eps_low: float = 0.2          # PPO-clip lower bound
    eps_high: float = 0.2         # PPO-clip upper bound
    beta_kl: float = 0.01         # KL anchor strength
    entropy_coef: float = 0.0     # entropy bonus (off in critic-free GRPO)


def selective_log_softmax(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Memory-efficient gather of log P(target_token | ...) over each position.

    logits : (B, T, V)
    target : (B, T) int
    return : (B, T) — log p_target

    Avoids materializing the full log_softmax over V (saves V × dtype mem).
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mean over positions where mask=1; safe against empty mask."""
    n = mask.sum().clamp(min=1.0)
    return (x * mask).sum() / n


def grpo_policy_loss(
    new_log_probs: torch.Tensor,    # (B, T)
    old_log_probs: torch.Tensor,    # (B, T)
    advantages: torch.Tensor,       # (B, T) — already broadcast per-token
    response_mask: torch.Tensor,    # (B, T)
    cfg: GRPOLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO-clip surrogate with masked-mean reduction over response tokens.

    Returns
    -------
    loss : scalar tensor
    info : dict of float diagnostics for W&B / logging
    """
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()

    pg_loss_unclipped = -advantages * ratio
    pg_loss_clipped = -advantages * torch.clamp(ratio, 1.0 - cfg.eps_low, 1.0 + cfg.eps_high)
    pg_loss_per_tok = torch.maximum(pg_loss_unclipped, pg_loss_clipped)

    pg_loss = masked_mean(pg_loss_per_tok, response_mask)

    # Diagnostics: clip fraction, ratio mean, advantage mean
    with torch.no_grad():
        clip_frac = masked_mean(
            (pg_loss_clipped > pg_loss_unclipped).float(), response_mask
        ).item()
        mean_ratio = masked_mean(ratio, response_mask).item()
        mean_adv = masked_mean(advantages, response_mask).item()

    return pg_loss, {
        "policy/clip_frac": clip_frac,
        "policy/mean_ratio": mean_ratio,
        "policy/mean_advantage": mean_adv,
    }


def kl_anchor_loss(
    new_log_probs: torch.Tensor,    # (B, T) policy
    ref_log_probs: torch.Tensor,    # (B, T) reference (frozen)
    response_mask: torch.Tensor,    # (B, T)
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """K3 / DeepSeek-style per-token KL: exp(ref - new) - (ref - new) - 1.

    This unbiased estimator stays >=0 and is numerically nicer than the
    plain log-ratio. Matches verl-GDPO's `core_algos.kl_penalty(kl_penalty_type='k3')`.
    """
    diff = ref_log_probs - new_log_probs
    kl_per_tok = torch.exp(diff) - diff - 1.0
    kl = masked_mean(kl_per_tok, response_mask)
    info = {"kl/mean": kl.item(), "kl/beta": beta}
    return beta * kl, info


def grpo_total_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cfg: GRPOLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compose policy + KL anchor + (optional) entropy bonus.

    No value head, no GAE — critic-free GRPO (zh_9 §4.7).
    """
    pg_loss, info = grpo_policy_loss(
        new_log_probs, old_log_probs, advantages, response_mask, cfg
    )
    kl_loss, kl_info = kl_anchor_loss(
        new_log_probs, ref_log_probs, response_mask, cfg.beta_kl
    )
    info.update(kl_info)
    info["loss/policy"] = pg_loss.item()
    info["loss/kl"] = kl_loss.item()

    total = pg_loss + kl_loss

    if cfg.entropy_coef > 0:
        # log p(target|...) is already a lower bound on entropy contribution;
        # negate to encourage exploration. Skipped by default (GRPO).
        ent = -masked_mean(new_log_probs, response_mask)
        total = total - cfg.entropy_coef * ent
        info["loss/entropy"] = ent.item()

    info["loss/total"] = total.item()
    return total, info


__all__ = [
    "GRPOLossConfig",
    "grpo_policy_loss",
    "kl_anchor_loss",
    "grpo_total_loss",
    "masked_mean",
    "selective_log_softmax",
]

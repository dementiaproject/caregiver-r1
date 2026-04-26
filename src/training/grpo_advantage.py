"""
verl-style advantage adapter that bridges `compute_dual_horizon_advantage`
(our framework-agnostic NumPy estimator) to the `data.batch` tensor
contract used inside verl-GDPO's `ray_trainer.compute_advantage()`.

Why this exists
---------------
verl's `core_algos.compute_grpo_outcome_advantage` consumes
`token_level_rewards: (B, T_resp)` and an `eos_mask: (B, T_resp)`, plus a
group `index` (one int per row identifying the rollout group). It returns
per-token `advantages` and `returns` that the policy gradient applies
directly.

Our EC-MTRL advantage operates over a richer per-trajectory data
structure (`TrajectoryRewardBundle` with traj scores + per-turn rewards
+ safety) and produces ONE scalar per (rollout, turn). To plug into verl
we therefore:

  1. Pack each verl-row's metadata into a `TrajectoryRewardBundle`
     (extracting traj scores from `data.batch['traj_scores_*']`,
     per-turn rewards from `data.batch['turn_rewards_*']`,
     and safety from `data.batch['safety_scores_*']`).
  2. Call `compute_dual_horizon_advantage(group=..., alpha, lambda_safety)`.
  3. Broadcast the per-(rollout, turn) advantage onto verl's per-token
     `(B, T_resp)` shape using the per-turn token spans (collected from
     the rollout actor, written to `data.non_tensor_batch['turn_token_spans']`).

How to integrate into a verl-GDPO fork
--------------------------------------
1. Apply the patch in `docs/verl_integration.md` to
   `verl/trainer/ppo/ray_trainer.py:compute_advantage()`. It adds a new
   branch:

       elif adv_estimator == "ec_mtrl_dual_horizon":
           from src.training.grpo_advantage import compute_advantage_dual_horizon
           advantages, returns = compute_advantage_dual_horizon(data, cfg)

2. Set `algorithm.adv_estimator = "ec_mtrl_dual_horizon"` in the verl
   Hydra config. Add `algorithm.alpha`, `algorithm.lambda_safety`,
   `algorithm.min_valid_turn` keys.

3. The reward function (rollout actor side) MUST populate the following
   per-row fields on `data.batch` / `data.non_tensor_batch`:

       data.batch['traj_scores_r_goal']        : (B,) int
       data.batch['traj_scores_r_fit']         : (B,) int
       data.batch['traj_scores_u_terminal']    : (B,) int
       data.batch['safety_c_safety']           : (B,) int 0/1
       data.batch['safety_c_cat']              : (B,) int 0/1  (= c_safety post-Decision-3)
       data.non_tensor_batch['turn_rewards']   : list[TurnRewards] length B
       data.non_tensor_batch['group_index']    : (B,) int (rollout group id)
       data.non_tensor_batch['trajectory_id']  : (B,) str
       data.non_tensor_batch['turn_token_spans']:
                                                 list[list[(start, end)]]
                                                 inner length = num_turns_i,
                                                 absolute token offsets WITHIN
                                                 the response slice.

This module is import-safe without verl installed (the verl-specific
imports are guarded inside the function body).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.rewards.mock_judge import SafetyScores, TrajectoryScores
from src.training.advantage import (
    GroupAdvantage,
    TrajectoryRewardBundle,
    TurnRewards,
    compute_dual_horizon_advantage,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch


# ---------------------------------------------------------------------------
# Pure-Python core: turn-level advantage matrix → token-level (B, T_resp)
# ---------------------------------------------------------------------------

def _broadcast_turn_advantage_to_tokens(
    group_advantage: GroupAdvantage,
    turn_token_spans: list[list[tuple[int, int]]],
    response_length: int,
) -> np.ndarray:
    """Project (rollout, turn) advantage onto per-token positions.

    Args
    ----
    group_advantage:     GroupAdvantage with `.advantage` shape (G, T_max).
    turn_token_spans:    For each rollout in the group, a list of
                         (start_token, end_token) pairs (one per actual turn,
                         absolute offsets within the response slice).
    response_length:     T_resp — number of token positions in each row's
                         response (uniform across rollouts after padding).

    Returns
    -------
    np.ndarray of shape (G, response_length): per-token advantage. Token
    positions outside any turn span receive 0.0 (they will be masked out by
    `eos_mask` during the policy gradient).
    """
    G = len(turn_token_spans)
    out = np.zeros((G, response_length), dtype=np.float32)
    A = group_advantage.advantage  # (G, T_max)

    for i, spans in enumerate(turn_token_spans):
        for t, (start, end) in enumerate(spans):
            if t >= A.shape[1]:
                # safety: turn count mismatch with advantage matrix; skip
                break
            adv_value = A[i, t]
            if not np.isfinite(adv_value):
                adv_value = 0.0
            start = max(0, int(start))
            end = min(response_length, int(end))
            if end > start:
                out[i, start:end] = adv_value
    return out


# ---------------------------------------------------------------------------
# verl entrypoint
# ---------------------------------------------------------------------------

@dataclass
class DualHorizonConfig:
    """Subset of `algorithm.*` keys we read from verl's Hydra config."""
    alpha: float = 1.0
    lambda_safety: float = 5.0
    min_valid_turn: int = 5
    whiten_advantages: bool = True


def compute_advantage_dual_horizon(
    data: Any,
    cfg: DualHorizonConfig | None = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """verl-compatible adapter — call from `ray_trainer.compute_advantage()`.

    Reads the per-row fields documented in this module's top docstring,
    bundles them by `group_index`, calls
    `compute_dual_horizon_advantage(group, alpha, lambda_safety)` once per
    group, and broadcasts the resulting (G, T_max) advantage onto the
    full (B, T_resp) token grid expected by verl.

    Returns (advantages, returns); `returns = advantages` (verl convention
    for outcome rewards — value head is disabled).
    """
    import torch  # local import: keeps `import verl_advantage` cheap

    cfg = cfg or DualHorizonConfig()

    # ---- 1. Extract per-row tensors / metadata ------------------------------
    response_length = int(data.batch["responses"].shape[-1])
    B = int(data.batch["responses"].shape[0])
    device = data.batch["responses"].device

    eos_mask = data.batch["attention_mask"][:, -response_length:]  # (B, T_resp)

    r_goal_np = data.batch["traj_scores_r_goal"].cpu().numpy().astype(np.int64)
    r_fit_np = data.batch["traj_scores_r_fit"].cpu().numpy().astype(np.int64)
    u_term_np = data.batch["traj_scores_u_terminal"].cpu().numpy().astype(np.int64)
    c_safety_np = data.batch["safety_c_safety"].cpu().numpy().astype(np.int64)
    c_cat_np = data.batch["safety_c_cat"].cpu().numpy().astype(np.int64)

    turn_rewards_list: list[TurnRewards] = list(data.non_tensor_batch["turn_rewards"])
    group_index_np = np.asarray(data.non_tensor_batch["group_index"], dtype=np.int64)
    traj_ids = list(data.non_tensor_batch["trajectory_id"])
    turn_token_spans: list[list[tuple[int, int]]] = list(
        data.non_tensor_batch["turn_token_spans"]
    )

    # ---- 2. Group rows by group_index ---------------------------------------
    groups: dict[int, list[int]] = {}
    for row_idx, g in enumerate(group_index_np.tolist()):
        groups.setdefault(int(g), []).append(row_idx)

    # ---- 3. Per-group: build bundles, compute advantage, broadcast ---------
    all_advantages_np = np.zeros((B, response_length), dtype=np.float32)
    for g_idx, rows in groups.items():
        bundles: list[TrajectoryRewardBundle] = []
        for r in rows:
            tr = turn_rewards_list[r]
            bundle = TrajectoryRewardBundle(
                trajectory_id=str(traj_ids[r]),
                num_turns=len(tr.r_distress),
                traj_scores=TrajectoryScores(
                    r_goal=int(r_goal_np[r]),
                    r_fit=int(r_fit_np[r]),
                    u_terminal=int(u_term_np[r]),
                    evidence_turn_indices=[],
                ),
                turn_rewards=tr,
                safety_scores=SafetyScores(c_safety=int(c_safety_np[r])),
                c_cat=int(c_cat_np[r]),
            )
            bundles.append(bundle)

        ga = compute_dual_horizon_advantage(
            group=bundles,
            alpha=float(cfg.alpha),
            lambda_safety=float(cfg.lambda_safety),
            min_valid_turn=int(cfg.min_valid_turn),
        )

        spans_for_group = [turn_token_spans[r] for r in rows]
        per_token = _broadcast_turn_advantage_to_tokens(
            group_advantage=ga,
            turn_token_spans=spans_for_group,
            response_length=response_length,
        )
        for local_i, r in enumerate(rows):
            all_advantages_np[r] = per_token[local_i]

    # ---- 4. (Optional) batch-wise whitening — match GDPO trl impl ---------
    advantages_t = torch.from_numpy(all_advantages_np).to(device).float()
    if cfg.whiten_advantages:
        advantages_t = _masked_whiten(advantages_t, eos_mask.float())

    # ---- 5. eos_mask gating -------------------------------------------------
    advantages_t = advantages_t * eos_mask.float()
    returns_t = advantages_t.clone()  # verl convention for outcome rewards

    return advantages_t, returns_t


# ---------------------------------------------------------------------------
# Helpers (no verl dependency)
# ---------------------------------------------------------------------------

def _masked_whiten(x: "torch.Tensor", mask: "torch.Tensor", eps: float = 1e-8) -> "torch.Tensor":
    """Standardize x to zero mean / unit std over `mask=1` positions."""
    import torch
    n = mask.sum()
    if n.item() < 2:
        return x  # not enough samples to whiten
    mean = (x * mask).sum() / n
    var = (((x - mean) * mask) ** 2).sum() / n
    std = torch.sqrt(var + eps)
    return (x - mean) / (std + eps) * mask


__all__ = [
    "DualHorizonConfig",
    "compute_advantage_dual_horizon",
]

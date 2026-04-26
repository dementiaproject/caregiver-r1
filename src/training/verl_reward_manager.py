"""
verl reward manager for caregiver-R1 EC-MTRL.

⚠️ Like the rollout worker, this is verl-API-touching code; signatures may
need light adjustment. Tested logic, untested integration.

Most of the per-trajectory reward signals are ALREADY computed inside the
rollout worker (`verl_rollout_worker.CaregiverMultiTurnRollout` already
calls the LLM judges and turn-level reward, and writes everything to
`data.batch` + `data.non_tensor_batch`). So this reward manager's job is
mostly **bookkeeping**: it produces the `token_level_rewards` field verl
expects, and it converts our advantage-side fields into a format
`compute_advantage_ec_mtrl` (the dispatch in `ray_trainer.compute_advantage`)
can consume.

Why split rollout and reward at all
-----------------------------------
verl's pipeline is rollout → reward → advantage → loss. Splitting the
work this way is what makes the multi-turn caregiver-DemMA loop fit:
- the rollout HAS to call the LLM judges (because to score a trajectory
  you need the full `<think>+<response>` and patient observations)
- BUT verl's `compute_advantage` checks `data.batch['token_level_rewards']`
- so we re-export the trajectory-level scores AS a token_level_rewards
  tensor (zero everywhere except the final-token of each row, which carries
  the bundled scalar reward — verl's standard "outcome reward" pattern)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class CaregiverRewardManager:
    """verl `RewardManager`-style component.

    Subclass / wrap as needed for your verl version:
      verl/utils/reward_score/__init__.py — register `name="caregiver"`
      verl/trainer/main_ppo.py            — invoke as
                                            `reward_fn = CaregiverRewardManager(cfg)`
                                            then `reward_fn(data)` in the trainer.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def __call__(self, data: Any) -> Any:
        """Populate `data.batch['token_level_rewards']` (B, T_resp).

        The advantage adapter (`compute_advantage_ec_mtrl`) already reads the
        per-row scalar fields from `data.batch` directly, so the
        `token_level_rewards` tensor we put here is mostly used by verl's
        own logging/diagnostics.

        Convention (matches verl's GRPO usage): the bundled outcome reward
        for each row is placed at the LAST non-pad position of the response.
        """
        import torch

        responses = data.batch["responses"]              # (B, T_resp)
        attention_mask = data.batch["attention_mask"]    # (B, T_prompt + T_resp)
        T_resp = responses.size(-1)
        response_mask = attention_mask[:, -T_resp:]      # (B, T_resp)

        # ---- 1. Collapse trajectory-level scores into a scalar reward -----
        #
        # We use the SAME composition as the smoke-run advantage estimator
        # falls back to for unit tests: a centered combination of R_goal,
        # R_fit, u_terminal (already pre-normalized by their rubric ranges).
        # This is purely a logging signal — the actual policy gradient signal
        # comes from `data.batch['advantages']` populated by the
        # `ec_mtrl_dual_horizon` advantage adapter, NOT from this scalar.
        r_goal = data.batch["traj_scores_r_goal"].float()
        r_fit = data.batch["traj_scores_r_fit"].float()
        u_term = data.batch["traj_scores_u_terminal"].float()
        c_safety = data.batch["safety_c_safety"].float()

        scalar = r_goal + r_fit + u_term - 5.0 * c_safety   # see lambda_violation
        scalar = scalar.to(responses.device)

        # Place scalar at the last non-pad position of each row's response
        last_idx = response_mask.sum(dim=-1).clamp(min=1).long() - 1   # (B,)
        token_level_rewards = torch.zeros_like(response_mask, dtype=torch.float)
        token_level_rewards.scatter_(
            dim=-1, index=last_idx.unsqueeze(-1),
            src=scalar.unsqueeze(-1),
        )
        data.batch["token_level_rewards"] = token_level_rewards
        # also expose `token_level_scores` (verl's diagnostic key)
        data.batch["token_level_scores"] = token_level_rewards

        # ---- 2. Diagnostic per-channel scores (for W&B) -------------------
        # These mirror the GDPO conventions (token_level_scores_<channel>).
        for name, t in [
            ("r_goal", r_goal), ("r_fit", r_fit),
            ("u_terminal", u_term), ("c_safety", c_safety),
        ]:
            tensor = torch.zeros_like(response_mask, dtype=torch.float)
            tensor.scatter_(
                dim=-1, index=last_idx.unsqueeze(-1),
                src=t.to(responses.device).unsqueeze(-1),
            )
            data.batch[f"token_level_scores_{name}"] = tensor

        log.info(
            "CaregiverRewardManager: B=%d  mean(R_goal)=%.2f  mean(R_fit)=%.2f  "
            "mean(u_term)=%.2f  c_safety_rate=%.2f",
            int(responses.shape[0]),
            float(r_goal.mean().item()),
            float(r_fit.mean().item()),
            float(u_term.mean().item()),
            float(c_safety.mean().item()),
        )

        return data


__all__ = ["CaregiverRewardManager"]

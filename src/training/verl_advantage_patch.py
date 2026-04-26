"""
The actual code to paste into verl-GDPO's `ray_trainer.compute_advantage()`.

This module exists so the verl patch is **one import line** instead of
copy-pasting 30 lines of business logic into the verl fork.

Apply it like this — open `verl/trainer/ppo/ray_trainer.py` in your
verl-GDPO fork, find `compute_advantage()` (around line 142), and add this
branch right BEFORE the `else: raise NotImplementedError` line:

    elif adv_estimator == 'ec_mtrl_dual_horizon':
        from caregiver_r1.training.verl_advantage_patch import compute_advantage_ec_mtrl
        data = compute_advantage_ec_mtrl(data, cfg=...)

(`cfg` is the trainer's algorithm config; see `_BUILD_CFG_FROM_TRAINER`
below for how to extract it from `self.config.algorithm`.)

A standalone unified-diff is in `docs/verl_patch.diff`.
"""

from __future__ import annotations

from typing import Any

from src.training.grpo_advantage import (
    DualHorizonConfig,
    compute_advantage_dual_horizon,
)


def _build_cfg_from_trainer(trainer_algorithm_cfg: Any) -> DualHorizonConfig:
    """Pull our knobs out of verl's `cfg.algorithm` Hydra section.

    Expected fields (set in `verl_configs/grpo_caregiver.yaml`):

        algorithm:
          adv_estimator: ec_mtrl_dual_horizon
          alpha: 1.0
          lambda_safety: 16.0  # = 3 + 2·α_max + margin (covers α∈[0.25, 4])
          min_valid_turn: 5
          whiten_advantages: true
    """
    return DualHorizonConfig(
        alpha=float(getattr(trainer_algorithm_cfg, "alpha", 1.0)),
        lambda_safety=float(getattr(trainer_algorithm_cfg, "lambda_safety", 5.0)),
        min_valid_turn=int(getattr(trainer_algorithm_cfg, "min_valid_turn", 5)),
        whiten_advantages=bool(
            getattr(trainer_algorithm_cfg, "whiten_advantages", True)
        ),
    )


def compute_advantage_ec_mtrl(data: Any, cfg: Any | None = None) -> Any:
    """Drop-in advantage branch — call this from `compute_advantage()`.

    `data` is verl's `DataProto`. The reward manager (see
    `verl_reward_manager.py`) MUST have already populated:

      data.batch['traj_scores_r_goal']      : (B,) int
      data.batch['traj_scores_r_fit']       : (B,) int
      data.batch['traj_scores_u_terminal']  : (B,) int
      data.batch['safety_c_safety']         : (B,) int 0/1
      data.batch['safety_c_cat']            : (B,) int 0/1
      data.non_tensor_batch['turn_rewards'] : list[TurnRewards] of length B
      data.non_tensor_batch['group_index']  : (B,) int (rollout group id)
      data.non_tensor_batch['trajectory_id']: (B,) str
      data.non_tensor_batch['turn_token_spans']:
                                              list[list[(start, end)]]
                                              token spans within response slice
    """
    dh_cfg = cfg if isinstance(cfg, DualHorizonConfig) else _build_cfg_from_trainer(cfg)
    advantages, returns = compute_advantage_dual_horizon(data, cfg=dh_cfg)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


__all__ = ["compute_advantage_ec_mtrl", "_build_cfg_from_trainer"]

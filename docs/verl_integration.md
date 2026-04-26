# Integrating EC-MTRL into verl-GDPO

This doc walks through how to take the framework-agnostic pieces in
`src/training/{advantage,grpo_advantage,grpo_loss}.py` and patch them
into a forked verl-GDPO so the trainer scales beyond a single GPU.

The single-process baseline in `scripts/train_grpo.py` is good for
smoke + 1-GPU sanity, but for 8×H100 / multi-node training we need verl's
Ray-managed actor system.

---

## 1. Fork verl-GDPO

```bash
git clone https://github.com/NVlabs/GDPO
cd GDPO/verl-GDPO
git checkout -b ec-mtrl
```

Install in your training env:
```bash
pip install -e .
```

## 2. Add our advantage estimator

Edit `verl/trainer/ppo/ray_trainer.py` and find the `compute_advantage`
function (around line 148–175 in the GDPO fork). Add a new branch:

```python
# verl/trainer/ppo/ray_trainer.py — compute_advantage()

elif adv_estimator == "ec_mtrl_dual_horizon":
    # Patch from caregiver-r1 — see src/training/grpo_advantage.py
    from caregiver_r1.training.grpo_advantage import (
        DualHorizonConfig,
        compute_advantage_dual_horizon,
    )
    ec_cfg = DualHorizonConfig(
        alpha=cfg.algorithm.alpha,
        lambda_safety=cfg.algorithm.lambda_safety,
        min_valid_turn=cfg.algorithm.min_valid_turn,
        whiten_advantages=cfg.algorithm.get("whiten_advantages", True),
    )
    advantages, returns = compute_advantage_dual_horizon(data, ec_cfg)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
```

## 3. Provide a multi-turn rollout actor

verl's default rollout (`vllm` or `hf`) is **single-turn**: prompt → response.
We need **multi-turn caregiver↔DemMA**.

Put your custom rollout actor in `verl_workers/caregiver_rollout_actor.py`
(module path you name), inheriting from verl's `BaseRolloutWorker` /
`BaseGenerator` (depending on verl version). The actor needs to:

  1. Receive a batch of scenarios.
  2. For each scenario, run `cfg.rollout.group_size` trajectories through
     the caregiver↔DemMA loop (port `scripts/run_smoke.py:synth_dummy_trajectory`).
  3. Per turn: call `policy.generate()` (vLLM-served caregiver), parse
     `<think>+<response>`, send `<response>` to DemMA.
  4. After each trajectory finishes, call the LLM judge (HTTP) for
     `R_goal / R_fit / u_terminal / c_safety`.
  5. Compute turn-level rewards via `src.rewards.turn_level.compute_turn_rewards`.
  6. Return a `data.batch` with the fields documented in
     `src/training/grpo_advantage.py` top docstring.

A good template is `verl-GDPO/verl/workers/rollout/vllm_rollout/vllm_rollout.py`.
Wrap that for the per-turn loop.

## 4. Reward function

verl's reward function is called per-trajectory. Implement it as
`src/training/verl_reward.py` (you write this once verl is set up):

```python
def compute_reward(data: DataProto) -> DataProto:
    # data.non_tensor_batch contains scenario, trajectories
    from src.rewards.vllm_judge_adapter import build_vllm_judges
    from src.rewards.turn_level import compute_turn_rewards
    # ... call judges + turn-level for each row ...
    # ... write traj_scores_*, safety_*, turn_rewards into data.batch ...
    return data
```

## 5. Hydra config

Create `verl_configs/grpo_caregiver.yaml`:

```yaml
algorithm:
  adv_estimator: ec_mtrl_dual_horizon
  alpha: 1.0
  lambda_safety: 16.0   # = 3 + 2·α_max + margin (cover α∈[0.25, 4])
  min_valid_turn: 5
  whiten_advantages: true

actor_rollout_ref:
  rollout:
    name: caregiver_multi_turn      # registered name of your custom actor
    n: 8                             # group_size
    max_turns: 6
  ...

reward_model:
  reward_manager: caregiver           # custom reward mgr that calls our judges
```

## 6. Launch

```bash
ray start --head --port=6379
python -m verl.trainer.main_ppo \
    --config-path=verl_configs --config-name=grpo_caregiver
```

## 7. Sanity checks before scaling

1. Run with `algorithm.alpha = 0.0` → behaves as pure trajectory-level
   GRPO; should match the verl GRPO baseline run.
2. Run with `n_gpus_per_node=1` for a smoke step; check W&B for
   `policy/clip_frac < 0.5`, `kl/mean ∈ [0.005, 0.1]`.
3. Confirm `c_safety_rate` from W&B matches the expected ~5% from the
   safety-judge prior; if it spikes to 50%+, your judge is malfunctioning.

## What lives where

| Module                              | Verl-side?  | Notes |
|-------------------------------------|-------------|-------|
| `src/training/advantage.py`         | shared      | pure NumPy, framework-free |
| `src/training/grpo_advantage.py`    | verl-adapter| reads `data.batch`, calls advantage, writes back |
| `src/training/grpo_loss.py`         | shared      | tensor PPO-clip + KL |
| `src/data/caregiver_client.py`      | rollout     | HTTP / mock client |
| `src/data/demma_*_client.py`        | rollout     | DemMA backends |
| `src/rewards/*`                     | reward      | judges + turn-level |
| `scripts/train_grpo.py`             | baseline    | single-GPU; bypassed by verl |
| `scripts/train_sft.py`              | independent | SFT happens BEFORE verl RL |

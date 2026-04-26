#!/usr/bin/env bash
# Launch verl-GDPO smoke training for caregiver-R1.
#
# Prerequisites
# -------------
# 1. verl-GDPO forked + patched (docs/verl_patch.diff applied).
# 2. caregiver-r1 repo on PYTHONPATH (or installed as `pip install -e .`).
# 3. Three vllm servers ready:
#       port 8000: caregiver SFT checkpoint  (will be the trained policy)
#       port 8001: judge model                (Qwen3-32B-Instruct or Qwen3-8B for smoke)
# 4. data/scenarios_train.jsonl exists.
# 5. ray cluster head: `ray start --head --port=6379` (single-node smoke).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERL_REPO="${VERL_REPO:-/path/to/verl-GDPO}"   # set this to your fork

# Make caregiver-r1 importable as `caregiver_r1`
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Optional: pin which GPUs verl can use. Leave commented to use all.
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd "${VERL_REPO}"

python -m verl.trainer.main_ppo \
    --config-path="${REPO_ROOT}/configs/verl" \
    --config-name=grpo_caregiver \
    trainer.experiment_name="grpo-smoke-$(date +%Y%m%d-%H%M%S)" \
    trainer.total_training_steps=10 \
    trainer.n_gpus_per_node="${N_GPUS:-1}" \
    "$@"

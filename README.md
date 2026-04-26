<div align="center">

# Caregiver-R1

### Environment-coherent multi-turn RL for dementia-care dialogue agents

Caregiver-R1 is a research training stack and testbed for caregiver policies
that interact with a frozen dementia-patient simulator and learn from both
trajectory-level outcomes and simulator-native turn-level behavioral signals.

[Method draft](paper/final_rps_method_section_locked_zh_9.md) ·
[Proposal](PROPOSAL.md) ·
[Engineering plan](ENGINEERING_PLAN.md) ·
[verl notes](docs/verl_integration.md) ·
[DemMA checkpoint](https://huggingface.co/hulehule/DemMA-Planner-SFT)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](#license)
[![Status](https://img.shields.io/badge/status-research%20preview-orange.svg)](#status)

</div>

---

## What Is Caregiver-R1?

Caregiver-R1 targets a narrow but difficult RL setting: multi-turn dementia-care
conversations where a patient sincerely states something that conflicts with
reality, such as wanting to visit a deceased parent or insisting medication has
already been taken. The caregiver must respond over several turns while balancing
comfort, autonomy, factual discipline, and safety.

The core idea is to treat DemMA's inline patient-behavior annotations as an
environment-native turn reward channel. Instead of paying for an extra LLM judge
at every turn, the patient simulator already emits structured motion, facial,
and sound labels along with each utterance. Caregiver-R1 maps those labels to
clinical ordinal tiers and uses tier deltas as turn-level credit.

In short: **DemMA provides the patient environment; EC-MTRL provides the
dual-horizon reward and advantage estimator; verl/GDPO provides the scalable RL
training path.**

> **Research preview.** This repository currently contains the simulator
> adapters, prompts, rubrics, reward code, smoke runners, and verl integration
> scaffolding. Paper-scale trained checkpoints and final evaluation results are
> not released yet.

## Why This Repo

- **A multi-turn clinical-care RL testbed, not static medical QA.** The task is
  an interactive caregiver-patient episode with short but meaningful horizon
  length.
- **Zero-extra-call turn rewards.** Turn-level `r_distress` and `r_resistance`
  are derived from DemMA inline labels rather than an external per-turn judge.
- **Dual-horizon credit assignment.** Trajectory-level judge rewards and
  turn-level PBRS state deltas are normalized separately and fused into one
  GRPO/GDPO-style advantage.
- **Binary hard-veto safety.** Catastrophic unsafe behavior is treated as a
  non-compensable constraint, not as another soft reward term.
- **Built for existing RL infrastructure.** The implementation is designed to
  plug into verl/GDPO-style critic-free training with PPO-clip and KL anchoring.

## Released Assets

| Asset | Status | Entry point |
|---|---:|---|
| DemMA mock / real / vLLM clients | Available | [`src/data/`](src/data/) |
| Unified caregiver prompt | Available | [`prompts/caregiver_system_prompt.md`](prompts/caregiver_system_prompt.md) |
| Clinical strategy cards | Available | [`prompts/strategy_cards/`](prompts/strategy_cards/) |
| Turn-level reward and anchors | Available | [`src/rewards/turn_level.py`](src/rewards/turn_level.py), [`src/rewards/clinical_anchors.yaml`](src/rewards/clinical_anchors.yaml) |
| LLM judge rubrics | Draft / lock candidate | [`prompts/rubrics/`](prompts/rubrics/) |
| CPU mock smoke config | Available | [`configs/smoke_cpu.yaml`](configs/smoke_cpu.yaml) |
| DemMA/vLLM smoke config | Available | [`configs/smoke_run.yaml`](configs/smoke_run.yaml) |
| verl integration notes and patch | Experimental | [`docs/verl_integration.md`](docs/verl_integration.md), [`docs/verl_patch.diff`](docs/verl_patch.diff) |
| Paper-scale trained checkpoints | Planned | forthcoming |
| Paper-scale evaluation tables | Planned | forthcoming |

## Method Summary

For each scenario `s`, Caregiver-R1 samples a group of caregiver trajectories
from the same unified prompt. The caregiver response is sent to DemMA; DemMA
returns the next patient utterance plus inline behavioral labels. A trajectory
judge scores the whole dialogue once, while the inline labels provide the
turn-level signal.

Let `G_s` be the rollout group for scenario `s`. For rollout `i` and turn `t`,
the reward stack produces:

- trajectory rewards: `R_goal`, `R_fit`, and `u_terminal`;
- turn rewards: `r_distress = D_{t-1} - D_t` and
  `r_resistance = b_t * (R_{t-1} - R_t)`;
- binary safety: `c_i ∈ {0, 1}`, where `1` means a catastrophic violation.

The EC-MTRL advantage used for policy optimization is:

$$
A^{\mathrm{EC}}_{i,t}
=
(1-c_i)\left(A^{\mathrm{traj}}_i+\alpha A^{\mathrm{turn}}_{i,t}\right)
-\lambda_{\mathrm{viol}}c_i.
$$

Here `A_traj` is the group-normalized trajectory advantage, `A_turn` is the
turn-position-normalized advantage, `alpha` controls the trajectory/turn mix,
and `lambda_viol` is a fixed safety floor. Current configs use
`lambda_viol = 16.0`, sized to keep catastrophic safety violations below clean
rollouts across the planned alpha sweep.

## Quick Start

The full stack is aimed at Linux GPU machines. A RunPod A100/H100 box is the
most straightforward development environment. macOS is fine for editing and
lightweight inspection, but `vllm` and GPU inference are Linux-first.

### 1. Install

```bash
git clone https://github.com/dementiaproject/caregiver-r1.git
cd caregiver-r1

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the real DemMA simulator:

```bash
python -m pip install -r requirements-demma.txt
python scripts/download_demma.py
```

On RunPod, put Hugging Face cache and temporary files under `/workspace` so the
checkpoint does not fill the container disk:

```bash
mkdir -p /workspace/hf_home /workspace/tmp
export HF_HOME=/workspace/hf_home
export HF_HUB_CACHE=/workspace/hf_home/hub
export TMPDIR=/workspace/tmp
export HF_HUB_DISABLE_XET=1
python scripts/download_demma.py
```

### 2. CPU Mock Smoke

This path uses a mock caregiver, mock DemMA, and mock judges. It is meant to
check schemas, reward plumbing, and advantage shapes without loading any model.

```bash
PYTHONPATH=. python scripts/run_smoke.py \
  --config configs/smoke_cpu.yaml \
  --dry-run \
  --n-groups 2
```

### 3. Real DemMA Smoke

After downloading `hulehule/DemMA-Planner-SFT`, use the main smoke config to
exercise the patient simulator. This loads DemMA on GPU and can take several
minutes because every trajectory is a multi-turn patient interaction.

```bash
PYTHONPATH=. python scripts/run_smoke.py \
  --config configs/smoke_run.yaml \
  --dry-run \
  --n-groups 1
```

For the first GPU check, keep `--n-groups 1`; the default rollout settings can
generate many DemMA turns.

### 4. verl Training Path

The paper-scale route is not a one-command release yet. The intended path is:

1. Fork / install the GDPO-compatible verl stack.
2. Apply [`docs/verl_patch.diff`](docs/verl_patch.diff).
3. Register the caregiver multi-turn rollout worker and reward manager.
4. Launch with [`configs/verl/grpo_caregiver.yaml`](configs/verl/grpo_caregiver.yaml).

See [`docs/verl_integration.md`](docs/verl_integration.md) for the current
integration notes. [`scripts/train_grpo.py`](scripts/train_grpo.py) is a
single-process scaffold and should be treated as a sanity harness, not the
paper-scale trainer.

## File Guide

Top-level files:

| Path | Purpose |
|---|---|
| [`README.md`](README.md) | Public project entry point. |
| [`PROPOSAL.md`](PROPOSAL.md) | Research proposal, current claims, risks, and architectural decisions. |
| [`ENGINEERING_PLAN.md`](ENGINEERING_PLAN.md) | Phase-by-phase implementation and compute plan. |
| [`AGENTS.md`](AGENTS.md) | Working instructions for coding agents in this repo. |
| [`pyproject.toml`](pyproject.toml) | Package metadata, dependencies, pytest, and lint config. |
| [`requirements-demma.txt`](requirements-demma.txt) | Extra dependencies for loading the real DemMA checkpoint. |
| [`data/scenarios_smoke.jsonl`](data/scenarios_smoke.jsonl) | Tiny checked-in scenario set for local smoke runs. |

Configs, scripts, and docs:

| Path | Purpose |
|---|---|
| [`configs/smoke_cpu.yaml`](configs/smoke_cpu.yaml) | Mock-only CPU smoke config. |
| [`configs/smoke_run.yaml`](configs/smoke_run.yaml) | GPU smoke config for DemMA/vLLM-style rollout checks. |
| [`configs/grpo_smoke.yaml`](configs/grpo_smoke.yaml) | Single-process GRPO scaffold config. |
| [`configs/sft.yaml`](configs/sft.yaml) | Phase-0 SFT config. |
| [`configs/verl/grpo_caregiver.yaml`](configs/verl/grpo_caregiver.yaml) | Experimental verl-GDPO training config. |
| [`scripts/download_demma.py`](scripts/download_demma.py) | Downloads `hulehule/DemMA-Planner-SFT`. |
| [`scripts/generate_scenarios.py`](scripts/generate_scenarios.py) | Builds structured dementia-care conflict scenarios. |
| [`scripts/run_smoke.py`](scripts/run_smoke.py) | End-to-end rollout/reward/advantage smoke runner. |
| [`scripts/preview_judge_prompt.py`](scripts/preview_judge_prompt.py) | Renders judge prompts for inspection. |
| [`scripts/train_sft.py`](scripts/train_sft.py) | SFT scaffold for caregiver warm-up. |
| [`scripts/train_grpo.py`](scripts/train_grpo.py) | Single-process GRPO scaffold, not the paper-scale trainer. |
| [`scripts/launch_verl_smoke.sh`](scripts/launch_verl_smoke.sh) | Ray/verl smoke launcher. |
| [`docs/verl_integration.md`](docs/verl_integration.md) | How the EC-MTRL pieces plug into verl. |
| [`docs/verl_patch.diff`](docs/verl_patch.diff) | Patch sketch for the verl advantage branch. |

Prompts and rubrics:

| Path | Purpose |
|---|---|
| [`prompts/caregiver_system_prompt.md`](prompts/caregiver_system_prompt.md) | Unified caregiver system prompt. |
| [`prompts/strategy_cards/`](prompts/strategy_cards/) | Ten clinical strategy cards used as an action-space prior. |
| [`prompts/rubrics/r_goal.yaml`](prompts/rubrics/r_goal.yaml) | Goal-progress judge rubric. |
| [`prompts/rubrics/r_fit.yaml`](prompts/rubrics/r_fit.yaml) | Clinical-fit and communication-quality judge rubric. |
| [`prompts/rubrics/u_terminal.yaml`](prompts/rubrics/u_terminal.yaml) | Terminal patient-state judge rubric. |
| [`prompts/rubrics/c_safety.yaml`](prompts/rubrics/c_safety.yaml) | Binary catastrophic-safety judge rubric. |
| [`prompts/rubrics/_judge_template.md`](prompts/rubrics/_judge_template.md) | Shared judge prompt template. |

Core Python modules:

| Path | Purpose |
|---|---|
| [`src/data/schemas.py`](src/data/schemas.py) | Pydantic schemas for scenarios, trajectories, labels, and observations. |
| [`src/data/caregiver_client.py`](src/data/caregiver_client.py) | Placeholder/mock/HTTP caregiver rollout clients. |
| [`src/data/demma_client.py`](src/data/demma_client.py) | DemMA client interface and CPU mock implementation. |
| [`src/data/demma_real_client.py`](src/data/demma_real_client.py) | Transformers in-process DemMA client. |
| [`src/data/demma_vllm_client.py`](src/data/demma_vllm_client.py) | vLLM DemMA client path. |
| [`src/rewards/clinical_anchors.yaml`](src/rewards/clinical_anchors.yaml) | Mapping from DemMA labels to clinical ordinal tiers. |
| [`src/rewards/turn_level.py`](src/rewards/turn_level.py) | `D_t`, `R_t`, care-bid mask, and PBRS turn rewards. |
| [`src/rewards/rubric.py`](src/rewards/rubric.py) | Rubric scoring helpers and aggregation modes. |
| [`src/rewards/judge_prompt.py`](src/rewards/judge_prompt.py) | Judge prompt assembly. |
| [`src/rewards/llm_judge.py`](src/rewards/llm_judge.py) | OpenAI-compatible LLM judge client and parser. |
| [`src/rewards/mock_judge.py`](src/rewards/mock_judge.py) | Deterministic mock trajectory and safety judges. |
| [`src/rewards/vllm_judge_adapter.py`](src/rewards/vllm_judge_adapter.py) | Adapter from vLLM judge outputs to reward bundles. |
| [`src/training/advantage.py`](src/training/advantage.py) | Framework-free EC-MTRL advantage implementation. |
| [`src/training/grpo_advantage.py`](src/training/grpo_advantage.py) | verl-facing adapter for token-level advantages. |
| [`src/training/grpo_loss.py`](src/training/grpo_loss.py) | PPO-clip + KL-anchor loss helper. |
| [`src/training/prompt_loader.py`](src/training/prompt_loader.py) | Loads and assembles caregiver prompts and strategy cards. |
| [`src/training/sft_extract.py`](src/training/sft_extract.py) | Converts DemMA-style dialogues into SFT examples. |
| [`src/training/verl_rollout_worker.py`](src/training/verl_rollout_worker.py) | Experimental caregiver-DemMA multi-turn rollout worker. |
| [`src/training/verl_reward_manager.py`](src/training/verl_reward_manager.py) | Experimental reward manager bridge for verl. |
| [`src/training/verl_advantage_patch.py`](src/training/verl_advantage_patch.py) | Convenience patch helpers for verl integration. |

Tests:

| Path | Purpose |
|---|---|
| [`tests/test_schemas.py`](tests/test_schemas.py) | Schema and label invariants. |
| [`tests/test_demma_client.py`](tests/test_demma_client.py) | DemMA mock and parsing behavior. |
| [`tests/test_turn_level.py`](tests/test_turn_level.py) | Turn-tier and PBRS reward behavior. |
| [`tests/test_advantage.py`](tests/test_advantage.py) | Dual-horizon advantage math and safety gating. |

## Status

<a id="status"></a>

| Layer | Current state |
|---|---|
| DemMA real client | Implemented; loads on the GPU smoke path |
| DemMA vLLM client | Implemented as a faster LM-only path; classifier-head parity is still a separate concern |
| Turn-level reward | Implemented with clinical-anchor YAML and PBRS-style deltas |
| Trajectory / safety judge rubrics | Written as lock candidates; final calibration still required |
| LLM judge client | Implemented for OpenAI-compatible vLLM endpoints |
| Advantage estimator | Implemented as framework-agnostic NumPy core plus verl adapter |
| CPU smoke | Available through `configs/smoke_cpu.yaml` |
| Full verl GRPO training | Experimental integration, not yet a polished public recipe |
| Paper-scale results | Not released |

## Safety And Intended Use

Caregiver-R1 is a research artifact for studying RL credit assignment and
simulated dementia-care dialogue. It is **not** a medical device, clinical
decision system, or caregiver replacement. The DemMA patient is a simulator;
all learned behaviors must be validated by domain experts before any real-world
interpretation.

The safety channel is intentionally binary and hard-vetoed: catastrophic
medical misinformation, unsafe permission, or coercive escalation should not be
recoverable by good style elsewhere in the conversation.

## Citation

<a id="citation"></a>

```bibtex
@misc{caregiver_r1_2026,
  title  = {EC-MTRL: Environment-Coherent Multi-Turn RL with Two-Horizon
            Reward Decoupling for Dementia-Care Dialogue},
  author = {Caregiver-R1 Team},
  year   = {2026},
  note   = {Research preview; preprint forthcoming.}
}
```

If you use the patient simulator, please also cite DemMA:

```bibtex
@inproceedings{demma2025,
  title     = {DemMA: A Multi-Agent Dementia-Patient Simulator for
               Caregiver Conversation Research},
  author    = {Hu et al.},
  booktitle = {Proceedings of ACL},
  year      = {2025}
}
```

## Acknowledgements

This project builds on [DemMA](https://aclanthology.org/2025.acl-long.0/),
[verl](https://github.com/volcengine/verl), and GDPO-style group-relative
optimization. The clinical strategy menu draws on caregiver communication
frameworks including NURSE, VERA, SPIKES, DICE, Reality Orientation,
Therapeutic Fibbing, Reminiscence Therapy, Montessori methods, Redirection, and
Non-committal response.

## License

<a id="license"></a>

This repository is released under the Apache 2.0 License. Trained checkpoints,
when released, will inherit the licenses of their base models.

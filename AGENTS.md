# Project handoff for any new agent thread

> Read this file FIRST in every new thread. It's the single source of truth
> for project state, architectural decisions, file layout, and how the user
> works. Cursor auto-loads this; for ChatGPT / Claude.ai, paste verbatim.

## Project: caregiver-r1 — EC-MTRL caregiver agent for dementia care

We're training an R1-style caregiver chatbot via multi-turn RL against
DemMA (a frozen LLM patient simulator from the user's lab). Algorithm
extends GDPO with two contributions: (1) multi-horizon rank-based reward
normalization (trajectory + per-turn percentile), (2) α-weighted
dual-horizon advantage with a binary hard-veto safety gate.

**Paper draft**: `paper/final_rps_method_section_locked_zh_9.md` (zh_9
locked; the doc has a top note pointing forward to `PROPOSAL.md §9` for
deltas not yet in zh_10).

## Architectural decisions (PROPOSAL.md §9)

| # | Decision | Status |
|---|---|---|
| 1 | Unified caregiver system prompt + temperature sampling for group diversity (drops the per-strategy prompt-conditioning that the original zh_9 design had). `strategy_id` is now just a rollout-index label. | ✅ locked |
| 2 | Single LLM safety judge replaces the old two-layer (regex `C_cat` + ordinal Lagrangian). | ⛔ superseded by Decision 3 |
| 3 | Safety is BINARY hard-veto (`c_safety ∈ {0,1}`). Any catastrophic violation zeroes performance reward AND adds a fixed `−λ_violation` floor penalty. **NO** dual-ascent Lagrangian. λ default 16. | ✅ locked |

## Repo layout — what's where

```
src/
  data/
    schemas.py                     Pydantic schemas (Persona/Scenario/Turn/Trajectory/Group/InlineAnnotation)
    demma_client.py                DemMAClient ABC + DemMAMockClient
    demma_real_client.py           transformers in-process DemMA (with classifier head)
    demma_vllm_client.py           vLLM offline LM-only DemMA
    caregiver_client.py            CaregiverClient ABC + Mock + HTTP (vllm serve)
  rewards/
    turn_level.py                  PBRS r_distress / r_resistance + care-bid mask
    clinical_anchors.yaml          DemMA label → distress/resistance tier lookup
    rubric.py                      Pydantic schemas for LLM judge rubrics
    judge_prompt.py                build_grader_prompt (NO <think> shown to judge!)
    llm_judge.py                   JudgeClient ABC + Mock + VllmJudgeClient (HTTP)
    mock_judge.py                  MockTrajectoryJudge / MockSafetyJudge for smoke
    vllm_judge_adapter.py          VllmTrajectoryJudge / VllmSafetyJudge wrappers
  training/
    advantage.py                   compute_dual_horizon_advantage (NumPy, framework-free)
    grpo_advantage.py              verl-style data.batch adapter
    grpo_loss.py                   PPO-clip + K3 KL anchor (tensor-only)
    prompt_loader.py               assembles caregiver_system_prompt + 10 strategy cards (sha256 frozen)
    sft_extract.py                 DemMA dialog → caregiver SFT JSONL via teacher LLM
    verl_advantage_patch.py        what to import inside verl ray_trainer.compute_advantage()
    verl_rollout_worker.py         CaregiverMultiTurnRollout (multi-turn caregiver↔DemMA actor)
    verl_reward_manager.py         CaregiverRewardManager (writes token_level_rewards)

prompts/
  caregiver_system_prompt.md       template w/ {strategy_cards} placeholder
  strategy_cards/01_..10_*.md      10 clinical strategy cards (NURSE/VERA/SPIKES/etc.)
  rubrics/_judge_template.md       static role section
  rubrics/{r_goal,r_fit,u_terminal,c_safety}.yaml   4 rubric definitions

scripts/
  run_smoke.py                     end-to-end pipeline smoke runner (mock|real|vllm dispatching)
  generate_scenarios.py            template-based 100/1500 scenarios (data/scenarios_*.jsonl)
  download_demma.py                pulls hulehule/DemMA-Planner-SFT
  preview_judge_prompt.py          renders a judge prompt against a fixture
  train_sft.py                     HF Trainer SFT warm-up
  train_grpo.py                    single-process baseline GRPO (rollout_group is NotImpl;
                                   user fills it on GPU using HF .generate(output_scores=True))
  launch_verl_smoke.sh             kicks off verl-GDPO Ray training

configs/
  smoke_run.yaml                   smoke runner (3 backends: demma/caregiver/judge each mock|real|vllm|http)
  grpo_smoke.yaml                  single-GPU baseline GRPO config
  sft.yaml                         SFT config
  verl/grpo_caregiver.yaml         full verl Hydra config

docs/
  verl_integration.md              fork+patch+launch recipe
  verl_patch.diff                  unified diff for verl-GDPO ray_trainer.py + worker registry

tests/                             74 tests passing
  test_schemas.py, test_turn_level.py, test_advantage.py
data/scenarios_smoke.jsonl         100 scenarios committed
```

## Status snapshot (engineering plan rollup)

```
A. Infra            ████████████████  100%  ✅
B. Data             ████████░░░░░░░░   50%  ⚠️ scenarios ✓; SFT data ❌ (needs DemMA dialog corpus)
C. Reward system    ████████████████  100%  ✅
D. SFT warm-up      ████████░░░░░░░░   50%  ⚠️ code ready; data ❌
E. RL trainer       ████████░░░░░░░░   60%  ⚠️ scaffold + verl files; train_grpo.rollout_group ❌
F. verl multi-GPU   ████░░░░░░░░░░░░   25%  ⚠️ scaffold committed, NOT GPU-tested
```

## Recent review (commit b04e688) — 4 issues fixed

The user had a reviewer flag 4 real bugs on 2026-04-26. All fixed:

1. **ground_truth leakage** — `caregiver_client.build_caregiver_user_message`
   used to inject `conflict_type / severity / risk_tier / ground_truth`
   into the user message. **Fixed**: only persona shown; conflict context
   removed entirely.
2. **Schema mismatch with Decision 1** — `Trajectory.strategy_id` was
   required and `Group._check_unique_strategies` was enforced.
   **Fixed**: `strategy_id: str = "r0"` Optional; uniqueness check replaced
   with non-empty check; `verl_rollout_worker._build_traj_stub` rebuilt
   to pass `seed`/`terminated_by` and the real per-turn annotations
   (was passing empty `InlineAnnotation`).
3. **Judge sees `<think>`** — `judge_prompt.render_trajectory_for_judge`
   used to include `<think>` lines. **Fixed**: only response + patient
   utterance + cues are shown to the judge.
4. **`λ_safety = 5` too small** — math: violator advantage `−λ` should be
   strictly below worst-clean `−3 − 2α`. **Fixed**: default bumped 5.0 →
   16.0 (covers α∈[0.25, 4] with margin ≥ 5).

## What runs RIGHT NOW

CPU dev box (this machine):
```bash
.venv/bin/python -m pytest tests/                                  # 74 pass
PYTHONPATH=. python scripts/run_smoke.py --dry-run --n-groups 5    # mock everywhere
python scripts/generate_scenarios.py --n 100 --out data/scenarios_smoke.jsonl
```

GPU box (RunPod, user has it):
- DemMA real client + vLLM client both verified end-to-end
- `run_smoke.py` with `demma.mode=vllm` + `judge.mode=mock` + `caregiver.mode=mock`
  ran successfully (~6 min for 5 groups × 30 turns)

## Pending GPU work (in priority order)

1. **Set up vLLM serves** for caregiver and judge (DemMA already in-process):
   ```bash
   vllm serve Qwen/Qwen3-8B --port 8000 --served-model-name caregiver-sft &
   vllm serve Qwen/Qwen3-8B --port 8001 &   # smoke; 32B for production
   ```
2. **Full smoke** with `caregiver.mode=http` + `judge.mode=vllm` to verify
   real reward signal looks reasonable.
3. **SFT data**: user has DemMA dialog corpus? If yes, run
   `src.training.sft_extract` against it. If no, we need synthetic
   generation alternative.
4. **Run `train_sft.py`** on the SFT JSONL.
5. **Fill `train_grpo.py:rollout_group()`** (~80 lines HF generate code; user can do this).
6. **Fork verl-GDPO + apply `docs/verl_patch.diff`** for multi-GPU scaling.
7. **Wire `_resync_caregiver_weights`** in `verl_rollout_worker.py` (LoRA push or vllm RPC).

## How the user works (preferences from observed behavior)

- **Direct, honest answers**. They push back hard when an answer is overly
  cautious, vague, or hedges. If you don't know, say so; if you have a
  guess, give it with the confidence level.
- **No deferring "until later"**. If they ask "why not now?", commit to
  doing it NOW unless there's a hard reason. They want progress, not
  TODOs.
- **Bilingual (zh + en)**. Replies in 中文 are fine; technical terms stay
  in English (vLLM, GRPO, KL, etc.).
- **They're a co-author of DemMA**. Trust their domain claims; verify their
  engineering claims by reading source. They sometimes propose engineering
  shortcuts that have subtle issues — say so directly.
- **Concise > comprehensive**. Tables and bullet points; trim ceremony.
  Don't explain things they already know.
- **Code first, doc second**. They prefer to see the diff and run it,
  not read 5 paragraphs of why.
- **They have GPU you don't**. Don't pretend to test things you can't;
  do mark code clearly as "needs GPU validation". They'll iterate.

## Common gotchas already burned

- DemMA SFT did NOT include `[ACT]` block in CE loss → LM hallucinates
  labels at inference. Solution: either use the classifier head (slow
  2-pass) OR write canonical 34-label vocab into prompt (current).
- `chat_template_kwargs={"enable_thinking": False}` for Qwen3 — DemMA
  was trained without thinking; default is True → wastes tokens.
- vLLM 0.8.3 falls back to Transformers backend for Qwen3; need ≥0.8.5.
  Current install lands on 0.19.x.
- `flash-attn` ABI breaks on torch upgrade; uninstall + reinstall when
  torch bumps.
- vLLM `LLM(...)` is in-process → cannot share GPU mem between caregiver
  training + DemMA inference + judge inference. Use HTTP via `vllm serve`
  for the latter two during training.

## Live transcript

The full multi-day chat history of this project lives at:

```
/Users/jiangwu/.cursor/projects/Users-jiangwu-caregiver-r1/agent-transcripts/
```

For ANY question about prior context (why was X chosen / what was tried /
what failed), search there before guessing. Format is JSONL of
user/assistant turns.

## Repo + recent commits

GitHub: `dementiaproject/caregiver-r1` (private)

Recent commits (most recent first):
- `b04e688`  fix(review): 4 reviewer-flagged issues
- `b93c8d7`  feat(verl): full verl-GDPO integration scaffolding
- `48ccced`  feat: end-to-end training scaffolding (waves 4-5)
- `0f9cb96`  feat(data): scenario generator + 100-scenario library
- `bfeff3c`  feat(rollout): real caregiver client (HTTP / mock)
- `c177fbf`  feat(judge): real vLLM HTTP judge client + smoke wiring

Run `git log --oneline -20` for more.

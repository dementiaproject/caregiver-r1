"""
verl multi-turn rollout worker for caregiver↔DemMA dialogues.

⚠️ This is verl-API-touching code; signatures may need light adjustment to
match your specific verl-GDPO commit. Tested logic, untested integration.

What it does
------------
verl's stock rollout workers (`vLLMRollout`, `HFRollout`) do single-turn
generation: prompt → response. We need:

    System prompt + scenario opener
        ↓ (caregiver.generate)
    <think>...</think><response>...</response>          ← turn 0 caregiver
        ↓ (DemMA.step)
    Patient utterance + InlineAnnotation                 ← turn 0 patient
        ↓ (caregiver.generate, history grows)
    Caregiver turn 1
        ...
    [up to max_turns or terminal condition]

This worker delegates the THREE LLM calls to **separate vLLM HTTP servers**
(caregiver / DemMA / judge), bypassing verl's hybrid weight-sync. The
upside is simplicity (no Megatron / FSDP weight-binding magic); the
downside is the caregiver vllm-serve has STALE WEIGHTS between training
steps unless you actively push checkpoints to it (see `_resync_caregiver_weights`).

Fitting into verl
-----------------
- Subclass `verl.workers.rollout.base.BaseRollout`.
- Override `generate_sequences(prompts: DataProto) -> DataProto`.
- Build the per-row `responses` tensor as the **concatenated tokens of all
  caregiver turns** (system / user / patient tokens are NOT in the response;
  they live in the prompt half via the chat template assembly).
- The actor's policy gradient is taken w.r.t. the response tokens only —
  matches our `caregiver_response` attention mask.
- Per-row metadata needed for the reward manager + advantage adapter is
  written to `non_tensor_batch`.

Outputs (the DataProto fields written by this worker)
-----------------------------------------------------
batch (TensorDict, keys are torch tensors padded to a uniform shape):
    prompts          : (B, T_prompt)
    responses        : (B, T_resp)        — caregiver tokens concatenated
                                            across all turns; right-padded
                                            with pad_token_id
    input_ids        : (B, T_prompt + T_resp) — concat of prompts+responses
    attention_mask   : (B, T_prompt + T_resp)
    position_ids     : (B, T_prompt + T_resp)

non_tensor_batch (dict-like):
    scenario_id      : (B,) str           — Scenario.scenario_id
    trajectory_id    : (B,) str
    group_index      : (B,) int
    uid              : (B,) int           — verl GRPO group key (= group_index)
    turn_count       : (B,) int           — number of completed turns
    turn_token_spans : list[list[(s,e)]]  — (start, end) WITHIN responses
    caregiver_thinks : list[list[str]]    — per-turn think strings (audit)
    patient_utts     : list[list[str]]    — per-turn patient text
    patient_anns     : list[list[InlineAnnotation]]
    cg_responses     : list[list[str]]

The reward manager reads these in
`src.training.verl_reward_manager.RewardManager.score()`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# Local imports — keep these inside methods to avoid hard verl dependency at
# `import caregiver_r1.training.verl_rollout_worker` time on a CPU box.

class CaregiverMultiTurnRollout:
    """Multi-turn caregiver rollout worker (subclass `BaseRollout` in verl).

    Use this as a template — copy into your verl fork as
    `verl/workers/rollout/caregiver_multi_turn_rollout.py` and make it a real
    subclass of `verl.workers.rollout.base.BaseRollout`.

    Constructor args (passed by verl when it instantiates the worker per the
    Hydra `actor_rollout_ref.rollout` section):

        config                : DictConfig (the rollout sub-tree)
        tokenizer             : the policy tokenizer
        scenario_provider     : a callable iter[Scenario] (loaded from
                                data/scenarios_train.jsonl by the trainer)

    The worker owns 3 HTTP clients:

        caregiver_client : src.data.caregiver_client.CaregiverHttpClient
                           (vllm serve hosting the SFT'd / RL-update target)
        demma_client     : src.data.demma_client.DemMAClient (any backend)
        judge_clients    : (TrajectoryJudge, SafetyJudge) from build_vllm_judges
    """

    def __init__(
        self,
        config: Any,
        tokenizer: Any,
        scenario_provider: Any,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.scenario_provider = scenario_provider

        from src.data.caregiver_client import CaregiverHttpClient
        from src.data.demma_client import DemMAMockClient
        from src.rewards.vllm_judge_adapter import build_vllm_judges

        cg_cfg = config.caregiver_http
        self.caregiver_client = CaregiverHttpClient(
            base_url=str(cg_cfg.base_url),
            model_name=str(cg_cfg.model_name),
            request_timeout_s=float(getattr(cg_cfg, "request_timeout_s", 60.0)),
            max_tokens=int(getattr(cg_cfg, "max_tokens", 1024)),
            temperature=float(getattr(cg_cfg, "temperature", 1.0)),
            top_p=float(getattr(cg_cfg, "top_p", 0.9)),
        )

        # DemMA dispatch — usually `vllm` mode for production, supplied by
        # the trainer config. Pull whatever the user has set.
        demma_mode = str(getattr(config, "demma_mode", "vllm"))
        if demma_mode == "vllm":
            from src.data.demma_vllm_client import DemMAVLLMClient
            d = config.demma_vllm
            self.demma_client = DemMAVLLMClient(
                model_path=str(d.model_path),
                long_memory_path=str(d.long_memory_path),
                short_memory_path=str(d.short_memory_path),
                patient_id=int(d.patient_id),
                dtype=str(d.dtype),
                gpu_memory_utilization=float(d.gpu_memory_utilization),
                max_model_len=int(d.max_model_len),
                max_new_tokens=int(d.max_new_tokens),
                temperature=float(d.temperature),
                top_p=float(d.top_p),
            )
        else:
            self.demma_client = DemMAMockClient()

        # Judges (HTTP)
        j = config.judge_vllm
        self.traj_judge, self.safety_judge = build_vllm_judges(
            base_url=str(j.base_url),
            model_name=str(j.model_name),
            request_timeout_s=float(getattr(j, "request_timeout_s", 120.0)),
            max_tokens=int(getattr(j, "max_tokens", 2048)),
            health_check_at_start=bool(getattr(j, "health_check_at_start", True)),
        )

        # Scenario iterator
        self._scenario_iter = iter(scenario_provider)

        # Generation knobs
        self.max_turns = int(getattr(config, "max_turns", 6))
        self.group_size = int(getattr(config, "group_size", 4))
        self.prompt_length = int(getattr(config, "prompt_length", 1024))
        self.response_length = int(getattr(config, "response_length", 4096))

    # ---- Weight sync helper (placeholder) ---------------------------------

    def _resync_caregiver_weights(self, actor_module) -> None:
        """Push current actor weights to the caregiver vllm-serve.

        Implementation depends on your serving layer:
          - if vllm serve was started with `--enable-lora`, push as a LoRA
          - if you launched a stock `vllm serve <ckpt>`, you need a sidecar
            that writes the checkpoint and restarts the server (slow)
          - if you use vllm's RPC `update_weights_from_external_launcher`,
            call that here

        For the smoke train, leaving weights stale between gradient steps is
        TOLERABLE for the first ~10 steps (small drift). Beyond that, weight
        sync must be wired or training quality degrades.
        """
        log.warning(
            "_resync_caregiver_weights: NOT IMPLEMENTED. Caregiver vllm-serve "
            "has stale weights. For >10-step training, wire weight sync."
        )

    # ---- generate_sequences (verl's main rollout entry point) -------------

    def generate_sequences(self, prompts) -> Any:  # prompts: DataProto
        """Multi-turn rollout.

        verl gives us a batch of starting prompts (one per scenario in the
        trainer's batch). For each, we run `group_size` trajectories of up
        to `max_turns` turns, collect everything, and return a verl
        `DataProto` whose `responses` are the concatenated caregiver tokens.

        Note the FAN-OUT: input batch is B scenarios → output batch is
        B*group_size rows.
        """
        import torch
        from tensordict import TensorDict
        from verl import DataProto

        from src.data.caregiver_client import CaregiverDialogueItem
        from src.data.demma_client import DialogueHistoryItem
        from src.rewards.turn_level import compute_turn_rewards
        from src.training.advantage import TrajectoryRewardBundle, TurnRewards

        scenarios = self._take_scenarios(prompts)   # one Scenario per row in input batch
        B_in = len(scenarios)
        G = self.group_size
        B_out = B_in * G

        # Per-row buffers
        all_prompt_ids: list[list[int]] = []
        all_response_ids: list[list[int]] = []
        all_turn_spans: list[list[tuple[int, int]]] = []
        all_scenario_id: list[str] = []
        all_traj_id: list[str] = []
        all_group_idx: list[int] = []
        all_turn_count: list[int] = []
        all_thinks: list[list[str]] = []
        all_patient_utts: list[list[str]] = []
        all_patient_anns: list[list[Any]] = []
        all_cg_responses: list[list[str]] = []
        all_traj_scores_rgoal: list[int] = []
        all_traj_scores_rfit: list[int] = []
        all_traj_scores_uterm: list[int] = []
        all_safety_c: list[int] = []
        all_safety_cat: list[int] = []
        all_turn_rewards: list[TurnRewards] = []

        # ---- Per-scenario rollouts ---------------------------------------
        for s_idx, scenario in enumerate(scenarios):
            for k in range(G):
                seed = (
                    int(getattr(self.config, "per_rollout_seed_offset", 1000))
                    + (hash(scenario.scenario_id) ^ (k * 7919))
                ) & 0xFFFFFFFF

                cg_history: list[CaregiverDialogueItem] = []
                dm_history: list[DialogueHistoryItem] = []
                annotations = []
                patient_utts: list[str] = []
                cg_responses: list[str] = []
                cg_thinks: list[str] = []

                response_ids: list[int] = []
                turn_spans: list[tuple[int, int]] = []

                for t in range(self.max_turns):
                    latest_patient = (
                        scenario.initial_patient_utterance if t == 0 else None
                    )
                    try:
                        cg_out = self.caregiver_client.step(
                            scenario=scenario,
                            history=cg_history,
                            latest_patient_utterance=latest_patient,
                            seed=seed + t,
                        )
                    except Exception as e:
                        log.warning("caregiver step failed s=%s k=%d t=%d: %s",
                                    scenario.scenario_id, k, t, e)
                        break

                    cg_thinks.append(cg_out.think)
                    cg_responses.append(cg_out.response)

                    # Tokenize the caregiver's full <think>+<response> XML blob
                    # so we can later recover per-turn token boundaries WITHIN
                    # the concatenated response sequence.
                    turn_text = (
                        f"<think>{cg_out.think}</think>"
                        f"<response>{cg_out.response}</response>"
                    )
                    turn_ids = self.tokenizer.encode(turn_text, add_special_tokens=False)
                    start = len(response_ids)
                    response_ids.extend(turn_ids)
                    end = len(response_ids)
                    turn_spans.append((start, end))

                    # DemMA reaction
                    try:
                        obs = self.demma_client.step(
                            persona=scenario.persona,
                            history=dm_history,
                            latest_caregiver_response=cg_out.response,
                            seed=seed + t,
                        )
                    except Exception as e:
                        log.warning("DemMA step failed s=%s k=%d t=%d: %s",
                                    scenario.scenario_id, k, t, e)
                        break

                    annotations.append(obs.annotation)
                    patient_utts.append(obs.utterance)
                    cg_history.append(CaregiverDialogueItem(
                        caregiver_response=cg_out.response,
                        patient_utterance=obs.utterance,
                    ))
                    dm_history.append(DialogueHistoryItem(
                        caregiver_response=cg_out.response,
                        patient_utterance=obs.utterance,
                        patient_annotation=obs.annotation,
                    ))

                # Truncate to max response_length (right-pad later)
                if len(response_ids) > self.response_length:
                    response_ids = response_ids[: self.response_length]
                    turn_spans = [
                        (s, e) for s, e in turn_spans if e <= self.response_length
                    ]

                # ---- Turn-level reward (rule-based) ----------------------
                if cg_responses:
                    tr = compute_turn_rewards(
                        annotations=annotations,
                        patient_texts=patient_utts,
                        caregiver_responses=cg_responses,
                        severity=scenario.severity,
                    )
                else:
                    tr = TurnRewards(r_distress=[], r_resistance=[], care_bid_mask=[])

                # ---- Trajectory-level + safety judges --------------------
                # Build a lightweight Trajectory stub for the judges.
                # IMPORTANT: pass the REAL `annotations` collected during the
                # rollout — earlier version threaded empty InlineAnnotations,
                # which starved the u_terminal / safety judges of patient cues.
                traj_stub = self._build_traj_stub(
                    trajectory_id=f"traj_{scenario.scenario_id}_k{k}",
                    scenario_id=scenario.scenario_id,
                    cg_history=cg_history,
                    cg_thinks=cg_thinks,
                    annotations=annotations,
                    seed=int(seed),
                    terminated_by=(
                        "max_turns" if len(cg_responses) >= self.max_turns
                        else "simulator_end"
                    ),
                )
                t_scores = self.traj_judge.score_trajectory(traj_stub)
                s_scores = self.safety_judge.score_trajectory(traj_stub)

                # ---- Build prompt token ids (system + scenario opener) --
                prompt_ids = self._build_prompt_ids(scenario)
                if len(prompt_ids) > self.prompt_length:
                    prompt_ids = prompt_ids[-self.prompt_length:]

                # ---- Append to row buffers -------------------------------
                all_prompt_ids.append(prompt_ids)
                all_response_ids.append(response_ids)
                all_turn_spans.append(turn_spans)
                all_scenario_id.append(scenario.scenario_id)
                all_traj_id.append(f"traj_{scenario.scenario_id}_k{k}")
                all_group_idx.append(s_idx)   # group key = scenario index
                all_turn_count.append(len(cg_responses))
                all_thinks.append(cg_thinks)
                all_patient_utts.append(patient_utts)
                all_patient_anns.append(annotations)
                all_cg_responses.append(cg_responses)
                all_traj_scores_rgoal.append(int(t_scores.r_goal))
                all_traj_scores_rfit.append(int(t_scores.r_fit))
                all_traj_scores_uterm.append(int(t_scores.u_terminal))
                all_safety_c.append(int(s_scores.c_safety))
                all_safety_cat.append(int(s_scores.c_safety))   # post-Decision-3: c_cat = c_safety
                all_turn_rewards.append(tr)

        # ---- Pad & build tensors -----------------------------------------
        prompts_t = self._pad_2d(all_prompt_ids, self.prompt_length, left_pad=True)
        responses_t = self._pad_2d(all_response_ids, self.response_length, left_pad=False)
        input_ids_t = torch.cat([prompts_t, responses_t], dim=-1)
        attention_mask_t = self._build_attention_mask(prompts_t, responses_t)
        position_ids_t = self._build_position_ids(attention_mask_t)

        batch = TensorDict(
            {
                "prompts": prompts_t,
                "responses": responses_t,
                "input_ids": input_ids_t,
                "attention_mask": attention_mask_t,
                "position_ids": position_ids_t,
                "traj_scores_r_goal": torch.tensor(all_traj_scores_rgoal, dtype=torch.long),
                "traj_scores_r_fit": torch.tensor(all_traj_scores_rfit, dtype=torch.long),
                "traj_scores_u_terminal": torch.tensor(all_traj_scores_uterm, dtype=torch.long),
                "safety_c_safety": torch.tensor(all_safety_c, dtype=torch.long),
                "safety_c_cat": torch.tensor(all_safety_cat, dtype=torch.long),
            },
            batch_size=B_out,
        )

        non_tensor = {
            "scenario_id": np.array(all_scenario_id, dtype=object),
            "trajectory_id": np.array(all_traj_id, dtype=object),
            "group_index": np.array(all_group_idx, dtype=np.int64),
            "uid": np.array(all_group_idx, dtype=np.int64),   # verl GRPO group key
            "turn_count": np.array(all_turn_count, dtype=np.int64),
            "turn_token_spans": all_turn_spans,                # list of lists
            "turn_rewards": all_turn_rewards,                  # list of TurnRewards
            "caregiver_thinks": all_thinks,
            "patient_utterances": all_patient_utts,
            "patient_annotations": all_patient_anns,
            "caregiver_responses": all_cg_responses,
        }

        return DataProto(batch=batch, non_tensor_batch=non_tensor)

    # ---- Helpers ------------------------------------------------------------

    def _take_scenarios(self, prompts) -> list:
        """Pull `len(prompts)` Scenarios off our scenario_provider iterator.

        verl's prompt batch carries the scenario id (we put it there during
        dataset prep — see verl_dataset.py — but for the SCAFFOLD we just
        iterate next() over the provider in order).
        """
        from src.data.schemas import Scenario
        out: list[Scenario] = []
        n = int(prompts.batch["input_ids"].shape[0])
        for _ in range(n):
            try:
                out.append(next(self._scenario_iter))
            except StopIteration:
                # restart epoch
                self._scenario_iter = iter(self.scenario_provider)
                out.append(next(self._scenario_iter))
        return out

    def _build_prompt_ids(self, scenario) -> list[int]:
        """Build the system-prompt + scenario-opener token ids that verl
        considers the row's "prompt". Match the chat template applied at
        inference time in CaregiverHttpClient."""
        from src.data.caregiver_client import build_caregiver_user_message
        from src.training.prompt_loader import load_caregiver_prompt
        sys_msg = load_caregiver_prompt()
        user_msg = build_caregiver_user_message(
            scenario=scenario,
            history=[],
            latest_patient_utterance=scenario.initial_patient_utterance,
        )
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _pad_2d(self, rows: list[list[int]], max_len: int, left_pad: bool):
        import torch
        pad_id = self.tokenizer.pad_token_id
        out = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
        for i, row in enumerate(rows):
            row = row[:max_len]
            if left_pad:
                out[i, max_len - len(row):] = torch.tensor(row, dtype=torch.long)
            else:
                out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        return out

    def _build_attention_mask(self, prompts_t, responses_t):
        import torch
        pad_id = self.tokenizer.pad_token_id
        mask = torch.cat(
            [(prompts_t != pad_id).long(), (responses_t != pad_id).long()],
            dim=-1,
        )
        return mask

    def _build_position_ids(self, attention_mask_t):
        import torch
        return (attention_mask_t.cumsum(-1) - 1).clamp(min=0)

    def _build_traj_stub(
        self,
        trajectory_id: str,
        scenario_id: str,
        cg_history,
        cg_thinks,
        annotations,
        seed: int,
        terminated_by: str,
    ):
        """Build a `Trajectory`-shape object the LLM judges can render.

        Schema is `extra="forbid"`, so we MUST pass exactly the fields the
        Trajectory pydantic model declares: trajectory_id, scenario_id,
        strategy_id (defaulted), turns, seed, terminated_by. The
        derivation of `num_turns` is a `@property` and must NOT be passed.
        """
        from src.data.schemas import (
            CaregiverOutput,
            InlineAnnotation,
            PatientObservation,
            Trajectory,
            Turn,
        )
        empty_ann = InlineAnnotation(motion=[], facial=[], sound=[])
        turns: list[Turn] = []
        for t, h in enumerate(cg_history):
            think_t = cg_thinks[t] if t < len(cg_thinks) else ""
            ann_t = annotations[t] if t < len(annotations) else empty_ann
            turns.append(
                Turn(
                    turn_index=t,
                    caregiver=CaregiverOutput(
                        think=think_t,
                        response=h.caregiver_response,
                        raw_text=f"<think>{think_t}</think><response>{h.caregiver_response}</response>",
                    ),
                    patient=PatientObservation(
                        utterance=h.patient_utterance,
                        annotation=ann_t,
                    ),
                )
            )
        return Trajectory(
            trajectory_id=trajectory_id,
            scenario_id=scenario_id,
            strategy_id="r0",
            turns=turns,
            seed=seed,
            terminated_by=terminated_by,  # type: ignore[arg-type]
        )


__all__ = ["CaregiverMultiTurnRollout"]

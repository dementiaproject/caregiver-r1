"""
Caregiver-R1 — Smoke Run Entry Script (Day 1)

Purpose
-------
End-to-end pipeline check for the EC-MTRL GRPO loop:
  1. Load Hydra config (configs/smoke_run.yaml)
  2. Build mock DemMA + mock judges + scenario set
  3. Build α-weighted dual-horizon advantage estimator
  4. Run a small number of GRPO steps with mock rewards
  5. Verify no crash / no NaN / KL stays in target range

What it does NOT do
-------------------
  * No real LLM judge call (judge.mode = mock)
  * No real DemMA simulator (demma.mode = mock)
  * No SFT warm-up (kl_anchor_to = base)
  * The "verl GRPO update" step is currently a no-op stub — Phase E.2 will
    plug verl's actor.update_actor() here once verl is installed and the
    config schema is finalized.

This file is intentionally framework-agnostic: all RL logic flows through
TrajectoryRewardBundle / GroupAdvantage so that swapping mock judge → real
judge or dummy update → real verl GRPO update is a one-line change.

Usage
-----
    cd caregiver-r1
    python scripts/run_smoke.py --config configs/smoke_run.yaml [--dry-run]

    --dry-run: skip the (currently stubbed) GRPO update; only roll out a
               handful of groups and verify reward / advantage shapes.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

# Local imports (pure-Python; no verl/torch yet)
from src.data import (
    DemMAClient,
    DemMAMockClient,
    InlineAnnotation,
    PatientObservation,
    Persona,
    Scenario,
    get_demma_real_client_cls,
    get_demma_vllm_client_cls,
)
from src.rewards.mock_judge import build_mock_judges
from src.rewards.vllm_judge_adapter import build_vllm_judges
from src.rewards.turn_level import (
    compute_care_bid_mask,
    compute_distress_tier,
    compute_resistance_tier,
    compute_turn_rewards,
)
from src.training.advantage import (
    TrajectoryRewardBundle,
    TurnRewards,
    compute_dual_horizon_advantage,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("smoke")


# ---------------------------------------------------------------------------
# Helpers — scenarios, mock turn-level rewards, mock C_cat
# ---------------------------------------------------------------------------

def load_scenarios(path: Path, n_max: int) -> list[Scenario]:
    """
    Load scenarios from JSONL. Falls back to N synthetic scenarios if the
    file does not yet exist (so smoke run works before
    scripts/build_smoke_scenarios.py is implemented).
    """
    if not path.exists():
        log.warning("scenarios file %s not found — generating %d synthetic scenarios", path, n_max)
        return [_synthetic_scenario(i) for i in range(n_max)]

    with path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    scenarios = [Scenario.model_validate(r) for r in records[:n_max]]
    log.info("loaded %d scenarios from %s", len(scenarios), path)
    return scenarios


def _synthetic_scenario(idx: int) -> Scenario:
    """Synthetic scenario for offline smoke run when no scenarios file exists."""
    rng = random.Random(idx)
    conflict_types = ["temporal", "identity", "event", "spatial", "medication"]
    conflict_type = rng.choice(conflict_types)
    severity = rng.choice([1, 2, 3])
    risk_tier = "high" if conflict_type in {"medication", "identity"} else rng.choice(["low", "medium"])
    return Scenario(
        scenario_id=f"smoke_{idx:04d}",
        persona=Persona(
            persona_id="demma_default",   # silences DemMARealClient persona-mismatch warning
            dementia_subtype="alzheimers_moderate",
            name="Mrs. Wang" if idx % 2 == 0 else "Mr. Lee",
            age=78 + (idx % 10),
        ),
        conflict_type=conflict_type,  # type: ignore[arg-type]
        severity=severity,             # type: ignore[arg-type]
        risk_tier=risk_tier,           # type: ignore[arg-type]
        initial_patient_utterance=f"[smoke conflict #{idx}] My mother visited me today.",
        ground_truth=f"[smoke gt #{idx}] Mother is deceased since 2014.",
    )


def mock_turn_rewards(num_turns: int, rng: random.Random) -> TurnRewards:
    """
    Until src.rewards.turn_level.compute_turn_rewards is implemented, fake
    plausible PBRS-style state-delta rewards in {-1, 0, +1} so the advantage
    estimator's percentile-rank logic gets a non-degenerate input.
    """
    r_distress = [rng.randint(-1, 1) for _ in range(num_turns)]   # D_{t-1} - D_t in {-3..3}, smoke uses tiny range
    care_bid = [rng.random() < 0.4 for _ in range(num_turns)]      # ~40% of turns have care-bid (rough prior)
    r_resistance = [(rng.randint(-1, 1) if mask else 0) for mask in care_bid]
    return TurnRewards(r_distress=r_distress, r_resistance=r_resistance, care_bid_mask=care_bid)


def derive_c_cat_from_safety(safety_scores) -> int:
    """Decision 3 (2026-04-26, see PROPOSAL §9): c_safety is now BINARY,
    so the gate input `c_cat` equals c_safety directly — no threshold.

        c_cat = c_safety   (both ∈ {0, 1})

    The advantage estimator gates the performance term to 0 when c_cat=1
    AND applies a fixed large `lambda_safety = λ_violation` penalty so
    violators rank strictly below every clean trajectory in the group.
    """
    return int(safety_scores.c_safety)


def build_demma_client(cfg: DictConfig) -> DemMAClient:
    """
    Dispatch on cfg.demma.mode:
      - "mock":  CPU-only random schema-valid annotations (default)
      - "real":  transformers in-process load of hulehule/DemMA-Planner-SFT
                 (requires GPU + `pip install -r requirements-demma.txt`
                 + `python scripts/download_demma.py`)
      - "vllm":  Phase F; not yet implemented (DemMAVLLMClient.step raises)
    """
    mode = str(cfg.demma.mode).lower()
    if mode == "mock":
        log.info("DemMA backend: MOCK (CPU)")
        return DemMAMockClient()
    if mode == "real":
        rcfg = cfg.demma.real
        log.info("DemMA backend: REAL (transformers in-process, device=%s, model_path=%s)",
                 rcfg.device, rcfg.model_path)
        cls = get_demma_real_client_cls()
        client = cls(
            model_path=rcfg.model_path,
            classifier_path=rcfg.classifier_path,
            long_memory_path=rcfg.long_memory_path,
            short_memory_path=rcfg.short_memory_path,
            patient_id=int(rcfg.patient_id),
            device=str(rcfg.device),
            dtype=str(rcfg.dtype),
            temperature=float(rcfg.temperature),
            top_p=float(rcfg.top_p),
            max_new_tokens=int(rcfg.max_new_tokens),
            action_threshold=float(rcfg.action_threshold),
            use_classifier=bool(getattr(rcfg, "use_classifier", False)),
        )
        if not client.health_check():
            raise FileNotFoundError(
                f"DemMA real-client artifacts missing under {rcfg.model_path}; "
                f"run `python scripts/download_demma.py` first."
            )
        return client
    if mode == "vllm":
        vcfg = cfg.demma.vllm
        log.info(
            "DemMA backend: VLLM (offline LLM, model=%s, gpu_mem=%.2f, tp=%d)",
            vcfg.model_path,
            float(vcfg.gpu_memory_utilization),
            int(getattr(vcfg, "tensor_parallel_size", 1)),
        )
        cls = get_demma_vllm_client_cls()
        client = cls(
            model_path=vcfg.model_path,
            long_memory_path=vcfg.long_memory_path,
            short_memory_path=vcfg.short_memory_path,
            patient_id=int(vcfg.patient_id),
            dtype=str(vcfg.dtype),
            gpu_memory_utilization=float(vcfg.gpu_memory_utilization),
            max_model_len=int(vcfg.max_model_len),
            temperature=float(vcfg.temperature),
            top_p=float(vcfg.top_p),
            max_new_tokens=int(vcfg.max_new_tokens),
            tensor_parallel_size=int(getattr(vcfg, "tensor_parallel_size", 1)),
            enforce_eager=bool(getattr(vcfg, "enforce_eager", False)),
        )
        if not client.health_check():
            raise FileNotFoundError(
                f"DemMA vLLM-client artifacts missing under {vcfg.model_path}; "
                f"run `python scripts/download_demma.py` first."
            )
        return client
    raise ValueError(f"unknown demma.mode={mode!r}; expected mock|real|vllm")


# ---------------------------------------------------------------------------
# Judge dispatcher (mock | vllm)
# ---------------------------------------------------------------------------

def build_judges(cfg: DictConfig, seed: int):
    """Dispatch on `cfg.judge.mode`:
      - "mock":  CPU random JSON, deterministic by seed (default for offline smoke)
      - "vllm":  4-rubric HTTP grading against an OpenAI-compatible vLLM endpoint
                 (requires `vllm serve <model>` running)
    """
    mode = str(getattr(cfg.judge, "mode", "mock")).lower()
    if mode == "mock":
        log.info("Judge backend: MOCK (CPU, seed=%d)", seed)
        traj, safety = build_mock_judges(seed=seed)
        return traj, safety
    if mode == "vllm":
        jcfg = cfg.judge.vllm
        log.info("Judge backend: VLLM (model=%s, base_url=%s)",
                 jcfg.model_name, jcfg.base_url)
        traj, safety = build_vllm_judges(
            base_url=str(jcfg.base_url),
            model_name=str(jcfg.model_name),
            request_timeout_s=float(getattr(jcfg, "request_timeout_s", 120.0)),
            max_tokens=int(getattr(jcfg, "max_tokens", 2048)),
            api_key=getattr(jcfg, "api_key", None),
            use_json_mode=bool(getattr(jcfg, "use_json_mode", True)),
            health_check_at_start=bool(getattr(jcfg, "health_check_at_start", True)),
        )
        return traj, safety
    raise ValueError(f"unknown judge.mode={mode!r}; expected mock|vllm")


# ---------------------------------------------------------------------------
# Caregiver client dispatcher (placeholder | mock | http)
# ---------------------------------------------------------------------------

def build_caregiver_client(cfg: DictConfig):
    """Dispatch on `cfg.caregiver.mode`:
      - "placeholder": return None → rollout uses canned strings (CPU debug)
      - "mock":         CaregiverMockClient (5 hand-written turns, CPU)
      - "http":         CaregiverHttpClient → vllm serve OpenAI endpoint
    """
    mode = str(getattr(cfg.caregiver, "mode", "placeholder")).lower()
    if mode == "placeholder":
        log.info("Caregiver backend: PLACEHOLDER (canned utterances; no model)")
        return None
    if mode == "mock":
        from src.data.caregiver_client import CaregiverMockClient
        log.info("Caregiver backend: MOCK (CPU template-based)")
        return CaregiverMockClient()
    if mode == "http":
        from src.data.caregiver_client import CaregiverHttpClient
        ccfg = cfg.caregiver.http
        log.info("Caregiver backend: HTTP (model=%s, base_url=%s)",
                 ccfg.model_name, ccfg.base_url)
        client = CaregiverHttpClient(
            base_url=str(ccfg.base_url),
            model_name=str(ccfg.model_name),
            request_timeout_s=float(getattr(ccfg, "request_timeout_s", 60.0)),
            max_tokens=int(getattr(ccfg, "max_tokens", 1024)),
            temperature=float(getattr(ccfg, "temperature", 1.0)),
            top_p=float(getattr(ccfg, "top_p", 0.9)),
            api_key=getattr(ccfg, "api_key", None),
        )
        if bool(getattr(ccfg, "health_check_at_start", True)):
            if not client.health_check():
                raise RuntimeError(
                    f"caregiver HTTP health-check failed at {ccfg.base_url} "
                    f"(model={ccfg.model_name}). Did you start `vllm serve {ccfg.model_name}`?"
                )
            log.info("Caregiver HTTP ready: %s @ %s", ccfg.model_name, ccfg.base_url)
        return client
    raise ValueError(f"unknown caregiver.mode={mode!r}; expected placeholder|mock|http")


def synth_dummy_trajectory(
    scenario: Scenario,
    strategy_id: str,
    num_turns: int,
    demma: DemMAClient,
    caregiver: "CaregiverClient | None",
    seed: int,
) -> tuple[str, int, list[InlineAnnotation], list[str], list[str], list[str]]:
    """
    Roll out a single trajectory of (caregiver ↔ DemMA) for up to `num_turns`
    turns and capture the per-turn quad needed by `compute_turn_rewards`
    (zh_9 §4.3) and the LLM judges (zh_9 §4.4):

        - patient annotation (InlineAnnotation: motion/facial/sound)
        - patient utterance text
        - caregiver <response> segment (forwarded to DemMA)
        - caregiver <think> segment (kept for judge / debug, NOT forwarded)

    The function maintains TWO history views:
      - `caregiver_history`: pairs the caregiver sees ((cg_resp, patient_utt));
        does NOT include patient annotation labels (zh_9 §4.2).
      - `demma_history`: pairs DemMA sees, including annotation labels.

    On turn 0 the caregiver responds to `scenario.initial_patient_utterance`
    (the designed conflict opener); subsequent turns respond to the previous
    DemMA-generated utterance, which already lives at the tail of
    `caregiver_history`.

    If `caregiver` is None we fall back to the deterministic placeholder
    behaviour from the original smoke (kept so a CPU box without any
    caregiver service can still exercise the reward / advantage path).

    Returns
    -------
    (trajectory_id, actual_num_turns,
     annotations, patient_texts,
     caregiver_responses, caregiver_thinks)
       all 4 lists aligned and of length `actual_num_turns`.
    """
    from src.data.caregiver_client import CaregiverDialogueItem  # local import: keeps top of file lean

    demma_history: list = []
    caregiver_history: list[CaregiverDialogueItem] = []
    annotations: list[InlineAnnotation] = []
    patient_texts: list[str] = []
    caregiver_responses: list[str] = []
    caregiver_thinks: list[str] = []

    import time

    for t in range(num_turns):
        # ---- Caregiver step -------------------------------------------------
        # The patient text the caregiver should respond to is either the
        # scenario opener (turn 0) or the most recent patient utterance, which
        # at this point has just been appended as the patient half of the LAST
        # caregiver_history item (so we pass latest_patient=None on turns >0).
        latest_patient = (
            scenario.initial_patient_utterance if t == 0 else None
        )

        if caregiver is not None:
            try:
                cg_t0 = time.monotonic()
                cg_out = caregiver.step(
                    scenario=scenario,
                    history=caregiver_history,
                    latest_patient_utterance=latest_patient,
                    seed=seed + t,
                )
                cg_dt = time.monotonic() - cg_t0
                cg_resp = cg_out.response
                cg_think = cg_out.think
            except Exception as e:
                log.warning("caregiver step failed at turn %d: %s", t, e)
                break
        else:
            # Placeholder fallback (no caregiver service configured)
            if t % 3 == 0:
                cg_resp = f"[turn {t}] please take your medication."
            elif t % 3 == 1:
                cg_resp = f"[turn {t}] would you like to come with me?"
            else:
                cg_resp = f"[turn {t}] beautiful weather today."
            cg_think = "(placeholder, no caregiver model)"
            cg_dt = 0.0

        # ---- DemMA step -----------------------------------------------------
        t_start = time.monotonic()
        try:
            obs = demma.step(
                persona=scenario.persona,
                history=demma_history,
                latest_caregiver_response=cg_resp,
                seed=seed + t,
            )
            dm_dt = time.monotonic() - t_start
            log.info(
                "  turn %d/%d  scenario=%s  strategy=%s  cg_dt=%.1fs  dm_dt=%.1fs  "
                "cg_chars=%d  patient_chars=%d  ann=(m=%d,f=%d,s=%d)",
                t + 1, num_turns, scenario.scenario_id, strategy_id,
                cg_dt, dm_dt, len(cg_resp), len(obs.utterance),
                len(obs.annotation.motion),
                len(obs.annotation.facial),
                len(obs.annotation.sound),
            )
            assert isinstance(obs, PatientObservation)
            assert isinstance(obs.annotation, InlineAnnotation)

            # Record this complete exchange.
            annotations.append(obs.annotation)
            patient_texts.append(obs.utterance)
            caregiver_responses.append(cg_resp)
            caregiver_thinks.append(cg_think)

            # Update both history views.
            caregiver_history.append(
                CaregiverDialogueItem(
                    caregiver_response=cg_resp,
                    patient_utterance=obs.utterance,
                )
            )
            # demma_history uses DialogueHistoryItem (includes annotation)
            from src.data.demma_client import DialogueHistoryItem
            demma_history.append(
                DialogueHistoryItem(
                    caregiver_response=cg_resp,
                    patient_utterance=obs.utterance,
                    patient_annotation=obs.annotation,
                )
            )
        except Exception as e:
            log.warning("DemMA step failed at turn %d: %s", t, e)
            break

    return (
        f"traj_{scenario.scenario_id}_{strategy_id}",
        len(annotations),
        annotations,
        patient_texts,
        caregiver_responses,
        caregiver_thinks,
    )


# ---------------------------------------------------------------------------
# One smoke "step" — rollout one group + compute advantage
# ---------------------------------------------------------------------------

def smoke_rollout_one_group(
    scenario: Scenario,
    cfg: DictConfig,
    demma: DemMAClient,
    caregiver,  # CaregiverClient | None
    traj_judge,
    safety_judge,
    rng: random.Random,
) -> tuple[list[TrajectoryRewardBundle], dict]:
    """
    Roll out group_size trajectories for one scenario under the **unified
    caregiver system prompt** (Decision 1, see PROPOSAL §9). Diversity within
    the group comes from per-rollout sampling seeds + decoder temperature,
    not from prompt variation. Each rollout's logical id is `traj_{scenario}_k`.

    Returns (bundles, debug_info).
    """
    group_size = cfg.rollout.group_size
    max_turns = cfg.rollout.max_turns

    # Derive a deterministic but distinct seed per (scenario, rollout-index k).
    seed_offset = int(getattr(cfg.rollout, "per_seed_offset", 7919))

    bundles: list[TrajectoryRewardBundle] = []
    captured_annotations: list[list[InlineAnnotation]] = []
    captured_patient_texts: list[list[str]] = []
    captured_caregiver_responses: list[list[str]] = []
    for k in range(group_size):
        # Combine scenario hash + per-k offset; keep within uint32.
        seed = (cfg.demma.per_rollout_seed_offset
                + (hash(scenario.scenario_id) ^ (k * seed_offset))) & 0xFFFFFFFF
        # Vary turn count slightly per trajectory to mimic real terminations.
        n_turns = rng.randint(max(3, max_turns - 3), max_turns)

        # Strategy id is logical only — every rollout uses the same unified
        # system prompt; "k" disambiguates them within the group.
        rollout_label = f"r{k}"

        traj_id, actual_turns, annotations, patient_texts, caregiver_responses, caregiver_thinks = (
            synth_dummy_trajectory(
                scenario, rollout_label, n_turns, demma, caregiver, seed
            )
        )
        captured_annotations.append(annotations)
        captured_patient_texts.append(patient_texts)
        captured_caregiver_responses.append(caregiver_responses)

        # NOTE: We DO NOT build the full Trajectory pydantic object here —
        # that requires real <think>/<response> generation from the caregiver
        # model, which is a verl integration concern (Phase E.1).
        # Mock judges only need num_turns; we pass a lightweight stub.
        traj_stub = _LightweightTrajectory(trajectory_id=traj_id, num_turns=actual_turns,
                                            rollout_index=k, scenario_id=scenario.scenario_id)

        # Real turn-level rewards via PBRS state-delta (zh_9 §4.3) — no longer mock.
        # Trajectory-level rewards (R_goal/R_fit/u_terminal) and C_cat are still
        # mock until P1/P2/P3 are written.
        if actual_turns > 0:
            turn_rewards = compute_turn_rewards(
                annotations=annotations,
                patient_texts=patient_texts,
                caregiver_responses=caregiver_responses,
                severity=scenario.severity,
            )
        else:
            # Empty rollout (DemMA crashed at turn 0) — fall back to empty rewards
            turn_rewards = mock_turn_rewards(0, rng)

        # Decision 2 (2026-04-25, see PROPOSAL §9): derive c_cat hard-veto
        # gate from the single safety judge's top tier (c_safety=3) instead
        # of a separate regex predicate channel.
        safety_scores = safety_judge.score_trajectory(traj_stub)            # type: ignore[arg-type]
        bundles.append(
            TrajectoryRewardBundle(
                trajectory_id=traj_id,
                num_turns=actual_turns,
                traj_scores=traj_judge.score_trajectory(traj_stub),         # type: ignore[arg-type]
                turn_rewards=turn_rewards,
                safety_scores=safety_scores,
                c_cat=derive_c_cat_from_safety(safety_scores),
            )
        )

    # Per-group diagnostic counters: D / R tier histograms + care-bid rate.
    # Recomputing tiers from the captured annotations is cheap (microseconds)
    # and keeps the diagnostic loop independent from compute_turn_rewards's
    # internal state.
    d_tier_counts = [0, 0, 0, 0]
    r_tier_counts = [0, 0, 0, 0]
    care_bid_count = 0
    total_turns = 0
    for bundle, ann_list, txt_list, resp_list in zip(
        bundles, captured_annotations, captured_patient_texts, captured_caregiver_responses
    ):
        for t in range(bundle.num_turns):
            d_tier_counts[compute_distress_tier(ann_list[t], txt_list[t])] += 1
            r_tier_counts[compute_resistance_tier(ann_list[t], txt_list[t])] += 1
            if compute_care_bid_mask(resp_list[t]):
                care_bid_count += 1
            total_turns += 1

    return bundles, {
        "scenario_id": scenario.scenario_id,
        "n_trajectories": len(bundles),
        "avg_turns": float(np.mean([b.num_turns for b in bundles])),
        "d_tier_counts": d_tier_counts,
        "r_tier_counts": r_tier_counts,
        "care_bid_count": care_bid_count,
        "total_turns": total_turns,
    }


class _LightweightTrajectory:
    """
    Stub satisfying the .num_turns attribute that mock judges read.
    Avoids constructing a full pydantic Trajectory until the caregiver model
    is wired up (Phase E.1 with verl). `rollout_index` (k) replaces the
    earlier `strategy_id` — under unified prompt + temperature sampling there
    is no strategy label, only a within-group rollout index.
    """
    def __init__(self, trajectory_id: str, num_turns: int, rollout_index: int, scenario_id: str) -> None:
        self.trajectory_id = trajectory_id
        self.num_turns = num_turns
        self.rollout_index = rollout_index
        self.scenario_id = scenario_id


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------

def evaluate_validation_gate(stats: dict, cfg: DictConfig) -> tuple[bool, list[str]]:
    """zh_9 §E.3 / configs/smoke_run.yaml validation_gate."""
    failures: list[str] = []

    if not stats.get("no_crash", True):
        failures.append("crash detected (smoke run did not finish all steps)")
    if stats.get("nan_advantages", 0) > 0:
        failures.append(f"NaN entries in finite advantage matrix: {stats['nan_advantages']}")

    finite_frac = stats.get("advantage_finite_fraction", 1.0)
    if finite_frac < cfg.validation_gate.advantage_finite_fraction_min:
        failures.append(
            f"advantage_finite_fraction={finite_frac:.3f} "
            f"< gate {cfg.validation_gate.advantage_finite_fraction_min}"
        )

    c_cat_rate = stats.get("c_cat_trigger_rate", 0.0)
    if c_cat_rate > cfg.validation_gate.c_cat_trigger_rate_max:
        failures.append(
            f"c_cat_trigger_rate={c_cat_rate:.3f} "
            f"> gate {cfg.validation_gate.c_cat_trigger_rate_max} (rules too strict?)"
        )

    return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Caregiver-R1 smoke run (Day 1, mock judges).")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke_run.yaml"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip GRPO update step; only verify rollout + advantage shapes.")
    parser.add_argument("--n-groups", type=int, default=10,
                        help="Number of scenarios (groups) to roll out for the smoke check.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    log.info("loaded config from %s", args.config)
    log.info("smoke run name: %s", cfg.run.name)

    # Make output dir
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "smoke_config_resolved.yaml")

    # RNG
    seed = cfg.run.seed
    rng = random.Random(seed)

    # Build components — dispatch on cfg.demma.mode + cfg.judge.mode + cfg.caregiver.mode
    demma = build_demma_client(cfg)
    caregiver = build_caregiver_client(cfg)
    traj_judge, safety_judge = build_judges(cfg, seed=seed)

    # Load (or synthesize) scenarios
    scenarios = load_scenarios(
        Path(cfg.data.scenarios_path),
        n_max=args.n_groups,
    )
    if not scenarios:
        log.error("no scenarios available; aborting")
        return 1

    # ---- Rollout + advantage loop ----
    crash = False
    nan_advantages = 0
    finite_advantage_frac_list: list[float] = []
    c_cat_trigger_count = 0
    total_trajectories = 0
    avg_a_traj_per_step: list[float] = []
    avg_a_turn_per_step: list[float] = []

    # New diagnostic accumulators (zh_9 §4.3 calibration sanity)
    a_turn_std_per_step: list[float] = []
    a_turn_nonzero_frac_per_step: list[float] = []
    d_tier_total = [0, 0, 0, 0]
    r_tier_total = [0, 0, 0, 0]
    care_bid_total = 0
    turns_total = 0

    t_start = time.time()
    for step_idx, scenario in enumerate(scenarios):
        try:
            bundles, info = smoke_rollout_one_group(scenario, cfg, demma, caregiver, traj_judge, safety_judge, rng)
            for tier in range(4):
                d_tier_total[tier] += info["d_tier_counts"][tier]
                r_tier_total[tier] += info["r_tier_counts"][tier]
            care_bid_total += info["care_bid_count"]
            turns_total += info["total_turns"]
            ga = compute_dual_horizon_advantage(
                bundles,
                alpha=cfg.reward.alpha,
                lambda_safety=cfg.reward.lambda_safety_init,
                min_valid_turn=cfg.reward.per_turn_min_valid,
            )

            # Stats
            adv = ga.advantages
            finite_mask = ~np.isnan(adv)
            finite_count = int(finite_mask.sum())
            total_count = adv.size
            finite_advantage_frac_list.append(finite_count / max(total_count, 1))
            nan_advantages += int(np.isnan(adv[finite_mask]).sum() if finite_count else 0)
            c_cat_trigger_count += int((ga.c_cat_gate == 0).sum())
            total_trajectories += len(bundles)
            avg_a_traj_per_step.append(float(np.mean(ga.a_traj_normalized)))
            # Mean of finite turn-level advantages (excluding NaN beyond T_i)
            finite_turn = ga.a_turn_normalized[~np.isnan(ga.a_turn_normalized)]
            avg_a_turn_per_step.append(float(np.mean(finite_turn)) if finite_turn.size else 0.0)
            # New diagnostics: turn-advantage std + non-zero fraction
            if finite_turn.size:
                a_turn_std_per_step.append(float(np.std(finite_turn)))
                a_turn_nonzero_frac_per_step.append(
                    float(np.mean(np.abs(finite_turn) > 1e-9))
                )
            else:
                a_turn_std_per_step.append(0.0)
                a_turn_nonzero_frac_per_step.append(0.0)

            if (step_idx + 1) % cfg.run.log_every_step == 0:
                log.info(
                    "step=%d/%d  scenario=%s  finite_adv=%.3f  c_cat_in_group=%d  a_traj_mean=%.3f  a_turn_mean=%.3f",
                    step_idx + 1, len(scenarios), scenario.scenario_id,
                    finite_advantage_frac_list[-1], int((ga.c_cat_gate == 0).sum()),
                    avg_a_traj_per_step[-1], avg_a_turn_per_step[-1],
                )

            # ---- GRPO update (Phase E.2 stub) ----
            if not args.dry_run:
                _stub_grpo_update(ga, cfg)

        except Exception as e:
            log.exception("crash at step %d, scenario %s: %s", step_idx, scenario.scenario_id, e)
            crash = True
            break

    elapsed = time.time() - t_start

    # ---- Aggregate stats + validation gate ----
    def _hist_fractions(counts: list[int], total: int) -> dict[str, float]:
        if total <= 0:
            return {f"tier_{i}": 0.0 for i in range(len(counts))}
        return {f"tier_{i}": counts[i] / total for i in range(len(counts))}

    stats = {
        "no_crash": not crash,
        "n_groups_processed": len(avg_a_traj_per_step),
        "total_trajectories": total_trajectories,
        "total_turns": turns_total,
        "advantage_finite_fraction": float(np.mean(finite_advantage_frac_list)) if finite_advantage_frac_list else 0.0,
        "nan_advantages": nan_advantages,
        "c_cat_trigger_rate": c_cat_trigger_count / max(total_trajectories, 1),
        "a_traj_mean_overall": float(np.mean(avg_a_traj_per_step)) if avg_a_traj_per_step else 0.0,
        "a_turn_mean_overall": float(np.mean(avg_a_turn_per_step)) if avg_a_turn_per_step else 0.0,
        # New diagnostics — mean=0 from CRank tells us nothing; these tell us
        # whether the turn-level signal is non-degenerate.
        "a_turn_std_overall":      float(np.mean(a_turn_std_per_step)) if a_turn_std_per_step else 0.0,
        "a_turn_nonzero_fraction": float(np.mean(a_turn_nonzero_frac_per_step)) if a_turn_nonzero_frac_per_step else 0.0,
        "d_tier_distribution":     _hist_fractions(d_tier_total, turns_total),
        "r_tier_distribution":     _hist_fractions(r_tier_total, turns_total),
        "care_bid_rate":           care_bid_total / max(turns_total, 1),
        "elapsed_sec": elapsed,
    }

    log.info("smoke run summary: %s", json.dumps(stats, indent=2))
    (out_dir / "smoke_stats.json").write_text(json.dumps(stats, indent=2))

    ok, failures = evaluate_validation_gate(stats, cfg)
    if ok:
        log.info("✅ smoke run passed all validation gates")
        return 0
    log.error("❌ smoke run failed %d gate(s):", len(failures))
    for f in failures:
        log.error("   - %s", f)
    return 2


def _stub_grpo_update(ga, cfg: DictConfig) -> None:
    """
    Phase E.2 will replace this with verl's actual GRPO actor update:

        from verl.trainer import GRPOTrainer
        trainer.update_actor(advantages=ga.advantages, ...)

    For Day-1 smoke we only assert shape sanity so a future verl integration
    fails loudly if the GroupAdvantage structure ever changes.
    """
    assert ga.advantages.ndim == 2, f"expected (N, T_max), got {ga.advantages.shape}"
    assert ga.a_traj_normalized.shape[0] == ga.advantages.shape[0]


if __name__ == "__main__":
    sys.exit(main())

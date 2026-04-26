"""
Adapters that wrap the low-level `VllmJudgeClient` into the higher-level
`TrajectoryJudge` / `SafetyJudge` Protocols used by the rollout pipeline
(zh_9 §4.4 / §4.5; mock_judge.py defines the protocols).

The split is:
    VllmJudgeClient.grade(rubric_name, traj)  → RubricGrade  (low-level, 1 call)
        ↓
    VllmTrajectoryJudge.score_trajectory(traj) → TrajectoryScores  (3 calls fused)
    VllmSafetyJudge.score_trajectory(traj)     → SafetyScores      (1 call)

This module is a pure orchestration layer: no LLM-specific logic lives here.
The point is so that downstream code (`run_smoke.py`, verl training loop)
sees a uniform `score_trajectory(traj) → scores` shape regardless of whether
the judges are mock or real.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

from src.data import Trajectory
from src.rewards.llm_judge import VllmJudgeClient
from src.rewards.mock_judge import (
    SafetyScores,
    TrajectoryScores,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory judge: fans out to 3 rubrics (R_goal, R_fit, u_terminal)
# ---------------------------------------------------------------------------

class VllmTrajectoryJudge:
    """
    Calls `VllmJudgeClient` three times — once per trajectory rubric — and
    fuses the per-rubric `RubricGrade.aggregated` integers into a single
    `TrajectoryScores` (the dataclass downstream code expects).

    Rubric → field mapping
    ----------------------
        r_goal      → TrajectoryScores.r_goal
        r_fit       → TrajectoryScores.r_fit
        u_terminal  → TrajectoryScores.u_terminal

    Evidence turns from all three rubrics are unioned and (de-duplicated,
    sorted) into `evidence_turn_indices` — only used for downstream W&B /
    audit, no effect on RL gradients.

    Notes
    -----
    - We deliberately do NOT batch the 3 calls into one prompt. HealthBench /
      RubRIX evidence shows that dedicated grader passes per rubric give
      better calibration than asking one prompt to score 3 unrelated rubrics.
      Cost: 3 HTTP calls per trajectory ≈ 3-6 seconds wall on a 32B judge.
    - On any rubric failure (parser exhaustion or transport error) we re-raise.
      The smoke run / verl reward fn should catch and treat as a poisoned
      trajectory (drop or log).
    """

    def __init__(self, client: VllmJudgeClient) -> None:
        self.client = client

    def score_trajectory(self, trajectory: Trajectory) -> TrajectoryScores:
        r_goal_grade, r_goal_stats = self.client.grade("r_goal", trajectory)
        r_fit_grade, r_fit_stats = self.client.grade("r_fit", trajectory)
        u_term_grade, u_term_stats = self.client.grade("u_terminal", trajectory)

        # Union evidence_turns across the 3 rubrics for a single audit list
        evidence_iter = itertools.chain.from_iterable(
            getattr(it, "evidence_turns", []) or []
            for grade in (r_goal_grade, r_fit_grade, u_term_grade)
            for it in grade.items
        )
        evidence = sorted(set(evidence_iter))

        log.debug(
            "VllmTrajectoryJudge: traj_id=%s  r_goal=%d  r_fit=%d  u_terminal=%d  "
            "retries=(%d,%d,%d)  prompt_chars=(%d,%d,%d)",
            trajectory.trajectory_id,
            r_goal_grade.aggregated,
            r_fit_grade.aggregated,
            u_term_grade.aggregated,
            r_goal_stats.n_retries, r_fit_stats.n_retries, u_term_stats.n_retries,
            r_goal_stats.prompt_chars, r_fit_stats.prompt_chars, u_term_stats.prompt_chars,
        )

        return TrajectoryScores(
            r_goal=r_goal_grade.aggregated,
            r_fit=r_fit_grade.aggregated,
            u_terminal=u_term_grade.aggregated,
            evidence_turn_indices=evidence,
        )


# ---------------------------------------------------------------------------
# Safety judge: single call to the c_safety binary-any-trigger rubric
# ---------------------------------------------------------------------------

class VllmSafetyJudge:
    """
    Wraps `VllmJudgeClient` for the c_safety rubric (Decision 3, binary
    hard-veto). The aggregator inside `aggregate_score` already collapses the
    4 catastrophic items into a single 0/1 via OR; we just lift it into a
    `SafetyScores` shell.

    One HTTP call per trajectory. Failure semantics same as
    VllmTrajectoryJudge.
    """

    def __init__(self, client: VllmJudgeClient) -> None:
        self.client = client

    def score_trajectory(self, trajectory: Trajectory) -> SafetyScores:
        grade, stats = self.client.grade("c_safety", trajectory)
        log.debug(
            "VllmSafetyJudge: traj_id=%s  c_safety=%d  retries=%d",
            trajectory.trajectory_id, grade.aggregated, stats.n_retries,
        )
        return SafetyScores(c_safety=grade.aggregated)


# ---------------------------------------------------------------------------
# Convenience builder (run_smoke.py / verl reward fn)
# ---------------------------------------------------------------------------

def build_vllm_judges(
    base_url: str,
    model_name: str,
    request_timeout_s: float = 120.0,
    max_tokens: int = 2048,
    api_key: str | None = None,
    use_json_mode: bool = True,
    extra_headers: dict[str, str] | None = None,
    health_check_at_start: bool = True,
) -> tuple[VllmTrajectoryJudge, VllmSafetyJudge]:
    """One-line factory that mirrors `build_mock_judges` for the real path.

    The two returned judges share ONE `VllmJudgeClient` (and thus one httpx
    Client + one keep-alive HTTP connection pool) — important when running
    on RunPod where socket churn can be expensive.
    """
    client = VllmJudgeClient(
        base_url=base_url,
        model_name=model_name,
        request_timeout_s=request_timeout_s,
        max_tokens=max_tokens,
        api_key=api_key,
        use_json_mode=use_json_mode,
        extra_headers=extra_headers,
    )
    if health_check_at_start:
        if not client.health_check():
            raise RuntimeError(
                f"vLLM judge health-check failed at {base_url} (model={model_name}). "
                f"Did you start `vllm serve {model_name} --port {_port_of(base_url)}`?"
            )
        log.info("vLLM judge ready: %s @ %s", model_name, base_url)
    return VllmTrajectoryJudge(client), VllmSafetyJudge(client)


def _port_of(url: str) -> str:
    """Best-effort port extraction for the helpful health-check error."""
    try:
        from urllib.parse import urlparse
        return str(urlparse(url).port or 8000)
    except Exception:
        return "8000"


__all__ = [
    "VllmTrajectoryJudge",
    "VllmSafetyJudge",
    "build_vllm_judges",
]

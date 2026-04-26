"""
Mock LLM judge — returns random rewards in the correct schema, for smoke runs.

zh_9 §4.4 specifies trajectory-level judge produces three rewards via a single
LLM call (R_goal, R_fit, u_terminal) plus a separate safety judge call for
c_safety. This module mocks BOTH, returning random values from the correct
ranges so the GRPO pipeline can be validated end-to-end before paying for real
LLM judge calls.

Usage:
    from src.rewards.mock_judge import MockTrajectoryJudge, MockSafetyJudge

    traj_judge = MockTrajectoryJudge(seed=0)
    safety_judge = MockSafetyJudge(seed=0)

    scores = traj_judge.score_trajectory(trajectory)   # → TrajectoryScores
    safety = safety_judge.score_trajectory(trajectory) # → SafetyScores

When swapping to a real judge (Phase F.2), implement the same Protocol:
    src.rewards.trajectory_judge.JudgeClient → drop-in replacement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from src.data import Trajectory


# ---------------------------------------------------------------------------
# Score schemas (matched to zh_9 §4.4 / §4.5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryScores:
    """zh_9 §4.4: three trajectory-level rewards from one judge call."""
    r_goal: int        # 0..8 (4 checklist items × {0,1,2})
    r_fit: int         # 0..8 (4 checklist items × {0,1,2})
    u_terminal: int    # 0..4 (2 dimensions × {0,1,2}, last 3 turns rubric)
    evidence_turn_indices: list[int]  # judge's evidence anchoring (zh_9 §4.4 RULERS)


@dataclass(frozen=True)
class SafetyScores:
    """Decision 3 (2026-04-26, see PROPOSAL §9): BINARY hard-veto safety.

    c_safety ∈ {0, 1}:
        0 = no catastrophic violation
        1 = ANY of 3 catastrophic items triggered (medication
            confirmation / unsafe permission / coercion-with-escalation)

    Decision 3 supersedes Decision 2's 4-tier ordinal scheme. Stylistic
    or empathy concerns (elderspeak, mild epistemic arrogance, uncritical
    affirmation outside Therapeutic-Fibbing conditions) are NOT graded
    here — they live as negative-points criteria in R_fit. This channel
    is reserved for true clinical red lines.

    See PROPOSAL §9 Decision 3 + ENGINEERING_PLAN §C.3 for the rationale.
    """
    c_safety: int      # 0 or 1 (BINARY)


# ---------------------------------------------------------------------------
# Protocol — real and mock judges share this interface
# ---------------------------------------------------------------------------

class TrajectoryJudge(Protocol):
    """Drop-in interface; real impl will be src.rewards.trajectory_judge.JudgeClient."""
    def score_trajectory(self, trajectory: Trajectory) -> TrajectoryScores: ...


class SafetyJudge(Protocol):
    """Drop-in interface; real impl will be src.rewards.safety_judge.SafetyJudgeClient."""
    def score_trajectory(self, trajectory: Trajectory) -> SafetyScores: ...


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------

class MockTrajectoryJudge:
    """
    Returns random TrajectoryScores. Use to validate the GRPO pipeline
    (advantage estimator, KL anchor, batch shapes) before paying for real
    LLM calls. Reward distribution is deliberately uniform — gives a
    well-behaved signal for end-to-end smoke run.

    Determinism: a single global RNG seeded once; calls are NOT reproducible
    across multi-process rollout (acceptable for smoke run).
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def score_trajectory(self, trajectory: Trajectory) -> TrajectoryScores:
        n_turns = trajectory.num_turns
        # Sample from full ranges so the smoke-run reward distribution covers
        # the same support as the real judge will eventually produce.
        r_goal = self._rng.randint(0, 8)
        r_fit = self._rng.randint(0, 8)
        u_term = self._rng.randint(0, 4)

        # "Evidence" is just a random subset of turn indices — only used by
        # downstream rubric audit (Phase F.4 hacking audit), no effect on RL.
        n_evidence = min(2, n_turns)
        evidence = self._rng.sample(range(n_turns), n_evidence) if n_turns else []

        return TrajectoryScores(
            r_goal=r_goal,
            r_fit=r_fit,
            u_terminal=u_term,
            evidence_turn_indices=sorted(evidence),
        )


class MockSafetyJudge:
    """Returns random SafetyScores. Same caveats as MockTrajectoryJudge."""

    def __init__(self, seed: int = 1) -> None:
        # Seed offset by 1 so traj/safety judges return uncorrelated values.
        self._rng = random.Random(seed)

    def score_trajectory(self, trajectory: Trajectory) -> SafetyScores:
        # Smoke-run prior: BINARY safety (Decision 3, 2026-04-26).
        #     c_safety = 0  (clean)        ~ 95%
        #     c_safety = 1  (catastrophic) ~  5%   ← triggers HARD VETO
        c = 1 if self._rng.random() < 0.05 else 0
        return SafetyScores(c_safety=c)


# ---------------------------------------------------------------------------
# Convenience builder for run_smoke.py
# ---------------------------------------------------------------------------

def build_mock_judges(seed: int = 0) -> tuple[MockTrajectoryJudge, MockSafetyJudge]:
    """One-line factory for the two mock judges used by smoke run."""
    return MockTrajectoryJudge(seed=seed), MockSafetyJudge(seed=seed + 1000)

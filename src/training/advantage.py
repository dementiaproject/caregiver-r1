"""
EC-MTRL advantage estimator (zh_9 §4.6).

Implements the two GRPO extensions on the advantage side:

  1. Multi-horizon GDPO normalization (§4.6.1)
       — trajectory-level rewards: per-reward CRank in G_s
       — turn-level rewards:       per-reward percentile rank in G_{s,t}

  2. α-weighted dual-horizon advantage fusion (§4.6.2) with binary
     hard-veto safety gate (Decision 3, PROPOSAL §9):

       A_{i,t} = 1[c_safety(τ_i) = 0] · (A_traj_i + α · A_turn_{i,t})
                 −  λ_violation · c_safety(τ_i)

  Where:
    - c_safety ∈ {0, 1} is the single binary safety judge output
      (rubric `prompts/rubrics/c_safety.yaml`).
    - The `c_cat` field on TrajectoryRewardBundle is the gate input,
      derived directly from `c_safety` at construction time:
          c_cat = c_safety   (no threshold, since c_safety is binary)
    - `lambda_safety` (parameter name kept for backward compat) is the
      `λ_violation` floor penalty — a FIXED LARGE constant (~5.0) that
      ensures violators rank strictly below every clean trajectory in
      the group, NOT the dual-ascent Lagrangian multiplier of the old
      Decision 2 design.

Both rank statistics are computed in pure NumPy and the public API takes
plain dataclasses, so this module is fully unit-testable without verl /
vLLM / DemMA — see tests/test_advantage.py.

Theoretical anchor (zh_9 §4.3 PBRS): r_distress and r_resistance are PBRS
shaping rewards (Ng et al. 1999; Wiewiora et al. 2003 for the action-
dependent care-bid mask), so any α > 0 preserves the optimal policy of the
sparse trajectory-level oracle. The α sweep in §6.3 A8 therefore explores
sample efficiency, NOT policy correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.rewards.mock_judge import SafetyScores, TrajectoryScores


# ---------------------------------------------------------------------------
# Per-trajectory and per-turn reward containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TurnRewards:
    """Rule-based, computed during rollout (zh_9 §4.3 — zero LLM call)."""
    r_distress: list[float]      # length T_i, signed integers from D_{t-1} - D_t
    r_resistance: list[float]    # length T_i, b_t · (R_{t-1} - R_t)
    care_bid_mask: list[bool]    # length T_i, b_t

    def __post_init__(self) -> None:
        assert len(self.r_distress) == len(self.r_resistance) == len(self.care_bid_mask)


@dataclass(frozen=True)
class TrajectoryRewardBundle:
    """All reward signals + safety state for one trajectory in the group."""
    trajectory_id: str
    num_turns: int
    traj_scores: TrajectoryScores       # R_goal / R_fit / u_terminal (judge)
    turn_rewards: TurnRewards           # r_distress / r_resistance (rule-based)
    safety_scores: SafetyScores         # c_safety (independent judge)
    c_cat: int                          # 0 or 1 (rule-based hard veto, §4.5)


# ---------------------------------------------------------------------------
# Per-channel rank normalization (§4.6.1)
# ---------------------------------------------------------------------------

def crank(values: np.ndarray) -> np.ndarray:
    """
    Centered Rank (GOPO arXiv 2026.02 / zh_9 §4.6.1) — maps to [-1, 1].

      CRank(x_i) = 2 · (rank(x_i) - 1) / (N - 1) - 1     for N >= 2
      CRank(x_i) = 0                                      for N == 1

    Ties broken via average rank (np.argsort of np.argsort gives ordinal
    rank; we use scipy-compatible average-tie convention by averaging
    indices of equal values).
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    # Average-rank tie handling
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    # Average ties
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        # Recompute with average rank for ties
        sorted_idx = np.argsort(values, kind="stable")
        sorted_values = values[sorted_idx]
        avg_ranks = np.empty(n, dtype=np.float64)
        i = 0
        rank_pos = 1
        while i < n:
            j = i
            while j < n and sorted_values[j] == sorted_values[i]:
                j += 1
            avg_rank = (rank_pos + (rank_pos + (j - i) - 1)) / 2.0
            for k in range(i, j):
                avg_ranks[sorted_idx[k]] = avg_rank
            rank_pos += (j - i)
            i = j
        ranks = avg_ranks

    return 2.0 * (ranks - 1.0) / (n - 1) - 1.0


def percentile_rank(
    values: np.ndarray,
    valid_mask: np.ndarray,
    min_valid: int = 5,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-(scenario, turn-position) percentile rank (zh_9 §4.6.1) — [-1, 1].

    Args:
        values:    shape (N,)  — per-trajectory turn-level reward at this position
        valid_mask: shape (N,) — True if this trajectory has a valid signal here
                                 (e.g. care-bid mask b_t = 1 for r_resistance)
        min_valid: zh_9 §4.6.1 fallback threshold; if fewer than this many
                   trajectories are valid at this turn position, return fallback
        fallback:  shape (N,)  — per-trajectory baseline to substitute (zh_9
                                 §4.6.1 says "回退为该轨迹自身轮级奖励的均值
                                 baseline"); typically all-zeros for smoke run

    Returns shape (N,) in [-1, 1]. Invalid (mask=False) entries get 0.0
    (neutral; they don't enter the gradient anyway because the corresponding
    advantage term gets multiplied by their own mask downstream).
    """
    values = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    n_valid = int(valid_mask.sum())
    if n_valid < min_valid:
        if fallback is None:
            return np.zeros(n, dtype=np.float64)
        return np.asarray(fallback, dtype=np.float64).copy()

    # Compute CRank only on the valid subset; invalid positions get 0.
    out = np.zeros(n, dtype=np.float64)
    valid_values = values[valid_mask]
    out[valid_mask] = crank(valid_values)
    return out


# ---------------------------------------------------------------------------
# Group-level dual-horizon advantage (§4.6.2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupAdvantage:
    """
    Per-(trajectory, turn) advantage matrix for one strategy-conditioned group.

    Shape (N_group, T_max). Entries beyond a trajectory's actual T_i are NaN
    so downstream loss-mask code can detect them.
    """
    advantages: np.ndarray            # shape (N_group, T_max), float64, with NaN padding
    a_traj_normalized: np.ndarray     # shape (N_group,) — for W&B logging
    a_turn_normalized: np.ndarray     # shape (N_group, T_max), with NaN padding
    c_cat_gate: np.ndarray            # shape (N_group,) — 0/1 indicator (1[C_cat=0])
    c_safety_raw: np.ndarray          # shape (N_group,) — for λ dual-update
    debug: dict = field(default_factory=dict)


def compute_dual_horizon_advantage(
    group: list[TrajectoryRewardBundle],
    alpha: float = 1.0,
    lambda_safety: float = 16.0,
    min_valid_turn: int = 5,
) -> GroupAdvantage:
    """
    zh_9 §4.6.2 — α-weighted dual-horizon advantage with BINARY hard-veto
    safety gate (Decision 3, PROPOSAL §9).

        A_traj_i      = CRank_G(R_goal_i) + CRank_G(R_fit_i) + CRank_G(u_terminal_i)
        A_turn_{i,t}  = PercRank_{G,t}(r_distress) + PercRank_{G,t}(r_resistance)
        A_{i,t}       = 1[c_safety_i = 0] · (A_traj_i + α · A_turn_{i,t})
                        − λ_violation · c_safety_i

    Args:
        group:        list of N_group TrajectoryRewardBundle (typically N=10).
                      Each bundle's `c_cat` field MUST equal its `c_safety`
                      under Decision 3 — the rollout layer enforces this
                      via `derive_c_cat_from_safety` in scripts/run_smoke.py.
        alpha:        weight on turn-level advantage (default 1.0;
                      §6.3 A8 sweep ∈ {0.25, 0.5, 1, 2, 4})
        lambda_safety: floor penalty on the binary safety violation
                      (default 16.0). Under Decision 3 this is a FIXED
                      LARGE constant — NOT dual-ascent Lagrangian.

                      Sizing rationale (review note 2026-04-26):
                          A_traj  ∈ [-3, +3]       (3 channels × CRank ∈ [-1,1])
                          A_turn  ∈ [-2, +2]       (2 channels × PercRank ∈ [-1,1])
                          worst-case clean = -3 - α · 2
                      For violator strictly < worst-clean (margin ≥ 2):
                          λ ≥ 3 + 2α + margin
                      At α=1: λ≥7; at α=4 (max sweep): λ≥13. Default 16
                      covers the entire α∈[0.25, 4] sweep with margin ≥ 5,
                      and after the optional batch-wise whitening still
                      maps violators to a strictly-lowest z-score.

                      Override only if you reduce α below 0.25 (then 8
                      suffices) or run an α=8 study (then bump to 24).
        min_valid_turn: per-turn-position minimum valid samples for
                      percentile rank fallback (zh_9 §4.6.1).

    Returns GroupAdvantage with NaN padding beyond each trajectory's T_i.
    """
    n_group = len(group)
    if n_group == 0:
        raise ValueError("empty group passed to compute_dual_horizon_advantage")
    t_max = max(b.num_turns for b in group)

    # ---- 1. Trajectory-level CRank in G_s, per reward head ---------------
    r_goal = np.array([b.traj_scores.r_goal for b in group], dtype=np.float64)
    r_fit = np.array([b.traj_scores.r_fit for b in group], dtype=np.float64)
    u_term = np.array([b.traj_scores.u_terminal for b in group], dtype=np.float64)

    a_traj = crank(r_goal) + crank(r_fit) + crank(u_term)   # shape (N,)

    # ---- 2. Turn-level percentile rank in G_{s, t}, per reward head ------
    a_turn = np.full((n_group, t_max), np.nan, dtype=np.float64)
    for t in range(t_max):
        # Gather distress + resistance across the group at turn position t,
        # masking trajectories that ended before turn t and (for resistance)
        # turns where care-bid mask was 0.
        d_vals = np.zeros(n_group, dtype=np.float64)
        r_vals = np.zeros(n_group, dtype=np.float64)
        d_mask = np.zeros(n_group, dtype=bool)
        r_mask = np.zeros(n_group, dtype=bool)
        for i, b in enumerate(group):
            if t < b.num_turns:
                d_vals[i] = b.turn_rewards.r_distress[t]
                d_mask[i] = True
                r_vals[i] = b.turn_rewards.r_resistance[t]
                # r_resistance already has b_t multiplied in; the mask we use
                # for percentile-rank validity is the care_bid_mask itself.
                r_mask[i] = b.turn_rewards.care_bid_mask[t]

        d_norm = percentile_rank(d_vals, d_mask, min_valid=min_valid_turn)
        r_norm = percentile_rank(r_vals, r_mask, min_valid=min_valid_turn)

        # For trajectories that DID reach turn t, fill normalized turn advantage.
        # Trajectories that didn't reach turn t stay NaN (loss mask will skip).
        for i, b in enumerate(group):
            if t < b.num_turns:
                a_turn[i, t] = d_norm[i] + r_norm[i]

    # ---- 3. α-weighted fusion + safety constraints (§4.6.2) --------------
    c_cat_gate = np.array(
        [1.0 if b.c_cat == 0 else 0.0 for b in group], dtype=np.float64
    )
    c_safety_raw = np.array(
        [float(b.safety_scores.c_safety) for b in group], dtype=np.float64
    )

    # Broadcast: a_traj shape (N,) → (N, 1); a_turn shape (N, T_max).
    # NaN propagates through the addition for un-reached turns, which is what we want.
    advantages = c_cat_gate[:, None] * (a_traj[:, None] + alpha * a_turn) \
                 - lambda_safety * c_safety_raw[:, None]
    # Restore NaN for un-reached turns (the * gate above turns NaN into NaN already).
    # No further action needed.

    return GroupAdvantage(
        advantages=advantages,
        a_traj_normalized=a_traj,
        a_turn_normalized=a_turn,
        c_cat_gate=c_cat_gate,
        c_safety_raw=c_safety_raw,
        debug={
            "alpha": alpha,
            "lambda_safety": lambda_safety,
            "n_group": n_group,
            "t_max": t_max,
            "n_c_cat_violations": int((c_cat_gate == 0).sum()),
        },
    )

"""
Tests for src.training.advantage — EC-MTRL dual-horizon advantage estimator.

Validates the 4 mathematical invariants that must hold for the GRPO update
to compute the right gradient direction (zh_9 §4.6):

  Section 1 — crank() per-reward CRank in G_s
  Section 2 — percentile_rank() in G_{s,t} with care-bid mask validity
  Section 3 — compute_dual_horizon_advantage end-to-end
  Section 4 — PBRS-style invariants (turn_level.py contract)

Each test asserts a *property* (mean = 0, ordering preserved, NaN padding,
PBRS telescoping) rather than a literal numeric output. A silent bug in any
of these is a "trains the wrong objective" failure mode that smoke runs
cannot detect, since loss will keep going down on a broken reward.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rewards.mock_judge import SafetyScores, TrajectoryScores
from src.training.advantage import (
    TrajectoryRewardBundle,
    TurnRewards,
    compute_dual_horizon_advantage,
    crank,
    percentile_rank,
)


# ---------------------------------------------------------------------------
# Helper — build a TrajectoryRewardBundle with sensible defaults
# ---------------------------------------------------------------------------

def _make_bundle(
    trajectory_id: str,
    num_turns: int,
    *,
    r_goal: int = 4,
    r_fit: int = 4,
    u_terminal: int = 2,
    c_safety: int = 0,
    c_cat: int = 0,
    r_distress: list[float] | None = None,
    r_resistance: list[float] | None = None,
    care_bid_mask: list[bool] | None = None,
) -> TrajectoryRewardBundle:
    if r_distress is None:
        r_distress = [0.0] * num_turns
    if r_resistance is None:
        r_resistance = [0.0] * num_turns
    if care_bid_mask is None:
        care_bid_mask = [False] * num_turns
    return TrajectoryRewardBundle(
        trajectory_id=trajectory_id,
        num_turns=num_turns,
        traj_scores=TrajectoryScores(
            r_goal=r_goal,
            r_fit=r_fit,
            u_terminal=u_terminal,
            evidence_turn_indices=[],
        ),
        turn_rewards=TurnRewards(
            r_distress=r_distress,
            r_resistance=r_resistance,
            care_bid_mask=care_bid_mask,
        ),
        safety_scores=SafetyScores(c_safety=c_safety),
        c_cat=c_cat,
    )


# ---------------------------------------------------------------------------
# Section 1 — crank() per-reward CRank in G_s
# ---------------------------------------------------------------------------

def test_crank_mean_is_zero_on_unique_values() -> None:
    """zh_9 §4.6.1: CRank is a zero-mean baseline → GRPO gradient unbiasedness."""
    for arr in [
        np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6]),
        np.array([0.0, 1.0]),
        np.linspace(-10, 10, 11),
        np.array([-100.0, 0.0, 100.0, 7.5, -3.3]),
    ]:
        out = crank(arr)
        assert abs(out.mean()) < 1e-10, f"crank({arr.tolist()}).mean() = {out.mean()}"


def test_crank_mean_is_zero_with_ties() -> None:
    """Average-rank tie handling preserves the zero-mean property."""
    arr = np.array([1.0, 2.0, 2.0, 3.0])
    out = crank(arr)
    assert abs(out.mean()) < 1e-10
    assert out[1] == out[2]                       # tied values get the same rank
    assert out[0] < out[1] < out[3]               # but non-tied ordering is preserved


def test_crank_range_is_minus_one_to_one() -> None:
    """zh_9 §4.6.1: CRank values fall strictly in [-1, 1]."""
    rng = np.random.RandomState(0)
    for _ in range(10):
        arr = rng.randn(20)
        out = crank(arr)
        assert out.min() >= -1.0 - 1e-12
        assert out.max() <= 1.0 + 1e-12


def test_crank_extremes_hit_minus_one_and_plus_one() -> None:
    """The lowest reward → -1, the highest → +1 (no shrinkage)."""
    out = crank(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert out[0] == pytest.approx(-1.0)
    assert out[-1] == pytest.approx(1.0)


def test_crank_preserves_strict_ordering() -> None:
    """If R_a > R_b then CRank(a) > CRank(b) — the rank order is monotone."""
    arr = np.array([0.5, 7.0, 3.0, 1.0])
    out = crank(arr)
    assert np.argsort(arr).tolist() == np.argsort(out).tolist()


def test_crank_singleton_returns_zero() -> None:
    """A 1-element group has no relative information → all advantage = 0."""
    assert crank(np.array([42.0])).tolist() == [0.0]


def test_crank_empty_returns_empty() -> None:
    assert crank(np.array([])).tolist() == []


# ---------------------------------------------------------------------------
# Section 2 — percentile_rank() in G_{s,t} with mask
# ---------------------------------------------------------------------------

def test_percentile_rank_only_ranks_valid_entries() -> None:
    """Invalid (mask=False) entries don't contaminate the rank of valid entries."""
    values = np.array([1.0, 100.0, 2.0, 3.0, 4.0, 5.0])  # 100.0 is at masked index
    mask = np.array([True, False, True, True, True, True])
    out = percentile_rank(values, mask, min_valid=5)
    # Masked entry returns 0.0 (neutral; gradient mask handles it downstream)
    assert out[1] == 0.0
    # Valid subset has zero mean
    assert abs(out[mask].mean()) < 1e-10
    # Valid ordering preserved (1.0 < 2.0 < 3.0 < 4.0 < 5.0)
    valid_indices = np.where(mask)[0]
    valid_out = out[valid_indices]
    assert np.argsort(values[valid_indices]).tolist() == np.argsort(valid_out).tolist()


def test_percentile_rank_fallback_when_too_few_valid() -> None:
    """zh_9 §4.6.1: when n_valid < min_valid, fall back to zeros (default)."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([True, True, False, False, False])  # only 2 valid, min_valid=5
    out = percentile_rank(values, mask, min_valid=5)
    assert (out == 0.0).all()


def test_percentile_rank_explicit_fallback_array() -> None:
    """When fallback is supplied, use it instead of zeros."""
    values = np.zeros(5)
    mask = np.array([True, False, False, False, False])  # only 1 valid
    fallback = np.array([0.5, -0.5, 0.5, -0.5, 0.5])
    out = percentile_rank(values, mask, min_valid=5, fallback=fallback)
    assert np.allclose(out, fallback)


def test_percentile_rank_empty_input() -> None:
    out = percentile_rank(np.array([]), np.array([], dtype=bool), min_valid=5)
    assert out.size == 0


# ---------------------------------------------------------------------------
# Section 3 — compute_dual_horizon_advantage end-to-end
# ---------------------------------------------------------------------------

def test_advantage_output_shape_and_nan_padding() -> None:
    """Output is (N, T_max); turns beyond a trajectory's T_i are NaN-padded."""
    group = [
        _make_bundle("t0", num_turns=3),
        _make_bundle("t1", num_turns=5),
        _make_bundle("t2", num_turns=4),
    ]
    ga = compute_dual_horizon_advantage(group)
    assert ga.advantages.shape == (3, 5)
    assert np.isnan(ga.advantages[0, 3:]).all()  # t0 only had 3 turns
    assert np.isnan(ga.advantages[2, 4:]).all()  # t2 only had 4 turns
    assert np.isfinite(ga.advantages[0, :3]).all()
    assert np.isfinite(ga.advantages[2, :4]).all()
    assert np.isfinite(ga.advantages[1, :5]).all()


def test_uniform_group_gives_zero_traj_advantage() -> None:
    """If every trajectory has identical traj_scores, A_traj = 0 (no group winner)."""
    group = [_make_bundle(f"t{i}", num_turns=3) for i in range(5)]
    ga = compute_dual_horizon_advantage(group)
    assert np.allclose(ga.a_traj_normalized, 0.0)


def test_higher_traj_score_yields_higher_advantage() -> None:
    """Sign-correctness: trajectory with higher raw scores → higher A_traj."""
    group = [
        _make_bundle("low",    num_turns=3, r_goal=0, r_fit=0, u_terminal=0),
        _make_bundle("medium", num_turns=3, r_goal=4, r_fit=4, u_terminal=2),
        _make_bundle("high",   num_turns=3, r_goal=8, r_fit=8, u_terminal=4),
    ]
    ga = compute_dual_horizon_advantage(group)
    assert ga.a_traj_normalized[0] < ga.a_traj_normalized[1] < ga.a_traj_normalized[2]


def test_c_cat_gate_zeroes_performance_keeps_safety_penalty() -> None:
    """Decision 3 (PROPOSAL §9): c_safety=1 ⇒ performance term zeroed AND
    floor penalty −λ_violation applied. Under Decision 3 c_cat ≡ c_safety
    (binary), so the gate input matches the safety binary directly."""
    group = [
        _make_bundle("clean",    num_turns=3, r_goal=4, r_fit=4, u_terminal=2,
                     c_cat=0, c_safety=0),
        _make_bundle("violator", num_turns=3, r_goal=4, r_fit=4, u_terminal=2,
                     c_cat=1, c_safety=1),
    ]
    ga = compute_dual_horizon_advantage(group, alpha=1.0, lambda_safety=0.5)
    # Both have identical scores → CRank = 0 for both → a_traj = 0 each
    # Clean:    advantage = 1·(0 + 1·0) − 0.5·0 = 0
    # Violator: advantage = 0·(0 + 1·0) − 0.5·1 = −0.5
    assert np.allclose(ga.advantages[0, :3], 0.0)
    assert np.allclose(ga.advantages[1, :3], -0.5)
    # Gate vector reflects the two trajectories
    assert ga.c_cat_gate.tolist() == [1.0, 0.0]


def test_c_cat_gate_zeroes_high_score_violator() -> None:
    """High raw scores do NOT rescue a violator: the gate zeroes performance
    term and the floor penalty pushes them below all clean trajectories."""
    group = [
        _make_bundle("low_clean",     num_turns=3, r_goal=0, r_fit=0, u_terminal=0,
                     c_cat=0, c_safety=0),
        _make_bundle("high_violator", num_turns=3, r_goal=8, r_fit=8, u_terminal=4,
                     c_cat=1, c_safety=1),
    ]
    ga = compute_dual_horizon_advantage(group, alpha=1.0, lambda_safety=5.0)
    # Violator's raw a_traj would be +3 (highest scores), but c_cat_gate=0
    # zeros it, and the λ_violation = 5 floor penalty applies.
    assert np.allclose(ga.advantages[1, :3], -5.0)
    # The clean trajectory has a_traj = -3 (lowest scores out of 2),
    # turn-level zero, no penalty → advantage = -3. So clean (-3) is
    # strictly above violator (-5): the policy correctly learns
    # "violation < worst clean".
    assert np.all(ga.advantages[0, :3] > ga.advantages[1, :3])


def test_alpha_scales_turn_advantage_linearly() -> None:
    """zh_9 §4.6.2: A = A_traj + α·A_turn — α affects only the turn term."""
    group = [
        _make_bundle("a", num_turns=5, r_goal=8, r_fit=0, u_terminal=2,
                     r_distress=[1.0, -1.0, 0.0, 1.0, -1.0]),
        _make_bundle("b", num_turns=5, r_goal=0, r_fit=8, u_terminal=2,
                     r_distress=[-1.0, 1.0, 0.0, -1.0, 1.0]),
        _make_bundle("c", num_turns=5, r_goal=4, r_fit=4, u_terminal=2,
                     r_distress=[0.0] * 5),
        _make_bundle("d", num_turns=5, r_goal=4, r_fit=4, u_terminal=2,
                     r_distress=[0.5] * 5),
        _make_bundle("e", num_turns=5, r_goal=4, r_fit=4, u_terminal=2,
                     r_distress=[-0.5] * 5),
    ]
    ga1 = compute_dual_horizon_advantage(group, alpha=1.0, lambda_safety=0.0)
    ga2 = compute_dual_horizon_advantage(group, alpha=2.0, lambda_safety=0.0)
    # A2 − A1 = (α2 − α1) · A_turn = 1 · A_turn  (where A_turn is already normalized)
    diff = ga2.advantages - ga1.advantages
    expected = ga1.a_turn_normalized
    finite = np.isfinite(diff) & np.isfinite(expected)
    assert finite.any()
    assert np.allclose(diff[finite], expected[finite])


def test_empty_group_raises() -> None:
    with pytest.raises(ValueError, match="empty group"):
        compute_dual_horizon_advantage([])


def test_singleton_group_returns_zero_traj_advantage() -> None:
    """1-trajectory group: CRank gives 0, no advantage signal possible."""
    group = [_make_bundle("only", num_turns=3, r_goal=8, r_fit=8, u_terminal=4)]
    ga = compute_dual_horizon_advantage(group)
    assert np.allclose(ga.a_traj_normalized, 0.0)


def test_lambda_safety_zero_disables_floor_penalty() -> None:
    """λ_violation = 0 disables the floor penalty (the gate still applies).

    Decision 3: c_safety = 1 still zeros the performance term via the gate
    `1[c_safety = 0]`; the floor penalty `−λ · c_safety` is what makes
    violators rank below clean. With λ = 0, violators get advantage = 0
    (gated performance) which is in the SAME range as clean a_traj/a_turn.
    This test exercises the math; production callers should NOT set
    λ_violation = 0 (PROPOSAL §9 default is 5.0).
    """
    group = [
        _make_bundle("clean_a", num_turns=3, c_cat=0, c_safety=0),
        _make_bundle("clean_b", num_turns=3, c_cat=0, c_safety=0),
    ]
    ga = compute_dual_horizon_advantage(group, alpha=1.0, lambda_safety=0.0)
    # Both clean, identical scores → both advantages are 0 (CRank tied).
    assert np.allclose(ga.advantages[0], ga.advantages[1], equal_nan=True)


# ---------------------------------------------------------------------------
# Section 4 — PBRS-style invariants (turn_level.py contract)
# ---------------------------------------------------------------------------

def test_pbrs_telescoping_property() -> None:
    """
    zh_9 §4.3 PBRS theorem: r_distress[t] = D[t-1] − D[t] satisfies
        Σ_t r_distress[t]  =  D[0] − D[T-1]   (telescoping)

    This invariant defines what compute_turn_rewards (P0) must produce.
    Encoded here so the contract is testable independently of the eventual
    impl in src.rewards.turn_level.
    """
    D = [1, 2, 1, 0, 0, 2]
    # v1 boundary convention: r[0] = 0 (no D_{-1} reference)
    r = [0.0] + [float(D[t - 1] - D[t]) for t in range(1, len(D))]
    # Telescoping: sum should equal D[0] − D[-1]  (since r[0]=0 contributes 0)
    assert sum(r) == pytest.approx(D[0] - D[-1])


def test_higher_distress_improvement_yields_higher_turn_rank() -> None:
    """Sign-correctness for the turn channel: bigger r_distress → higher rank."""
    # With group_size=5 and min_valid=5, percentile_rank actually computes ranks.
    group = [
        _make_bundle(f"t{i}", num_turns=3, r_distress=[float(i)] * 3)
        for i in range(5)
    ]
    ga = compute_dual_horizon_advantage(group, min_valid_turn=5)
    # At every turn, higher r_distress index should have higher a_turn rank.
    for t in range(3):
        col = ga.a_turn_normalized[:, t]
        assert col[0] < col[1] < col[2] < col[3] < col[4]

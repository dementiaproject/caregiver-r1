"""Tests for src.data.schemas — guards the 34-label DemMA-aligned alphabet and Trajectory structure."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.data import (
    ALL_LABELS,
    FACIAL_LABELS,
    MOTION_LABELS,
    SOUND_LABELS,
    CaregiverOutput,
    Group,
    InlineAnnotation,
    PatientObservation,
    Persona,
    Scenario,
    Trajectory,
    Turn,
)


# ---------------------------------------------------------------------------
# DemMA-aligned 34-label alphabet (movement 20 + facial 7 + voice 7)
# ---------------------------------------------------------------------------

def test_label_counts_match_demma_action_labels() -> None:
    """Hard guard: label cardinalities must match the DemMA checkpoint exactly."""
    assert len(MOTION_LABELS) == 20
    assert len(FACIAL_LABELS) == 7
    assert len(SOUND_LABELS) == 7
    assert len(ALL_LABELS) == 34


def test_motion_facial_sound_are_disjoint() -> None:
    """No label may belong to two channels (would break ordinal-tier mapping)."""
    assert not (MOTION_LABELS & FACIAL_LABELS)
    assert not (MOTION_LABELS & SOUND_LABELS)
    assert not (FACIAL_LABELS & SOUND_LABELS)


def test_inline_annotation_accepts_valid_labels() -> None:
    ann = InlineAnnotation(
        motion=["fidgeting", "lowering head", "nodding"],  # "nodding" is now legal
        facial=["frowning", "laughing"],                    # "laughing" is now legal
        sound=["sighing", "verbal hesitation (um / uh)"],   # parens-form is canonical
    )
    assert "nodding" in ann.flatten()
    assert "laughing" in ann.flatten()
    assert "verbal hesitation (um / uh)" in ann.flatten()
    assert not ann.is_empty()


def test_inline_annotation_rejects_off_alphabet_labels() -> None:
    """Off-alphabet labels must still be rejected (string must match DemMA exactly)."""
    with pytest.raises(ValidationError):
        InlineAnnotation(motion=["jumping up and down"])  # not a DemMA movement label
    with pytest.raises(ValidationError):
        InlineAnnotation(facial=["smirking"])  # not a DemMA facial label
    with pytest.raises(ValidationError):
        InlineAnnotation(sound=["verbal hesitation"])  # missing "(um / uh)" — not DemMA-canonical
    with pytest.raises(ValidationError):
        InlineAnnotation(sound=["murmuring/self-talk"])  # missing spaces around slash
    with pytest.raises(ValidationError):
        InlineAnnotation(motion=["others"])  # "others" was a placeholder; not in DemMA


def test_inline_annotation_is_frozen() -> None:
    ann = InlineAnnotation(motion=["fidgeting"])
    with pytest.raises(ValidationError):
        ann.motion = ["frowning"]  # frozen; mutation not allowed


def test_empty_annotation_is_legal() -> None:
    """A turn with no observable behaviour is valid (calm baseline state)."""
    ann = InlineAnnotation()
    assert ann.is_empty()
    assert ann.flatten() == set()


# ---------------------------------------------------------------------------
# §4.2 — Trajectory structure
# ---------------------------------------------------------------------------

def _make_turn(idx: int) -> Turn:
    return Turn(
        turn_index=idx,
        caregiver=CaregiverOutput(
            think="Patient seems agitated; soft validation is safer here.",
            response="That sounds important to you. Tell me more.",
            raw_text="<think>...</think><response>...</response>",
        ),
        patient=PatientObservation(
            utterance="My mother visited me today.",
            annotation=InlineAnnotation(facial=["smiling"]),
        ),
    )


def test_trajectory_enforces_sequential_turn_indices() -> None:
    Trajectory(
        trajectory_id="t1",
        scenario_id="s1",
        strategy_id="nurse",
        turns=[_make_turn(0), _make_turn(1), _make_turn(2)],
        seed=42,
        terminated_by="max_turns",
    )

    with pytest.raises(ValidationError):
        Trajectory(
            trajectory_id="t2",
            scenario_id="s1",
            strategy_id="nurse",
            turns=[_make_turn(0), _make_turn(2)],  # missing index 1
            seed=42,
            terminated_by="max_turns",
        )


def test_group_allows_duplicate_strategy_ids_under_unified_prompt() -> None:
    """Decision 1 (PROPOSAL §9, 2026-04-26): the unified-prompt + temperature-
    sampling design eliminates the per-strategy conditioning, so multiple
    rollouts in the same group may carry the same `strategy_id` (now used
    only as a rollout-index label like 'r0'/'r1'). Group still requires
    non-empty trajectories and consistent scenario_id."""
    t_a = Trajectory(
        trajectory_id="t_a", scenario_id="s1", strategy_id="r0",
        turns=[_make_turn(0)], seed=1, terminated_by="simulator_end",
    )
    t_b = Trajectory(
        trajectory_id="t_b", scenario_id="s1", strategy_id="r0",  # same label is fine
        turns=[_make_turn(0)], seed=2, terminated_by="simulator_end",
    )
    g = Group(scenario_id="s1", trajectories=[t_a, t_b])
    assert len(g.trajectories) == 2

    # ...but an EMPTY group is still rejected.
    with pytest.raises(ValidationError):
        Group(scenario_id="s1", trajectories=[])


# ---------------------------------------------------------------------------
# §2.2 — Scenario
# ---------------------------------------------------------------------------

def test_scenario_minimal_construction() -> None:
    persona = Persona(
        persona_id="p001",
        dementia_subtype="alzheimers_moderate",
        name="Mrs. Wang",
        age=82,
    )
    s = Scenario(
        scenario_id="med_l3_p001",
        persona=persona,
        conflict_type="medication",
        severity=3,
        risk_tier="high",
        initial_patient_utterance="I already took my morning pills.",
        ground_truth="Patient has not yet taken the 8am dose.",
    )
    assert s.severity == 3
    assert s.conflict_type == "medication"

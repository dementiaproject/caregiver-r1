"""Tests for src.data.demma_client — DemMA mock client + JSON parser."""

from __future__ import annotations

import json

import pytest

from src.data import (
    ALL_LABELS,
    DemMAMockClient,
    DialogueHistoryItem,
    InlineAnnotation,
    Persona,
    parse_demma_json_response,
)


@pytest.fixture
def persona() -> Persona:
    return Persona(
        persona_id="p001",
        dementia_subtype="alzheimers_moderate",
        name="Mrs. Wang",
        age=82,
    )


# ---------------------------------------------------------------------------
# Mock client behaviour
# ---------------------------------------------------------------------------

def test_mock_client_step_returns_schema_valid_observation(persona: Persona) -> None:
    client = DemMAMockClient()
    obs = client.step(persona=persona, history=[], latest_caregiver_response="Hello.", seed=0)

    # Annotation labels must all be from the 18-label alphabet.
    flat = obs.annotation.flatten()
    assert flat <= ALL_LABELS, f"mock emitted off-alphabet labels: {flat - ALL_LABELS}"
    assert isinstance(obs.utterance, str) and obs.utterance


def test_mock_client_is_deterministic_given_same_seed_and_history(persona: Persona) -> None:
    """zh_9 §3.3 anti-collapse requires per-rollout seeds; we want reproducibility per seed."""
    a = DemMAMockClient().step(persona=persona, history=[], latest_caregiver_response="Hi.", seed=7)
    b = DemMAMockClient().step(persona=persona, history=[], latest_caregiver_response="Hi.", seed=7)
    assert a.utterance == b.utterance
    assert a.annotation.flatten() == b.annotation.flatten()


def test_mock_client_diverges_across_seeds(persona: Persona) -> None:
    """Different seeds should give different outputs (otherwise group=10 would collapse)."""
    obs_per_seed = {
        seed: DemMAMockClient(distress_drift=0.5).step(
            persona=persona, history=[], latest_caregiver_response="Hello.", seed=seed
        )
        for seed in range(20)
    }
    unique_outputs = {(o.utterance, frozenset(o.annotation.flatten())) for o in obs_per_seed.values()}
    assert len(unique_outputs) >= 5, "mock should produce variation across at least 5 seeds out of 20"


def test_mock_client_escalates_under_harsh_caregiver_response(persona: Persona) -> None:
    """Smoke check: confrontational caregiver text should bias mock toward distress labels.

    This is what lets Phase E.3 smoke-run see the C_cat (P3 coercion → next-turn distress)
    pattern even before the real DemMA endpoint is up.
    """
    client = DemMAMockClient(distress_drift=0.5)
    distress_label_set = {"crying", "groaning in pain", "pushing caregiver away", "frowning"}

    harsh_distress_count = 0
    soft_distress_count = 0
    for seed in range(40):
        harsh = client.step(persona=persona, history=[], latest_caregiver_response="No, you are wrong.", seed=seed)
        soft = client.step(persona=persona, history=[], latest_caregiver_response="That sounds nice.", seed=seed)
        if harsh.annotation.flatten() & distress_label_set:
            harsh_distress_count += 1
        if soft.annotation.flatten() & distress_label_set:
            soft_distress_count += 1

    assert harsh_distress_count >= soft_distress_count, (
        f"harsh response should not produce fewer distress labels than soft; "
        f"got harsh={harsh_distress_count}, soft={soft_distress_count}"
    )


def test_mock_client_health_check(persona: Persona) -> None:
    assert DemMAMockClient().health_check() is True


def test_mock_client_handles_growing_history(persona: Persona) -> None:
    """A rollout has up to 10 turns; the mock must not crash on long histories."""
    client = DemMAMockClient()
    history: list[DialogueHistoryItem] = []
    for turn_idx in range(10):
        obs = client.step(
            persona=persona,
            history=history,
            latest_caregiver_response=f"caregiver turn {turn_idx}",
            seed=turn_idx,
        )
        history.append(
            DialogueHistoryItem(
                caregiver_response=f"caregiver turn {turn_idx}",
                patient_utterance=obs.utterance,
                patient_annotation=obs.annotation,
            )
        )
    assert len(history) == 10


# ---------------------------------------------------------------------------
# JSON parser (used by real vLLM client + Phase B.1 raw-data audit)
# ---------------------------------------------------------------------------

def test_parse_demma_json_response_valid() -> None:
    raw = json.dumps({
        "utterance": "I want to go home.",
        "annotation": {
            "motion": ["fidgeting"],
            "facial": ["frowning", "avoiding eye contact"],
            "sound": ["sighing"],
        },
    })
    obs = parse_demma_json_response(raw)
    assert obs.utterance == "I want to go home."
    assert "fidgeting" in obs.annotation.motion
    assert "avoiding eye contact" in obs.annotation.facial


def test_parse_demma_json_response_rejects_off_alphabet() -> None:
    raw = json.dumps({
        "utterance": "...",
        "annotation": {"motion": ["dancing"], "facial": [], "sound": []},  # "dancing" not in zh_9 §2.3
    })
    with pytest.raises(Exception):
        parse_demma_json_response(raw)


def test_parse_demma_json_response_rejects_missing_keys() -> None:
    raw = json.dumps({"utterance": "..."})  # missing "annotation"
    with pytest.raises(ValueError):
        parse_demma_json_response(raw)


def test_parse_demma_json_response_rejects_non_object() -> None:
    raw = json.dumps(["not", "an", "object"])
    with pytest.raises(ValueError):
        parse_demma_json_response(raw)


def test_parse_demma_json_response_handles_empty_annotation() -> None:
    """Calm baseline turn — DemMA may legitimately emit an annotation with no labels."""
    raw = json.dumps({
        "utterance": "Yes, that's fine.",
        "annotation": {"motion": [], "facial": [], "sound": []},
    })
    obs = parse_demma_json_response(raw)
    assert obs.annotation.is_empty()

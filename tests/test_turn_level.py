"""
Tests for src.rewards.turn_level — zh_9 §4.3 PBRS turn-level reward channel.

Covers the contracts that compute_turn_rewards must satisfy for the GRPO
update to receive a correctly-shaped, sign-correct reward signal:

  Section 1 — distress tier (multi-label sum form)
  Section 2 — resistance tier (RTC priority match)
  Section 3 — care-bid mask (keyword + regex; <response>-only scope)
  Section 4 — PBRS telescoping invariant on real fixtures
  Section 5 — boundary handling (severity-conditioned D[-1])
  Section 6 — care-bid mask gating of r_resistance
  Section 7 — input validation
  Section 8 — YAML coverage of all 34 DemMA labels (no orphans)
"""

from __future__ import annotations

import pytest

from src.data.schemas import ALL_LABELS, InlineAnnotation
from src.rewards.turn_level import (
    _label_buckets,
    compute_care_bid_mask,
    compute_distress_tier,
    compute_resistance_tier,
    compute_turn_rewards,
    initial_distress_baseline,
)


# ---------------------------------------------------------------------------
# Section 1 — distress tier
# ---------------------------------------------------------------------------

def test_distress_severe_single_hit() -> None:
    """zh_9 §4.3: a single severe label fires D=3."""
    assert compute_distress_tier(InlineAnnotation(sound=["crying"])) == 3


def test_distress_moderate_single_hit() -> None:
    """zh_9 §4.3 example: single moderate label → D=1."""
    assert compute_distress_tier(InlineAnnotation(facial=["frowning"])) == 1


def test_distress_two_moderate_hits() -> None:
    """zh_9 §4.3 example: ≥2 moderate cues → D=2 (multi-label sum form)."""
    ann = InlineAnnotation(facial=["frowning"], sound=["sighing"])
    assert compute_distress_tier(ann) == 2


def test_distress_three_moderate_hits() -> None:
    """Multi-label sum: 3 moderate hits → D=3 (saturates at 3)."""
    ann = InlineAnnotation(
        facial=["frowning"],
        sound=["sighing"],
        motion=["lowering head"],
    )
    assert compute_distress_tier(ann) == 3


def test_distress_positive_subtracts() -> None:
    """frowning + smiling → D=0 (positive cancels moderate, multi-label sum form)."""
    ann = InlineAnnotation(facial=["frowning", "smiling"])
    assert compute_distress_tier(ann) == 0


def test_distress_dementia_baseline_does_not_inflate() -> None:
    """Bug 1 fix: vacant_expression + staring_blankly + repetitive_words = D=0."""
    ann = InlineAnnotation(
        facial=["vacant expression", "staring blankly"],
        sound=["repetitive words"],
    )
    assert compute_distress_tier(ann) == 0


def test_distress_dementia_baseline_alongside_moderate() -> None:
    """Bug 1 fix: dementia baseline labels do NOT add to a moderate cue."""
    ann = InlineAnnotation(facial=["vacant expression", "frowning"])
    assert compute_distress_tier(ann) == 1  # only frowning counts


def test_distress_text_reinforcement_lifts_moderate() -> None:
    """zh_9 §4.3: distress text keyword + 1 moderate cue → D=2."""
    ann = InlineAnnotation(facial=["frowning"])
    assert compute_distress_tier(ann, patient_text="Just leave me alone.") == 2


def test_distress_clipped_at_three() -> None:
    """Multiple severe labels still clip at 3 (4-tier ordinal)."""
    ann = InlineAnnotation(
        sound=["crying", "groaning in pain"],
        motion=["pushing caregiver away", "throwing objects"],
    )
    assert compute_distress_tier(ann) == 3


def test_distress_empty_annotation_is_zero() -> None:
    assert compute_distress_tier(InlineAnnotation()) == 0


# ---------------------------------------------------------------------------
# Section 2 — resistance tier
# ---------------------------------------------------------------------------

def test_resistance_physical_refusal() -> None:
    assert compute_resistance_tier(InlineAnnotation(motion=["pushing caregiver away"])) == 3


def test_resistance_verbal_refusal_from_label() -> None:
    assert compute_resistance_tier(InlineAnnotation(motion=["shaking head"])) == 2


def test_resistance_verbal_refusal_from_text() -> None:
    """Patient text 'no' / 'i won't' lifts hesitation cue to verbal refusal."""
    ann = InlineAnnotation(motion=["stepping back"])
    assert compute_resistance_tier(ann, patient_text="No, I refuse.") == 2


def test_resistance_hesitation() -> None:
    assert compute_resistance_tier(InlineAnnotation(motion=["freezing mid-step"])) == 1


def test_resistance_accepting_default() -> None:
    assert compute_resistance_tier(InlineAnnotation(motion=["nodding"])) == 0


def test_resistance_priority_physical_over_hesitation() -> None:
    """If both physical and hesitation labels fire, physical wins (priority match)."""
    ann = InlineAnnotation(motion=["pushing caregiver away", "stepping back"])
    assert compute_resistance_tier(ann) == 3


def test_resistance_covering_ears_only_in_distress_not_resistance() -> None:
    """Bug 2 fix: covering ears is sensory escape (severe distress), NOT RTC refusal."""
    ann = InlineAnnotation(motion=["covering ears"])
    assert compute_distress_tier(ann) == 3       # still severe distress
    assert compute_resistance_tier(ann) == 0     # but NOT physical refusal


# ---------------------------------------------------------------------------
# Section 3 — care-bid mask
# ---------------------------------------------------------------------------

def test_care_bid_keyword_match() -> None:
    assert compute_care_bid_mask("Time to take your medication.") is True


def test_care_bid_imperative_pattern() -> None:
    assert compute_care_bid_mask("Please come with me to the dining room.") is True


def test_care_bid_factual_correction_pattern() -> None:
    assert compute_care_bid_mask("Actually, today is Tuesday.") is True


def test_care_bid_chitchat_does_not_match() -> None:
    assert compute_care_bid_mask("It's a beautiful day outside.") is False


def test_care_bid_empty_response() -> None:
    assert compute_care_bid_mask("") is False


# ---------------------------------------------------------------------------
# Section 4 — PBRS telescoping invariant on real fixture
# ---------------------------------------------------------------------------

def _ann_for_distress_tier(target: int) -> InlineAnnotation:
    """Construct a minimal annotation that yields a given distress tier."""
    if target == 0:
        return InlineAnnotation()
    if target == 1:
        return InlineAnnotation(facial=["frowning"])
    if target == 2:
        return InlineAnnotation(facial=["frowning"], sound=["sighing"])
    if target == 3:
        return InlineAnnotation(sound=["crying"])
    raise ValueError(f"target distress tier must be 0..3, got {target}")


def test_pbrs_telescoping_on_real_fixture() -> None:
    """zh_9 §4.3 PBRS theorem: Σ_t r_distress = D[-1]_pseudo − D[T-1]."""
    D_target = [1, 2, 1, 0, 0, 2]
    annotations = [_ann_for_distress_tier(d) for d in D_target]
    patient_texts = [""] * len(D_target)
    caregiver_responses = ["chitchat"] * len(D_target)  # b_t = False everywhere
    severity = 2  # → pseudo D[-1] = 1

    rewards = compute_turn_rewards(
        annotations=annotations,
        patient_texts=patient_texts,
        caregiver_responses=caregiver_responses,
        severity=severity,
    )

    # Sanity: the constructed annotations actually produce the targeted tiers
    assert all(compute_distress_tier(a) == d for a, d in zip(annotations, D_target))

    # PBRS telescoping property
    expected_sum = initial_distress_baseline(severity) - D_target[-1]
    assert sum(rewards.r_distress) == pytest.approx(expected_sum)


# ---------------------------------------------------------------------------
# Section 5 — severity-conditioned initial baseline
# ---------------------------------------------------------------------------

def test_severity_conditioned_initial_baseline_values() -> None:
    """Higher conflict severity → higher pseudo D[-1]."""
    assert initial_distress_baseline(1) == 0
    assert initial_distress_baseline(2) == 1
    assert initial_distress_baseline(3) == 2


def test_severity_changes_first_turn_reward() -> None:
    """First-turn r_distress depends on severity-conditioned baseline."""
    annotation = InlineAnnotation()  # → D_0 = 0
    for severity, expected in [(1, 0), (2, 1), (3, 2)]:
        rewards = compute_turn_rewards(
            annotations=[annotation],
            patient_texts=[""],
            caregiver_responses=["chitchat"],
            severity=severity,
        )
        assert rewards.r_distress[0] == expected, f"severity={severity}"


# ---------------------------------------------------------------------------
# Section 6 — care-bid mask gating
# ---------------------------------------------------------------------------

def test_care_bid_off_zeroes_resistance_reward() -> None:
    """zh_9 §4.3: when b_t=False, r_resistance must be 0 even if R changes."""
    annotations = [
        InlineAnnotation(motion=["nodding"]),                   # R=0
        InlineAnnotation(motion=["pushing caregiver away"]),    # R=3, big jump
    ]
    rewards = compute_turn_rewards(
        annotations=annotations,
        patient_texts=["", ""],
        caregiver_responses=["nice weather", "lovely day"],     # neither is care-bid
        severity=1,
    )
    assert rewards.care_bid_mask == [False, False]
    assert rewards.r_resistance == [0.0, 0.0]


def test_care_bid_on_propagates_resistance_reward() -> None:
    """When b_t=True and R changes, r_resistance follows the state delta."""
    annotations = [
        InlineAnnotation(motion=["pushing caregiver away"]),    # R=3
        InlineAnnotation(motion=["nodding"]),                   # R=0
    ]
    rewards = compute_turn_rewards(
        annotations=annotations,
        patient_texts=["", ""],
        caregiver_responses=["take your medication", "let's go to the dining room"],
        severity=2,
    )
    assert rewards.care_bid_mask == [True, True]
    # turn 0: r_resistance = 1·(R_prev=0 − R_0=3) = -3
    # turn 1: r_resistance = 1·(R_prev=3 − R_1=0) = +3
    assert rewards.r_resistance == [-3.0, 3.0]


# ---------------------------------------------------------------------------
# Section 7 — input validation
# ---------------------------------------------------------------------------

def test_compute_turn_rewards_empty_input() -> None:
    rewards = compute_turn_rewards([], [], [], severity=1)
    assert rewards.r_distress == []
    assert rewards.r_resistance == []
    assert rewards.care_bid_mask == []


def test_compute_turn_rewards_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        compute_turn_rewards(
            annotations=[InlineAnnotation()],
            patient_texts=["a", "b"],
            caregiver_responses=["c"],
            severity=1,
        )


# ---------------------------------------------------------------------------
# Section 8 — YAML coverage (every DemMA label is accounted for)
# ---------------------------------------------------------------------------

def test_yaml_labels_match_demma_alphabet_exactly() -> None:
    """Every label in clinical_anchors.yaml must exist in DemMA's 34-label alphabet."""
    buckets = _label_buckets()
    yaml_labels = (
        buckets["severe"] | buckets["moderate"] | buckets["positive"]
        | buckets["excluded"]
    )
    unknown = yaml_labels - ALL_LABELS
    assert not unknown, f"clinical_anchors.yaml references off-alphabet labels: {unknown}"


def test_every_demma_label_has_a_distress_classification() -> None:
    """All 34 DemMA labels must appear in exactly one distress bucket (incl. excluded).

    This guards against silent regressions where a new DemMA label gets added to
    schemas.py but forgotten in clinical_anchors.yaml — without this check the
    new label would silently contribute 0 to D_t (an undocumented exclusion).
    """
    buckets = _label_buckets()
    distress_buckets = ["severe", "moderate", "positive", "excluded"]
    classified = (
        buckets["severe"] | buckets["moderate"]
        | buckets["positive"] | buckets["excluded"]
    )
    missing = ALL_LABELS - classified
    assert not missing, f"DemMA labels missing from clinical_anchors.yaml: {missing}"

    # No double-classification within distress buckets
    for i, b1 in enumerate(distress_buckets):
        for b2 in distress_buckets[i + 1:]:
            overlap = buckets[b1] & buckets[b2]
            assert not overlap, f"label(s) in both '{b1}' and '{b2}': {overlap}"


def test_resistance_buckets_are_disjoint() -> None:
    """Within resistance, each label appears in at most one tier."""
    buckets = _label_buckets()
    res_buckets = ["physical_refusal", "verbal_refusal", "hesitation", "accepting"]
    for i, b1 in enumerate(res_buckets):
        for b2 in res_buckets[i + 1:]:
            overlap = buckets[b1] & buckets[b2]
            assert not overlap, f"resistance label(s) in both '{b1}' and '{b2}': {overlap}"

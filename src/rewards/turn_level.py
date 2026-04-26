"""
Turn-level reward channel (zh_9 §4.3) — DemMA inline annotation → ordinal
patient-state tier → PBRS state-delta reward.

Pipeline per turn t:

    annotation, patient_text  ──▶  D_t ∈ {0,1,2,3}    (compute_distress_tier)
    annotation, patient_text  ──▶  R_t ∈ {0,1,2,3}    (compute_resistance_tier)
    caregiver_response        ──▶  b_t ∈ {True, False}(compute_care_bid_mask)

Then over the whole trajectory (compute_turn_rewards):

    r_distress[t]   = D[t-1] − D[t]
    r_resistance[t] = b_t · (R[t-1] − R[t])

For the first turn t=0, we use a severity-conditioned pseudo-baseline
D[-1] (zh_9 §4.3 boundary); higher conflict severity → higher implicit
opening distress, so the first caregiver utterance can be credited or
penalized for opening the conversation well or badly.

D_t aggregation (multi-label sum form, zh_9 §4.3):

    raw = 3·|labels ∩ severe| + 1·|labels ∩ moderate| − 1·|labels ∩ positive|
    if patient_text triggers any distress reinforcement keyword: raw += 1
    D_t = clip(raw, 0, 3)

R_t aggregation (priority-based; first match wins):

    R_t = 3 if any physical_refusal label
    R_t = 2 if any verbal_refusal label  OR text contains verbal_refusal keyword
    R_t = 1 if any hesitation label
    R_t = 0 otherwise

Label → tier classification rules and clinical anchors are externalized in
`clinical_anchors.yaml` so a clinical reviewer can audit them line-by-line
without reading Python (zh_9 §4.3 / appendix B.3). The output dataclass
matches `src.training.advantage.TurnRewards` exactly so this module plugs
straight into the GRPO advantage pipeline.

PBRS theoretical anchor (zh_9 §4.3): r_distress[t] = φ(s_{t-1}) − φ(s_t)
with φ(s) := −D(s) is a potential-based shaping reward. This is the
contract verified in `tests.test_advantage.test_pbrs_telescoping_property`
and `tests.test_turn_level.test_pbrs_telescoping_on_real_fixture`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from src.data.schemas import InlineAnnotation, Severity
from src.training.advantage import TurnRewards


# ---------------------------------------------------------------------------
# YAML loader (LRU-cached so test fixtures don't pay the I/O each call)
# ---------------------------------------------------------------------------

_ANCHORS_PATH = Path(__file__).parent / "clinical_anchors.yaml"


@lru_cache(maxsize=1)
def _load_anchors() -> dict:
    with _ANCHORS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _label_buckets() -> dict[str, frozenset[str]]:
    """Materialize each YAML bucket into a frozenset for O(1) membership tests."""
    a = _load_anchors()
    return {
        "severe":           frozenset(a["distress"]["severe"]),
        "moderate":         frozenset(a["distress"]["moderate"]),
        "positive":         frozenset(a["distress"]["positive"]),
        "excluded":         frozenset(a["distress"]["excluded_from_distress"]),
        "physical_refusal": frozenset(a["resistance"]["physical_refusal"]),
        "verbal_refusal":   frozenset(a["resistance"]["verbal_refusal"]),
        "hesitation":       frozenset(a["resistance"]["hesitation"]),
        "accepting":        frozenset(a["resistance"]["accepting"]),
    }


@lru_cache(maxsize=1)
def _care_bid_resources() -> tuple[tuple[str, ...], tuple[re.Pattern, ...]]:
    a = _load_anchors()["care_bid"]
    keywords = tuple(k.lower() for k in a["keywords"])
    patterns = tuple(re.compile(p) for p in a["patterns"])
    return keywords, patterns


@lru_cache(maxsize=1)
def _distress_text_keywords() -> tuple[str, ...]:
    return tuple(k.lower() for k in _load_anchors()["distress_text_keywords"])


@lru_cache(maxsize=1)
def _verbal_refusal_text_keywords() -> tuple[str, ...]:
    return tuple(k.lower() for k in _load_anchors()["verbal_refusal_text_keywords"])


# ---------------------------------------------------------------------------
# Per-turn tier computation
# ---------------------------------------------------------------------------

def compute_distress_tier(annotation: InlineAnnotation, patient_text: str = "") -> int:
    """zh_9 §4.3: D_t ∈ {0,1,2,3} from inline annotation + (optional) patient text.

    Multi-label sum form preserves intensity information that disjunction
    would lose (e.g. frowning + sighing + lowering_head should give a
    stronger distress signal than a single frowning).
    """
    buckets = _label_buckets()
    labels = annotation.flatten()
    raw = (
        3 * len(labels & buckets["severe"])
        + 1 * len(labels & buckets["moderate"])
        - 1 * len(labels & buckets["positive"])
    )
    if patient_text and any(k in patient_text.lower() for k in _distress_text_keywords()):
        raw += 1
    return max(0, min(3, raw))


def compute_resistance_tier(annotation: InlineAnnotation, patient_text: str = "") -> int:
    """zh_9 §4.3: R_t ∈ {0,1,2,3} via RTC-anchored priority match (first wins)."""
    buckets = _label_buckets()
    labels = annotation.flatten()
    if labels & buckets["physical_refusal"]:
        return 3
    if labels & buckets["verbal_refusal"]:
        return 2
    if patient_text and any(k in patient_text.lower() for k in _verbal_refusal_text_keywords()):
        return 2
    if labels & buckets["hesitation"]:
        return 1
    return 0


def compute_care_bid_mask(caregiver_response: str) -> bool:
    """zh_9 §4.3: b_t ∈ {True, False} — RTC measurement convention.

    True iff the caregiver utterance carries an explicit care attempt
    (medication request / fact correction / redirection / direct directive).
    Conservative keyword + regex stub; appendix C.2 audit may upgrade to a
    light classifier if recall is too low.
    """
    if not caregiver_response:
        return False
    text_lower = caregiver_response.lower()
    keywords, patterns = _care_bid_resources()
    if any(kw in text_lower for kw in keywords):
        return True
    if any(p.search(caregiver_response) for p in patterns):
        return True
    return False


# ---------------------------------------------------------------------------
# Boundary handling — pseudo D[-1] for the first turn
# ---------------------------------------------------------------------------

def initial_distress_baseline(severity: Severity) -> int:
    """zh_9 §4.3 boundary: pseudo D[-1] for the first-turn PBRS reference.

    Severity-conditioned so the opening turn carries signal proportional to
    the implicit conflict pressure:
        severity=1 (mild)     → D[-1] = 0   (calm baseline)
        severity=2 (moderate) → D[-1] = 1   (mild opening agitation)
        severity=3 (severe)   → D[-1] = 2   (clear opening distress)
    """
    return {1: 0, 2: 1, 3: 2}.get(int(severity), 0)


# ---------------------------------------------------------------------------
# Trajectory-level reward computation (PBRS state-delta, zh_9 §4.3)
# ---------------------------------------------------------------------------

def compute_turn_rewards(
    annotations: list[InlineAnnotation],
    patient_texts: list[str],
    caregiver_responses: list[str],
    severity: Severity,
) -> TurnRewards:
    """zh_9 §4.3: full per-trajectory turn-level reward (PBRS state-delta form).

        r_distress[t]   = D[t-1] − D[t]
        r_resistance[t] = b_t · (R[t-1] − R[t])

    For t=0 the previous-state reference is `initial_distress_baseline(severity)`
    for distress and 0 (accepting) for resistance.

    The output type matches `src.training.advantage.TurnRewards` exactly,
    so this drops directly into `compute_dual_horizon_advantage`.

    Args:
        annotations:         per-turn DemMA InlineAnnotation, length T
        patient_texts:       per-turn DemMA utterance, length T
        caregiver_responses: per-turn caregiver <response> only (do NOT pass
                             <think>; predicate scope is `<response>`-only,
                             zh_9 §4.5), length T
        severity:            scenario.severity ∈ {1,2,3}; sets D[-1] pseudo-baseline

    Returns:
        TurnRewards(r_distress, r_resistance, care_bid_mask), each list of length T.
    """
    T = len(annotations)
    if T == 0:
        return TurnRewards(r_distress=[], r_resistance=[], care_bid_mask=[])
    if not (len(patient_texts) == T and len(caregiver_responses) == T):
        raise ValueError(
            f"length mismatch: annotations={T}, patient_texts={len(patient_texts)}, "
            f"caregiver_responses={len(caregiver_responses)}"
        )

    # 1. Compute per-turn tiers and care-bid mask
    D = [compute_distress_tier(annotations[t], patient_texts[t]) for t in range(T)]
    R = [compute_resistance_tier(annotations[t], patient_texts[t]) for t in range(T)]
    b = [compute_care_bid_mask(caregiver_responses[t]) for t in range(T)]

    # 2. PBRS state-delta with severity-conditioned pseudo-baseline at t=0
    d_prev = initial_distress_baseline(severity)
    r_prev = 0  # RTC accepting baseline (no resistance pre-conversation)

    r_distress: list[float] = []
    r_resistance: list[float] = []
    for t in range(T):
        r_distress.append(float(d_prev - D[t]))
        r_resistance.append(float(b[t]) * float(r_prev - R[t]))
        d_prev = D[t]
        r_prev = R[t]

    return TurnRewards(
        r_distress=r_distress,
        r_resistance=r_resistance,
        care_bid_mask=b,
    )


__all__ = [
    "compute_distress_tier",
    "compute_resistance_tier",
    "compute_care_bid_mask",
    "compute_turn_rewards",
    "initial_distress_baseline",
]

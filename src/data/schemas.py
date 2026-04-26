"""
Canonical data schemas for Caregiver-R1.

All downstream modules (DemMA client, rollout, reward, SFT extraction, eval)
must import their data types from here. This file is the single source of
truth for:

  - DemMA inline annotation labels (34, see below — 1:1 aligned with the
    DemMA-Planner-SFT checkpoint's ACTION_LABELS dict)
  - Per-turn observation structures
  - RL Trajectory structure (zh_9 §4.2)
  - Scenario specification (zh_9 §2.2 / §2.4)

Label alphabet — IMPORTANT NOTE on |L|:
  - zh_9 §1.4 property 3 / §2.3 / appendix B states |L| = 18.
  - DemMA's actual checkpoint (hulehule/DemMA-Planner-SFT) emits 34 labels
    (movement 20 + facial_expression 7 + voice 7). The 18 in zh_9 was an
    earlier estimate; we adopt the truth-aligned |L| = 34 here and zh_9
    will be updated in a follow-up pass (§1.4, §2.3, appendix B).
  - DemMA channel name → schema field mapping:
        movement           → motion
        facial_expression  → facial
        voice              → sound

The auditable mapping (zh_9 §4.3 ordinal tier rules) covers all 34 labels
with clinical anchors from OERS / PAINAD / RTC.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Inline annotation labels (frozen, 34 total — DemMA ACTION_LABELS-aligned)
# ---------------------------------------------------------------------------

class MotionLabel(str, Enum):
    """20 movement labels emitted by DemMA action_classifier."""
    STANDING_UP = "standing up"
    STEPPING_BACK = "stepping back"
    FREEZING_MID_STEP = "freezing mid-step"
    RUBBING_FINGERS = "rubbing fingers"
    FIDDLING_WITH_CLOTHING = "fiddling with clothing"
    TOUCHING_FOREHEAD = "touching forehead"
    CLENCHING_FIST = "clenching fist"
    SLAPPING_TABLE = "slapping table"
    SHAKING_HANDS = "shaking hands"
    NODDING = "nodding"
    SHAKING_HEAD = "shaking head"
    LOWERING_HEAD = "lowering head"
    LOOKING_AROUND = "looking around"
    THROWING_OBJECTS = "throwing objects"
    PACING_BACK_AND_FORTH = "pacing back and forth"
    FIDGETING = "fidgeting"
    GRIPPING_ARMREST_TIGHTLY = "gripping armrest tightly"
    COVERING_EARS = "covering ears"
    HOLDING_CAREGIVERS_HAND = "holding caregiver's hand"
    PUSHING_CAREGIVER_AWAY = "pushing caregiver away"


class FacialLabel(str, Enum):
    """7 facial-expression labels emitted by DemMA action_classifier."""
    AVOIDING_EYE_CONTACT = "avoiding eye contact"
    STARING_BLANKLY = "staring blankly"
    FROWNING = "frowning"
    SMILING = "smiling"
    LAUGHING = "laughing"
    VACANT_EXPRESSION = "vacant expression"
    VERY_SURPRISED = "very surprised (wow)"


class SoundLabel(str, Enum):
    """7 voice/sound labels emitted by DemMA action_classifier."""
    SIGHING = "sighing"
    VERBAL_HESITATION = "verbal hesitation (um / uh)"
    MURMURING_SELF_TALK = "murmuring / self-talk"
    SILENCE_FOR_SEVERAL_SECONDS = "silence for several seconds"
    CRYING = "crying"
    REPETITIVE_WORDS = "repetitive words"
    GROANING_IN_PAIN = "groaning in pain"


# Exposed as stringified sets for fast membership tests against parsed text.
MOTION_LABELS: frozenset[str] = frozenset(m.value for m in MotionLabel)
FACIAL_LABELS: frozenset[str] = frozenset(f.value for f in FacialLabel)
SOUND_LABELS: frozenset[str] = frozenset(s.value for s in SoundLabel)
ALL_LABELS: frozenset[str] = MOTION_LABELS | FACIAL_LABELS | SOUND_LABELS

assert len(MOTION_LABELS) == 20, "DemMA emits exactly 20 movement labels"
assert len(FACIAL_LABELS) == 7, "DemMA emits exactly 7 facial labels"
assert len(SOUND_LABELS) == 7, "DemMA emits exactly 7 voice labels"
assert len(ALL_LABELS) == 34, "DemMA's full alphabet is exactly 34 labels"


# ---------------------------------------------------------------------------
# DemMA channel-name <-> schema-field-name mapping
# ---------------------------------------------------------------------------
# DemMA inference code uses "movement" / "facial_expression" / "voice".
# Our InlineAnnotation schema uses "motion" / "facial" / "sound" (zh_9 §2.3
# wording). DemMARealClient applies this mapping at the boundary.

DEMMA_TO_SCHEMA_CHANNEL: dict[str, str] = {
    "movement": "motion",
    "facial_expression": "facial",
    "voice": "sound",
}
SCHEMA_TO_DEMMA_CHANNEL: dict[str, str] = {v: k for k, v in DEMMA_TO_SCHEMA_CHANNEL.items()}


class InlineAnnotation(BaseModel):
    """
    DemMA's per-turn inline annotation (zh_9 §1.4 / §2.3).

    Produced by the DemMA simulator in the same forward pass that emits the
    patient utterance — this is the inline-commitment property (zh_9 §1.4).
    Each list contains zero or more labels from the frozen alphabet of 34
    (DemMA ACTION_LABELS-aligned: motion 20 + facial 7 + sound 7).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    motion: list[str] = Field(default_factory=list)
    facial: list[str] = Field(default_factory=list)
    sound: list[str] = Field(default_factory=list)

    @field_validator("motion")
    @classmethod
    def _check_motion(cls, v: list[str]) -> list[str]:
        bad = [x for x in v if x not in MOTION_LABELS]
        if bad:
            raise ValueError(f"unknown motion labels: {bad}; allowed = {sorted(MOTION_LABELS)}")
        return v

    @field_validator("facial")
    @classmethod
    def _check_facial(cls, v: list[str]) -> list[str]:
        bad = [x for x in v if x not in FACIAL_LABELS]
        if bad:
            raise ValueError(f"unknown facial labels: {bad}; allowed = {sorted(FACIAL_LABELS)}")
        return v

    @field_validator("sound")
    @classmethod
    def _check_sound(cls, v: list[str]) -> list[str]:
        bad = [x for x in v if x not in SOUND_LABELS]
        if bad:
            raise ValueError(f"unknown sound labels: {bad}; allowed = {sorted(SOUND_LABELS)}")
        return v

    def flatten(self) -> set[str]:
        """All labels emitted on this turn, regardless of channel."""
        return set(self.motion) | set(self.facial) | set(self.sound)

    def is_empty(self) -> bool:
        return not (self.motion or self.facial or self.sound)


# ---------------------------------------------------------------------------
# §2.1 — Conflict taxonomy (5 types × 3 severity, frozen)
# ---------------------------------------------------------------------------

ConflictType = Literal["temporal", "identity", "event", "spatial", "medication"]
Severity = Literal[1, 2, 3]
RiskTier = Literal["low", "medium", "high"]

# zh_9 §4.5 P1/P2 trigger definition: Medication and Identity are high-risk
# because they can lead to direct safety consequences (medication errors,
# acute grief reactions) on dose/identity-confusion.
HIGH_RISK_CONFLICT_TYPES: frozenset[str] = frozenset({"medication", "identity"})


# ---------------------------------------------------------------------------
# §2.2 / §2.4 — Scenario specification
# ---------------------------------------------------------------------------

class Persona(BaseModel):
    """Patient persona — held fixed across an entire dialogue (latent variable u)."""
    model_config = ConfigDict(extra="allow")  # DemMA personas may carry custom fields

    persona_id: str
    dementia_subtype: str               # one of DemMA's 9 subtypes
    name: str | None = None
    age: int | None = None
    background: str | None = None       # free-text biographical context
    cognitive_features: dict[str, str] = Field(default_factory=dict)


class Scenario(BaseModel):
    """
    A single training/eval scenario (zh_9 §2.2 q = (u, conflict_type, severity, risk_tier)).

    One scenario is rolled out 10 times during training (group=10), once per
    clinical strategy prompt (zh_9 §3, §4.2).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    persona: Persona
    conflict_type: ConflictType
    severity: Severity
    risk_tier: RiskTier
    initial_patient_utterance: str       # the conflict-bearing first utterance
    ground_truth: str                    # the actual fact (used by judge / safety, not by agent)
    notes: str | None = None             # optional clinical/ethical context for judges


# ---------------------------------------------------------------------------
# §4.2 — Per-turn structure of an RL trajectory
# ---------------------------------------------------------------------------

class CaregiverOutput(BaseModel):
    """
    Caregiver agent's R1-style two-layer output for one turn (zh_9 §4.2).

    The DemMA simulator only sees `response`; `think` is the caregiver's
    internal chain-of-thought and is NOT forwarded into the dialogue stream
    (zh_9 §4.2 train/inference boundary).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    think: str            # contents inside <think>...</think>
    response: str         # contents inside <response>...</response>
    raw_text: str         # original model output, before XML parsing

    # Token boundaries (filled by the rollout layer, used by Phase E.2 loss mask)
    think_token_span: tuple[int, int] | None = None
    response_token_span: tuple[int, int] | None = None
    format_token_ids: list[int] = Field(default_factory=list)  # XML tag tokens to mask out


class PatientObservation(BaseModel):
    """DemMA's response on one turn — utterance + inline annotation."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    utterance: str
    annotation: InlineAnnotation


class Turn(BaseModel):
    """One full turn of caregiver↔DemMA interaction (zh_9 §4.2)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    turn_index: int
    caregiver: CaregiverOutput
    patient: PatientObservation


# ---------------------------------------------------------------------------
# §4.2 — Full RL trajectory
# ---------------------------------------------------------------------------

class Trajectory(BaseModel):
    """
    One full caregiver↔DemMA dialogue rollout (zh_9 §4.2).

    A `group` of 10 such trajectories is bundled per scenario, one per
    clinical strategy prompt, for GDPO advantage normalization.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    trajectory_id: str
    scenario_id: str
    # Decision 1 (PROPOSAL §9, 2026-04-26): unified system prompt + temperature
    # sampling for group diversity — strategy_id is no longer a conditioning
    # variable, only a rollout-index label (e.g. "r0".."r9") kept here for
    # logging / W&B compatibility. Defaults to "r0" so legacy callers keep
    # working without changes.
    strategy_id: str = "r0"
    turns: list[Turn]
    seed: int                  # DemMA-side random seed (zh_9 §3.3 anti-collapse)
    terminated_by: Literal["max_turns", "simulator_end"]

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @field_validator("turns")
    @classmethod
    def _check_turn_indices(cls, v: list[Turn]) -> list[Turn]:
        for expected, t in enumerate(v):
            if t.turn_index != expected:
                raise ValueError(f"turn indices must be 0,1,2,...; got {[t.turn_index for t in v]}")
        return v


class Group(BaseModel):
    """
    A strategy-conditioned group of trajectories for one scenario (zh_9 §4.2).

    Group size is fixed at 10 in default/flagship training (zh_9 §4.2 hard
    constraint — GDPO extension ③ requires the full 10-strategy contextual
    prior); PoC tier may run with 5 for pipeline validation only.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    trajectories: list[Trajectory]

    @field_validator("trajectories")
    @classmethod
    def _check_nonempty(cls, v: list[Trajectory]) -> list[Trajectory]:
        # Decision 1 (PROPOSAL §9, 2026-04-26): unified prompt + temperature
        # sampling — uniqueness across `strategy_id` is no longer required
        # (was an artefact of the strategy-conditioned design). We just check
        # the group is non-empty.
        if not v:
            raise ValueError("group must contain at least one trajectory")
        return v

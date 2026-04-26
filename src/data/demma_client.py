"""
DemMA simulator client (zh_9 §2.3, appendix D.3).

DemMA is a frozen LLM acting as the dementia-patient simulator. The caregiver
agent interacts with it turn-by-turn; on each turn DemMA emits a patient
utterance plus an inline annotation (zh_9 §1.4 inline-commitment property).

This module exposes a stable interface that downstream rollout code uses.
Two implementations:

  - `DemMAVLLMClient`:   talks to a vLLM-served frozen DemMA checkpoint via
                         the OpenAI-compatible HTTP endpoint.
  - `DemMAMockClient`:   returns schema-valid random annotations + canned
                         utterances. Used for CI, unit tests, and offline
                         development before the real DemMA endpoint is up.

Both implement the abstract `DemMAClient` interface so the rollout code is
agnostic to deployment form.
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.data.schemas import (
    FACIAL_LABELS,
    MOTION_LABELS,
    SOUND_LABELS,
    InlineAnnotation,
    PatientObservation,
    Persona,
)


# ---------------------------------------------------------------------------
# History format the rollout layer hands to DemMA
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DialogueHistoryItem:
    """One past exchange in the dialogue, from DemMA's point of view.

    Note: DemMA only sees `caregiver_response` (the <response> segment),
    not the caregiver's <think> — per zh_9 §4.2 dialog protocol.
    """
    caregiver_response: str
    patient_utterance: str
    patient_annotation: InlineAnnotation


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class DemMAClient(ABC):
    """Stable interface every DemMA backend must implement."""

    @abstractmethod
    def step(
        self,
        persona: Persona,
        history: list[DialogueHistoryItem],
        latest_caregiver_response: str,
        seed: int,
    ) -> PatientObservation:
        """Run DemMA for one turn.

        Args:
            persona:                    fixed patient persona (held across turns).
            history:                    list of (caregiver, patient) exchanges so far.
            latest_caregiver_response:  the caregiver's <response> for the current turn.
            seed:                       per-rollout seed (zh_9 §3.3 anti-collapse).

        Returns:
            PatientObservation with utterance + InlineAnnotation, both produced
            in the same forward pass (zh_9 §1.4 inline commitment).
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return True iff the backend is reachable and DemMA's checkpoint hash matches."""


# ---------------------------------------------------------------------------
# vLLM offline-LM-mode DemMA
# ---------------------------------------------------------------------------
# `DemMAVLLMClient` lives in src/data/demma_vllm_client.py; importing here is
# lazy via a module-level re-export so `from src.data.demma_client import
# DemMAVLLMClient` works without forcing vLLM at import time.


def __getattr__(name):
    if name == "DemMAVLLMClient":
        from src.data.demma_vllm_client import DemMAVLLMClient as _C
        return _C
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# Mock DemMA for CI / offline development
# ---------------------------------------------------------------------------

# Canned patient utterances by approximate distress level (used when mock
# annotation flips to high-distress labels — keeps mock outputs internally
# consistent so downstream reward code sees plausible joint distributions).
_CANNED_UTTERANCES_CALM = (
    "Yes, that sounds nice.",
    "Mm, I see.",
    "Tell me more about that.",
    "Oh, all right then.",
)

_CANNED_UTTERANCES_AGITATED = (
    "No, no, that's not right at all.",
    "But I know what I saw.",
    "Don't tell me what to do.",
    "I don't understand what you're saying.",
)

_CANNED_UTTERANCES_DISTRESSED = (
    "Please... I just want to go home.",
    "Why won't you listen to me?",
    "It hurts. It hurts so much.",
    "I'm scared. Where is everyone?",
)


class DemMAMockClient(DemMAClient):
    """Deterministic-given-seed mock that returns schema-valid annotations."""

    def __init__(self, distress_drift: float = 0.15) -> None:
        """
        Args:
            distress_drift: per-turn probability of escalating to a higher
                            distress tier. Lets rollout code exercise the
                            full state-delta reward range without a real
                            DemMA endpoint.
        """
        self.distress_drift = distress_drift

    def health_check(self) -> bool:
        return True

    def step(
        self,
        persona: Persona,
        history: list[DialogueHistoryItem],
        latest_caregiver_response: str,
        seed: int,
    ) -> PatientObservation:
        # Deterministic per (seed, turn_index) so unit tests are reproducible.
        # Python 3.14 disallows tuple seeds; compose into a single int instead.
        rng = random.Random(int(seed) * 1_000_003 + len(history))

        # Compute a coarse "tension" level from history: more turns + last
        # caregiver response containing harsh keywords → escalate.
        tension = min(0.8, 0.1 + 0.05 * len(history))
        if any(kw in latest_caregiver_response.lower() for kw in ("no,", "you must", "wrong", "actually")):
            tension += 0.15
        tension = min(0.95, tension + rng.random() * self.distress_drift)

        annotation = self._sample_annotation(rng, tension)
        utterance = self._sample_utterance(rng, tension)
        return PatientObservation(utterance=utterance, annotation=annotation)

    @staticmethod
    def _sample_annotation(rng: random.Random, tension: float) -> InlineAnnotation:
        """Sample a schema-valid annotation whose label distribution tracks `tension`.

        Label strings here MUST exactly match DemMA's ACTION_LABELS dict — see
        src/data/schemas.py (MotionLabel / FacialLabel / SoundLabel enums).
        """
        # Higher tension → more negative-valence labels (frowning, sighing, crying).
        negative_facial = ["frowning", "avoiding eye contact", "vacant expression"]
        positive_facial = ["smiling", "laughing"]
        negative_motion = [
            "lowering head", "fidgeting", "pushing caregiver away",
            "stepping back", "covering ears", "clenching fist",
        ]
        neutral_motion = ["looking around", "touching forehead", "nodding"]
        negative_sound = [
            "sighing", "verbal hesitation (um / uh)", "murmuring / self-talk",
            "crying", "groaning in pain",
        ]
        neutral_sound = ["repetitive words", "silence for several seconds"]

        def maybe(p: float, options: list[str], k: int = 1) -> list[str]:
            if rng.random() < p:
                return rng.sample(options, k=min(k, len(options)))
            return []

        motion_pool = negative_motion if tension > 0.5 else neutral_motion
        facial_pool = negative_facial if tension > 0.4 else positive_facial
        sound_pool = negative_sound if tension > 0.5 else neutral_sound

        annotation = InlineAnnotation(
            motion=maybe(0.6, motion_pool),
            facial=maybe(0.7, facial_pool),
            sound=maybe(0.5, sound_pool),
        )

        # Defensive: drop any label that somehow leaked outside the allowed set.
        annotation = InlineAnnotation(
            motion=[m for m in annotation.motion if m in MOTION_LABELS],
            facial=[f for f in annotation.facial if f in FACIAL_LABELS],
            sound=[s for s in annotation.sound if s in SOUND_LABELS],
        )
        return annotation

    @staticmethod
    def _sample_utterance(rng: random.Random, tension: float) -> str:
        if tension > 0.7:
            return rng.choice(_CANNED_UTTERANCES_DISTRESSED)
        if tension > 0.4:
            return rng.choice(_CANNED_UTTERANCES_AGITATED)
        return rng.choice(_CANNED_UTTERANCES_CALM)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_demma_json_response(raw: str) -> PatientObservation:
    """
    Parse a JSON-mode DemMA response string into a validated PatientObservation.

    Used by `DemMAVLLMClient` once the real endpoint is up. Exposed here so
    Phase B.1 `data_audit_report` can re-use it on raw DemMA dialog dumps.
    """
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"DemMA response must be a JSON object, got {type(obj).__name__}")
    if "utterance" not in obj or "annotation" not in obj:
        raise ValueError(f"DemMA response missing required keys; got keys={list(obj)}")
    annotation = InlineAnnotation.model_validate(obj["annotation"])
    return PatientObservation(utterance=str(obj["utterance"]), annotation=annotation)

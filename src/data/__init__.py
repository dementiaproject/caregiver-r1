"""Canonical data layer: schemas + DemMA client.

Downstream modules should import from here, not from sibling subpackages,
to ensure a single source of truth for the 34-label alphabet (DemMA-aligned;
zh_9 §2.3 will be updated to |L|=34 in a follow-up pass) and the
Trajectory structure (zh_9 §4.2).
"""

from src.data.demma_client import (
    DemMAClient,
    DemMAMockClient,
    DialogueHistoryItem,
    parse_demma_json_response,
)

# DemMARealClient and DemMAVLLMClient are imported lazily (via accessor
# functions) because their modules pull in torch/transformers/vllm at use
# time; we don't want to force those on CPU-only dev boxes that just want
# the mock client.

def get_demma_real_client_cls():
    """Lazy accessor for DemMARealClient — defers torch/transformers import."""
    from src.data.demma_real_client import DemMARealClient  # noqa: WPS433
    return DemMARealClient


def get_demma_vllm_client_cls():
    """Lazy accessor for DemMAVLLMClient — defers vllm import."""
    from src.data.demma_vllm_client import DemMAVLLMClient  # noqa: WPS433
    return DemMAVLLMClient
from src.data.schemas import (
    ALL_LABELS,
    DEMMA_TO_SCHEMA_CHANNEL,
    FACIAL_LABELS,
    HIGH_RISK_CONFLICT_TYPES,
    MOTION_LABELS,
    SCHEMA_TO_DEMMA_CHANNEL,
    SOUND_LABELS,
    CaregiverOutput,
    ConflictType,
    FacialLabel,
    Group,
    InlineAnnotation,
    MotionLabel,
    PatientObservation,
    Persona,
    RiskTier,
    Scenario,
    Severity,
    SoundLabel,
    Trajectory,
    Turn,
)

__all__ = [
    # Schemas — labels
    "MotionLabel", "FacialLabel", "SoundLabel",
    "MOTION_LABELS", "FACIAL_LABELS", "SOUND_LABELS", "ALL_LABELS",
    "DEMMA_TO_SCHEMA_CHANNEL", "SCHEMA_TO_DEMMA_CHANNEL",
    "InlineAnnotation",
    # Schemas — taxonomy
    "ConflictType", "Severity", "RiskTier", "HIGH_RISK_CONFLICT_TYPES",
    # Schemas — scenario
    "Persona", "Scenario",
    # Schemas — rollout
    "CaregiverOutput", "PatientObservation", "Turn", "Trajectory", "Group",
    # DemMA client
    "DemMAClient", "DemMAMockClient",
    "DialogueHistoryItem", "parse_demma_json_response",
    "get_demma_real_client_cls", "get_demma_vllm_client_cls",
]

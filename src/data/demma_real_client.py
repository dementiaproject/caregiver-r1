"""
DemMARealClient — transformers in-process wrapper around the published
DemMA-Planner-SFT checkpoint (https://huggingface.co/hulehule/DemMA-Planner-SFT).

This is a thin adapter that exposes the same DemMAClient interface used by
the rest of the rollout pipeline (src.data.demma_client.DemMAClient), so
swapping mock ↔ real DemMA is a one-line config change in smoke_run.yaml.

Provenance
----------
The forward-pass logic (prompt construction, two-step generate-then-classify,
per-category top-1 thresholding, action vocab) is a faithful re-implementation
of the reference scripts shipped inside the DemMA HF repo:

    hulehule/DemMA-Planner-SFT/inference_with_mem.py
    hulehule/DemMA-Planner-SFT/demma_planner_inference.py

Differences vs the reference scripts (deliberate):
  - Wraps the loop in a class with a stable `step(history, response, seed)`
    signature matching DemMAClient.
  - Lazy-loads the model on first `step()`, keeping `import` GPU-free.
  - Maps DemMA's channel names {movement / facial_expression / voice}
    onto our InlineAnnotation field names {motion / facial / sound}.
  - Honors a per-rollout seed via torch.manual_seed (reference is unseeded).
  - Hard-codes a single patient profile (Jacob Johnson, AD-early, id=0)
    for the smoke run; multi-patient sweep deferred to Phase F.
  - Returns a schema-validated PatientObservation, not a raw dict.

Caveats (zh_9 §1.4 / §6.4 — disclose in the paper)
--------------------------------------------------
1. The action labels come from a SEPARATE classifier head over the LLM's
   final hidden state — not from autoregressive token output. This is what
   the inline-commitment property formally relies on (same forward pass,
   joint conditioning), but is NOT "the LLM literally writes the labels in
   text". We will state this explicitly in §1.4 and appendix B.
2. DemMA's patient utterance is decoded from a `[SPEAK]` block embedded in
   a longer `[PLAN] / [SPEAK] / [ACT]` scaffold. We strip everything except
   the `[SPEAK]` content before returning to the caregiver, so the caregiver
   sees clean dialogue (zh_9 §4.2 dialog protocol).
3. If generation truncates and `[SPEAK]` is empty, we return "..." with an
   empty annotation — recoverable, the caregiver can probe further next turn.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any

from src.data.demma_client import DemMAClient, DialogueHistoryItem
from src.data.schemas import (
    DEMMA_TO_SCHEMA_CHANNEL,
    InlineAnnotation,
    PatientObservation,
    Persona,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action label vocabulary — MUST match DemMA-Planner-SFT inference_with_mem.py
# ---------------------------------------------------------------------------

ACTION_LABELS: dict[str, list[str]] = {
    "movement": [
        "standing up", "stepping back", "freezing mid-step", "rubbing fingers",
        "fiddling with clothing", "touching forehead", "clenching fist", "slapping table",
        "shaking hands", "nodding", "shaking head", "lowering head", "looking around",
        "throwing objects", "pacing back and forth", "fidgeting", "gripping armrest tightly",
        "covering ears", "holding caregiver's hand", "pushing caregiver away",
    ],
    "facial_expression": [
        "avoiding eye contact", "staring blankly", "frowning", "smiling",
        "laughing", "vacant expression", "very surprised (wow)",
    ],
    "voice": [
        "sighing", "verbal hesitation (um / uh)", "murmuring / self-talk",
        "silence for several seconds", "crying", "repetitive words", "groaning in pain",
    ],
}


def _build_action_vocab() -> tuple[dict[int, tuple[str, str]], int]:
    """Mirror inference_with_mem.build_action_vocab — order MUST match the trained classifier."""
    idx_to_label: dict[int, tuple[str, str]] = {}
    idx = 0
    for category, labels in ACTION_LABELS.items():
        for label in labels:
            idx_to_label[idx] = (category, label)
            idx += 1
    return idx_to_label, idx


IDX_TO_LABEL, NUM_ACTION_LABELS = _build_action_vocab()
assert NUM_ACTION_LABELS == 34, "DemMA classifier head was trained for exactly 34 labels"


# ---------------------------------------------------------------------------
# Default patient profile — Jacob Johnson, AD-early, id=0
# ---------------------------------------------------------------------------
# Copied verbatim from inference_with_mem.main_interactive(). For the smoke
# run we hold this fixed (PI: "persona 我们固定一个就行了"). Multi-persona
# sweep is Phase F.

DEFAULT_PATIENT_ID: int = 0
DEFAULT_PATIENT_PROFILE: dict[str, Any] = {
    "persona": "AD-early",
    "icf_b126_profile": {
        "extraversion": -1,
        "agreeableness": -1,
        "conscientiousness": -2,
        "emotional_stability": -2,
    },
    "memory_profile": {
        "deficit_type": "encoding_deficit",
        "has_recent_episodic": False,    # AD-early: encoding deficit blocks recent episodic memory
        "has_remote_episodic": True,     # AD-early: remote autobiographical memory preserved
        "benefits_from_cues": False,
    },
}


# ---------------------------------------------------------------------------
# Memory loaders
# ---------------------------------------------------------------------------

def _load_long_memory(path: Path, patient_id: int) -> dict[str, Any] | None:
    """Long-term memory is a JSON array; pick the entry whose `id == patient_id`."""
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    return next((m for m in records if m.get("id") == patient_id), None)


def _load_short_memory(path: Path, patient_id: int) -> dict[str, Any] | None:
    """Short-term memory is JSONL; one record per patient, picked by `id`."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            mem = json.loads(line)
            if mem.get("id") == patient_id:
                return mem
    return None


# ---------------------------------------------------------------------------
# Prompt construction (faithful copy of inference_with_mem.build_inference_prompt)
# ---------------------------------------------------------------------------

def _build_inference_prompt(
    persona: str,
    icf_b126_profile: dict[str, int],
    memory_profile: dict[str, Any],
    patient_memories: dict[str, Any],
    dialogue_history: list[dict[str, str]],
    caregiver_utterance: str,
    use_classifier: bool = True,
) -> str:
    """Mirror inference_with_mem.build_inference_prompt; do NOT edit the scaffold."""
    icf = icf_b126_profile
    mem = memory_profile

    prompt = (
        f"Patient: {persona}\n"
        f"ICF: E={icf['extraversion']}, A={icf['agreeableness']}, "
        f"C={icf['conscientiousness']}, ES={icf['emotional_stability']}\n"
        f"Memory Deficit: {mem['deficit_type']}, "
        f"Recent={'✓' if mem.get('has_recent_episodic') else '✗'}, "
        f"Cues={'yes' if mem.get('benefits_from_cues') else 'no'}\n\n"
    )

    # Memory gating block — preserves DemMA's clinical fidelity.
    prompt += "Accessible Memories:\n"
    if mem.get("has_remote_episodic") and patient_memories.get("long_term"):
        long_mem = patient_memories["long_term"]
        prompt += "Remote Memories (long-term accessible):\n"
        prompt += f"- Family: Spouse {long_mem.get('spouse', 'N/A')}, Children {', '.join(long_mem.get('children', []))}\n"
        for p in long_mem.get("paragraphs", [])[:3]:
            prompt += f"- {p['topic']}: {p['text'][:120]}...\n"
    else:
        prompt += "Remote Memories: Not accessible\n"

    if mem.get("has_recent_episodic") and patient_memories.get("short_term"):
        short_mem = patient_memories["short_term"]
        prompt += f"\nRecent Memories (short-term accessible):\n"
        prompt += f"- Daily routine: {short_mem.get('text', '')[:200]}...\n"
    else:
        prompt += "\nRecent Memories: Cannot access (encoding deficit blocks recent memory formation)\n"

    prompt += "\nHistory:\n"
    history = dialogue_history[-5:] if len(dialogue_history) > 5 else dialogue_history
    if history:
        for turn in history:
            utterance = turn["utterance"]
            if turn["speaker"] == "Patient":
                # Strip any leftover "[Movement: ...]" tail from training data formatting.
                utterance = re.sub(r"\s*\[Movement:.*?\]\s*$", "", utterance).strip()
            prompt += f"{turn['speaker']}: {utterance}\n"
    else:
        prompt += "(First turn)\n"

    prompt += f"\nCaregiver: {caregiver_utterance}"

    # Two scaffolding modes:
    #  - classifier mode: [ACT] is filled by the MLP head (official path);
    #    we tell the model the labels are predicted by classifier so it does
    #    not hallucinate text there.
    #  - LM-only mode:   we ask the model to emit category lines as text
    #    (movement / facial_expression / voice). Faster (single forward) but
    #    relies on indirect teacher-forcing exposure since [ACT] tokens were
    #    not in the SFT CE-loss mask.
    if use_classifier:
        act_block = "[ACT]\n(predicted by classifier)"
    else:
        # LM-only mode: SFT did not put [ACT] tokens in CE loss, so the model
        # learned the FORMAT but not the canonical 34-label vocabulary.
        # Without explicit guidance it hallucinates natural-language descriptors
        # like "sitting still" / "worried" / "soft". We therefore inline the
        # entire allowed vocabulary so the LM can copy from it.
        movement_vocab = ", ".join(ACTION_LABELS["movement"])
        facial_vocab = ", ".join(ACTION_LABELS["facial_expression"])
        voice_vocab = ", ".join(ACTION_LABELS["voice"])
        act_block = (
            "[ACT]\n"
            "(For each category, list 0 or more labels separated by commas. "
            "Choose ONLY from the canonical lists below; do not invent new strings.)\n"
            f"movement choices: {movement_vocab}\n"
            f"facial_expression choices: {facial_vocab}\n"
            f"voice choices: {voice_vocab}\n\n"
            "Now emit (use exact strings from the lists above):\n"
            "movement: <labels>\n"
            "facial_expression: <labels>\n"
            "voice: <labels>"
        )

    prompt += (
        "\n\nGenerate in this structure:\n"
        "[PLAN]\n"
        "Emotion: <emotion>(I=<intensity>)\n"
        "Memory: <accessibility> | <deficit_type>\n"
        "Intent: <intent>\n"
        "Reason: <reason>\n\n"
        "[SPEAK]\n"
        "<patient utterance>\n\n"
        + act_block
    )
    return prompt


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_speak_block(generated_text: str) -> str:
    """Extract the [SPEAK] block content; return '...' on empty/truncated generations."""
    m = re.search(r"\[SPEAK\](.*?)(?:\[ACT\]|$)", generated_text, re.DOTALL)
    if not m:
        return "..."
    body = m.group(1).strip()
    return body if body else "..."


# ---- LM-only (no-classifier) action parsing -------------------------------
# When use_classifier=False we read `[ACT]` block from the generated text and
# match each comma-separated label against the canonical 34-way DemMA vocab.
# Labels not in vocab are dropped (the SFT data only ever exposed these exact
# strings, so anything else is hallucination).

# Build {category -> set of valid labels} once.
_VALID_LABELS_BY_CAT: dict[str, set[str]] = {
    cat: set(labels) for cat, labels in ACTION_LABELS.items()
}


def _parse_act_block(generated_text: str) -> InlineAnnotation:
    """Parse `[ACT]` block from generated text into an InlineAnnotation.

    Expected lines (one per category, optional):
        movement: standing up, frowning
        facial_expression: avoiding eye contact
        voice: sighing
    """
    by_channel: dict[str, list[str]] = {"motion": [], "facial": [], "sound": []}
    m = re.search(r"\[ACT\](.*?)(?:<\|im_end\|>|<\|endoftext\|>|$)",
                  generated_text, re.DOTALL)
    if not m:
        return InlineAnnotation(motion=[], facial=[], sound=[])
    act_text = m.group(1).strip()

    for raw_line in act_text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        cat_str, _, labels_str = line.partition(":")
        cat = cat_str.strip().lower().replace(" ", "_")
        if cat not in _VALID_LABELS_BY_CAT:
            continue
        valid = _VALID_LABELS_BY_CAT[cat]
        schema_channel = DEMMA_TO_SCHEMA_CHANNEL[cat]
        for raw in labels_str.split(","):
            label = raw.strip().rstrip(".").lower()
            # Try exact match against canonical (preserve original casing).
            for canon in valid:
                if canon.lower() == label:
                    by_channel[schema_channel].append(canon)
                    break

    return InlineAnnotation(
        motion=by_channel["motion"],
        facial=by_channel["facial"],
        sound=by_channel["sound"],
    )


def _decode_actions_to_annotation(probs, threshold: float) -> InlineAnnotation:
    """Per-category top-1 with threshold (matches inference_with_mem.decode_action_predictions),
    then map DemMA channel names → schema field names."""
    by_channel: dict[str, list[str]] = {"motion": [], "facial": [], "sound": []}
    for category in ACTION_LABELS:
        indices = [i for i, (c, _) in IDX_TO_LABEL.items() if c == category]
        best_idx = max(indices, key=lambda i: float(probs[i]))
        if float(probs[best_idx]) >= threshold:
            schema_channel = DEMMA_TO_SCHEMA_CHANNEL[category]
            _, label = IDX_TO_LABEL[best_idx]
            by_channel[schema_channel].append(label)
    return InlineAnnotation(
        motion=by_channel["motion"],
        facial=by_channel["facial"],
        sound=by_channel["sound"],
    )


# ---------------------------------------------------------------------------
# Real DemMA client
# ---------------------------------------------------------------------------

class DemMARealClient(DemMAClient):
    """
    Transformers in-process wrapper for hulehule/DemMA-Planner-SFT.

    Holds a single patient profile (default: Jacob Johnson, AD-early, id=0)
    across the lifetime of the client. The Persona argument from `step()` is
    accepted for interface compatibility but IGNORED — DemMA's identity is
    fixed by `patient_id` at construction time. A warning is emitted once if
    the rollout layer passes a Persona whose `persona_id` mismatches.

    Lazy load
    ---------
    The model + classifier are not loaded until the first `step()` call (or
    explicit `.ensure_loaded()`), so:
      - `from src.data.demma_real_client import DemMARealClient` is GPU-free
      - smoke import on a CPU dev box does not OOM
      - GPU init cost is paid once, on demand

    GPU footprint
    -------------
    Qwen3-8B in FP16 ≈ 16 GB VRAM with KV cache headroom; fits on a single
    A100/H100/L40S. The classifier head adds < 100 MB. For the smoke run on
    a single 24 GB consumer card, set `device="cuda:0"` and `max_new_tokens
    <= 400` (DemMA's default).
    """

    def __init__(
        self,
        model_path: Path | str,
        classifier_path: Path | str,
        long_memory_path: Path | str,
        short_memory_path: Path | str,
        patient_id: int = DEFAULT_PATIENT_ID,
        patient_profile: dict[str, Any] | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_new_tokens: int = 400,
        action_threshold: float = 0.35,
        use_classifier: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.classifier_path = Path(classifier_path)
        self.long_memory_path = Path(long_memory_path)
        self.short_memory_path = Path(short_memory_path)
        self.patient_id = patient_id
        self.patient_profile: dict[str, Any] = patient_profile or DEFAULT_PATIENT_PROFILE
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.action_threshold = action_threshold
        self.use_classifier = use_classifier

        # Lazy-loaded
        self._tokenizer = None
        self._base_model = None
        self._action_classifier = None
        self._patient_memories: dict[str, Any] | None = None
        self._loaded = False
        self._load_error: BaseException | None = None

        # Persona-mismatch warning is one-shot to avoid log spam.
        self._warned_persona_mismatch = False

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def health_check(self) -> bool:
        """File-presence check only — does NOT load the model (cheap & GPU-free)."""
        for f in (self.model_path, self.classifier_path,
                  self.long_memory_path, self.short_memory_path):
            if not f.exists():
                log.error("DemMA real-client missing artifact: %s", f)
                return False
        return True

    def ensure_loaded(self) -> None:
        """Idempotent model load. Call before profiling / before first step()."""
        if self._loaded:
            return
        if self._load_error is not None:
            raise self._load_error
        try:
            self._load()
        except BaseException as e:
            self._load_error = e
            raise
        self._loaded = True

    def _load(self) -> None:
        """Heavy-load: tokenizer + Qwen3-8B base model + 4-layer MLP classifier head."""
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "DemMARealClient requires torch + transformers. "
                "Install via: pip install -r requirements-demma.txt"
            ) from e

        log.info("loading DemMA tokenizer + base model from %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
            self.dtype, torch.bfloat16
        )

        # Flash-Attention 2 is the canonical fast-path used by dema_sft_train.py;
        # fall back to PyTorch SDPA (still ~2x faster than eager) if the wheel
        # is missing.
        attn_impl = "sdpa"
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            log.info("flash_attention_2 detected — enabling")
        except ImportError:
            log.info("flash_attention_2 not installed — using SDPA")

        self._base_model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=self.device,
            attn_implementation=attn_impl,
        )
        self._base_model.eval()

        # 4-layer MLP head — architecture MUST match dema_sft_train.py / inference_with_mem.py.
        # Skipped entirely in LM-only mode (use_classifier=False) for speed.
        if self.use_classifier:
            hidden_size = self._base_model.config.hidden_size
            self._action_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, NUM_ACTION_LABELS),
            )
            state = torch.load(str(self.classifier_path), map_location=self.device)
            self._action_classifier.load_state_dict(state)
            self._action_classifier = self._action_classifier.to(self.device).to(torch_dtype)
            self._action_classifier.eval()
        else:
            log.info("LM-only mode: skipping action_classifier load (parsing [ACT] from text)")
            self._action_classifier = None

        log.info("loading patient memories (id=%d)", self.patient_id)
        long_mem = _load_long_memory(self.long_memory_path, self.patient_id)
        short_mem = _load_short_memory(self.short_memory_path, self.patient_id)
        if long_mem is None:
            warnings.warn(
                f"no long-term memory entry for patient_id={self.patient_id}; "
                "remote memory will be reported as 'Not accessible'"
            )
        if short_mem is None and self.patient_profile["memory_profile"].get("has_recent_episodic"):
            warnings.warn(
                f"no short-term memory entry for patient_id={self.patient_id}; "
                "recent memory will be reported as 'Cannot access'"
            )
        self._patient_memories = {"long_term": long_mem, "short_term": short_mem}

        log.info(
            "DemMA loaded: model=%s | classifier=34-way MLP | persona=%s (id=%d)",
            self.model_path, self.patient_profile["persona"], self.patient_id,
        )

    # -------------------------------------------------------------------
    # The single rollout-facing entry point
    # -------------------------------------------------------------------

    def step(
        self,
        persona: Persona,
        history: list[DialogueHistoryItem],
        latest_caregiver_response: str,
        seed: int,
    ) -> PatientObservation:
        self.ensure_loaded()

        # Persona-fixed semantics: warn once if the rollout layer passes a different one.
        if (not self._warned_persona_mismatch
                and persona.persona_id not in {"", "demma_default", f"demma_id_{self.patient_id}"}):
            warnings.warn(
                f"DemMARealClient holds patient_id={self.patient_id} fixed; "
                f"caller passed persona_id={persona.persona_id!r} which is ignored. "
                "Set Scenario.persona.persona_id='demma_default' to silence."
            )
            self._warned_persona_mismatch = True

        # History adapter: our (caregiver_response, patient_utterance) pairs
        # → DemMA's flat [{speaker, utterance}, ...] format.
        flat_history: list[dict[str, str]] = []
        for item in history:
            flat_history.append({"speaker": "Caregiver", "utterance": item.caregiver_response})
            flat_history.append({"speaker": "Patient", "utterance": item.patient_utterance})

        return self._generate_one_turn(flat_history, latest_caregiver_response, seed)

    # -------------------------------------------------------------------
    # Internal: one DemMA turn = generate utterance + classify actions
    # -------------------------------------------------------------------

    def _generate_one_turn(
        self,
        flat_history: list[dict[str, str]],
        caregiver_utterance: str,
        seed: int,
    ) -> PatientObservation:
        import torch  # local import: torch is loaded only once `ensure_loaded()` succeeds

        # Best-effort reproducibility under temperature sampling (zh_9 §3.3 anti-collapse).
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        user_content = _build_inference_prompt(
            persona=self.patient_profile["persona"],
            icf_b126_profile=self.patient_profile["icf_b126_profile"],
            memory_profile=self.patient_profile["memory_profile"],
            patient_memories=self._patient_memories,  # type: ignore[arg-type]
            dialogue_history=flat_history,
            caregiver_utterance=caregiver_utterance,
            use_classifier=self.use_classifier,
        )
        messages = [
            {"role": "system", "content": "You are a dementia patient planner."},
            {"role": "user", "content": user_content},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._base_model.device)

        # ---- Step 1: autoregressive generation of [PLAN]/[SPEAK]/[ACT] ----
        with torch.no_grad():
            outputs = self._base_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        generated_ids = outputs[0]

        # ---- Decode utterance ---
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=False)
        m = re.search(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)",
                      generated_text, re.DOTALL)
        response_text = m.group(1).strip() if m else generated_text.split("assistant")[-1].strip()
        utterance = _parse_speak_block(response_text)

        # ---- Decode actions: two paths ----
        if self.use_classifier:
            # Step 2: re-feed full sequence to grab final-token hidden state.
            # SFT trained the head on hidden state at attention_mask.sum()-1
            # (the <|im_end|> position after teacher-forcing). generate() does
            # NOT compute a hidden state for the EOS token it emits, so this
            # 2nd forward is the only way to recover the training-time signal.
            with torch.no_grad():
                attention_mask = (generated_ids != self._tokenizer.pad_token_id).long()
                full_outputs = self._base_model(
                    input_ids=generated_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = full_outputs.hidden_states[-1]
                seq_length = int(attention_mask.sum().item()) - 1
                special_tokens = {self._tokenizer.eos_token_id, self._tokenizer.pad_token_id}
                if hasattr(self._tokenizer, "im_end_id") and self._tokenizer.im_end_id is not None:
                    special_tokens.add(self._tokenizer.im_end_id)
                actual_last_idx = seq_length
                for i in range(seq_length, max(0, seq_length - 10), -1):
                    if int(generated_ids[i].item()) not in special_tokens:
                        actual_last_idx = i
                        break
                last_valid_hidden = hidden_states[0, actual_last_idx, :]
                action_logits = self._action_classifier(last_valid_hidden.unsqueeze(0))
                action_probs = torch.sigmoid(action_logits).squeeze(0).float().cpu().numpy()
            annotation = _decode_actions_to_annotation(
                action_probs, threshold=self.action_threshold,
            )
        else:
            # LM-only path: just parse the [ACT] block out of the generated text.
            # No second forward pass; expects model to emit lines like
            # "movement: standing up, frowning".
            annotation = _parse_act_block(response_text)

        return PatientObservation(utterance=utterance, annotation=annotation)

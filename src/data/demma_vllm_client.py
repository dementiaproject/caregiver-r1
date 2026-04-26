"""
DemMAVLLMClient — vLLM offline-LM wrapper around hulehule/DemMA-Planner-SFT.

Same prompt + parsing as `DemMARealClient` (faithful re-impl of the official
`inference_with_mem.py`), but runs the base LLM through **vLLM's offline
`LLM` class** instead of `transformers.generate(batch=1)`. Expected
throughput improvement on a single H100/A100 vs transformers SDPA:

    transformers SDPA  batch=1   ~40   tok/s   →   ~9-15 s / turn
    transformers FA2   batch=1   ~80   tok/s   →   ~3-5  s / turn
    vLLM offline       batch=1   ~150  tok/s   →   ~1-2  s / turn
    vLLM offline       batch=8   ~600+ tok/s   →   ~0.3  s / turn-equiv

Mode constraint
---------------
This backend is **LM-only** (no MLP classifier head). vLLM does not expose
hidden states from arbitrary layers, so the official 2-pass classifier path
is incompatible. We rely on the model emitting [ACT] block as text and
parse it with `_parse_act_block` (same as DemMARealClient(use_classifier=False)).

If you need the canonical classifier path for fidelity / paper reporting,
fall back to `DemMARealClient(use_classifier=True)`.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

from src.data.demma_client import DemMAClient, DialogueHistoryItem
from src.data.demma_real_client import (
    DEFAULT_PATIENT_ID,
    DEFAULT_PATIENT_PROFILE,
    _build_inference_prompt,
    _load_long_memory,
    _load_short_memory,
    _parse_act_block,
    _parse_speak_block,
)
from src.data.schemas import PatientObservation, Persona

log = logging.getLogger(__name__)


class DemMAVLLMClient(DemMAClient):
    """
    Offline-mode vLLM wrapper for hulehule/DemMA-Planner-SFT.

    Lifecycle
    ---------
    Lazy-loads vLLM on first `step()` (or explicit `ensure_loaded()`), so
    importing this module is GPU-free. The vLLM engine consumes a fixed
    fraction of GPU memory (`gpu_memory_utilization`, default 0.85) for its
    KV-cache pool — keep this in mind when sharing the GPU with a training
    process or another vLLM server.

    Persona
    -------
    Holds a single `patient_id` (default 0 = Jacob Johnson, AD-early) for
    the lifetime of the client; the `persona` argument to `step()` is
    accepted for interface symmetry but ignored. A one-shot warning is
    emitted if the rollout layer passes a mismatched `persona_id`.
    """

    def __init__(
        self,
        model_path: Path | str,
        long_memory_path: Path | str,
        short_memory_path: Path | str,
        patient_id: int = DEFAULT_PATIENT_ID,
        patient_profile: dict[str, Any] | None = None,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_new_tokens: int = 400,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        show_progress: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.long_memory_path = Path(long_memory_path)
        self.short_memory_path = Path(short_memory_path)
        self.patient_id = patient_id
        self.patient_profile: dict[str, Any] = patient_profile or DEFAULT_PATIENT_PROFILE
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self.show_progress = show_progress

        # Lazy-loaded
        self._llm = None
        self._tokenizer = None
        self._patient_memories: dict[str, Any] | None = None
        self._loaded = False
        self._load_error: BaseException | None = None  # cached so we fail fast on retry

        self._warned_persona_mismatch = False

    # ---- lifecycle ----------------------------------------------------

    def health_check(self) -> bool:
        for f in (self.model_path, self.long_memory_path, self.short_memory_path):
            if not Path(f).exists():
                log.error("DemMAVLLMClient missing artifact: %s", f)
                return False
        return True

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        if getattr(self, "_load_error", None) is not None:
            # Don't retry — re-raise the cached load error so the caller
            # gets a fast, deterministic failure instead of looping
            # through full vLLM engine init on every turn.
            raise self._load_error
        try:
            self._load()
        except BaseException as e:
            self._load_error = e
            raise
        self._loaded = True

    def _load(self) -> None:
        try:
            from vllm import LLM
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "DemMAVLLMClient requires vllm + transformers. "
                "Install via: pip install vllm transformers"
            ) from e

        log.info("loading DemMA tokenizer from %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        log.info(
            "starting vLLM engine (model=%s, dtype=%s, gpu_mem=%.2f, tp=%d, max_len=%d)",
            self.model_path, self.dtype, self.gpu_memory_utilization,
            self.tensor_parallel_size, self.max_model_len,
        )
        self._llm = LLM(
            model=str(self.model_path),
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            enforce_eager=self.enforce_eager,
            trust_remote_code=True,
        )

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
            "DemMA-vLLM ready: model=%s | persona=%s (id=%d)",
            self.model_path, self.patient_profile["persona"], self.patient_id,
        )

    # ---- step ---------------------------------------------------------

    def step(
        self,
        persona: Persona,
        history: list[DialogueHistoryItem],
        latest_caregiver_response: str,
        seed: int,
    ) -> PatientObservation:
        from vllm import SamplingParams  # local import: vllm only loaded on first step

        self.ensure_loaded()

        if (not self._warned_persona_mismatch
                and persona.persona_id not in
                {"", "demma_default", f"demma_id_{self.patient_id}"}):
            warnings.warn(
                f"DemMAVLLMClient holds patient_id={self.patient_id} fixed; "
                f"caller passed persona_id={persona.persona_id!r} which is ignored. "
                "Set Scenario.persona.persona_id='demma_default' to silence."
            )
            self._warned_persona_mismatch = True

        flat_history: list[dict[str, str]] = []
        for item in history:
            flat_history.append({"speaker": "Caregiver", "utterance": item.caregiver_response})
            flat_history.append({"speaker": "Patient", "utterance": item.patient_utterance})

        user_content = _build_inference_prompt(
            persona=self.patient_profile["persona"],
            icf_b126_profile=self.patient_profile["icf_b126_profile"],
            memory_profile=self.patient_profile["memory_profile"],
            patient_memories=self._patient_memories,  # type: ignore[arg-type]
            dialogue_history=flat_history,
            caregiver_utterance=latest_caregiver_response,
            use_classifier=False,
        )

        chat = [
            {"role": "system", "content": "You are a dementia patient planner."},
            {"role": "user", "content": user_content},
        ]

        sampling = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            seed=int(seed),
        )

        # vLLM's `chat()` applies the chat template internally and runs one
        # offline batch (here: single-turn batch of 1). Output is a list of
        # `RequestOutput`; `.outputs[0].text` is the generated continuation
        # only (prompt is stripped automatically).
        log.debug("vLLM chat() turn-start (history_len=%d, prompt_chars=%d)",
                  len(flat_history), len(user_content))
        request_outputs = self._llm.chat(
            messages=chat,
            sampling_params=sampling,
            use_tqdm=self.show_progress,
            # Qwen3 chat template defaults `enable_thinking=True` which
            # injects <think>...</think>; DemMA SFT was trained without
            # thinking, so we explicitly turn it off here. Otherwise the
            # model may spend all 256 max_tokens inside <think> and never
            # emit [SPEAK]/[ACT].
            chat_template_kwargs={"enable_thinking": False},
        )
        text = request_outputs[0].outputs[0].text
        n_tokens = len(request_outputs[0].outputs[0].token_ids)
        log.debug("vLLM chat() turn-end (generated %d tokens)", n_tokens)

        # ONE-SHOT debug: dump the first generation we ever see so the user
        # can confirm the model emits [PLAN]/[SPEAK]/[ACT] structure.
        if not getattr(self, "_logged_first_generation", False):
            self._logged_first_generation = True
            preview = text[:1500].replace("\n", "\\n")
            log.warning(
                "DemMA first generation preview (%d tokens, %d chars):\n%s",
                n_tokens, len(text), preview,
            )
        utterance = _parse_speak_block(text)
        annotation = _parse_act_block(text)
        return PatientObservation(utterance=utterance, annotation=annotation)

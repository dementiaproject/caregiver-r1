"""
Caregiver agent client — generates the R1-style `<think>+<response>`
output for one turn given the scenario context + dialogue history.

zh_9 §4.2: the caregiver agent emits two segments per turn,

    <think>...</think>      ← internal chain-of-thought (NOT shown to DemMA)
    <response>...</response>← caregiver utterance forwarded into dialogue

Both segments are required; the rollout layer parses them out, forwards
ONLY `response` to DemMA, and stores the raw text + parsed segments on
`CaregiverOutput` (src/data/schemas.py).

Two backends:

  - CaregiverHttpClient: HTTP to a `vllm serve` endpoint (canonical for
    smoke + verl rollout actor; OpenAI-compatible /v1/chat/completions)
  - CaregiverMockClient: deterministic template-based responses (CPU,
    no GPU, used to validate pipeline plumbing without serving a model)

For verl training the rollout adapter will instead route through verl's
internal model interface (the policy actor) — that path is in Wave 4.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.data.schemas import CaregiverOutput, PatientObservation, Scenario
from src.training.prompt_loader import (
    caregiver_prompt_sha256,
    load_caregiver_prompt,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# History format the caregiver sees
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaregiverDialogueItem:
    """One past exchange from the caregiver's point of view.

    Note: caregiver only sees the patient's TEXT — the inline annotation
    labels (motion/facial/sound) are reward-side ground truth and are NOT
    forwarded into the caregiver's prompt (zh_9 §4.2).
    """
    caregiver_response: str
    patient_utterance: str


# ---------------------------------------------------------------------------
# XML parsing — robust to whitespace, missing closes, optional <think>
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_RESPONSE_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL)
_RESPONSE_OPEN_ONLY = re.compile(r"<response>(.*?)$", re.DOTALL)


def parse_caregiver_output(raw_text: str) -> CaregiverOutput:
    """Parse caregiver model output into structured CaregiverOutput.

    Tolerates several common malformations:
      - missing </response> close tag (truncated generation)
      - missing <think>...</think> entirely (we synthesize "(no think emitted)")
      - leading/trailing whitespace around tags

    Raises ValueError ONLY when neither <response>...</response> nor
    <response>... is present — without the response segment we have nothing
    to forward to DemMA.
    """
    think_match = _THINK_RE.search(raw_text)
    think = think_match.group(1).strip() if think_match else "(no think emitted)"

    response_match = _RESPONSE_RE.search(raw_text)
    if response_match:
        response = response_match.group(1).strip()
    else:
        # Fall back: <response>... without close (truncation)
        open_match = _RESPONSE_OPEN_ONLY.search(raw_text)
        if open_match:
            response = open_match.group(1).strip()
        else:
            raise ValueError(
                f"caregiver output missing <response> tag entirely; "
                f"raw_text[:500]={raw_text[:500]!r}"
            )

    if not response:
        # Empty response is recoverable — treat as silence/probe-acknowledged
        response = "..."

    return CaregiverOutput(
        think=think,
        response=response,
        raw_text=raw_text,
    )


# ---------------------------------------------------------------------------
# Conversation builder — turns scenario+history into the chat user message
# ---------------------------------------------------------------------------

def build_caregiver_user_message(
    scenario: Scenario,
    history: list[CaregiverDialogueItem],
    latest_patient_utterance: str | None,
) -> str:
    """Assemble the per-turn USER message for the caregiver chat completion.

    Layout (matches RLVER / Doctor-R1 style):

        # Patient
        <persona summary>

        # Conflict
        <conflict context>

        # Conversation so far
        Caregiver: ...
        Patient:   ...
        Caregiver: ...
        Patient:   <latest>

        Generate your next turn in this format:
        <think>your reasoning</think><response>your reply to the patient</response>

    On the FIRST turn (history empty + latest_patient_utterance from
    scenario.initial_patient_utterance), we still emit a "Conversation so far"
    block so the model sees a uniform layout — saves alignment headaches.
    """
    # IMPORTANT: ground_truth, conflict_type, severity, risk_tier are all
    # REWARD-SIDE annotations (judge / safety / Scenario.model_dump for
    # diagnostics). They are NOT shown to the caregiver — that would be a
    # direct answer leak (caregiver could state facts perfectly every turn,
    # making R_goal trivial; this is the canonical reward-hacking vector).
    # Only the persona fields a real bedside caregiver would have on intake
    # (name, age, dementia subtype) plus the conversation history are shown.
    parts: list[str] = []
    parts.append("# Patient")
    parts.append(f"- name: {scenario.persona.name}")
    parts.append(f"- age: {scenario.persona.age}")
    parts.append(f"- dementia subtype: {scenario.persona.dementia_subtype}")
    parts.append("")
    parts.append("# Conversation so far")
    if not history:
        parts.append("(beginning of conversation)")
    else:
        for item in history:
            parts.append(f"Caregiver: {item.caregiver_response}")
            parts.append(f"Patient: {item.patient_utterance}")
    if latest_patient_utterance is not None:
        parts.append(f"Patient: {latest_patient_utterance}")
    parts.append("")
    parts.append(
        "Generate your next turn in EXACTLY this XML format (one <think> "
        "block, one <response> block, no other text outside the tags):"
    )
    parts.append("<think>your concise clinical reasoning here</think><response>"
                 "your one-or-two-sentence reply to the patient</response>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class CaregiverClient(ABC):
    """Stable interface every caregiver backend must implement."""

    @abstractmethod
    def step(
        self,
        scenario: Scenario,
        history: list[CaregiverDialogueItem],
        latest_patient_utterance: str | None,
        seed: int,
    ) -> CaregiverOutput:
        """Run the caregiver for one turn.

        Args:
            scenario:                  fixed scenario (persona + conflict).
            history:                   past (caregiver, patient) exchanges.
            latest_patient_utterance:  None on the very first turn before
                                       any caregiver-issued response (use
                                       scenario.initial_patient_utterance);
                                       otherwise the most recent patient text.
            seed:                      per-rollout seed (zh_9 §3.3).

        Returns:
            CaregiverOutput(think, response, raw_text). The rollout layer
            forwards ONLY `response` to DemMA.
        """

    def health_check(self) -> bool:
        """Cheap probe — default True; HTTP backends override."""
        return True


# ---------------------------------------------------------------------------
# Mock backend — template-based, no GPU
# ---------------------------------------------------------------------------

class CaregiverMockClient(CaregiverClient):
    """Deterministic mock: emits canned `<think>+<response>` per turn.

    Used to validate pipeline plumbing on a CPU box where no caregiver model
    is available. Cycles through a small library of clinically-plausible
    responses so the mock smoke run still produces non-trivial conversation
    text (instead of '[turn 0] please take your medication').
    """

    _RESPONSES: tuple[tuple[str, str], ...] = (
        ("Open with NURSE — name the feeling first, no facts yet.",
         "It sounds like that's really weighing on you right now."),
        ("Validate first (VERA). Lower the affect before any redirection.",
         "I hear how upsetting this is — you're not alone in this."),
        ("Ask permission before any concrete action (SPIKES Setup).",
         "Would it be okay if we sit here together for a moment?"),
        ("Soft factual anchor — co-read the chart instead of contradicting.",
         "Let me check the chart with you so we can see what was planned."),
        ("Bridge to a calming activity tied to a remote-memory cue.",
         "Tell me about her while we look out the window."),
    )

    def __init__(self) -> None:
        self._counter = 0

    def step(
        self,
        scenario: Scenario,
        history: list[CaregiverDialogueItem],
        latest_patient_utterance: str | None,
        seed: int,
    ) -> CaregiverOutput:
        idx = (seed + self._counter) % len(self._RESPONSES)
        self._counter += 1
        think, response = self._RESPONSES[idx]
        raw_text = f"<think>{think}</think><response>{response}</response>"
        return parse_caregiver_output(raw_text)


# ---------------------------------------------------------------------------
# HTTP backend — vllm serve, OpenAI-compatible /v1/chat/completions
# ---------------------------------------------------------------------------

class CaregiverHttpClient(CaregiverClient):
    """Real caregiver via `vllm serve` over OpenAI-compatible HTTP.

    System message is the canonical caregiver system prompt assembled by
    `prompt_loader.load_caregiver_prompt()` (frozen + sha256-tracked across
    a run; see zh_9 §4.2).

    User message is rebuilt every turn by `build_caregiver_user_message`.

    Sampling: default temperature 1.0 (vLLM Qwen3 default), top_p 0.9 — match
    standard GRPO rollout semantics (Decision 1, unified prompt + temp
    sampling for group diversity).
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        request_timeout_s: float = 60.0,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.9,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.request_timeout_s = request_timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.api_key = api_key
        self.extra_headers = dict(extra_headers or {})

        # Lazy-loaded
        self._client = None
        self._system_prompt: str | None = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError(
                "CaregiverHttpClient requires httpx. Install via: pip install httpx"
            ) from e
        self._client = httpx.Client(timeout=self.request_timeout_s)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "CaregiverHttpClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = load_caregiver_prompt()
            log.info(
                "CaregiverHttpClient loaded system prompt (sha256=%s, %d chars)",
                caregiver_prompt_sha256()[:16], len(self._system_prompt),
            )
        return self._system_prompt

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "Reply with the single word: OK"},
                    {"role": "user", "content": "ping"},
                ],
                "temperature": 0.0,
                "max_tokens": 4,
            }
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            r = client.post(f"{self.base_url}/v1/chat/completions",
                            json=payload, headers=headers, timeout=15.0)
            return r.status_code == 200 and bool(r.json().get("choices"))
        except Exception:
            return False

    def step(
        self,
        scenario: Scenario,
        history: list[CaregiverDialogueItem],
        latest_patient_utterance: str | None,
        seed: int,
    ) -> CaregiverOutput:
        client = self._get_client()
        user_msg = build_caregiver_user_message(
            scenario=scenario,
            history=history,
            latest_patient_utterance=latest_patient_utterance,
        )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": int(seed),
            # Qwen3 thinking is REPLACED by our own <think>+<response> protocol;
            # turn off the chat-template thinking block so we don't get nested
            # <think> markers.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        response = client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"caregiver vLLM returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        try:
            raw_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"caregiver response missing choices[0].message.content; got: {data!r}"
            ) from e

        return parse_caregiver_output(raw_text)


__all__ = [
    "CaregiverClient",
    "CaregiverMockClient",
    "CaregiverHttpClient",
    "CaregiverDialogueItem",
    "parse_caregiver_output",
    "build_caregiver_user_message",
]

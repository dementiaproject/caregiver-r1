"""
LLM judge client: takes a Rubric + a Trajectory, calls the LLM judge,
parses the structured JSON response, and returns both per-item grades
and the aggregated trajectory-level reward.

Architecture:

    Rubric + Trajectory
            │
            ▼
    build_grader_prompt()  ──►  prompt string
            │
            ▼
    JudgeClient.score()    ──►  raw JSON from LLM
            │
            ▼
    parse + validate (Pydantic)  ──►  RubricGrade
            │
            ▼
    aggregate_score()      ──►  trajectory-level reward (int)

Two backends:

    MockLlmJudgeClient   — synthetic JSON, deterministic by seed (CPU)
    VllmJudgeClient      — vLLM HTTP endpoint, real LLM calls (P4 / GPU)

The VllmJudgeClient body raises NotImplementedError until GPU goes online;
its public interface is final, so dependent code can be written against
it now.
"""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.data.schemas import Trajectory
from src.rewards.judge_prompt import build_grader_prompt
from src.rewards.rubric import (
    BooleanCriterion,
    OrdinalItem,
    Rubric,
    load_rubric,
)


# ---------------------------------------------------------------------------
# Response schemas (what the LLM judge must return per item)
# ---------------------------------------------------------------------------

class BooleanItemGrade(BaseModel):
    """Per-item grade for a weighted_boolean rubric (R_goal, R_fit)."""
    model_config = ConfigDict(extra="forbid")

    id: str
    evidence_turns: list[int] = Field(default_factory=list)
    explanation: str
    criteria_met: bool


class OrdinalItemGrade(BaseModel):
    """Per-item grade for an ordinal rubric (u_terminal, c_safety)."""
    model_config = ConfigDict(extra="forbid")

    id: str
    evidence_turns: list[int] = Field(default_factory=list)
    explanation: str
    score: int


ItemGrade = BooleanItemGrade | OrdinalItemGrade


class RubricGrade(BaseModel):
    """Full result of one judge call: per-item grades + aggregated score."""
    model_config = ConfigDict(extra="forbid")

    rubric: str
    items: list[ItemGrade]
    aggregated: int                         # final trajectory-level reward
    raw_score: float | None = None          # pre-clip raw value (for diagnostics)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_score(rubric: Rubric, items: list[ItemGrade]) -> tuple[int, float]:
    """Apply the rubric's aggregation rule to the per-item grades.

    Returns (aggregated_int, raw_float) where:
      - aggregated_int is the value stored on RubricGrade.aggregated
      - raw_float is the pre-clip value (useful for W&B diagnostics)
    """
    by_id = {g.id: g for g in items}
    if rubric.is_boolean:
        # Common path for weighted_boolean (R_goal/R_fit) and binary_any_trigger (c_safety):
        # both consume BooleanItemGrade per item. Aggregation differs.
        flags: list[bool] = []
        weighted_raw = 0.0
        for c in rubric.items:
            assert isinstance(c, BooleanCriterion)
            grade = by_id.get(c.id)
            if not isinstance(grade, BooleanItemGrade):
                raise ValueError(f"missing or wrong-type grade for boolean criterion {c.id!r}")
            flags.append(grade.criteria_met)
            if grade.criteria_met:
                weighted_raw += c.points

        if rubric.aggregation.type == "binary_any":
            # Decision 3 c_safety: ANY criterion met → 1, else 0.
            c_safety = 1 if any(flags) else 0
            return c_safety, float(c_safety)

        # weighted_sum: raw = sum(c.points for c in criteria if criteria_met[c])
        norm = rubric.aggregation.paper_normalized_range or rubric.aggregation.range
        if norm is None:
            return int(round(weighted_raw)), weighted_raw
        clipped = max(norm[0], min(norm[1], weighted_raw))
        return int(round(clipped)), weighted_raw

    # ordinal_per_dimension (u_terminal)
    scores: list[int] = []
    for it in rubric.items:
        assert isinstance(it, OrdinalItem)
        grade = by_id.get(it.id)
        if not isinstance(grade, OrdinalItemGrade):
            raise ValueError(f"missing or wrong-type grade for ordinal item {it.id!r}")
        # validate score is one of the rubric's defined levels
        valid = {lvl.score for lvl in it.levels}
        if grade.score not in valid:
            raise ValueError(
                f"item {it.id!r} score {grade.score} not in defined levels {sorted(valid)}"
            )
        scores.append(grade.score)

    if rubric.aggregation.type == "sum":
        raw = float(sum(scores))
    else:
        raise ValueError(
            f"unsupported aggregation type {rubric.aggregation.type!r} for ordinal rubric "
            f"(only 'sum' is supported now; 'max' was retired with Decision 3)"
        )

    return int(raw), raw


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.MULTILINE)


def _strip_json_fence(raw: str) -> str:
    """Remove ```json fences if the LLM wrapped the JSON in markdown."""
    return _JSON_FENCE_RE.sub("", raw.strip()).strip()


def parse_judge_response(rubric: Rubric, raw_text: str) -> RubricGrade:
    """Parse the LLM's response text into a typed RubricGrade.

    Validation steps:
      1. Strip markdown fences and parse JSON
      2. Verify `rubric` field matches expected name
      3. Verify item count matches and IDs cover all rubric items
      4. Coerce each item to BooleanItemGrade or OrdinalItemGrade
      5. Aggregate

    Raises `ValueError` (with a clear message) on any malformed response;
    callers (the JudgeClient retry wrapper) should catch and retry up to
    a small number of attempts before raising a hard error.
    """
    cleaned = _strip_json_fence(raw_text)
    try:
        obj: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"judge response is not valid JSON: {e}\nraw: {raw_text[:500]!r}") from e

    if not isinstance(obj, dict):
        raise ValueError(f"judge response must be a JSON object, got {type(obj).__name__}")

    if obj.get("rubric") != rubric.name:
        raise ValueError(
            f"judge response rubric mismatch: expected {rubric.name!r}, got {obj.get('rubric')!r}"
        )

    raw_items = obj.get("items")
    if not isinstance(raw_items, list):
        raise ValueError(f"judge response 'items' must be a list, got {type(raw_items).__name__}")

    expected_ids = rubric.item_ids()
    if len(raw_items) != len(expected_ids):
        raise ValueError(
            f"judge returned {len(raw_items)} items, expected {len(expected_ids)} "
            f"({expected_ids})"
        )

    parsed: list[ItemGrade] = []
    for idx, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"item[{idx}] must be a JSON object")
        if rubric.is_boolean:
            parsed.append(BooleanItemGrade.model_validate(raw_item))
        else:
            parsed.append(OrdinalItemGrade.model_validate(raw_item))

    # Verify all expected IDs are present (order may differ in malformed responses)
    returned_ids = sorted(g.id for g in parsed)
    if returned_ids != sorted(expected_ids):
        missing = set(expected_ids) - set(returned_ids)
        extra = set(returned_ids) - set(expected_ids)
        raise ValueError(
            f"judge response item IDs do not cover the rubric: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    # Re-order to canonical (rubric-defined) order so aggregation is deterministic
    by_id = {g.id: g for g in parsed}
    parsed = [by_id[i] for i in expected_ids]

    aggregated, raw_score = aggregate_score(rubric, parsed)
    return RubricGrade(
        rubric=rubric.name,
        items=parsed,
        aggregated=aggregated,
        raw_score=raw_score,
    )


# ---------------------------------------------------------------------------
# Judge client interface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeCallStats:
    """Accounting info per judge call — for W&B / cost tracking."""
    rubric: str
    n_items: int
    n_turns: int
    prompt_chars: int
    response_chars: int
    n_retries: int = 0


class JudgeClient(ABC):
    """Abstract base. Concrete subclasses talk to a real or mock LLM."""

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Send the prompt to the LLM, return raw response text."""
        ...

    def grade(
        self,
        rubric_name: str,
        trajectory: Trajectory,
        max_retries: int = 3,
    ) -> tuple[RubricGrade, JudgeCallStats]:
        """End-to-end: build prompt, call LLM, parse, aggregate.

        On parse failure, retries up to `max_retries` times with the same
        prompt (LLMs at temperature 0 may still occasionally produce
        malformed JSON). Persistent failure raises ValueError.
        """
        rubric = load_rubric(rubric_name)
        prompt = build_grader_prompt(rubric, trajectory)
        last_err: Exception | None = None
        last_response = ""
        for attempt in range(max_retries + 1):
            response = self._call_llm(prompt)
            last_response = response
            try:
                grade = parse_judge_response(rubric, response)
                stats = JudgeCallStats(
                    rubric=rubric_name,
                    n_items=len(rubric.items),
                    n_turns=trajectory.num_turns,
                    prompt_chars=len(prompt),
                    response_chars=len(response),
                    n_retries=attempt,
                )
                return grade, stats
            except (ValueError, Exception) as e:
                last_err = e
                continue
        raise ValueError(
            f"judge failed to produce valid JSON for rubric {rubric_name!r} after "
            f"{max_retries + 1} attempts: {last_err}\n"
            f"final response: {last_response[:500]!r}"
        )


# ---------------------------------------------------------------------------
# Mock backend (CPU, deterministic by seed)
# ---------------------------------------------------------------------------

class MockLlmJudgeClient(JudgeClient):
    """Synthesize plausible judge responses without calling any LLM.

    Deterministic per (seed, rubric_name, trajectory_id) so smoke runs
    are reproducible. Used until a real vLLM endpoint is available.

    Distribution priors (rough sanity targets — not hand-tuned to anything):
      - boolean rubrics: each criterion is "met" with prob 0.6 (positives)
        or 0.1 (negatives). Yields R_goal/R_fit ≈ middle of the range.
      - ordinal rubrics: each item draws score uniformly from its defined
        levels, with a small bias toward 1 (mid). Yields u_terminal ≈ 2,
        c_safety ≈ 1 in expectation.
    """

    def __init__(self, seed: int = 0) -> None:
        self._base_seed = seed

    def _rng_for(self, prompt: str) -> random.Random:
        # Use prompt hash so identical calls return identical results.
        return random.Random((self._base_seed * 1_000_003 + hash(prompt)) & 0xFFFFFFFF)

    def _call_llm(self, prompt: str) -> str:
        rubric_name = self._infer_rubric_name(prompt)
        rubric = load_rubric(rubric_name)
        rng = self._rng_for(prompt)

        items_out: list[dict[str, Any]] = []
        for it in rubric.items:
            evidence = sorted(rng.sample(range(0, 10), k=2)) if rng.random() < 0.95 else []
            explanation = (
                f"[mock judge] Synthesized grade for item {it.id!r}; in production this "
                f"would cite verbatim quotes from turns {evidence}."
            )
            if isinstance(it, BooleanCriterion):
                # Mock prior depends on rubric semantics:
                #  - For c_safety (binary_any_trigger): each catastrophic
                #    criterion fires with low prob (~2%) so the OR-aggregated
                #    c_safety lands around 5–8% across 3 items — the rough
                #    rate of catastrophic violations we expect from a poorly
                #    aligned policy.
                #  - For weighted_boolean (R_goal / R_fit): positive criteria
                #    fire ~60%, negatives ~10% (HealthBench-style mid-range
                #    aggregate).
                if rubric.aggregation.type == "binary_any":
                    p_met = 0.02
                elif it.points >= 0:
                    p_met = 0.6
                else:
                    p_met = 0.1
                criteria_met = rng.random() < p_met
                items_out.append({
                    "id": it.id,
                    "evidence_turns": evidence,
                    "explanation": explanation,
                    "criteria_met": criteria_met,
                })
            else:
                assert isinstance(it, OrdinalItem)
                possible = [lvl.score for lvl in it.levels]
                # Bias toward mid-low scores so c_safety ≈ 1 in expectation
                weights = [0.5, 0.3, 0.15, 0.05][: len(possible)]
                score = rng.choices(possible, weights=weights, k=1)[0]
                items_out.append({
                    "id": it.id,
                    "evidence_turns": evidence,
                    "explanation": explanation,
                    "score": score,
                })

        return json.dumps({"rubric": rubric.name, "items": items_out})

    @staticmethod
    def _infer_rubric_name(prompt: str) -> str:
        """Recover the rubric name from the prompt (the meta-template embeds
        `# Rubric: <name>` in the body). Falls back to first known rubric
        name found anywhere in the prompt text."""
        m = re.search(r"# Rubric:\s*([a-z_]+)", prompt)
        if m:
            return m.group(1)
        # Fallback — search for any rubric file name
        for candidate in ("r_goal", "r_fit", "u_terminal", "c_safety"):
            if candidate in prompt:
                return candidate
        raise ValueError("could not infer rubric name from prompt")


# ---------------------------------------------------------------------------
# vLLM backend (real LLM judge over OpenAI-compatible HTTP)
# ---------------------------------------------------------------------------

# Static system prompt: short, JSON-mode-friendly. The full role/discipline
# instructions already live in the user prompt (assembled by
# build_grader_prompt → _read_template_static_sections), so the system message
# only needs to assert the JSON contract.
_JUDGE_SYSTEM_MESSAGE = (
    "You are a strict, evidence-based trajectory judge. "
    "Follow the role and scoring discipline given in the user message verbatim. "
    "Return ONLY a single valid JSON object — no preamble, no markdown fences, "
    "no commentary. Cite specific turn indices in evidence_turns and quote "
    "verbatim from those turns in explanation."
)


class VllmJudgeClient(JudgeClient):
    """Real LLM judge over an OpenAI-compatible HTTP endpoint (vLLM serve).

    Talks to `{base_url}/v1/chat/completions` with:
      - system message: a short JSON-contract reminder (the full role+rubric
        instructions are in the user message via build_grader_prompt)
      - user message: the full grader prompt
      - temperature 0 (deterministic grading)
      - response_format=json_object when use_json_mode=True (vLLM ≥0.4 supports
        this for many serving stacks; if your server doesn't, set it False)
      - Authorization: Bearer header if api_key is set

    Retries are handled at the JudgeClient.grade() layer (parser-failure
    retries); transport-level errors (timeout, 5xx) are NOT retried here on
    purpose — they bubble up so the smoke run fails loudly instead of silently
    eating budget.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        request_timeout_s: float = 120.0,
        max_tokens: int = 2048,
        api_key: str | None = None,
        use_json_mode: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.request_timeout_s = request_timeout_s
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.use_json_mode = use_json_mode
        self.extra_headers = dict(extra_headers or {})

        self._client = None  # lazy httpx.Client

    def _get_client(self):
        """Lazy-instantiate the httpx.Client so importing this module is
        network-free / dependency-free if no judge is actually called."""
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError(
                "VllmJudgeClient requires httpx. Install via: pip install httpx"
            ) from e
        self._client = httpx.Client(timeout=self.request_timeout_s)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "VllmJudgeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if self.use_json_mode:
            # vLLM serve maps this onto its guided-decoding (json) backend.
            payload["response_format"] = {"type": "json_object"}

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        response = client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        # Surface vLLM error bodies (they contain useful diagnostic info,
        # e.g. "context length exceeded" or "invalid response_format")
        if response.status_code >= 400:
            raise RuntimeError(
                f"vLLM judge returned HTTP {response.status_code}: {response.text[:500]}"
            )
        data = response.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"vLLM judge response missing choices[0].message.content; got: {data!r}"
            ) from e

    def health_check(self) -> bool:
        """Cheap probe — POST a tiny prompt and verify we get a 200 + non-empty
        body. Catches misconfigured base_url / wrong model_name / serve down
        BEFORE the smoke run wastes a full rollout on broken judges.
        """
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


__all__ = [
    "BooleanItemGrade",
    "OrdinalItemGrade",
    "ItemGrade",
    "RubricGrade",
    "JudgeCallStats",
    "JudgeClient",
    "MockLlmJudgeClient",
    "VllmJudgeClient",
    "aggregate_score",
    "parse_judge_response",
]

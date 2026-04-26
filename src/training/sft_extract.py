"""
SFT data extractor — turns DemMA training-corpus dialogs into caregiver SFT
examples in the canonical `<think>+<response>` R1 format.

Background
----------
The DemMA training corpus (DemMA-Planner-SFT/data) contains paired
caregiver-patient dialogs annotated with planner steps. Each turn has:
  - caregiver_utterance       (text)
  - patient_utterance + action labels  (DemMA's target)
  - planner_rationale         (reasoning DemMA used)

For caregiver SFT we need DIFFERENT structure: input = (scenario, history),
output = `<think>caregiver reasoning</think><response>caregiver utterance</response>`.

We approximate the caregiver's `<think>` by:
  - Either (a) RAG-style: condition on the patient's planner_rationale and
    invert (treat planner output as caregiver "what does patient need" hint)
  - Or (b) Synthesize: use a strong teacher (Qwen3-32B-Instruct) to generate
    `<think>` blocks given the dialog turn.

For the SFT seed pass we use approach (b). This module takes raw DemMA
dialogs and a teacher LLM endpoint, emits a JSONL of caregiver-side SFT
examples ready to feed to `scripts/train_sft.py`.

Output schema (one JSON object per line):
    {
      "scenario_summary": "Patient (Mrs. Chen, 78, AD-mod) believes her
                           deceased mother visited today.",
      "history": [
        {"caregiver": "...", "patient": "..."},
        ...
      ],
      "latest_patient": "What time is it? My class starts in 10 minutes.",
      "target_think": "She is anxious about going to school. Use NURSE
                       (name the feeling) before any redirection.",
      "target_response": "It sounds like you really want to be on time.
                          Tell me what you're hoping to do today."
    }

Usage
-----
    python -m src.training.sft_extract \\
        --dialog-corpus data/demma_dialogs.jsonl \\
        --teacher-base-url http://localhost:8001 \\
        --teacher-model Qwen/Qwen3-32B-Instruct \\
        --out data/sft_caregiver.jsonl \\
        --max-examples 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher prompt — instructs the teacher to invent caregiver <think>+<response>
# ---------------------------------------------------------------------------

SYSTEM_TEACHER = """You are a senior dementia-care clinician writing a TRAINING
example for a junior caregiver agent. Given the patient's last utterance and
the prior dialogue, produce the caregiver's next move in EXACTLY this XML
format (no other text):

<think>
A 1-3 sentence clinical reasoning trace. Cite which strategy you are using
(NURSE / VERA / SPIKES / Therapeutic Fibbing / GO / DEEP / etc.). Mention
the patient's emotional state and what risk you are managing.
</think>
<response>
The caregiver's actual reply to the patient. 1-2 sentences, conversational,
no jargon, do not contradict the patient harshly.
</response>

Strict rules:
- NEVER include any text outside the two XML blocks.
- The <response> is what the patient HEARS; never expose the strategy name
  or any clinical labels there.
- If the situation involves a clinical red line (medication refusal,
  identity confusion about a deceased loved one), follow the patient's
  affect first; do NOT push facts in the same turn.
"""


# ---------------------------------------------------------------------------
# Conversion: DemMA dialog turn → SFT example with teacher-generated think
# ---------------------------------------------------------------------------

def _summarize_persona(sample: dict[str, Any]) -> str:
    """Build the 1-line scenario_summary from DemMA dialog metadata."""
    persona = sample.get("persona", "AD-early")
    icf = sample.get("icf_b126_profile", {})
    mem = sample.get("memory_profile", {})
    return (
        f"Patient ({persona}; ICF: E={icf.get('extraversion','?')}, "
        f"A={icf.get('agreeableness','?')}, C={icf.get('conscientiousness','?')}, "
        f"ES={icf.get('emotional_stability','?')}; deficit: "
        f"{mem.get('deficit_type','?')})"
    )


def _build_teacher_user_message(
    scenario_summary: str,
    history: list[dict[str, str]],
    latest_patient: str,
) -> str:
    parts = [f"# Patient context\n{scenario_summary}", "", "# Conversation so far"]
    if not history:
        parts.append("(beginning)")
    else:
        for h in history:
            parts.append(f"Caregiver: {h['caregiver']}")
            parts.append(f"Patient: {h['patient']}")
    parts.append(f"Patient: {latest_patient}")
    parts.append("")
    parts.append("Now write the caregiver's next move:")
    return "\n".join(parts)


def call_teacher(
    base_url: str,
    model: str,
    system: str,
    user: str,
    api_key: str | None = None,
    max_tokens: int = 512,
    timeout_s: float = 60.0,
) -> str:
    """One OpenAI-compatible chat call. Returns assistant content text."""
    import httpx
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=timeout_s) as c:
        r = c.post(f"{base_url.rstrip('/')}/v1/chat/completions",
                   json=payload, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"teacher HTTP {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Iteration over DemMA dialog corpus
# ---------------------------------------------------------------------------

def iter_demma_dialog_turns(corpus_path: Path) -> Iterable[dict[str, Any]]:
    """Yield one (sample, history, latest_patient) tuple per turn.

    Expected DemMA corpus format (JSONL, one dialog per line):
        {
          "persona": "AD-early",
          "icf_b126_profile": {...},
          "memory_profile": {...},
          "dialogue_history": [
              {"speaker": "Patient",   "utterance": "..."},
              {"speaker": "Caregiver", "utterance": "..."},
              ...
          ],
          "caregiver_utterance": "the next caregiver move (training target)",
          "patient_utterance":   "patient's reply to the above caregiver move"
        }

    For SFT we want the input to be (everything BEFORE caregiver_utterance)
    and the output is the caregiver_utterance (we fabricate <think> from it
    using the teacher).
    """
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            yield sample


def extract_one(
    sample: dict[str, Any],
    teacher_base_url: str,
    teacher_model: str,
    teacher_api_key: str | None,
) -> dict[str, Any] | None:
    history_raw = sample.get("dialogue_history", [])
    # Compress history into (caregiver, patient) pairs by walking the turns
    history_pairs: list[dict[str, str]] = []
    pending_caregiver: str | None = None
    latest_patient: str | None = None
    for turn in history_raw:
        if turn["speaker"] == "Caregiver":
            pending_caregiver = turn["utterance"]
        else:  # Patient
            if pending_caregiver is not None:
                history_pairs.append({
                    "caregiver": pending_caregiver,
                    "patient": turn["utterance"],
                })
                pending_caregiver = None
            else:
                # Patient spoke first; treat as scenario opener
                latest_patient = turn["utterance"]

    # Final patient line is what the caregiver_utterance responds to
    latest_patient = sample.get("latest_patient_utterance") or latest_patient
    if latest_patient is None and history_pairs:
        latest_patient = history_pairs[-1]["patient"]
    if latest_patient is None:
        return None  # no patient context — skip

    target_response = sample.get("caregiver_utterance")
    if not target_response:
        return None

    summary = _summarize_persona(sample)
    user_msg = _build_teacher_user_message(summary, history_pairs, latest_patient)

    try:
        raw = call_teacher(
            base_url=teacher_base_url, model=teacher_model,
            system=SYSTEM_TEACHER, user=user_msg, api_key=teacher_api_key,
        )
    except Exception as e:
        log.warning("teacher call failed: %s", e)
        return None

    # We trust the teacher's <response> AND <think> over the corpus's
    # caregiver_utterance because the corpus utterance has no <think>
    # paired with it. The corpus utterance is kept as a quality fallback
    # in `target_response_corpus` for downstream filtering.
    from src.data.caregiver_client import parse_caregiver_output
    try:
        out = parse_caregiver_output(raw)
    except ValueError:
        return None

    return {
        "scenario_summary": summary,
        "history": history_pairs,
        "latest_patient": latest_patient,
        "target_think": out.think,
        "target_response": out.response,
        "target_response_corpus": target_response,   # for audit
        "teacher_raw": raw,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialog-corpus", type=Path, required=True)
    ap.add_argument("--teacher-base-url", default="http://localhost:8001")
    ap.add_argument("--teacher-model", default="Qwen/Qwen3-32B-Instruct")
    ap.add_argument("--teacher-api-key", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=5000)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_emitted = 0
    n_skipped = 0
    with args.out.open("w", encoding="utf-8") as f_out:
        for sample in iter_demma_dialog_turns(args.dialog_corpus):
            if n_emitted >= args.max_examples:
                break
            ex = extract_one(
                sample=sample,
                teacher_base_url=args.teacher_base_url,
                teacher_model=args.teacher_model,
                teacher_api_key=args.teacher_api_key,
            )
            if ex is None:
                n_skipped += 1
                continue
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_emitted += 1
            if n_emitted % 100 == 0:
                log.info("extracted %d/%d (skipped %d so far)",
                         n_emitted, args.max_examples, n_skipped)

    log.info("done. emitted=%d skipped=%d → %s", n_emitted, n_skipped, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

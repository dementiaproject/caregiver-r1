"""
Grader-prompt builder: takes a Rubric + a Trajectory and assembles the
single string that gets sent to the LLM judge.

Composition (matching `prompts/rubrics/_judge_template.md`):

  ┌────────────────────────────────────────────────────────────┐
  │  ROLE + INPUT FORMAT + SCORING DISCIPLINE                  │  ← static, from
  │  (read once from _judge_template.md)                       │    template file
  ├────────────────────────────────────────────────────────────┤
  │  # Conversation                                            │
  │  Turn 0:                                                   │
  │    caregiver <think>...</think><response>...</response>    │  ← rendered from
  │    patient: "..."                                          │    Trajectory
  │    annotation: motion=[...], facial=[...], sound=[...]     │
  │  Turn 1: ...                                               │
  ├────────────────────────────────────────────────────────────┤
  │  # Rubric: <name>                                          │
  │  <description>                                             │  ← from Rubric
  │                                                            │
  │  ## Items                                                  │
  │   1. <id>: <text>                                          │
  │      - 0: <anchor>                                         │
  │      - 1: <anchor>                                         │
  │      - 2: <anchor>                                         │  (boolean rubrics
  │   2. ...                                                   │   skip the levels
  ├────────────────────────────────────────────────────────────┤   block; show
  │  # Required output format                                  │   `points` instead)
  │  <boolean or ordinal example, picked by scoring_mode>      │
  │                                                            │
  │  Return ONLY the JSON object.                              │
  └────────────────────────────────────────────────────────────┘

Output: a single string ready to send as the user message of a chat
completion call. The system message comes from the same template file
(extracted as the ROLE section).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.schemas import Trajectory
from src.rewards.rubric import (
    BooleanCriterion,
    OrdinalItem,
    Rubric,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGE_TEMPLATE_FILE = REPO_ROOT / "prompts" / "rubrics" / "_judge_template.md"


# ---------------------------------------------------------------------------
# Trajectory rendering
# ---------------------------------------------------------------------------

def render_trajectory_for_judge(trajectory: Trajectory) -> str:
    """Render a Trajectory as a numbered turn-by-turn block.

    The judge sees the full caregiver `<think>+<response>` AND the patient
    side (utterance + inline annotation labels). Turn indices are 0-based
    so the judge's `evidence_turns` field aligns with `Turn.turn_index`.
    """
    lines: list[str] = []
    for turn in trajectory.turns:
        ann = turn.patient.annotation
        ann_parts = []
        if ann.motion:
            ann_parts.append(f"motion={ann.motion}")
        if ann.facial:
            ann_parts.append(f"facial={ann.facial}")
        if ann.sound:
            ann_parts.append(f"sound={ann.sound}")
        ann_str = " | ".join(ann_parts) if ann_parts else "(no observable cues)"

        # SECURITY: never expose caregiver.think to the judge — the policy
        # could otherwise stuff <think> with self-serving claims (e.g. "this
        # was a low-distress turn") and influence u_terminal / R_fit through
        # a backchannel the response itself doesn't carry. The judge only
        # sees what the patient hears (response) plus the observable cues.
        lines.append(f"## Turn {turn.turn_index}")
        lines.append(f"caregiver said: \"{turn.caregiver.response}\"")
        lines.append(f"patient said: \"{turn.patient.utterance}\"")
        lines.append(f"patient cues: {ann_str}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Rubric body rendering
# ---------------------------------------------------------------------------

def _render_boolean_criterion(idx: int, c: BooleanCriterion) -> str:
    sign = "+" if c.points >= 0 else ""
    polarity = "DESIRABLE" if c.points >= 0 else "UNDESIRABLE (criteria_met=true means the negative behavior was observed)"
    body = c.text.strip()
    tag_str = f" [tags: {', '.join(c.tags)}]" if c.tags else ""
    return (
        f"{idx}. **{c.id}**  ({polarity}, points={sign}{c.points:g}){tag_str}\n"
        f"   {body}"
    )


def _render_ordinal_item(idx: int, item: OrdinalItem) -> str:
    body = item.text.strip()
    tag_str = f" [tags: {', '.join(item.tags)}]" if item.tags else ""
    rubrix_str = f"  [RubRIX dim: {item.rubrix_dim}]" if item.rubrix_dim else ""
    out = [f"{idx}. **{item.id}**{tag_str}{rubrix_str}", f"   {body}"]
    out.append("   Levels:")
    for lvl in item.levels:
        anchor = lvl.anchor.strip().replace("\n", " ")
        out.append(f"     - {lvl.score}: {anchor}")
    return "\n".join(out)


def _render_rubric_body(rubric: Rubric) -> str:
    lines: list[str] = []
    lines.append(f"# Rubric: {rubric.name}  (version {rubric.version}, scoring_mode={rubric.scoring_mode})")
    lines.append("")
    lines.append(rubric.description.strip())
    lines.append("")
    lines.append("## Items")
    lines.append("")
    for i, item in enumerate(rubric.items, start=1):
        if isinstance(item, BooleanCriterion):
            lines.append(_render_boolean_criterion(i, item))
        else:
            lines.append(_render_ordinal_item(i, item))
        lines.append("")
    lines.append(f"## Aggregation")
    lines.append(f"- type: `{rubric.aggregation.type}`")
    if rubric.aggregation.range:
        lines.append(f"- range: {rubric.aggregation.range}")
    if rubric.aggregation.raw_range:
        lines.append(f"- raw_range (pre-clip): {rubric.aggregation.raw_range}")
    if rubric.aggregation.hard_veto_threshold is not None:
        lines.append(f"- hard_veto_threshold: {rubric.aggregation.hard_veto_threshold}  ← top tier triggers HARD VETO")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output-format example block (boolean vs ordinal)
# ---------------------------------------------------------------------------

_BOOLEAN_EXAMPLE = '''```json
{
  "rubric": "%s",
  "items": [
    {
      "id": "%s",
      "evidence_turns": [3, 5],
      "explanation": "Turn 3: \\"<verbatim caregiver response>\\" — this met the criterion because …",
      "criteria_met": true
    }
  ]
}
```'''

_ORDINAL_EXAMPLE = '''```json
{
  "rubric": "%s",
  "items": [
    {
      "id": "%s",
      "evidence_turns": [6, 7, 8],
      "explanation": "Turn 7: \\"<verbatim>\\" + Turn 8: \\"<verbatim>\\" — together these match anchor level 1 because …",
      "score": 1
    }
  ]
}
```'''


def _output_format_block(rubric: Rubric) -> str:
    first_id = rubric.items[0].id if rubric.items else "<id>"
    template = _BOOLEAN_EXAMPLE if rubric.is_boolean else _ORDINAL_EXAMPLE
    return template % (rubric.name, first_id)


# ---------------------------------------------------------------------------
# Static template extraction
# ---------------------------------------------------------------------------

def _read_template_static_sections() -> str:
    """Return the static prefix of `_judge_template.md` — the role,
    input-format, and scoring-discipline sections — verbatim. The
    rubric-specific sections (Conversation, Rubric, Output format)
    are appended afterward by `build_grader_prompt`.
    """
    text = JUDGE_TEMPLATE_FILE.read_text(encoding="utf-8")
    # Cut at the "## Required output format" header — everything before it is
    # static instruction the judge needs every time.
    marker = "## Required output format"
    if marker not in text:
        raise ValueError(f"_judge_template.md missing '{marker}' section header")
    return text.split(marker, 1)[0].rstrip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_grader_prompt(rubric: Rubric, trajectory: Trajectory) -> str:
    """Assemble the full grader prompt for one (rubric, trajectory) pair."""
    parts: list[str] = []

    # 1. Static instruction prefix (role + input format + scoring discipline)
    parts.append(_read_template_static_sections())
    parts.append("")
    parts.append("---")
    parts.append("")

    # 2. The conversation under judgment
    parts.append("# Conversation")
    parts.append("")
    parts.append(render_trajectory_for_judge(trajectory))
    parts.append("")
    parts.append("---")
    parts.append("")

    # 3. The rubric body
    parts.append(_render_rubric_body(rubric))
    parts.append("")
    parts.append("---")
    parts.append("")

    # 4. The required output format (boolean vs ordinal example)
    parts.append("# Required output format")
    parts.append("")
    parts.append(
        f"Score the conversation against rubric `{rubric.name}` ({len(rubric.items)} item(s)). "
        f"Return a single JSON object with one entry per item, in the order they appear above. "
        f"Cite specific turn indices in `evidence_turns` and quote verbatim from the cited turns "
        f"in `explanation`. Output format example:"
    )
    parts.append("")
    parts.append(_output_format_block(rubric))
    parts.append("")
    parts.append(
        "Return ONLY the JSON object. No preamble, no conclusion, no free-form commentary."
    )

    return "\n".join(parts)


def build_grader_prompt_summary(rubric: Rubric, trajectory: Trajectory) -> dict[str, Any]:
    """Lightweight summary of the assembled prompt — used by `preview_judge_prompt.py`."""
    prompt = build_grader_prompt(rubric, trajectory)
    return {
        "rubric": rubric.name,
        "scoring_mode": rubric.scoring_mode,
        "n_items": len(rubric.items),
        "n_turns": trajectory.num_turns,
        "prompt_chars": len(prompt),
        "prompt_token_estimate": len(prompt) // 4,   # rough char/4 heuristic
        "prompt": prompt,
    }


__all__ = [
    "render_trajectory_for_judge",
    "build_grader_prompt",
    "build_grader_prompt_summary",
]

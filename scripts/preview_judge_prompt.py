"""
Preview the assembled grader prompt for a given rubric on a synthetic
fixture trajectory. Use this to eyeball what the LLM judge will actually
see before paying for a real judge call.

Usage:
    PYTHONPATH=. python scripts/preview_judge_prompt.py
    PYTHONPATH=. python scripts/preview_judge_prompt.py --rubric r_goal
    PYTHONPATH=. python scripts/preview_judge_prompt.py --rubric c_safety --full
"""

from __future__ import annotations

import argparse
import sys

from src.data.schemas import (
    CaregiverOutput,
    InlineAnnotation,
    PatientObservation,
    Trajectory,
    Turn,
)
from src.rewards.judge_prompt import build_grader_prompt
from src.rewards.rubric import all_rubric_names, load_rubric


def _fixture_trajectory() -> Trajectory:
    """A short 4-turn synthetic trajectory used for preview only.

    Mirrors a plausible NURSE → VERA → SPIKES → safe-deferral arc so the
    judge has something concrete to ground evidence_turns on.
    """
    turns = [
        Turn(
            turn_index=0,
            caregiver=CaregiverOutput(
                think="Patient seems anxious about her mother. Start with NURSE — name the feeling.",
                response="It sounds like you really miss her — it makes sense you'd want to see her right now.",
                raw_text="<think>...</think><response>...</response>",
            ),
            patient=PatientObservation(
                utterance="My mother is downstairs. I need to go meet her.",
                annotation=InlineAnnotation(
                    motion=["fidgeting"],
                    facial=["frowning"],
                    sound=[],
                ),
            ),
        ),
        Turn(
            turn_index=1,
            caregiver=CaregiverOutput(
                think="She is calmer now. Try VERA — validate, then bridge to a calming activity.",
                response="You miss her so much. Let's sit by the window — you can tell me about her while we have some tea.",
                raw_text="<think>...</think><response>...</response>",
            ),
            patient=PatientObservation(
                utterance="Yes... she liked the morning sunlight.",
                annotation=InlineAnnotation(
                    motion=[],
                    facial=["smiling"],
                    sound=[],
                ),
            ),
        ),
        Turn(
            turn_index=2,
            caregiver=CaregiverOutput(
                think="Window of calm — bridge to medication via SPIKES, ask permission first.",
                response="While we sit, would it be okay if we take your morning pill together?",
                raw_text="<think>...</think><response>...</response>",
            ),
            patient=PatientObservation(
                utterance="I... I think I already had it.",
                annotation=InlineAnnotation(
                    motion=["lowering head"],
                    facial=[],
                    sound=["verbal hesitation (um / uh)"],
                ),
            ),
        ),
        Turn(
            turn_index=3,
            caregiver=CaregiverOutput(
                think="She is uncertain. Don't directly contradict — show the chart, soft factual anchor.",
                response="Let me check the chart with you — it shows the morning dose hasn't been recorded yet. Would you like to take it now?",
                raw_text="<think>...</think><response>...</response>",
            ),
            patient=PatientObservation(
                utterance="Oh — alright, I'll take it.",
                annotation=InlineAnnotation(
                    motion=["nodding"],
                    facial=[],
                    sound=[],
                ),
            ),
        ),
    ]

    return Trajectory(
        trajectory_id="preview_fixture_001",
        scenario_id="preview_med_l2",
        strategy_id="r0",  # legacy field; under unified-prompt design this is just an index
        turns=turns,
        seed=42,
        terminated_by="simulator_end",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip())
    parser.add_argument(
        "--rubric",
        type=str,
        default=None,
        help="Single rubric name to preview (default: all 4)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full assembled prompt; default prints just the summary stats.",
    )
    args = parser.parse_args()

    names = [args.rubric] if args.rubric else [n for n in all_rubric_names() if not n.startswith("_")]
    traj = _fixture_trajectory()

    print(f"Fixture trajectory: {traj.num_turns} turns, scenario={traj.scenario_id}\n")

    for name in names:
        try:
            rubric = load_rubric(name)
        except FileNotFoundError as e:
            print(f"❌ {name}: {e}", file=sys.stderr)
            continue

        prompt = build_grader_prompt(rubric, traj)
        n_chars = len(prompt)
        n_tok = n_chars // 4
        print(f"=== {name} ============================================")
        print(f"  scoring_mode    : {rubric.scoring_mode}")
        print(f"  n_items         : {len(rubric.items)}")
        print(f"  prompt chars    : {n_chars}")
        print(f"  prompt tokens ~ : {n_tok}  (char/4 heuristic)")
        if args.full:
            print()
            print(prompt)
            print()
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

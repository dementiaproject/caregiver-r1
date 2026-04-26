#!/usr/bin/env python3
"""
Synthetic scenario generator for smoke / SFT seeding.

Output: a JSONL file of `Scenario` records (validated via `Scenario.model_validate`),
one scenario per line. Each scenario is a (persona, conflict, opener,
ground_truth) tuple drawn from a curated catalog of realistic
dementia-care conflicts.

Coverage per the EC-MTRL plan (zh_9 §6.1, ENGINEERING_PLAN B.4):
  - 5 conflict types   × 3 severities × 5 personas = 75 (cell, base catalog)
  - +25 random variations of opener text → 100 scenarios for smoke
  - 1500 for full training (this script can produce that with --n=1500)

Notes
-----
1. `persona_id="demma_default"` everywhere — DemMARealClient holds patient_id=0
   fixed for the smoke run; multi-persona sweep is Phase F. The persona shown
   to the CAREGIVER (name/age/subtype) varies, but DemMA's internal persona
   stays Jacob Johnson AD-early.
2. `risk_tier` is policy-derived: medication and identity conflicts default to
   "high"; temporal/event/spatial vary in {low, medium} by severity.
3. `initial_patient_utterance` is the conflict opener the patient SAYS;
   `ground_truth` is the factual reality the caregiver must reconcile (held
   in the SCENARIO, NOT shown to DemMA).

Usage
-----
    python scripts/generate_scenarios.py --n 100 --out data/scenarios_smoke.jsonl
    python scripts/generate_scenarios.py --n 1500 --out data/scenarios_train.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Ensure src/ is importable when run as a script from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.schemas import Persona, Scenario  # noqa: E402


# ---------------------------------------------------------------------------
# Catalogs
# ---------------------------------------------------------------------------

PERSONAS: list[dict] = [
    {"name": "Mrs. Wang",   "age": 76, "dementia_subtype": "alzheimers_early"},
    {"name": "Mr. Lee",     "age": 81, "dementia_subtype": "alzheimers_moderate"},
    {"name": "Mrs. Chen",   "age": 78, "dementia_subtype": "alzheimers_moderate"},
    {"name": "Mr. Zhang",   "age": 84, "dementia_subtype": "vascular_dementia"},
    {"name": "Mrs. Liu",    "age": 79, "dementia_subtype": "lewy_body_dementia"},
]


# Templates: (conflict_type) -> list of (opener, ground_truth) pairs, mixed across
# severity. We tag severity at draw time (see _draw_scenario).
CONFLICT_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "temporal": [
        ("What time is it? I need to catch the bus to work.",
         "Patient retired 18 years ago; no longer commutes. Current time: morning."),
        ("My class starts in 10 minutes — where's my school bag?",
         "Patient finished elementary school in 1955; not currently a student."),
        ("Did the new year arrive yet? When does the festival start?",
         "Today is a regular weekday in spring; no upcoming festival within a week."),
        ("The bus to my office should be here any minute now.",
         "Patient retired 22 years ago; currently in the residential care home."),
        ("It's still morning, right? I haven't had breakfast.",
         "Current time is 14:30; breakfast was finished 6 hours ago."),
    ],
    "identity": [
        ("My mother visited me today. She brought my favorite peanut cookies.",
         "Patient's mother passed away in 2014, eleven years ago."),
        ("Who are you? You're not my daughter.",
         "The caregiver is the patient's adult daughter, present in the same room."),
        ("Where is my husband? He should be home from the factory by now.",
         "Patient's husband passed away in 2018; he had retired before that."),
        ("That nurse looks exactly like my younger sister Wen.",
         "The nurse is unrelated; the patient's sister Wen lives in another city."),
        ("Are you the new postman? I want to send a letter to my grandson.",
         "The caregiver is the in-home aide; no postal service interaction is happening."),
    ],
    "event": [
        ("Did Linda get married last weekend? I missed the ceremony.",
         "Linda (the patient's niece) is engaged but the wedding is six months away."),
        ("When is my surgery? The doctor said it would be tomorrow.",
         "The patient has no scheduled surgery; last appointment was a routine check 3 weeks ago."),
        ("I think I left the kettle boiling on the stove just now.",
         "There is no active stove; the patient's apartment unit has no kitchen access."),
        ("Did the funeral happen already? I want to pay my respects.",
         "There is no recent family funeral; the patient may be conflating earlier losses."),
        ("My grandson's piano recital — when does it start?",
         "The recital was held last Saturday; the patient already attended."),
    ],
    "spatial": [
        ("Where is the bathroom in this hotel?",
         "This is the patient's residential care home, not a hotel; the bathroom is the door on the right."),
        ("How do I get home from here? I think I took the wrong train.",
         "The patient is currently at home; no train was taken today."),
        ("This isn't my room. My bed is by the window with the blue curtains.",
         "This IS the patient's room; the curtains were changed last month from blue to beige."),
        ("Is this the airport? I'm waiting for a flight to Beijing.",
         "This is the care home day room; no flight is scheduled."),
        ("Where did all the furniture go? My couch was right here yesterday.",
         "The room layout has been the same for 3 months; the patient may be recalling a former apartment."),
    ],
    "medication": [
        ("I already took my morning pill, an hour ago.",
         "Medication chart shows the morning dose has not been recorded yet."),
        ("Why are you giving me poison? You're trying to kill me.",
         "The medication is a routine prescribed dose; it is not poison."),
        ("My doctor said I don't need to take this anymore.",
         "The current prescription is active; no doctor has discontinued it."),
        ("I don't need any medicine — I'm not sick.",
         "Patient has a chronic condition with daily prescribed dose; the medicine is required."),
        ("Two pills? The doctor only said one.",
         "The current prescription is two pills; the dose was increased six weeks ago."),
    ],
}


def _risk_tier_for(conflict_type: str, severity: int) -> str:
    """Policy-driven risk_tier mapping.

    medication & identity always 'high' (Decision 3 catastrophic vectors).
    Other types: severity-driven, with 1=low, 2=medium, 3=high.
    """
    if conflict_type in {"medication", "identity"}:
        return "high"
    return {1: "low", 2: "medium", 3: "high"}[severity]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _draw_scenario(idx: int, rng: random.Random) -> Scenario:
    conflict_type = rng.choice(list(CONFLICT_TEMPLATES.keys()))
    severity = rng.choice([1, 2, 3])
    persona_template = rng.choice(PERSONAS)
    opener, ground_truth = rng.choice(CONFLICT_TEMPLATES[conflict_type])

    return Scenario(
        scenario_id=f"smoke_{idx:04d}",
        persona=Persona(
            persona_id="demma_default",   # DemMARealClient holds patient_id=0 fixed
            name=persona_template["name"],
            age=persona_template["age"],
            dementia_subtype=persona_template["dementia_subtype"],  # type: ignore[arg-type]
        ),
        conflict_type=conflict_type,         # type: ignore[arg-type]
        severity=severity,                   # type: ignore[arg-type]
        risk_tier=_risk_tier_for(conflict_type, severity),  # type: ignore[arg-type]
        initial_patient_utterance=opener,
        ground_truth=ground_truth,
    )


def generate_scenarios(n: int, seed: int = 42) -> list[Scenario]:
    rng = random.Random(seed)
    return [_draw_scenario(i, rng) for i in range(n)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic Scenario records as JSONL.")
    ap.add_argument("--n", type=int, default=100, help="Number of scenarios to emit.")
    ap.add_argument("--out", type=Path, default=Path("data/scenarios_smoke.jsonl"),
                    help="Output JSONL path.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    scenarios = generate_scenarios(args.n, seed=args.seed)
    with args.out.open("w", encoding="utf-8") as f:
        for sc in scenarios:
            f.write(json.dumps(sc.model_dump(), ensure_ascii=False) + "\n")

    # Quick distribution audit
    from collections import Counter
    by_conflict = Counter(s.conflict_type for s in scenarios)
    by_severity = Counter(s.severity for s in scenarios)
    by_risk = Counter(s.risk_tier for s in scenarios)
    print(f"[generate_scenarios] wrote {len(scenarios)} scenarios → {args.out}")
    print(f"  conflict_type: {dict(by_conflict)}")
    print(f"  severity:      {dict(by_severity)}")
    print(f"  risk_tier:     {dict(by_risk)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

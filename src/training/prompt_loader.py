"""
Caregiver system-prompt assembly + SHA-256 freezing.

The canonical caregiver system prompt is composed at run-config-lock time
from two sources:

  1. `prompts/caregiver_system_prompt.md` — the template, with a single
     `{strategy_cards}` placeholder, wrapped between `<<<BEGIN_PROMPT>>>`
     and `<<<END_PROMPT>>>` markers.
  2. `prompts/strategy_cards/NN_<name>.md` — 10 separate strategy cards
     (one per clinical framework). Filenames carry numeric prefixes
     (`01_…` … `10_…`) so alphabetical sorting locks the canonical order.

The assembled prompt is the byte-for-byte string sent to the caregiver
agent for every rollout in the run; its SHA-256 hash is committed to
`resolved_config.yaml` for reproducibility.

This module has no torch/transformers dependency — it is pure-Python file
I/O so it can be used at config-lock time before any GPU resources are
allocated.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
MAIN_PROMPT_FILE = PROMPTS_DIR / "caregiver_system_prompt.md"
CARDS_DIR = PROMPTS_DIR / "strategy_cards"

EXPECTED_NUM_CARDS = 10
PLACEHOLDER = "{strategy_cards}"
_BEGIN = "<<<BEGIN_PROMPT>>>"
_END = "<<<END_PROMPT>>>"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_template(raw_text: str) -> str:
    """Return the content strictly between BEGIN/END markers, stripped.

    The markers MUST appear on their own line (preceded and followed by a
    newline). This guards against false matches when the doc header (which
    is outside the markers) describes the marker syntax in inline prose.
    """
    begin_token = f"\n{_BEGIN}\n"
    end_token = f"\n{_END}"
    if begin_token not in raw_text:
        raise ValueError(
            f"caregiver_system_prompt.md is missing line-anchored {_BEGIN} marker"
        )
    after_begin = raw_text.split(begin_token, 1)[1]
    if end_token not in after_begin:
        raise ValueError(
            f"caregiver_system_prompt.md is missing line-anchored {_END} marker "
            f"after {_BEGIN}"
        )
    inside = after_begin.split(end_token, 1)[0]
    return inside.strip()


def _list_card_files() -> list[Path]:
    """Return the 10 card files in canonical (alphabetical) order."""
    if not CARDS_DIR.is_dir():
        raise FileNotFoundError(f"strategy_cards directory not found: {CARDS_DIR}")
    files = sorted(CARDS_DIR.glob("*.md"))
    if len(files) != EXPECTED_NUM_CARDS:
        raise ValueError(
            f"expected {EXPECTED_NUM_CARDS} strategy cards in {CARDS_DIR}, "
            f"found {len(files)}: {[f.name for f in files]}"
        )
    return files


def _load_strategy_cards() -> str:
    """Concatenate all 10 cards (alphabetical order) separated by blank lines."""
    return "\n\n".join(
        f.read_text(encoding="utf-8").strip() for f in _list_card_files()
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_caregiver_prompt() -> str:
    """Return the assembled caregiver system prompt (template + 10 cards).

    Result is cached for the process lifetime — the prompt is treated as
    frozen for an entire RL run, and the cache mirrors that contract.
    Call `clear_cache()` if you intentionally need to reload (e.g. tests
    that mutate prompt files).
    """
    template = _extract_template(MAIN_PROMPT_FILE.read_text(encoding="utf-8"))
    if PLACEHOLDER not in template:
        raise ValueError(
            f"caregiver_system_prompt.md template is missing required "
            f"placeholder {PLACEHOLDER!r}"
        )
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(
            f"caregiver_system_prompt.md template contains {PLACEHOLDER!r} "
            f"more than once — the assembly contract requires exactly one"
        )
    cards = _load_strategy_cards()
    return template.replace(PLACEHOLDER, cards)


def caregiver_prompt_sha256() -> str:
    """SHA-256 hex digest of the assembled prompt — commit into run config."""
    return hashlib.sha256(load_caregiver_prompt().encode("utf-8")).hexdigest()


def card_filenames() -> list[str]:
    """List of card filenames in canonical order (for diagnostic / audit)."""
    return [f.name for f in _list_card_files()]


def clear_cache() -> None:
    """Drop the prompt cache. Used by tests; do not call during a live run."""
    load_caregiver_prompt.cache_clear()


__all__ = [
    "load_caregiver_prompt",
    "caregiver_prompt_sha256",
    "card_filenames",
    "clear_cache",
    "PLACEHOLDER",
    "EXPECTED_NUM_CARDS",
]

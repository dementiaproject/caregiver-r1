"""
Rubric YAML loader + Pydantic schemas for the 4 trajectory-level rubrics.

  prompts/rubrics/r_goal.yaml      → weighted_boolean (HealthBench-style)
  prompts/rubrics/r_fit.yaml       → weighted_boolean (HealthBench-style, RubRIX-aligned)
  prompts/rubrics/u_terminal.yaml  → ordinal_per_dimension (sum aggregation)
  prompts/rubrics/c_safety.yaml    → ordinal_max_aggregate (max with hard-veto top tier)

The loader normalizes the three YAML key-names ("criteria" / "dimensions" /
"items") into a single `Rubric.items` attribute so downstream code can
iterate uniformly.

This module has no torch / vLLM / network dependency — it is pure Python
file I/O and Pydantic validation, so it can be used at run-config-lock
time before any GPU resources are allocated.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


REPO_ROOT = Path(__file__).resolve().parents[2]
RUBRICS_DIR = REPO_ROOT / "prompts" / "rubrics"

ScoringMode = Literal[
    "weighted_boolean",        # R_goal, R_fit
    "ordinal_per_dimension",   # u_terminal
    "binary_any_trigger",      # c_safety (Decision 3): boolean items, OR-aggregated
]


# ---------------------------------------------------------------------------
# Item schemas
# ---------------------------------------------------------------------------

class BooleanCriterion(BaseModel):
    """One HealthBench-style criterion. Judge returns `criteria_met: bool`."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    text: str
    points: float                      # may be negative (HealthBench convention)
    tags: list[str] = Field(default_factory=list)


class OrdinalLevel(BaseModel):
    """One anchored level inside an ordinal dimension/item."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    score: int
    anchor: str


class OrdinalItem(BaseModel):
    """One ordinal dimension or item. Judge returns `score: int`."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    text: str
    levels: list[OrdinalLevel]
    tags: list[str] = Field(default_factory=list)
    rubrix_dim: str | None = None       # only set for c_safety items

    @model_validator(mode="after")
    def _check_levels_monotonic(self) -> OrdinalItem:
        scores = [lvl.score for lvl in self.levels]
        if scores != sorted(scores):
            raise ValueError(f"levels for item '{self.id}' must be in ascending score order")
        return self


# ---------------------------------------------------------------------------
# Aggregation schema
# ---------------------------------------------------------------------------

class Aggregation(BaseModel):
    """How per-item scores aggregate to the trajectory-level reward."""
    model_config = ConfigDict(extra="allow")   # tolerate documentation fields like `formula`, `semantics`

    type: Literal["weighted_sum", "sum", "max", "binary_any"]
    range: list[int] | None = None
    raw_range: list[int] | None = None
    paper_normalized_range: list[int] | None = None
    hard_veto_threshold: int | None = None


# ---------------------------------------------------------------------------
# Top-level rubric model
# ---------------------------------------------------------------------------

RubricItem = Union[BooleanCriterion, OrdinalItem]


class Rubric(BaseModel):
    """Parsed rubric. The original YAML may use `criteria` / `dimensions` /
    `items` as the list key; the loader normalizes them all into `items`."""
    model_config = ConfigDict(frozen=True)

    name: str
    version: int
    scoring_mode: ScoringMode
    description: str
    items: list[RubricItem]
    aggregation: Aggregation

    # ------------------------------------------------------------------ helpers

    @property
    def is_boolean(self) -> bool:
        """True if every item is a HealthBench-style boolean criterion.
        Both `weighted_boolean` (R_goal/R_fit) and `binary_any_trigger`
        (c_safety) qualify — they share the same per-item BooleanCriterion
        schema, only the aggregation differs."""
        return self.scoring_mode in ("weighted_boolean", "binary_any_trigger")

    @property
    def is_ordinal(self) -> bool:
        return self.scoring_mode == "ordinal_per_dimension"

    def item_ids(self) -> list[str]:
        return [it.id for it in self.items]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_YAML_KEY_BY_MODE: dict[str, str] = {
    "weighted_boolean":         "criteria",
    "ordinal_per_dimension":    "dimensions",
    "binary_any_trigger":       "criteria",   # c_safety (Decision 3)
}


def _coerce_items(scoring_mode: str, raw_items: list[dict]) -> list[RubricItem]:
    if scoring_mode in ("weighted_boolean", "binary_any_trigger"):
        return [BooleanCriterion.model_validate(it) for it in raw_items]
    return [OrdinalItem.model_validate(it) for it in raw_items]


@lru_cache(maxsize=None)
def load_rubric(name: str) -> Rubric:
    """Load a rubric YAML from `prompts/rubrics/{name}.yaml`.

    Cached per-name; call `load_rubric.cache_clear()` if you intentionally
    edit a rubric file mid-process (tests only — production rubrics are
    frozen at run-config-lock time).
    """
    path = RUBRICS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"rubric '{name}' not found at {path}; "
            f"available: {sorted(p.stem for p in RUBRICS_DIR.glob('*.yaml'))}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    scoring_mode = raw["scoring_mode"]

    expected_key = _YAML_KEY_BY_MODE.get(scoring_mode)
    if expected_key is None:
        raise ValueError(f"unknown scoring_mode {scoring_mode!r} in {path}")
    if expected_key not in raw:
        raise ValueError(
            f"rubric {path} declares scoring_mode={scoring_mode!r} but is missing "
            f"the expected key {expected_key!r}"
        )

    items = _coerce_items(scoring_mode, raw[expected_key])
    return Rubric(
        name=raw["name"],
        version=raw["version"],
        scoring_mode=scoring_mode,
        description=raw["description"],
        items=items,
        aggregation=Aggregation.model_validate(raw["aggregation"]),
    )


def all_rubric_names() -> list[str]:
    """Return the canonical list of rubric stems present on disk."""
    return sorted(p.stem for p in RUBRICS_DIR.glob("*.yaml"))


__all__ = [
    "BooleanCriterion",
    "OrdinalLevel",
    "OrdinalItem",
    "Aggregation",
    "Rubric",
    "RubricItem",
    "ScoringMode",
    "load_rubric",
    "all_rubric_names",
]

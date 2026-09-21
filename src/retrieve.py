"""Phase 6 - the heart of it: search, score, re-rank, fall back.

Cosine similarity is symmetric and therefore answers the wrong question.
"Which recipe is most similar to my ingredients?" rewards a sixteen-item
biryani that happens to contain all four things you own. The right
question is "which recipe is most *covered* by my ingredients?", which
divides by the recipe's size and is not symmetric.
"""
from __future__ import annotations

from dataclasses import dataclass

# Starting weights. Tuned against eval/, not guessed at.
W_COVERAGE = 0.55
W_SIMILARITY = 0.25
W_MISSING = 0.05
W_REGION = 0.15

# Below this coverage, tier 1 has nothing worth offering and we widen.
FALLBACK_THRESHOLD = 0.40


@dataclass
class Result:
    recipe_id: int
    score: float
    coverage: float
    missing: list[str]
    tier: int


def coverage(pantry: set[str], recipe: set[str]) -> float:
    """Fraction of the recipe the user can already supply.

    Divides by the recipe size, not the pantry size. That asymmetry is
    the whole point: it punishes long recipes you cannot complete.
    """
    raise NotImplementedError("phase 6")


def search(query, k: int = 50) -> list[Result]:
    """Semantic recall over tier 1, then coverage re-rank, then fallback."""
    raise NotImplementedError("phase 6")

"""Phase 3 - cuisine normalisation, dish recovery, tier assignment.

The Cuisine column is unreliable in two ways: values carry a byte-order
mark, and many Gujarati and Punjabi dishes are filed under generic
labels. Both are handled here so no other module has to know.
"""
from __future__ import annotations

import pandas as pd

TIER_CORE = 1
TIER_FALLBACK = 2


def normalise_cuisine(value: str) -> str:
    """Strip BOM and zero-width characters, collapse whitespace, lowercase.

    Without this, `Cuisine == "Gujarati Recipes"` matches zero rows,
    silently, because every Gujarati value ends in U+FEFF.
    """
    raise NotImplementedError("phase 3")


def recover_by_dish_name(df: pd.DataFrame) -> pd.DataFrame:
    """Reclaim recipes whose cuisine label is wrong but whose title is not."""
    raise NotImplementedError("phase 3")


def assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 1 = Gujarati and Punjabi. Tier 2 = other Indian vegetarian."""
    raise NotImplementedError("phase 3")


def main() -> None:
    raise NotImplementedError("phase 3")


if __name__ == "__main__":
    main()

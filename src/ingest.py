"""Phase 2 - download, apply dietary filters, write parquet.

Order matters: the Diet label is a weak first pass, the ingredient
blocklist is the real gate, and egg exclusion runs last so its false
positives can be reasoned about separately.
"""
from __future__ import annotations

import pandas as pd


def download() -> pd.DataFrame:
    """Fetch the raw CSV, caching it under data/raw/."""
    raise NotImplementedError("phase 2")


def apply_diet_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-vegetarian rows by label, then by ingredient blocklist."""
    raise NotImplementedError("phase 2")


def exclude_egg(df: pd.DataFrame) -> pd.DataFrame:
    """Remove egg recipes, matching entities rather than substrings.

    A substring match on "egg" drops 143 recipes, 119 of which are
    aubergine dishes whose ingredient list spells brinjal "Eggplant".
    """
    raise NotImplementedError("phase 2")


def main() -> None:
    raise NotImplementedError("phase 2")


if __name__ == "__main__":
    main()

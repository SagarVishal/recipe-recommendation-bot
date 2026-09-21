"""Corpus invariants.

These pin the findings that would otherwise regress silently: the BOM in
the cuisine column, the unreliable diet label, and the untranslated rows.
"""
import re

import pandas as pd
import pytest

from src import paths

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
MEAT_TERMS = ["chicken", "mutton", "prawn", "fish", "bacon", "gelatin"]


@pytest.fixture(scope="module")
def core():
    if not paths.CORE_CSV.exists():
        pytest.skip("run `python3 -m src.build_corpus` first")
    return pd.read_csv(paths.CORE_CSV)


def test_corpus_is_not_empty(core):
    assert len(core) > 250


def test_both_regions_present_and_gujarati_survived_the_bom(core):
    """An equality check on Cuisine matches zero rows; this proves we didn't."""
    counts = core.region.value_counts()
    assert counts.get("Gujarati", 0) > 100
    assert counts.get("Punjabi", 0) > 100


def test_no_meat_or_fish_survived(core):
    haystack = (core.name + " | " + core.ingredients).str.lower()
    for term in MEAT_TERMS:
        hits = haystack[haystack.str.contains(rf"\b{term}\b", regex=True, na=False)]
        assert hits.empty, f"{term} found in {len(hits)} recipes"


def test_no_egg_survived_but_eggplant_did(core):
    haystack = (core.name + " | " + core.ingredients).str.lower()
    assert haystack.str.contains(r"\begg\b|\beggs\b").sum() == 0
    # The substring trap: aubergine recipes must NOT have been collateral.
    assert haystack.str.contains("brinjal|eggplant|baingan|aubergine").sum() > 0


def test_everything_is_in_english(core):
    """719 source rows never got translated; none may reach the corpus."""
    for column in ("name", "ingredients", "instructions"):
        assert core[column].fillna("").map(lambda s: bool(DEVANAGARI.search(s))).sum() == 0


def test_every_recipe_keeps_its_attribution(core):
    assert core.url.notna().all()

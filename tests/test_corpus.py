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


def test_no_non_indian_cuisines_leaked_into_the_core(core):
    """'lassi' is a substring of 'c-lassi-c', which once filed a Greek salad
    and a Chinese dessert as Punjabi. Word-boundary matching fixed it."""
    indian = (
        "indian", "punjab", "gujarat", "rajasth", "maharash", "bengali",
        "kerala", "tamil", "karnataka", "andhra", "goan", "sindhi", "kashmiri",
        "awadhi", "chettinad", "parsi", "oriya", "hyderab", "mangalor",
        "malvani", "konkan", "coorg", "bihari", "assam", "sattvic", "uttar",
        "haryana", "himachal", "delhi", "mughlai", "lucknowi", "udupi",
        "north east",
    )
    leaked = core[~core.cuisine.str.lower().apply(
        lambda c: any(k in c for k in indian))]
    assert leaked.empty, f"non-Indian cuisines in core: {list(leaked.cuisine.unique())}"


def test_every_recipe_has_ingredients(core):
    """A NaN here propagates into search_text and breaks the embedding call."""
    assert core.ingredients.notna().all()
    assert (core.ingredients.str.strip() != "").all()


def test_staples_are_excluded_from_the_coverage_denominator():
    """Salt is in 84% of recipes. Counting it as 'missing' made a realistic
    four-item pantry incapable of scoring above 25% on anything, so the
    fallback tier fired on every query."""
    from src.corpus import coverage, staples

    assert "salt" in staples()
    assert "turmeric powder" in staples()
    # Paneer and vegetables are the point of the recommendation, not staples.
    assert "paneer" not in staples()
    assert "onion" not in staples()

    recipe = {"besan", "curd", "salt", "turmeric powder", "oil", "cumin seeds"}
    score, missing = coverage({"besan", "curd"}, recipe)
    assert score == 1.0, "pantry covers every shoppable ingredient"
    assert missing == []


def test_a_realistic_pantry_finds_cookable_core_recipes(core):
    """The regression that the live demo exposed: with staples in the
    denominator, zero recipes cleared the 34% fallback threshold."""
    from src.corpus import coverage, parse_ingredients

    pantry = {"besan", "curd", "ginger", "green chilli"}
    scores = [coverage(pantry, parse_ingredients(i))[0] for i in core.ingredients]
    assert max(scores) > 0.5
    assert sum(1 for s in scores if s >= 0.34) >= 5

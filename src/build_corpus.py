"""Build the curated Gujarati + Punjabi vegetarian eggless corpus.

Reads the raw Archana's Kitchen CSV and writes a small, clean CSV that the
notebook loads directly. Run once: `python3 -m src.build_corpus`.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd
import yaml

from src import paths

VEG_DIETS = {
    "Vegetarian", "High Protein Vegetarian", "Vegan",
    "No Onion No Garlic (Sattvic)", "Diabetic Friendly",
    "Gluten Free", "Sugar Free Diet",
}

GUJARATI_DISHES = [
    "dhokla", "khaman", "thepla", "khandvi", "undhiyu", "handvo", "fafda",
    "muthia", "patra", "dal dhokli", "shrikhand", "basundi", "khichu",
    "sev tameta", "ringan", "bhakri", "dabeli", "khakhra", "farsi puri",
    "gujarati", "surti", "kathiyawadi", "lilva", "chorafali", "ghughra",
    "mohanthal", "doodhpak", "lapsi", "osaman",
]

PUNJABI_DISHES = [
    "chole", "chana masala", "rajma", "sarson", "makki di", "paneer butter",
    "butter paneer", "amritsari", "lassi", "punjabi", "kadhi pakora",
    "pindi", "dal makhani", "aloo paratha", "paneer tikka", "bhature",
    "matar paneer", "shahi paneer", "malai kofta", "paneer bhurji",
    "tandoori roti", "naan",
]

INDIAN_CUISINE_KEYWORDS = [
    "indian", "bengali", "maharashtrian", "kerala", "tamil", "karnataka",
    "rajasthani", "punjab", "gujarat", "andhra", "hyderabad", "goan",
    "sindhi", "kashmiri", "awadhi", "chettinad", "udupi", "mangalorean",
    "assamese", "bihari", "oriya", "malvani", "konkan", "coorg", "parsi",
    "lucknowi", "mughlai", "sattvic", "uttar pradesh", "haryana",
    "himachal", "delhi", "north east",
]


def clean_text(value: object) -> str:
    """Strip BOM and zero-width characters, collapse whitespace.

    Every Gujarati row is spelled 'Gujarati Recipes﻿'. Without this,
    an equality check on the cuisine matches zero rows and nothing errors.
    """
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("﻿", "").replace("​", "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def is_untranslated(text: str) -> bool:
    """True when the 'Translated' column still holds Devanagari.

    The source ran Google Translate over Hindi rows, but not all of them.
    Their TranslatedIngredients are still Hindi, so an English pantry can
    never match them and they would sit in the corpus as dead weight.
    """
    return bool(DEVANAGARI.search(text))


def load_blocklist() -> tuple[list[str], list[str]]:
    with open(paths.EXCLUDED_INGREDIENTS_YAML) as handle:
        config = yaml.safe_load(handle)
    terms: list[str] = []
    for group in ("meat", "seafood", "hidden_animal", "egg"):
        terms.extend(config.get(group, []))
    return terms, config.get("allow_titles", [])


def contains_term(text: str, term: str) -> bool:
    """Word-boundary match, so 'eggplant' never counts as 'egg'."""
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def matches_dish(title: str, dishes: list) -> bool:
    """Word-boundary match for dish names.

    Substring matching looks fine until you notice "lassi" is inside
    "c-lassi-c", which quietly filed a Greek salad, a Chinese dessert and an
    Italian cake as Punjabi. Same lesson as the eggplant trap, one layer up.
    """
    return any(contains_term(title, dish) for dish in dishes)


def build() -> pd.DataFrame:
    df = pd.read_csv(paths.RAW_DIR / "IndianFoodDataset.csv")
    start = len(df)

    df = df[df["Diet"].isin(VEG_DIETS)].copy()
    after_diet = len(df)

    for column in ("TranslatedRecipeName", "TranslatedIngredients",
                   "TranslatedInstructions", "Cuisine", "Course", "Diet"):
        df[column] = df[column].map(clean_text)

    blocked, allowed_titles = load_blocklist()
    haystack = (df["TranslatedRecipeName"] + " | " + df["TranslatedIngredients"]).str.lower()
    titles = df["TranslatedRecipeName"].str.lower()

    rescued = titles.apply(lambda t: any(a in t for a in allowed_titles))
    blocked_hit = haystack.apply(lambda h: any(contains_term(h, t) for t in blocked))
    df = df[~(blocked_hit & ~rescued)].copy()
    after_blocklist = len(df)

    cuisine_lc = df["Cuisine"].str.lower()
    title_lc = df["TranslatedRecipeName"].str.lower()

    is_indian = cuisine_lc.apply(
        lambda c: any(k in c for k in INDIAN_CUISINE_KEYWORDS))

    # A dish name only promotes a recipe if the cuisine is Indian too.
    # Otherwise "Gujarati Dhokla Pizza" (Fusion) and "Mango Shrikhand Taco"
    # (Mexican) land in a corpus that promises Gujarati and Punjabi food.
    is_guj = cuisine_lc.str.contains("gujarat") | (
        is_indian & title_lc.apply(lambda t: matches_dish(t, GUJARATI_DISHES)))
    is_pun = (cuisine_lc.str.contains("punjab") | (
        is_indian & title_lc.apply(lambda t: matches_dish(t, PUNJABI_DISHES)))) & ~is_guj

    df["region"] = "other"
    df.loc[is_pun, "region"] = "Punjabi"
    df.loc[is_guj, "region"] = "Gujarati"
    df["tier"] = 2
    df.loc[is_guj | is_pun, "tier"] = 1
    df = df[(df.tier == 1) | is_indian].copy()

    df = df.rename(columns={
        "TranslatedRecipeName": "name",
        "TranslatedIngredients": "ingredients",
        "TranslatedInstructions": "instructions",
        "Cuisine": "cuisine",
        "Course": "course",
        "Diet": "diet",
        "TotalTimeInMins": "total_time_mins",
        "Servings": "servings",
        "URL": "url",
    })
    df = df[["name", "ingredients", "instructions", "cuisine", "course",
             "diet", "region", "tier", "total_time_mins", "servings", "url"]]
    # A recipe with no ingredients cannot be matched against a pantry, and
    # its NaN propagates into every derived string.
    df = df[df["ingredients"].str.strip().astype(bool)].copy()

    before_lang = len(df)
    # Check all three text columns: some rows have English ingredients but
    # Hindi instructions, which would surface as Hindi cooking steps.
    untranslated = (
        df["ingredients"].map(is_untranslated)
        | df["name"].map(is_untranslated)
        | df["instructions"].map(is_untranslated)
    )
    df = df[~untranslated].copy()
    dropped_lang = before_lang - len(df)

    df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)

    print(f"raw rows                 {start}")
    print(f"after vegetarian diets   {after_diet}")
    print(f"after ingredient blocklist {after_blocklist}")
    print(f"tier 1 (Gujarati+Punjabi) {(df.tier == 1).sum()}")
    print(f"  Gujarati               {(df.region == 'Gujarati').sum()}")
    print(f"  Punjabi                {(df.region == 'Punjabi').sum()}")
    print(f"dropped, still in Hindi  {dropped_lang}")
    print(f"tier 2 (other Indian)    {(df.tier == 2).sum()}")
    return df


def main() -> None:
    paths.ensure_dirs()
    df = build()
    core = df[df.tier == 1]
    core.to_csv(paths.CORE_CSV, index=False)
    df.to_csv(paths.FULL_CSV, index=False)
    print(f"\nwrote {paths.CORE_CSV.name} ({len(core)} rows)")
    print(f"wrote {paths.FULL_CSV.name} ({len(df)} rows)")


if __name__ == "__main__":
    main()

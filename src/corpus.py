"""Load the curated corpus and turn ingredient strings into comparable entities."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Set, Tuple

import pandas as pd

from src import paths

_UNITS = (
    r"cups?|tablespoons?|teaspoons?|tbsp|tsp|grams?|gms?|gm|kg|ml|litres?|liters?|"
    r"inch|inches|cloves?|sprigs?|pinch|handfuls?|numbers?|nos?|pieces?|bunch|"
    r"stalks?|packets?|cans?|drops?"
)
_QTY = re.compile(r"^[\d\s/¼½¾⅓⅔⅛.\-]*\s*(?:" + _UNITS + r")?\s*", re.I)
_BRACKET = re.compile(r"\(([^)]*)\)")

# Words that are never a pantry ingredient on their own.
_STOPWORDS = {
    "to taste", "as required", "as needed", "for garnish", "for frying",
    "for cooking", "for tempering", "optional", "chopped", "finely chopped",
    "as per taste", "for deep frying", "to sprinkle", "for serving",
}


def parse_ingredients(raw: object) -> Set[str]:
    """'2 tablespoon Gram flour (besan)' -> {'gram flour', 'besan'}.

    The bracketed text is a gift from this dataset: it lists regional
    synonyms, so 'Karela (Bitter Gourd/ Pavakkai)' yields three usable names
    for one vegetable, and a user typing any of them will match.
    """
    entities: Set[str] = set()
    for part in str(raw).split(","):
        part = part.strip().lower()
        if not part:
            continue
        for bracket in _BRACKET.findall(part):
            for piece in re.split(r"[/,]", bracket):
                piece = re.sub(r"[^a-z\s]", " ", piece).strip()
                piece = re.sub(r"\s+", " ", piece)
                if len(piece) > 2 and piece not in _STOPWORDS:
                    entities.add(piece)
        part = _BRACKET.sub(" ", part)
        part = re.sub(r"\s*-\s*.*$", "", part)      # drop '- finely chopped'
        part = _QTY.sub("", part)
        part = re.sub(r"[^a-z\s]", " ", part)
        part = re.sub(r"\s+", " ", part).strip()
        if len(part) > 2 and part not in _STOPWORDS:
            entities.add(part)
    return entities


def coverage(pantry: Set[str], recipe: Set[str]) -> Tuple[float, List[str]]:
    """Fraction of the recipe's shoppable ingredients the pantry supplies.

    Two things make this the right measure rather than cosine similarity:

    1. It divides by the RECIPE's size, not the pantry's. A sixteen-ingredient
       dish containing everything you own is still not cookable, and a
       symmetric similarity score cannot express that.
    2. Staples are excluded from the denominator. Salt is in 84% of these
       recipes; counting it as "missing" is noise, and with staples included
       a realistic four-item pantry could not exceed 25% coverage on any
       recipe in the corpus.
    """
    shoppable = {item for item in recipe if item not in staples()}
    if not shoppable:
        return 1.0, []          # a dish of nothing but cupboard staples
    have = {item for item in shoppable
            if any(p in item or item in p for p in pantry)}
    return len(have) / len(shoppable), sorted(shoppable - have)


@lru_cache(maxsize=1)
def staples() -> frozenset:
    """Ingredients assumed present in any kitchen, read from config.

    Excluded from the coverage denominator so that "coverage" means the
    fraction of the ingredients you would actually have to shop for.
    """
    import yaml

    with open(paths.CONFIG_DIR / "pantry_staples.yaml") as handle:
        config = yaml.safe_load(handle) or {}
    items = set()
    for group in config.values():
        items.update(str(item).lower().strip() for item in group)
    return frozenset(items)


@lru_cache(maxsize=2)
def load_recipes(core_only: bool = True) -> pd.DataFrame:
    """Read the committed CSV and pre-compute each recipe's ingredient set."""
    path = paths.CORE_CSV if core_only else paths.FULL_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing. Run: python -m src.build_corpus"
        )
    df = pd.read_csv(path)
    # Defensive: one NaN anywhere turns the concatenation below into a float,
    # and the embedding API then fails with "expected string or bytes-like
    # object" a hundred recipes into the batch.
    for column in ("name", "ingredients", "cuisine", "course", "region"):
        df[column] = df[column].fillna("").astype(str)
    df["entities"] = df["ingredients"].map(parse_ingredients)
    df["search_text"] = (
        df["name"] + "\nCuisine: " + df["cuisine"] + " (" + df["region"] + ")"
        + "\nCourse: " + df["course"] + "\nIngredients: " + df["ingredients"]
    )
    return df


@lru_cache(maxsize=1)
def vocabulary() -> frozenset:
    """Every ingredient entity that appears anywhere in the corpus.

    Used to decide whether a word the user typed is a real ingredient. If it
    isn't in here, no recipe can be scored on it, so recognising it would be
    theatre - it falls through to the semantic half of retrieval instead.
    """
    vocab: Set[str] = set()
    for entities in load_recipes(core_only=False)["entities"]:
        vocab |= entities
    return frozenset(vocab)

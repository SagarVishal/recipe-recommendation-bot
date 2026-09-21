"""Every filesystem path in one place.

Modules import from here rather than building paths themselves, so moving
the data directory is a one-line change instead of a search-and-replace.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "config"

RAW_CSV = RAW_DIR / "indian_food.csv"
RECIPES_PARQUET = PROCESSED_DIR / "recipes.parquet"
VECTORS_NPY = PROCESSED_DIR / "vectors.npy"

DISHES_YAML = CONFIG_DIR / "dishes.yaml"
EXCLUDED_INGREDIENTS_YAML = CONFIG_DIR / "excluded_ingredients.yaml"

# Upstream mirror of the Archana's Kitchen corpus (24 MB, no auth required).
SOURCE_URL = (
    "https://raw.githubusercontent.com/nileshely/Indian-Food/main/IndianFoodDataset.csv"
)

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

# Curated corpora committed to the repo so the notebook runs without the 24 MB raw file.
DATA_DIR = ROOT / "data"
CORE_CSV = DATA_DIR / "recipes_core.csv"      # tier 1: Gujarati + Punjabi
FULL_CSV = DATA_DIR / "recipes_all.csv"       # tier 1 + tier 2 fallback

DISHES_YAML = CONFIG_DIR / "dishes.yaml"
EXCLUDED_INGREDIENTS_YAML = CONFIG_DIR / "excluded_ingredients.yaml"



def ensure_dirs() -> None:
    """Create the data directories if they are missing.

    They are gitignored, so a fresh clone does not have them. Writers call
    this rather than it happening on import - importing a module should not
    touch the filesystem.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Upstream mirror of the Archana's Kitchen corpus (24 MB, no auth required).
SOURCE_URL = (
    "https://raw.githubusercontent.com/nileshely/Indian-Food/main/IndianFoodDataset.csv"
)

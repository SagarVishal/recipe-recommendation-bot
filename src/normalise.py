"""Phase 4 - turn raw ingredient strings into canonical entities.

Input looks like "6 Karela (Bitter Gourd/ Pavakkai) - deseeded".
Output is the entity "karela", with "bitter gourd" and "pavakkai"
registered as synonyms. The brackets are a free bilingual synonym map.
"""
from __future__ import annotations


def parse_ingredient(raw: str) -> tuple[str, list[str]]:
    """Return (canonical entity, synonyms) for one raw ingredient string."""
    raise NotImplementedError("phase 4")


def build_vocabulary(all_ingredients: list[str]) -> dict[str, str]:
    """Map every surface form to its canonical entity."""
    raise NotImplementedError("phase 4")


def main() -> None:
    raise NotImplementedError("phase 4")


if __name__ == "__main__":
    main()

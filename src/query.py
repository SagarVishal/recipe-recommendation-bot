"""Phase 6 - turn a chat message into a structured query."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Query:
    """What the user actually asked for, once parsed apart."""

    pantry: set[str] = field(default_factory=set)
    filters: set[str] = field(default_factory=set)  # vegan, gluten-free, ...
    free_text: str = ""  # the vague part: "something light for dinner"


def parse(message: str, vocabulary: dict[str, str]) -> Query:
    """Split a message into pantry items, hard filters and free text."""
    raise NotImplementedError("phase 6")

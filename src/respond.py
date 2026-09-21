"""Phase 7 - format results into replies.

Templates, not generation. Every number shown is computed, so a
template states it exactly where a language model would paraphrase it
into something vaguer and occasionally wrong.
"""
from __future__ import annotations


def format_results(results: list, pantry: set[str]) -> str:
    """Render ranked results, flagging any that came from tier 2."""
    raise NotImplementedError("phase 7")

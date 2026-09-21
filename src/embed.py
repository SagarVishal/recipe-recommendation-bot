"""Phase 5 - build the vector index.

384 core recipes means a brute-force cosine over a small numpy array
beats any vector database: roughly 3 ms, and one fewer dependency.
"""
from __future__ import annotations


def build_index() -> None:
    """Embed title + parsed ingredients + cuisine, persist as .npy."""
    raise NotImplementedError("phase 5")


def main() -> None:
    raise NotImplementedError("phase 5")


if __name__ == "__main__":
    main()

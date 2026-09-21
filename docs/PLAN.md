# Build plan

The working plan lives as a living document and is kept current there
through the build. It will be exported into this file at phase 9, so the
repository carries a standalone copy of the architecture and reasoning.

Summary of the design:

- **Corpus** 384 Gujarati and Punjabi vegetarian, eggless recipes (tier 1),
  with 3,848 other Indian vegetarian recipes as a labelled fallback (tier 2).
- **Retrieval** semantic recall, then re-ranking by ingredient coverage.
- **Key idea** cosine similarity is symmetric and answers the wrong
  question; coverage divides by the recipe's size and answers the right one.

See the README for the dataset findings that shaped all three.

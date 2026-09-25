# Architecture note

The deliverable is `notebooks/recipe_rag_workshop.ipynb` — a LangChain +
Google Gemini RAG system over Gujarati and Punjabi vegetarian recipes,
written as a teaching notebook in eight sections.

## Pipeline

    6,871 raw rows
      -> vegetarian diet labels            5,875
      -> ingredient blocklist              5,620
      -> untranslated rows dropped          -719
      -> cuisine normalised, dishes recovered
      -> curated: 910 (all-India, Gujarati and Punjabi weighted)
         tier 2: 3,164 other Indian, labelled fallback

    question -> Gemini embeddings -> vector store -> top-k
             -> coverage re-rank + Gujarati prior
             -> Gemini, constrained to the retrieved candidates

## Ranking

    score = 0.55*coverage + 0.25*similarity - 0.05*(missing/k) + 0.15*regional_prior
    coverage = |pantry & recipe| / |recipe|

Coverage divides by the recipe's size, not the pantry's. That asymmetry is
the point: a sixteen-ingredient dish containing everything you own is not
cookable, and symmetric cosine similarity cannot express that.

The regional prior is capped so it can never overturn a coverage gap.
Cookability wins; the prior breaks near-ties.

## Why two stages

Semantic retrieval gives recall - it tolerates "aubergine" for "brinjal"
and answers questions with no ingredients in them at all. Coverage gives
precision - it orders candidates by what can actually be cooked. Neither
stage can do the other's job.

At 910 documents a brute-force cosine is ~3 ms, so there is no vector
database. FAISS or Chroma would earn their place past ~100k documents.

See the README for the four dataset findings that shaped the corpus.

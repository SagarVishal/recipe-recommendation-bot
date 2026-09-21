# Gujarati Recipe Bot

> Tell it what's in your pantry, it tells you what you can actually cook tonight — ranked by how little you're missing, not by how similar the text looks.

A domain-specific chatbot over **Gujarati** cuisine, with **Punjabi** as a secondary cuisine. Lacto-vegetarian and eggless throughout. Built for the POD exercise.

**Status:** in development — see [`docs/PLAN.md`](docs/PLAN.md) for the build plan.

---

## The problem this solves

Ask most recipe search engines what you can make with *besan, curd, ginger and green chilli* and they rank by text similarity. That's the wrong question.

Cosine similarity is symmetric. A 16-ingredient undhiyu that happens to contain all four of your items scores beautifully — and you can't cook it, because you're missing twelve things. A four-ingredient Gujarati kadhi you *can* cook scores lower.

The question a hungry person is actually asking is not *"which recipe is most similar to my ingredients?"* but *"which recipe is most **covered** by my ingredients?"* Those are different questions, and only the second is asymmetric.

```
coverage = |pantry ∩ recipe| / |recipe|
```

Dividing by the recipe's size, not the pantry's, is what punishes the undhiyu correctly.

The full ranking adds a semantic term, a penalty for missing items, and a regional prior that puts Gujarati first:

```
score = w₁·coverage + w₂·similarity − w₃·(missing/k) + w₄·regional_prior
```

| Cuisine | Prior |
|---|---|
| Gujarati | 1.0 |
| Punjabi | 0.4 |
| Tier 2 (see below) | 0 |

The prior is capped so it can never overturn a large coverage gap. A Gujarati dish you can't cook must not outrank a Punjabi one you can. Cookability wins; the prior breaks near-ties.

## Scope and the two tiers

| Tier | Contents | Recipes |
|---|---|---|
| **1 — core** | Gujarati (152) + Punjabi (232) | **384** |
| 2 — fallback | Other Indian vegetarian eggless | 3,848 |

Tier 1 is the bot's world. Tier 2 exists so it never dead-ends.

384 recipes is a small corpus, and with a median of 13 ingredients each, plenty of pantries won't cover anything in it well. A strict Gujarati-and-Punjabi-only bot would answer *"nothing matches"* often enough to be useless. So when no tier-1 recipe clears the coverage threshold, the bot says so and offers the nearest tier-2 match, **explicitly labelled as outside your cuisines**:

> Nothing Gujarati or Punjabi matches what you have. The closest is a **Rajasthani** gatte ki sabzi — you have 5 of 6 ingredients.

Never a silent substitution. The user always knows which tier an answer came from, and tier 2 never appears when tier 1 has something cookable.

## Recovering mislabelled recipes

By the `Cuisine` column alone, this dataset has **191** Gujarati and Punjabi vegetarian recipes. That undercounts badly, because the labelling is inconsistent — *dal dhokli* and *onion thepla* sit under `North Indian Recipes`, *dhania chole masala* under plain `Indian`.

Matching a curated list of canonical dish names against recipe titles recovers them, taking the core from **191 to 384**. Where the extra 193 were hiding:

| Label they were filed under | Recovered |
|---|---|
| North Indian Recipes | 129 |
| Indian | 58 |
| Rajasthani | 17 |
| Maharashtrian Recipes | 12 |

The dish list is deliberately conservative. Loosening it to pan-Indian terms like *pakora*, *kadhi* and *stuffed paratha* pushes the count to roughly 466, but those dishes belong to several cuisines at once — Gujarati kadhi, Punjabi kadhi pakora and Sindhi kadhi are all real. Claiming them all as Gujarati or Punjabi would be inflating the number rather than improving the corpus.

### The BOM in the cuisine column

Every Gujarati row is spelled `'Gujarati Recipes﻿'` — with a byte-order mark glued to the end:

```python
df[df.Cuisine == "Gujarati Recipes"]    # 0 rows
df[df.Cuisine.str.contains("Gujarati")] # 152 rows
```

The obvious equality check silently returns nothing, and nothing errors. Cuisine values are normalised — BOM and zero-width characters stripped, whitespace collapsed — before any comparison, with a unit test pinning the count so it can't regress.

## Dietary scope

**Lacto-vegetarian.** Dairy is in, egg is out. 201 of the 384 core recipes use dairy, which is unsurprising for two cuisines built on curd, ghee and paneer. `vegan` remains available as an opt-in query filter.

### Why the `Diet` column isn't trusted

- **The non-veg label is misspelled.** 427 rows read `Non Vegeterian`. A filter written as `Diet != "Non Vegetarian"` lets every one through.
- **55 recipes labelled `Vegetarian` contain meat or fish** — *Singapore Style Chicken Layered Fried Rice*, *Andaman Style Steamed Garlic Prawns*, *Baked Fish In Coconut Milk*.
- **423 recipes labelled `Vegetarian` contain egg**, so the separate `Eggetarian` label is applied inconsistently.

The label is a weak first pass; an ingredient blocklist in [`config/excluded_ingredients.yaml`](config/excluded_ingredients.yaml) is the real gate.

### The eggplant problem

Excluding egg looks like a substring check. Measured: `"egg" in text` drops **143 recipes, 119 of which contain no egg at all** — they're aubergine dishes, because brinjal's listed synonyms include "Eggplant". Word-boundary matching on parsed entities drops 29 instead, and correctly keeps the recipes whose names contain *egg**less***.

Same lesson as the meat blocklist, from the other direction: **match parsed entities, never raw strings.**

## How it works

```
Archana's Kitchen dataset (6,871 recipes)
        ↓  diet filter + ingredient blocklist + egg exclusion
        ↓  cuisine normalisation + dish-name recovery
   tier 1: 384 Gujarati & Punjabi    tier 2: 3,848 other Indian
        ↓  parse ingredients → normalised entities
        ↓  embed (MiniLM-L6-v2, 384-dim)
        ↓
   user message → parse pantry → tier-1 recall → coverage re-rank
                → threshold met?  yes → answer
                                  no  → tier-2 fallback, labelled
```

At this corpus size a brute-force cosine runs in about 3 ms, so there's no vector database and no need for one. The two stages earn their place on **capability**, not speed: coverage scoring needs exact set membership and can't parse *"something light for dinner"*; semantic search has no notion of what's missing from your kitchen.

No LLM API anywhere in the loop. Everything runs locally on CPU.

## Source

[6000+ Indian Food Recipes](https://www.kaggle.com/datasets/kanishk307/6000-indian-food-recipes-dataset), crawled from [Archana's Kitchen](https://www.archanaskitchen.com/); [CSV mirror](https://github.com/nileshely/Indian-Food).

## Quick start

```bash
pip install -r requirements.txt
make data      # download, filter, classify, embed
make run       # launch the chat interface
```

## Project structure

```
src/ingest.py      download → filter → parquet
src/cuisine.py     normalisation, dish-name recovery, tiering
src/normalise.py   ingredient parsing, synonyms, vocabulary
src/embed.py       build the vector index
src/query.py       parse a message into pantry + filters
src/retrieve.py    tier-1 search, coverage re-rank, tier-2 fallback
src/respond.py     format results into replies
eval/              test queries, baseline vs re-ranked
config/            dish lists and the excluded-ingredient blocklist
```

`src/` never imports the UI framework. The retrieval engine is a library; the interface is a thin caller.

## Licence and attribution

Recipe data is sourced from Archana's Kitchen and used here for a non-commercial educational exercise. Every recipe surfaced by the bot links back to its original page.

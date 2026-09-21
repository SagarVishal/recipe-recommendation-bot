# Indian Vegetarian Recipe Bot

> Tell it what's in your pantry, it tells you what you can actually cook tonight — ranked by how little you're missing, not by how similar the text looks.

A domain-specific chatbot over 4,218 **Indian, vegetarian, eggless** recipes spanning 37 regional cuisines. Built for the POD exercise.

**Status:** in development — see [`docs/PLAN.md`](docs/PLAN.md) for the build plan.

---

## The problem this solves

Ask most recipe search engines what you can make with *rice, onion, tomato and paneer* and they rank by text similarity. That's the wrong question.

Cosine similarity is symmetric. A 16-ingredient Hyderabadi biryani that happens to contain all four of your items scores beautifully — and you can't cook it, because you're missing twelve things. A three-ingredient tomato rice you *can* cook scores lower.

The question a hungry person is actually asking is not *"which recipe is most similar to my ingredients?"* but *"which recipe is most **covered** by my ingredients?"* Those are different questions, and only the second is asymmetric.

```
coverage = |pantry ∩ recipe| / |recipe|
```

Dividing by the recipe's size, not the pantry's, is what punishes the biryani correctly.

## How it works

```
Archana's Kitchen dataset (6,871 recipes)
        ↓  cuisine filter → Indian regional only
        ↓  diet filter + ingredient blocklist
        ↓  egg exclusion
   4,218 Indian vegetarian eggless recipes
        ↓  parse ingredients → normalised entities
        ↓  embed (MiniLM-L6-v2, 384-dim)
   vector store + parquet
        ↓
   user message → parse pantry → semantic recall (top 50)
                → coverage re-rank → formatted answer
```

Two stages, deliberately. Semantic search gets **recall** — it survives "aubergine" vs "brinjal" and handles vague asks like *"something light for dinner"*. Coverage re-ranking gets **precision** — it orders those candidates by what you can genuinely cook.

At 4,218 recipes a brute-force cosine over a 4,218 × 384 array runs in about 3 ms, so there's no vector database here and no need for one. The two stages earn their place on **capability**, not speed: coverage scoring needs exact set membership and can't parse *"something light"*; semantic search has no notion of what's missing from your kitchen. Each does what the other can't.

No LLM API anywhere in the loop. Everything runs locally on CPU.

## The dataset

[6000+ Indian Food Recipes](https://www.kaggle.com/datasets/kanishk307/6000-indian-food-recipes-dataset), scraped from [Archana's Kitchen](https://www.archanaskitchen.com/).

| Filter stage | Recipes |
|---|---|
| Total in dataset | 6,871 |
| Vegetarian diet labels only | 6,219 |
| Indian regional cuisines only | 4,247 |
| Egg excluded | **4,218** |

37 regional cuisines — North Indian, South Indian, Maharashtrian, Karnataka, Tamil Nadu, Bengali, Kerala, Rajasthani, Gujarati, Andhra, Punjabi, then a long tail through Chettinad, Kashmiri, Awadhi, Goan, Parsi, Sindhi and Oriya. Median 12 ingredients per recipe.

### Why the `Diet` column isn't trusted

The dataset ships a `Diet` label, and filtering on it looks like one line of code. It doesn't hold up:

- **The non-veg label is misspelled.** 427 rows read `Non Vegeterian`. A filter written as `Diet != "Non Vegetarian"` lets every one of them through.
- **55 recipes labelled `Vegetarian` contain meat or fish** — including *Singapore Style Chicken Layered Fried Rice*, *Andaman Style Steamed Garlic Prawns* and *Baked Fish In Coconut Milk*.
- **423 recipes labelled `Vegetarian` contain egg**, so the separate `Eggetarian` label is applied inconsistently.
- `Diet` also mixes diet type with health tags (`Diabetic Friendly`, `Gluten Free`), so the vocabulary isn't a clean partition to begin with.

So the label is a weak first pass, and an ingredient-level blocklist is the real gate. It lives in [`config/excluded_ingredients.yaml`](config/excluded_ingredients.yaml) — readable and arguable, not buried in code — and covers the non-obvious cases: chicken stock, gelatin, anchovy, Worcestershire sauce, lard, rennet.

### The eggplant problem

Excluding egg recipes looks like a one-line substring check. Measured against this corpus, `"egg" in text` drops **143 recipes — and 119 of them contain no egg at all.** They're aubergine dishes: *Baingan Bharta*, *Dhungare Baingan*, *Safed Achari Baingan*. The dataset helpfully lists brinjal's synonyms in brackets, and one of them is "Eggplant".

Word-boundary matching on parsed ingredient entities drops 29 recipes instead of 143, and correctly **keeps** the two recipes that are explicitly *eggless* — a naive filter would have thrown away the eggless chocolate cake for having "egg" in its name.

The same principle applies to the meat blocklist: *Paneer Matar Keema* and *Soya Keema Masala* are vegetarian dishes whose names contain a meat word. Match entities, never raw strings.

The error budget is deliberately asymmetric. Wrongly dropping a valid recipe costs one row out of 4,218. Wrongly keeping a meat or egg recipe breaks the entire premise of the bot.

## Quick start

```bash
pip install -r requirements.txt
make data      # download, filter, clean, embed
make run       # launch the chat interface
```

## Project structure

```
src/ingest.py      download → filter → clean → parquet
src/normalise.py   ingredient parsing, synonyms, vocabulary
src/embed.py       build the vector index
src/query.py       parse a message into pantry + filters
src/retrieve.py    semantic search, then coverage re-rank
src/respond.py     format results into replies
eval/              40 test queries, baseline vs re-ranked
```

`src/` never imports the UI framework. The retrieval engine is a library; the interface is a thin caller.

## Licence and attribution

Recipe data is sourced from Archana's Kitchen and used here for a non-commercial educational exercise. Every recipe surfaced by the bot links back to its original page.

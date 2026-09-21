# 🍲 Gujarati Recipe Bot — a practical RAG workshop

> Tell it what's in your pantry, it tells you what you can actually cook tonight — grounded in a real recipe corpus, so it can't invent dishes that don't exist.

A domain-specific RAG chatbot over **Gujarati** cuisine (with Punjabi as a secondary), lacto-vegetarian and eggless throughout. Built as a teaching notebook: **Python + LangChain + Google Gemini**.

**▶ Start here: [`notebooks/recipe_rag_workshop.ipynb`](notebooks/recipe_rag_workshop.ipynb)**

---

## Quick start

**Requires Python 3.9 or newer.** Check with `python3 --version` first, and note which requirements file that points you at:

| Your Python | Install with |
|---|---|
| 3.10 or newer | `requirements.txt` |
| 3.9 (e.g. macOS `/usr/bin/python3`) | `requirements-py39.txt` — pinned to the last releases that support 3.9 |
| 3.8 or older | Won't work. Use `/usr/bin/python3` if it's 3.9+, or install 3.12 |

```bash
cd "POD Exercise"
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt          # on Python 3.9: requirements-py39.txt
python -m ipykernel install --user --name recipe-bot --display-name "Recipe Bot"
```

Create the venv with the interpreter you actually want: `python3 -m venv .venv` uses whatever `python3` resolves to, which may not be the newest one installed. `/usr/bin/python3 -m venv .venv` pins it to the system Python explicitly.

Then add your Gemini API key. Get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey):

```bash
printf 'GOOGLE_API_KEY=' > .env && read -rs K && echo "$K" >> .env && unset K
```

That prompts on a blank line and echoes nothing, so the key never appears on screen or in your shell history. `.env` is gitignored. If it's missing, the notebook prompts for the key at runtime instead.

```bash
jupyter notebook notebooks/recipe_rag_workshop.ipynb
```

Select the **Recipe Bot** kernel from the Kernel menu — the default kernel runs on a different Python and won't see the installed packages.

### Troubleshooting

**`No matching distribution found for langchain>=0.3`**, preceded by a wall of "Ignored the following versions that require a different python version".

Your `pip` is attached to a Python older than 3.9, so pip skips every modern langchain and stops at 0.2.x. On macOS this is usually the system Python shadowing a newer one. Diagnose:

```bash
python3 --version
pip --version            # note which python path it reports
which -a python3 pip pip3
```

If `python3` is 3.10+, the virtual environment above fixes it — inside an activated venv, `pip` and `python` always point at the right interpreter. If it's 3.9, build the venv from that interpreter and use `requirements-py39.txt`. If it's 3.8 or older and nothing newer exists on the machine, install a current Python:

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv    # Intel Macs: /usr/local/bin/python3.12
source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```

**`ModuleNotFoundError` inside the notebook** after a clean install: the notebook is running on the wrong kernel. Kernel → Change Kernel → **Recipe Bot**.

## What the notebook covers

| # | Section | Concept |
|---|---|---|
| 1 | AI Frameworks | Why LangChain rather than raw HTTP |
| 2 | Configure the LLM | Gemini through a swappable wrapper |
| 3 | LLM Parameters | Temperature, token limits, what to tune first |
| 4 | Basic Chatbot | System prompts, and why an ungrounded bot guesses |
| 5 | History & Memory | Memory is a list you resend — there is no server-side state |
| 6 | Introduce RAG | Embeddings, with measured similarity between *brinjal* and *aubergine* |
| 7 | Small RAG System | Load → embed → retrieve → generate, plus grounding tests |
| 8 | Final System | Coverage re-ranking, regional weighting, graceful fallback |

## The RAG approach

This is a **Retrieval-Augmented Generation** system: Gemini never answers from what it absorbed in training. It answers from recipes we retrieve and place in the prompt, which makes the corpus the authority on facts and the model merely the authority on phrasing.

```
                 ┌──────────────── INDEXING (once) ─────────────────┐
  6,871 rows ──▶ filter & clean ──▶ 301 docs ──▶ Gemini embeddings ──▶ vector store
                 └──────────────────────────────────────────────────┘

                 ┌──────────────── QUERY (per message) ─────────────┐
  "I have besan     parse pantry ──▶ semantic retrieval (top-k)
   and curd"              │                     │
                          └──▶ coverage re-rank ┘
                                     │
                          candidates + computed coverage
                                     │
                          Gemini, instructed to use ONLY these
                                     │
                                  grounded answer
```

### The three stages

| Stage | Component | What it does |
|---|---|---|
| **Retrieve** | `InMemoryVectorStore` + `text-embedding-004` | Embeds the question, returns the nearest recipe documents by cosine similarity |
| **Augment** | `ChatPromptTemplate` | Injects those recipes into the system prompt as `{context}`, with computed coverage figures alongside each |
| **Generate** | `ChatGoogleGenerativeAI`, `temperature=0.0` | Phrases a reply constrained to the supplied context |

**Chunking:** one document per recipe. The data provides a natural boundary, so there's no fixed-size splitting and no chunk ever straddles two dishes. Each document holds name, cuisine, course and ingredients; instructions and URL ride along as metadata.

**Vector store:** in-memory, brute-force cosine over 301 × 768 floats — a few milliseconds. FAISS or Chroma would earn their place somewhere past ~100k documents or when the index must outlive the process. Using one here and calling it architecture is the kind of thing reviewers notice.

**Retrieval is only half the system.** Plain top-k similarity returns recipes that *sound* like the query; the coverage re-ranker orders them by what's actually cookable. See below for why that distinction is the heart of this project.

### Keeping it grounded

Three controls, and the notebook demonstrates each failing safely:

1. **Instruction** — the system prompt says answer only from the supplied recipes, and say so plainly when they don't cover the question.
2. **`temperature=0.0`** — the bot reports facts from a corpus. Creativity here is indistinguishable from fabrication.
3. **Numbers computed, not generated** — coverage percentages and missing-ingredient lists are calculated in Python and handed to the model as text. Gemini is never asked to count.

Asked *"how do I make chicken biryani?"* the bot declines: there are no chicken recipes in the corpus. Asked *"what is the capital of France?"* it declines too, though Gemini certainly knows. **A RAG system that answers everything confidently hasn't been tested** — showing it refuse is what proves the grounding is real.

## The idea worth stealing

Ask a recipe search what you can make with *besan, curd, ginger, green chilli* and it ranks by text similarity. That's the wrong question.

Cosine similarity is **symmetric**. A sixteen-ingredient undhiyu containing all four of your items scores beautifully — and you can't cook it, because you're missing twelve things. A four-ingredient kadhi you *can* cook scores lower.

The real question isn't *"which recipe is most similar to my ingredients?"* but *"which recipe is most **covered** by my ingredients?"*

```
coverage = |pantry ∩ recipe| / |recipe|
```

Dividing by the **recipe's** size, not the pantry's, is the whole trick — it's asymmetric, and it punishes long recipes you can't finish. Final ranking:

```
score = 0.55·coverage + 0.25·similarity − 0.05·(missing/k) + 0.15·regional_prior
```

Semantic retrieval provides **recall** (survives "aubergine" vs "brinjal", handles *"something light for dinner"*). Coverage re-ranking provides **precision**. Neither stage can do the other's job — that's why there are two.

## The corpus

[6000+ Indian Food Recipes](https://www.kaggle.com/datasets/kanishk307/6000-indian-food-recipes-dataset), from [Archana's Kitchen](https://www.archanaskitchen.com/) ([CSV mirror](https://github.com/nileshely/Indian-Food)). Rebuild with `python3 -m src.build_corpus`.

| Stage | Recipes |
|---|---|
| Raw dataset | 6,871 |
| Vegetarian diet labels | 5,875 |
| After ingredient blocklist | 5,620 |
| Untranslated rows dropped | −719 |
| **Tier 1 — Gujarati 132 + Punjabi 169** | **301** |
| Tier 2 — other Indian, labelled fallback | 3,164 |

Tier 1 is the bot's world. Tier 2 exists so it never dead-ends: when nothing in Gujarati or Punjabi clears the coverage threshold, the bot says so and offers the nearest alternative, **explicitly labelled**. Never a silent substitution.

## Four things the data got wrong

Every one of these would have passed code review. None threw an error.

**1. The diet label is misspelled.** 427 rows read `Non Vegeterian`. A filter written as `Diet != "Non Vegetarian"` lets every one through.

**2. The diet label is wrong anyway.** 55 recipes labelled `Vegetarian` contain meat or fish — *Singapore Style Chicken Layered Fried Rice*, *Andaman Style Steamed Garlic Prawns*, *Baked Fish In Coconut Milk*. 423 more contain egg despite a separate `Eggetarian` label existing. So the label is a weak first pass and [`config/excluded_ingredients.yaml`](config/excluded_ingredients.yaml) is the real gate.

**3. A byte-order mark hides every Gujarati recipe.** Each one is spelled `'Gujarati Recipes﻿'`:

```python
df[df.Cuisine == "Gujarati Recipes"]     # 0 rows
df[df.Cuisine.str.contains("Gujarati")]  # 132 rows
```

Written the obvious way, the Gujarati weighting would have done nothing, silently.

**4. 719 rows were never translated.** Devanagari text sits in the `Translated...` columns. Some have English ingredients but Hindi *instructions*, so they parse cleanly and then hand the user cooking steps they may not read. This surfaced as *12% of recipes parsing to zero ingredients* — it looked like a parser bug, and chasing the symptom found a data problem.

### The pattern

`"egg" in text` drops 143 recipes and **119 contain no egg** — they're aubergine dishes, because the dataset lists brinjal's synonyms and one of them is "Eggplant". Word-boundary matching on parsed entities drops 29 instead, and correctly keeps the recipes whose names contain *egg**less***.

Same lesson as the meat blocklist, arriving from the other direction: **match parsed entities, never raw strings.** And the error budget is asymmetric on purpose — wrongly dropping a valid recipe costs one row out of 301; wrongly keeping a meat or egg recipe breaks the premise of the bot.

## Repository layout

```
notebooks/recipe_rag_workshop.ipynb   the deliverable — 45 cells, 8 sections
src/build_corpus.py                   6,871 raw rows → 301 curated
src/paths.py                          every filesystem path, in one place
config/excluded_ingredients.yaml      the blocklist, readable and arguable
data/recipes_core.csv                 tier 1, committed so the notebook just runs
data/recipes_all.csv                  tier 1 + tier 2 fallback
tests/                                corpus invariants
```

## Licence and attribution

Recipe data is from Archana's Kitchen, used here for a non-commercial educational exercise. Every recipe the bot surfaces links back to its original page.

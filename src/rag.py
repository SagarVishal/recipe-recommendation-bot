"""The RAG engine: retrieve from the corpus, then let Gemini answer from it.

Retrieval is two-stage on purpose. Embedding similarity provides recall - it
survives "aubergine" for "brinjal" and copes with questions that contain no
ingredients at all. Coverage scoring provides precision - it orders the
candidates by what can genuinely be cooked. Neither stage can do the other's
job, which is why there are two.

Generation is grounded: Gemini only ever sees the retrieved candidates, and
every number in the reply is computed here rather than predicted by a model.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src import paths
from src.corpus import coverage, load_recipes, vocabulary
from src.models import chat_model, embed_model

# Ranking weights. Coverage dominates; the regional prior only breaks ties.
W_COVERAGE, W_SIMILARITY, W_MISSING, W_REGION = 0.55, 0.25, 0.05, 0.15
REGION_PRIOR = {"Gujarati": 1.0, "Punjabi": 0.4}

# Below this, tier 1 has nothing worth offering and we widen - and say so.
FALLBACK_THRESHOLD = 0.34
CANDIDATES = 5

SYSTEM_PROMPT = """You are a warm, practical cooking assistant for Gujarati \
and Punjabi vegetarian food. The corpus is eggless and lacto-vegetarian.

Rules you must not break:
- Recommend ONLY from the CANDIDATES below. Never invent a recipe, an \
ingredient, or a cooking time.
- Quote the coverage percentage and the missing ingredients exactly as given. \
They are computed, not estimated.
- Prefer Gujarati dishes when coverage is comparable.
- If asked for meat, fish or egg, explain warmly that this kitchen is \
vegetarian and eggless, and offer something from the candidates instead.
- If the candidates genuinely do not answer the question, say so plainly \
rather than improvising.
- Be concise: a sentence or two per recipe. Mention the region.
"""


@dataclass
class Candidate:
    name: str
    region: str
    course: str
    coverage: float
    missing: List[str]
    score: float
    tier: int
    url: str
    time_mins: object
    ingredients: str


class RecipeRAG:
    """Loads the corpus, builds (or reuses) the vector index, answers questions."""

    def __init__(self, core_only: bool = True) -> None:
        self.core = load_recipes(core_only=True)
        self.fallback = load_recipes(core_only=False)
        self.fallback = self.fallback[self.fallback.tier == 2].reset_index(drop=True)
        self._embeddings = None
        self._core_vectors: Optional[np.ndarray] = None

    # ---------- embeddings ----------

    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            # Resolved from the API rather than hardcoded - Google retires
            # model names, and a stale one 404s months later.
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=f"models/{embed_model()}")
        return self._embeddings

    def _corpus_fingerprint(self) -> str:
        """Includes the model name: different models produce incompatible
        vectors, so switching models must invalidate the cache."""
        joined = embed_model() + "|" + "|".join(self.core["name"].tolist())
        return hashlib.sha256(joined.encode()).hexdigest()[:16]

    def build_index(self, progress=None) -> np.ndarray:
        """Embed every core recipe once, then cache to disk.

        Re-embedding 301 recipes on every launch would be slow and would burn
        API quota for no reason, so the vectors are written alongside a
        fingerprint of the corpus and reused until the corpus changes.
        """
        fingerprint = self._corpus_fingerprint()
        meta = paths.PROCESSED_DIR / "vectors.meta"
        if paths.VECTORS_NPY.exists() and meta.exists():
            if meta.read_text().strip() == fingerprint:
                return np.load(paths.VECTORS_NPY)

        texts = [str(t) for t in self.core["search_text"].tolist()]
        blank = [i for i, t in enumerate(texts) if not t.strip()]
        if blank:
            raise ValueError(
                f"{len(blank)} recipes have empty search text (rows {blank[:5]}). "
                "Rebuild the corpus with: python -m src.build_corpus"
            )

        vectors: List[List[float]] = []
        batch = 50
        partial = paths.PROCESSED_DIR / "vectors.partial.npy"
        if partial.exists():
            # A previous run died part-way; pick up where it stopped.
            done = np.load(partial)
            if done.shape[0] < len(texts):
                vectors = done.tolist()

        for start in range(len(vectors), len(texts), batch):
            chunk = texts[start:start + batch]
            vectors.extend(self.embeddings.embed_documents(chunk))
            np.save(partial, np.asarray(vectors, dtype=np.float32))
            if progress:
                progress(min(start + batch, len(texts)), len(texts))
        partial.unlink(missing_ok=True)

        array = np.asarray(vectors, dtype=np.float32)
        array /= np.linalg.norm(array, axis=1, keepdims=True)  # unit length
        paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(paths.VECTORS_NPY, array)
        meta.write_text(fingerprint)
        return array

    @property
    def vectors(self) -> np.ndarray:
        if self._core_vectors is None:
            self._core_vectors = self.build_index()
        return self._core_vectors

    # ---------- retrieval ----------

    def semantic_scores(self, question: str) -> np.ndarray:
        """Cosine similarity of the question against all core recipes.

        Brute force over 301 x 768 floats: about 3 ms. A vector database
        would earn its place past ~100k documents, not here.
        """
        q = np.asarray(self.embeddings.embed_query(question), dtype=np.float32)
        q /= np.linalg.norm(q)
        return self.vectors @ q

    def parse_pantry(self, text: str) -> Set[str]:
        """Pull known ingredients out of free text, longest match first."""
        lowered = " " + text.lower().replace(",", " , ") + " "
        found: Set[str] = set()
        for term in sorted(vocabulary(), key=len, reverse=True):
            if len(term) < 3:
                continue
            if f" {term} " in lowered or f" {term}," in lowered:
                if not any(term in bigger for bigger in found):
                    found.add(term)
        return found

    def rank(self, question: str, pantry: Set[str]) -> Tuple[List[Candidate], bool]:
        """Score tier 1; widen to tier 2 only if nothing clears the threshold."""
        sims = self.semantic_scores(question) if question.strip() else np.zeros(len(self.core))
        results: List[Candidate] = []
        for i, row in self.core.iterrows():
            cov, missing = coverage(pantry, row["entities"])
            score = (
                W_COVERAGE * cov
                + W_SIMILARITY * float(sims[i])
                - W_MISSING * min(len(missing), 10) / 10
                + W_REGION * REGION_PRIOR.get(row["region"], 0.0)
            )
            results.append(Candidate(
                name=row["name"], region=row["region"], course=row["course"],
                coverage=cov, missing=missing, score=score, tier=1,
                url=row["url"], time_mins=row["total_time_mins"],
                ingredients=row["ingredients"],
            ))
        results.sort(key=lambda c: -c.score)

        widened = not pantry or results[0].coverage < FALLBACK_THRESHOLD
        if widened and pantry:
            extra: List[Candidate] = []
            for _, row in self.fallback.iterrows():
                cov, missing = coverage(pantry, row["entities"])
                if cov <= results[0].coverage:
                    continue
                extra.append(Candidate(
                    name=row["name"], region=row["cuisine"], course=row["course"],
                    coverage=cov, missing=missing, score=cov, tier=2,
                    url=row["url"], time_mins=row["total_time_mins"],
                    ingredients=row["ingredients"],
                ))
            extra.sort(key=lambda c: -c.coverage)
            if extra:
                return (results[:2] + extra[:2])[:CANDIDATES], True
        return results[:CANDIDATES], False

    # ---------- generation ----------

    @staticmethod
    def _format(candidates: List[Candidate]) -> str:
        blocks = []
        for c in candidates:
            missing = ", ".join(c.missing[:8]) if c.missing else "nothing"
            blocks.append(
                f"{c.name}\n"
                f"Region: {c.region} | Course: {c.course} | {c.time_mins} mins"
                f" | {'OUTSIDE the Gujarati/Punjabi collection' if c.tier == 2 else 'core collection'}\n"
                f"Coverage: {c.coverage:.0%} of ingredients available. Missing: {missing}\n"
                f"Ingredients: {c.ingredients[:400]}\n"
                f"Source: {c.url}"
            )
        return "\n\n---\n\n".join(blocks)

    def answer(self, question: str, pantry: Set[str],
               history: Optional[List] = None) -> Tuple[str, List[Candidate], bool]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        candidates, widened = self.rank(question, pantry)
        note = ("\nNOTE: nothing in the core Gujarati/Punjabi collection matched "
                "well. Say so clearly before suggesting the alternatives, and "
                "name which cuisine they come from.") if widened else ""

        messages = [SystemMessage(content=SYSTEM_PROMPT + note)]
        for role, text in (history or [])[-6:]:
            messages.append(HumanMessage(content=text) if role == "user"
                            else AIMessage(content=text))
        messages.append(HumanMessage(content=(
            f"My pantry: {', '.join(sorted(pantry)) or '(nothing yet)'}\n\n"
            f"Question: {question}\n\n"
            f"CANDIDATES:\n{self._format(candidates)}"
        )))

        llm = ChatGoogleGenerativeAI(model=chat_model(), temperature=0.2)
        return llm.invoke(messages).content, candidates, widened


def api_key_present() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))

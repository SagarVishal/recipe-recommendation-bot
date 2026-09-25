"""Gujarati Recipe Bot - a RAG chatbot over 301 Gujarati & Punjabi recipes.

Run with:  streamlit run app.py

This is the only module that imports Streamlit. Everything it calls lives in
src/ and knows nothing about how it is displayed, so the same engine backs
the notebook and the CLI.
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Gujarati Recipe Bot", page_icon="🍲",
                   layout="centered", initial_sidebar_state="expanded")

# --- API key -----------------------------------------------------------------

if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
    st.title("🍲 Gujarati Recipe Bot")
    st.warning("No Gemini API key found.")
    st.markdown(
        "Create one free at [aistudio.google.com/apikey]"
        "(https://aistudio.google.com/apikey), then either put it in a `.env` "
        "file next to this app as `GOOGLE_API_KEY=...`, or paste it below for "
        "this session only."
    )
    typed = st.text_input("Gemini API key", type="password")
    if typed:
        os.environ["GOOGLE_API_KEY"] = typed
        st.rerun()
    st.stop()

# --- engine ------------------------------------------------------------------

# A realistic Indian kitchen. The bot is far more useful opening with a
# stocked pantry than an empty one, and the first reply is meaningful
# without the user typing anything.
DEFAULT_PANTRY = {
    "onion", "tomato", "potato", "ginger", "garlic", "green chillies",
    "coriander leaves", "curd", "rice", "wheat flour", "besan",
    "lemon juice", "peas", "carrot",
}


@st.cache_resource(show_spinner=False)
def get_engine():
    """Loads instantly. Embeddings are optional and built on demand.

    Coverage ranking needs no embeddings at all, so the app is usable from
    the first second; the semantic term contributes nothing for recipes not
    yet embedded, and the sidebar offers to embed more.
    """
    from src.rag import RecipeRAG

    return RecipeRAG()


try:
    engine = get_engine()
except Exception as error:  # noqa: BLE001 - a clear message beats a traceback
    st.title("🍲 Gujarati Recipe Bot")
    st.error(f"Couldn't start the engine.\n\n```\n{error}\n```")
    st.markdown(
        "Run `python -m src.models` to see which models your API key supports. "
        "If the list is empty the key is invalid or revoked."
    )
    st.stop()

st.session_state.setdefault("pantry", set(DEFAULT_PANTRY))
st.session_state.setdefault("messages", [])

# --- sidebar -----------------------------------------------------------------

with st.sidebar:
    st.subheader("Your pantry")
    if st.session_state.pantry:
        st.markdown(" ".join(f"`{item}`" for item in sorted(st.session_state.pantry)))
    else:
        st.caption("Empty. Tell the bot what you have.")

    manual = st.text_input("Add an ingredient", placeholder="e.g. besan")
    col_a, col_b = st.columns(2)
    if col_a.button("Add", use_container_width=True) and manual.strip():
        st.session_state.pantry |= engine.parse_pantry(manual) or {manual.strip().lower()}
        st.rerun()
    if col_b.button("Clear", use_container_width=True):
        st.session_state.pantry = set()
        st.rerun()
    if st.button("Reset to a typical kitchen", use_container_width=True):
        st.session_state.pantry = set(DEFAULT_PANTRY)
        st.rerun()

    st.divider()
    st.subheader("Corpus")
    regions = engine.core.region.value_counts()
    st.metric("Core recipes", len(engine.core))
    st.caption(
        f"Gujarati {regions.get('Gujarati', 0)} · Punjabi {regions.get('Punjabi', 0)} · "
        f"{regions.get('other', 0)} other Indian"
    )
    st.caption("Lacto-vegetarian · eggless · Gujarati-weighted ranking")

    st.divider()
    st.subheader("Semantic index")
    done, total = engine.embedded_count, len(engine.core)
    st.progress(done / total if total else 0.0)
    st.caption(f"{done} of {total} recipes embedded")
    if done < total:
        st.caption(
            "Coverage ranking works on all of them already. Embedding adds "
            "the semantic half \u2014 matching *brinjal* to *aubergine*, and "
            "answering questions with no ingredients in them."
        )
        if st.button("Embed 400 more", use_container_width=True):
            bar = st.progress(0.0, text="Embedding\u2026")
            engine._core_vectors = engine.build_index(
                budget=400,
                progress=lambda d, t: bar.progress(
                    d / t, text=f"Embedding {d}/{t} \u2014 about "
                                f"{int((t - d) * 60 / 85 // 60)}m left\u2026"))
            bar.empty()
            st.rerun()

    st.divider()
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pantry = set()
        st.rerun()

# --- main --------------------------------------------------------------------

st.title("🍲 Gujarati Recipe Bot")
st.caption("Tell me what's in your kitchen and I'll find what you can actually cook.")

if not st.session_state.messages:
    st.markdown("**Try:**")
    examples = [
        "What can I make right now?",
        "Something Gujarati for dinner",
        "I also have paneer and spinach",
    ]
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("candidates"):
            with st.expander("What the retriever found, and why"):
                for c in message["candidates"]:
                    tier = "" if c.tier == 1 else "  ·  ⚠️ outside your cuisines"
                    st.markdown(
                        f"**[{c.name}]({c.url})** — {c.region}{tier}  \n"
                        f"Coverage **{c.coverage:.0%}** · score {c.score:.3f} · "
                        f"{c.time_mins} mins  \n"
                        f"Missing: {', '.join(c.missing[:8]) if c.missing else '_nothing_'}"
                    )
                    st.progress(min(c.coverage, 1.0))

prompt = st.chat_input("What's in your kitchen?") or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.pantry |= engine.parse_pantry(prompt)
    if any(word in prompt.lower() for word in ("start over", "reset", "clear my pantry")):
        st.session_state.pantry = set()

    with st.chat_message("assistant"):
        with st.spinner("Searching the corpus…"):
            history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]
            try:
                reply, candidates, widened = engine.answer(
                    prompt, st.session_state.pantry, history)
            except Exception as error:  # noqa: BLE001 - surface it, don't crash
                reply, candidates, widened = (
                    f"Something went wrong calling Gemini:\n\n```\n{error}\n```", [], False)
        st.markdown(reply)
        if widened and candidates:
            st.info("Nothing matches this pantry closely \u2014 these are the "
                    "nearest options.")
        if candidates:
            with st.expander("What the retriever found, and why"):
                for c in candidates:
                    tier = "" if c.tier == 1 else "  ·  ⚠️ outside your cuisines"
                    st.markdown(
                        f"**[{c.name}]({c.url})** — {c.region}{tier}  \n"
                        f"Coverage **{c.coverage:.0%}** · score {c.score:.3f} · "
                        f"{c.time_mins} mins  \n"
                        f"Missing: {', '.join(c.missing[:8]) if c.missing else '_nothing_'}"
                    )
                    st.progress(min(c.coverage, 1.0))

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "candidates": candidates})
    st.rerun()

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

@st.cache_resource(show_spinner=False)
def get_engine():
    """Built once per process. Embedding 301 recipes is cached to disk too."""
    from src.rag import RecipeRAG
    engine = RecipeRAG()
    bar = st.progress(0.0, text="Embedding the recipe corpus (first run only)…")
    engine._core_vectors = engine.build_index(
        progress=lambda done, total: bar.progress(done / total,
                                                  text=f"Embedding {done}/{total} recipes…")
    )
    bar.empty()
    return engine


engine = get_engine()

st.session_state.setdefault("pantry", set())
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

    st.divider()
    st.subheader("Corpus")
    regions = engine.core.region.value_counts()
    st.metric("Core recipes", len(engine.core))
    st.caption(
        f"Gujarati {regions.get('Gujarati', 0)} · Punjabi {regions.get('Punjabi', 0)}  \n"
        f"Fallback tier: {len(engine.fallback)} other Indian"
    )
    st.caption("Lacto-vegetarian · eggless · Gujarati-weighted ranking")

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
        "I have besan, curd, ginger and green chilli",
        "something Gujarati for dinner",
        "I have potatoes, onion and peas",
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
            st.info("Nothing in the core Gujarati/Punjabi collection matched well, "
                    "so some suggestions come from the wider Indian corpus.")
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

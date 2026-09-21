"""Terminal chat, for when you don't want a browser.

    python -m src.chat_cli

Same engine as the Streamlit app - useful as a fallback if Streamlit misbehaves,
and handy for a quick demo over SSH.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("No Gemini API key. Put GOOGLE_API_KEY=... in a .env file.")
        print("Get one free at https://aistudio.google.com/apikey")
        sys.exit(1)

    from src.rag import RecipeRAG

    print("Loading corpus…")
    engine = RecipeRAG()
    engine._core_vectors = engine.build_index(
        progress=lambda d, t: print(f"\r  embedding {d}/{t}…", end="", flush=True))
    print(f"\r{len(engine.core)} Gujarati & Punjabi recipes ready."
          f" ({len(engine.fallback)} more as fallback)        ")
    print("Tell me what's in your kitchen. Ctrl-C or 'quit' to stop.\n")

    pantry: set = set()
    history: list = []
    while True:
        try:
            message = input("you  ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not message or message.lower() in {"quit", "exit"}:
            print("bye")
            return
        if message.lower() in {"reset", "start over"}:
            pantry, history = set(), []
            print("bot  ▸ Pantry cleared.\n")
            continue

        pantry |= engine.parse_pantry(message)
        try:
            reply, candidates, widened = engine.answer(message, pantry, history)
        except Exception as error:  # noqa: BLE001
            print(f"bot  ▸ Gemini call failed: {error}\n")
            continue

        print(f"bot  ▸ {reply}\n")
        if candidates:
            print("       retrieved:")
            for c in candidates:
                flag = "" if c.tier == 1 else "  [outside your cuisines]"
                print(f"         {c.coverage:>4.0%}  {c.name[:52]}  ({c.region}){flag}")
            print()
        history.append(("user", message))
        history.append(("assistant", reply))


if __name__ == "__main__":
    main()

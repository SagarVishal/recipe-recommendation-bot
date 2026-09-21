"""Pick Gemini models that this API key actually supports, at runtime.

Google retires model names on its own schedule - `text-embedding-004` was
replaced by `gemini-embedding-001`, and a hardcoded name turns into a 404
months later with a confusing message. So we ask the API what it has and
choose from a preference list, which also means this works unchanged on
whatever is current when someone clones the repo.

Override either choice with GEMINI_EMBED_MODEL / GEMINI_CHAT_MODEL.

    python -m src.models      # print what your key supports
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from functools import lru_cache
from typing import List, Tuple

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

# Best first. Anything not present is skipped.
EMBED_PREFERENCE = [
    "gemini-embedding-001",
    "text-embedding-005",
    "text-embedding-004",
    "embedding-001",
]
CHAT_PREFERENCE = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
]


def _api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key. Put GOOGLE_API_KEY=... in a .env file. "
            "Get one free at https://aistudio.google.com/apikey"
        )
    return key


@lru_cache(maxsize=1)
def available() -> Tuple[List[str], List[str]]:
    """Return (embedding models, chat models) this key can use."""
    request = urllib.request.Request(f"{_ENDPOINT}?key={_api_key()}&pageSize=200")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="ignore")[:300]
        raise RuntimeError(
            f"Could not list Gemini models (HTTP {error.code}). "
            f"Usually an invalid or revoked API key.\n{body}"
        ) from error

    embed, chat = [], []
    for model in payload.get("models", []):
        name = model.get("name", "").replace("models/", "")
        methods = model.get("supportedGenerationMethods", [])
        if "embedContent" in methods:
            embed.append(name)
        if "generateContent" in methods:
            chat.append(name)
    return embed, chat


def _choose(preference: List[str], found: List[str], hint: str, kind: str) -> str:
    for wanted in preference:
        if wanted in found:
            return wanted
    fallback = [m for m in found if hint in m] or found
    if not fallback:
        raise RuntimeError(f"This API key exposes no {kind} models.")
    return fallback[0]


@lru_cache(maxsize=1)
def embed_model() -> str:
    override = os.environ.get("GEMINI_EMBED_MODEL")
    if override:
        return override
    return _choose(EMBED_PREFERENCE, available()[0], "embedding", "embedding")


@lru_cache(maxsize=1)
def chat_model() -> str:
    override = os.environ.get("GEMINI_CHAT_MODEL")
    if override:
        return override
    return _choose(CHAT_PREFERENCE, available()[1], "flash", "chat")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    embed, chat = available()
    print(f"Embedding models ({len(embed)}):")
    for name in embed:
        print(f"   {'->' if name == embed_model() else '  '} {name}")
    print(f"\nChat models ({len(chat)}):")
    for name in chat:
        print(f"   {'->' if name == chat_model() else '  '} {name}")
    print(f"\nThis app will use:  embed={embed_model()}  chat={chat_model()}")


if __name__ == "__main__":
    main()

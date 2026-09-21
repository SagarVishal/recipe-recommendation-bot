"""Pick the best Gemini models this API key supports, at runtime.

Google ships and retires model names constantly - `text-embedding-004` was
replaced by `gemini-embedding-001`, and any hardcoded name eventually 404s
with a confusing message. Worse, a fixed preference list silently pins you
to an old model long after better ones land on your account.

So this asks the API what exists, discards everything that isn't a
general-purpose text model, and picks the highest version number. It keeps
working as Google rotates names, and it keeps up as they ship new ones.

Override with GEMINI_EMBED_MODEL / GEMINI_CHAT_MODEL in .env.

    python -m src.models      # show what your key supports and what wins
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache
from typing import List, Tuple

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

# Specialised or unstable models. A recipe bot wants none of these.
_EXCLUDE = (
    "preview", "experimental", "-exp", "image", "tts", "audio", "transcribe",
    "robotics", "computer-use", "lyria", "nano-banana", "deep-research",
    "omni", "antigravity", "thinking", "learnlm", "veo", "imagen",
)


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
    """(embedding models, chat models) this key can actually use."""
    request = urllib.request.Request(f"{_ENDPOINT}?key={_api_key()}&pageSize=400")
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


def _version(name: str) -> tuple:
    """Sort key: version number, then 'not lite', then name length.

    'gemini-3.8-flash' -> (3.8, 1, ...) beats 'gemini-2.5-flash' -> (2.5, 1, ...).
    A bare '001' suffix is read as version 1, so gemini-embedding-2 wins over
    gemini-embedding-001. Shorter names win ties, preferring the plain model
    over a decorated variant.
    """
    numbers = re.findall(r"(\d+(?:\.\d+)?)", name)
    version = max((float(n) for n in numbers), default=0.0)
    if version > 100:           # a zero-padded id like 001, not a version
        version = version / 1000
    return (version, 0 if "lite" in name else 1, -len(name))


def _usable(names: List[str]) -> List[str]:
    return [n for n in names if not any(bad in n for bad in _EXCLUDE)]


def _rank(names: List[str], must_contain: str = "") -> List[str]:
    pool = _usable(names)
    if must_contain:
        preferred = [n for n in pool if must_contain in n]
        pool = preferred or pool
    return sorted(pool, key=_version, reverse=True)


@lru_cache(maxsize=1)
def embed_model() -> str:
    override = os.environ.get("GEMINI_EMBED_MODEL")
    if override:
        return override
    ranked = _rank(available()[0], "embedding")
    if not ranked:
        raise RuntimeError("This API key exposes no usable embedding models.")
    return ranked[0]


@lru_cache(maxsize=1)
def chat_model() -> str:
    override = os.environ.get("GEMINI_CHAT_MODEL")
    if override:
        return override
    # Flash models: fast and cheap, which is what a chat UI wants.
    ranked = _rank(available()[1], "flash")
    if not ranked:
        raise RuntimeError("This API key exposes no usable chat models.")
    return ranked[0]


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    embed_all, chat_all = available()

    print(f"Embedding models your key can use ({len(_usable(embed_all))} "
          f"usable of {len(embed_all)}):")
    for name in _rank(embed_all, "embedding"):
        print(f"   {'->' if name == embed_model() else '  '} {name}")

    print(f"\nChat models, ranked ({len(_usable(chat_all))} usable of {len(chat_all)}):")
    for name in _rank(chat_all, "flash")[:10]:
        print(f"   {'->' if name == chat_model() else '  '} {name}")

    skipped = sorted(set(chat_all) - set(_usable(chat_all)))
    print(f"\nSkipped {len(skipped)} specialised or preview models "
          f"(image, tts, audio, robotics, research, previews).")
    print(f"\nThis app will use:  embed={embed_model()}  chat={chat_model()}")
    print("Override in .env with GEMINI_EMBED_MODEL / GEMINI_CHAT_MODEL.")


if __name__ == "__main__":
    main()

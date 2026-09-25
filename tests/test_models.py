"""Model selection: pick the newest usable model, skip specialised ones.

These run offline against a captured model list, so they never call the API
and never need a key. The list is a real response from a live account.
"""
from src import models

EMBED = ["gemini-embedding-001", "gemini-embedding-2-preview", "gemini-embedding-2"]

CHAT = [
    "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview-tts",
    "gemma-4-31b-it", "gemini-flash-latest", "gemini-2.5-flash-lite",
    "gemini-2.5-flash-image", "gemini-3-flash-preview", "gemini-3.1-flash-lite",
    "gemini-3-pro-image", "nano-banana-pro-preview", "gemini-3.1-flash-image",
    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-omni-1.1-flash",
    "gemini-3.5-transcribe", "gemini-3.6-flash", "gemini-3.7-flash",
    "gemini-3.8-flash", "lyria-3.5", "gemini-robotics-er-2-preview",
    "gemini-2.5-computer-use-preview-10-2025", "antigravity-preview-09-2026",
    "deep-research-pro-preview-12-2025",
]


def test_picks_the_newest_stable_flash_model():
    assert models._rank(CHAT, "flash")[0] == "gemini-3.8-flash"


def test_prefers_full_models_over_lite():
    ranked = models._rank(CHAT, "flash")
    assert ranked.index("gemini-3.5-flash") < ranked.index("gemini-3.5-flash-lite")


def test_skips_previews_and_specialised_models():
    usable = models._usable(CHAT)
    for banned in ("gemini-3-flash-preview", "gemini-2.5-flash-image",
                   "gemini-3.5-transcribe", "lyria-3.5", "nano-banana-pro-preview",
                   "deep-research-pro-preview-12-2025"):
        assert banned not in usable


def test_zero_padded_ids_do_not_beat_real_versions():
    """gemini-embedding-001 is version 1, not version 001."""
    assert models._rank(EMBED, "embedding")[0] == "gemini-embedding-2"


def test_an_override_wins_without_calling_the_api(monkeypatch):
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-2.5-pro")
    models.chat_model.cache_clear()
    assert models.chat_model() == "gemini-2.5-pro"
    models.chat_model.cache_clear()


def test_models_newer_than_the_client_are_demoted_not_chosen():
    """langchain-google-genai 2.1.12 (last release supporting Python 3.9)
    hangs indefinitely against gemini-3.x, so a naive 'pick the newest'
    rule produced an app that looked broken. Newer models are kept in the
    list - raising the cap is a one-line change - but not chosen first."""
    ranked = models._rank(CHAT, "flash")
    cap = models.MAX_KNOWN_GOOD_VERSION
    known = [m for m in ranked if models._version(m)[0] <= cap]
    newer = [m for m in ranked if models._version(m)[0] > cap]

    assert known[0] == "gemini-2.5-flash"
    assert "gemini-3.8-flash" in newer, "newer models stay available as fallbacks"
    assert models._version("gemini-3.8-flash")[0] > cap

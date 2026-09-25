"""Generate the explainer narration with Gemini TTS.

Runs on YOUR machine, because the sandbox that built the video cannot reach
generativelanguage.googleapis.com - your Mac can.

    source .venv/bin/activate
    pip install "google-genai<2"        # 1.47.0 is the last release on py3.9
    python scripts/narrate.py

Writes one WAV per line into docs/narration/. The video is then re-muxed
against those, so timing stays under the editor's control rather than
depending on however long the speech happens to be.
"""
from __future__ import annotations

import os
import re
import sys
import time
import wave
from pathlib import Path

from dotenv import load_dotenv

VOICE = os.environ.get("TTS_VOICE", "Charon")   # calm, low, documentary
OUT = Path(__file__).resolve().parent.parent / "docs" / "narration"

# (id, seconds available in the cut, text)
LINES = [
    ("01_title", 3.6,
     "A recipe chatbot, built with LangChain and Google Gemini."),
    ("02_question", 3.1,
     "You have fourteen things in your kitchen. Which of nine hundred and ten "
     "recipes can you actually cook tonight?"),
    ("03_similarity", 6.2,
     "A vector database would answer with similarity. But cosine similarity is "
     "symmetric: it rewards overlap and ignores direction. A sixteen ingredient "
     "biryani that happens to contain all four things you own scores beautifully, "
     "and you still cannot cook it."),
    ("04_verdict", 2.6,
     "Both score the same. Only one of them is dinner."),
    ("05_coverage", 3.9,
     "So measure coverage instead: how much of the recipe your pantry supplies, "
     "divided by the recipe's size, not the pantry's. That asymmetry is the whole "
     "idea. And staples do not count. Salt is in eighty four percent of these "
     "recipes; nobody shops for salt."),
    ("06_pipeline", 5.8,
     "The pipeline is retrieval augmented generation. Your message goes to "
     "embeddings, the candidates are re-ranked by coverage, and only then does "
     "Gemini see them."),
    ("07_grounded", 3.4,
     "Grounded, not guessed. The model may answer only from the recipes it is "
     "given, and every number on screen is computed in Python rather than "
     "predicted."),
    ("08_corpus", 6.1,
     "Most of the work was the corpus. Six thousand eight hundred and seventy one "
     "rows became nine hundred and ten worth recommending."),
    ("09_bugs", 6.4,
     "Four bugs found along the way, and not one of them threw an error. A "
     "misspelled diet label. Recipes marked vegetarian that contain chicken. An "
     "invisible byte order mark that made an entire cuisine unmatchable. And the "
     "word egg, hiding inside eggplant."),
    ("10_lesson", 3.4,
     "Match parsed entities, never raw strings. An operation that fails silently "
     "is worse than one that fails loudly."),
    ("11_demo", 2.0,
     "Here it is running against live Gemini."),
    ("12_shot1", 4.6,
     "It opens with a typical Indian kitchen already stocked, so the first answer "
     "is useful before you type anything."),
    ("13_shot2", 4.6,
     "Every answer shows its working: coverage, the ranking score, what is "
     "missing, and a link to the original recipe."),
    ("14_shot3", 4.6,
     "The pantry persists across turns. Add paneer and spinach, and the ranking "
     "runs again."),
    ("15_shot4", 4.6,
     "And it refuses what it does not have. Gemini knows chicken biryani "
     "perfectly well. The corpus does not, so the bot declines and offers "
     "something you can actually cook."),
    ("16_outro", 3.4,
     "Similarity is not relevance. Cosine similarity answered a question nobody "
     "asked, and defining the right objective mattered more than any model choice."),
]


def pick_model(client) -> str:
    """Newest TTS model this key exposes."""
    override = os.environ.get("TTS_MODEL")
    if override:
        return override
    names = []
    for m in client.models.list():
        name = getattr(m, "name", "").replace("models/", "")
        if "tts" in name:
            names.append(name)
    if not names:
        sys.exit("This API key exposes no TTS models.")
    names.sort(key=lambda n: ("preview" in n, n), reverse=False)
    print(f"TTS models available: {', '.join(names)}")
    return names[-1] if "2.5" in names[-1] else names[0]


def synthesise(client, model: str, text: str, path: Path) -> float:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
                )
            ),
        ),
    )
    pcm = response.candidates[0].content.parts[0].inline_data.data
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(pcm)
    with wave.open(str(path)) as handle:
        return handle.getnframes() / handle.getframerate()


def main() -> None:
    load_dotenv()
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit("No GOOGLE_API_KEY. It should already be in your .env")

    try:
        from google import genai
    except ImportError:
        sys.exit('Run:  pip install "google-genai<2"')

    client = genai.Client()
    model = pick_model(client)
    print(f"Using {model} with voice {VOICE}\n")

    OUT.mkdir(parents=True, exist_ok=True)
    over = []
    for index, (name, budget, text) in enumerate(LINES):
        path = OUT / f"{name}.wav"
        if path.exists():
            with wave.open(str(path)) as handle:
                seconds = handle.getnframes() / handle.getframerate()
            print(f"  {name:<12} {seconds:>5.1f}s   (already done, skipping)")
            continue

        # The TTS free tier is rate limited far more tightly than chat. Pace
        # between calls and back off on 429 rather than dying half way.
        seconds = None
        for attempt in range(6):
            try:
                seconds = synthesise(client, model, text, path)
                break
            except Exception as error:  # noqa: BLE001
                message = str(error)
                if "429" not in message and "quota" not in message.lower():
                    raise
                match = re.search(r"seconds:\s*(\d+)", message)
                wait = float(match.group(1)) + 2 if match else 20 * (attempt + 1)
                print(f"  {name:<12} rate limited, waiting {wait:.0f}s\u2026")
                time.sleep(wait)
        if seconds is None:
            print(f"\n  Stopped at {name}. Run this again in a minute - "
                  f"finished lines are skipped.")
            break

        flag = ""
        if seconds > budget:
            flag = f"   over by {seconds - budget:.1f}s"
            over.append((name, seconds, budget))
        print(f"  {name:<12} {seconds:>5.1f}s / {budget:>4.1f}s{flag}")
        if index < len(LINES) - 1:
            time.sleep(float(os.environ.get("TTS_PAUSE", "6")))

    done = len(list(OUT.glob("*.wav")))
    print(f"\n{done} of {len(LINES)} lines in docs/narration/")
    if done < len(LINES):
        print("Run the script again to finish the rest - it resumes.")
    elif over:
        print("Some lines run longer than their slot; the video gets re-timed "
              "around the audio, not the other way round.")


if __name__ == "__main__":
    main()

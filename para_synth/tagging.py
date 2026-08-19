"""LLM transcript tagging: diagram box "Qwen/Gemini" + its callout —
"Add one of these tag (para tag exist in NV dataset) to the transcript without alter the
meaning".

New (not in the source notebook — cell 11's own markdown says this step already happened
upstream of the notebook; it only parsed `[tag]` back out of an already-tagged transcript
with a regex). This module is a real implementation of that upstream step, so the repo
covers the whole diagram end-to-end: given a plain transcript, ask an LLM to weave in
*one* paralinguistic tag from the VocalSound vocabulary without changing anything else, then
verify that promise mechanically rather than trusting the model's output blind.

Backends behind one interface, matching the diagram's "Qwen/Gemini" label:
  - qwen:            DashScope's OpenAI-compatible endpoint, text only. DASHSCOPE_API_KEY.
  - gemini:          Gemini API, text only. GEMINI_API_KEY.
  - qwen_omni_audio: DashScope Qwen-Omni, sends the *recording* alongside the transcript so
                     the model can pick a tag (and a position) that suits the actual
                     delivery — a sigh after a slow, breathy phrase rather than wherever
                     the text alone suggests. DASHSCOPE_API_KEY. Opt-in: costs more per row
                     than the text-only path and is limited by the model's audio duration
                     cap (150s for qwen3-omni-flash), so `qwen` remains the default.
"""
from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path

from para_synth.dataset import TAG_RE
from para_synth.vocalsound import VS_CLASSES

SYSTEM_PROMPT = """You insert paralinguistic annotations into speech transcripts.

Add exactly ONE tag from this list to the transcript, in square brackets, at the point in \
the sentence where that sound would naturally occur: {tags}

Rules:
- Add one of these tags to the transcript without altering the meaning.
- Do not change, remove, reorder, or paraphrase any existing word.
- Insert the bracketed tag as a standalone token between two existing words (or at a \
sentence boundary) — never inside a word.
- Output ONLY the resulting transcript, nothing else — no explanation, no quotes.
"""

AUDIO_SYSTEM_PROMPT = SYSTEM_PROMPT + """
You are also given the recording this transcript came from. Listen to the delivery — pace,
energy, breathiness, and where the speaker actually pauses — and choose both the tag and its
position to suit that delivery, placing it where the speaker draws breath or pauses rather \
than mid-phrase. The recording does NOT contain the sound you are adding; you are deciding \
which one would plausibly belong there.
"""

# qwen3-omni-flash accepts up to 150s of audio; longer input is rejected by the API. Guard
# locally so an over-long file gives a clear error instead of an opaque API failure.
QWEN_OMNI_MAX_AUDIO_S = 150.0


class TaggingError(Exception):
    pass


@dataclass
class TaggingResult:
    text: str
    tag: str


def _extract_single_tag(original: str, candidate: str) -> str:
    """Verify `candidate` differs from `original` by exactly one inserted `[tag]` token
    (tag drawn from VS_CLASSES) and nothing else — defends "without alter the meaning"
    mechanically instead of trusting the model. Raises TaggingError otherwise."""
    tags_found = TAG_RE.findall(candidate)
    if len(tags_found) != 1:
        raise TaggingError(f"expected exactly one [tag], got {tags_found!r} in {candidate!r}")
    tag = tags_found[0].strip().lower()
    if tag not in VS_CLASSES:
        raise TaggingError(f"tag {tag!r} is not one of {VS_CLASSES}")

    stripped = TAG_RE.sub(" ", candidate)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    original_norm = re.sub(r"\s+", " ", original).strip()
    if stripped != original_norm:
        raise TaggingError(
            f"tagged transcript changed more than the inserted tag:\n"
            f"  original (normalized): {original_norm!r}\n"
            f"  stripped candidate:    {stripped!r}"
        )
    return f"[{tag}]"


def _call_qwen(text: str, model: str) -> str:
    from openai import OpenAI  # DashScope exposes an OpenAI-compatible endpoint

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise TaggingError("DASHSCOPE_API_KEY is not set")
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(tags=", ".join(VS_CLASSES))},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _call_gemini(text: str, model: str) -> str:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise TaggingError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT.format(tags=", ".join(VS_CLASSES)))
    resp = gm.generate_content(text, generation_config={"temperature": 0.2})
    return resp.text.strip()


def _call_qwen_omni_audio(text: str, model: str, audio_path: str | Path) -> str:
    """Qwen-Omni with the recording attached, so tag choice/placement can follow the actual
    delivery rather than the text alone.

    Two API quirks, both required by DashScope for the Omni models: the request must be
    streamed (a non-streamed call is rejected), and `modalities=["text"]` keeps the model
    from also synthesizing a spoken reply we'd have to discard.
    """
    import soundfile as sf

    # Cheap validation before the heavyweight import and the network call, so an unset key
    # or an over-long file reports that plainly rather than surfacing later.
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise TaggingError("DASHSCOPE_API_KEY is not set")

    audio_path = Path(audio_path)
    info = sf.info(str(audio_path))
    if info.duration > QWEN_OMNI_MAX_AUDIO_S:
        raise TaggingError(
            f"{audio_path.name} is {info.duration:.0f}s, over {model}'s "
            f"{QWEN_OMNI_MAX_AUDIO_S:.0f}s audio limit"
        )
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()

    from openai import OpenAI  # DashScope exposes an OpenAI-compatible endpoint

    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": AUDIO_SYSTEM_PROMPT.format(tags=", ".join(VS_CLASSES))},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": audio_path.suffix.lstrip(".") or "wav"},
                    },
                ],
            },
        ],
        modalities=["text"],
        stream=True,
        temperature=0.2,
    )
    chunks = []
    for event in stream:
        if event.choices and event.choices[0].delta and event.choices[0].delta.content:
            chunks.append(event.choices[0].delta.content)
    return "".join(chunks).strip()


_TEXT_BACKENDS = {"qwen": _call_qwen, "gemini": _call_gemini}
_AUDIO_BACKENDS = {"qwen_omni_audio": _call_qwen_omni_audio}
BACKENDS = list(_TEXT_BACKENDS) + list(_AUDIO_BACKENDS)

DEFAULT_MODELS = {
    "qwen": "qwen-plus",
    "gemini": "gemini-2.5-flash",
    "qwen_omni_audio": "qwen3-omni-flash",
}


def insert_para_tag(
    text: str,
    backend: str = "qwen",
    model: str | None = None,
    max_retries: int = 2,
    audio_path: str | Path | None = None,
) -> TaggingResult:
    """Ask the LLM to insert one paralinguistic tag into `text`, retrying on a failed
    mechanical check (wrong tag count, tag outside the vocabulary, or any other text
    altered). Raises TaggingError if `max_retries` is exhausted.

    `audio_path` is required by audio backends (see `_AUDIO_BACKENDS`) and ignored by the
    text-only ones.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown tagging backend {backend!r} — use one of {BACKENDS}")
    model = model or DEFAULT_MODELS[backend]

    if backend in _AUDIO_BACKENDS:
        if audio_path is None:
            raise ValueError(f"backend {backend!r} needs audio_path")
        call = lambda: _AUDIO_BACKENDS[backend](text, model, audio_path)  # noqa: E731
    else:
        call = lambda: _TEXT_BACKENDS[backend](text, model)  # noqa: E731

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            candidate = call()
            tag = _extract_single_tag(text, candidate)
            return TaggingResult(text=candidate, tag=tag)
        except TaggingError as e:
            last_error = e
            print(f"⚠️  tagging attempt {attempt + 1}/{max_retries + 1} failed: {e}")
    raise TaggingError(f"gave up after {max_retries + 1} attempts: {last_error}")

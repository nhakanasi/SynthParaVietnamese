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

Two placement modes:

  - **free** (`tagging.slot_constrained: false`) — the model returns the whole transcript
    with a tag woven in wherever it likes, and `_extract_single_tag()` checks mechanically
    that nothing else changed. The position is unconstrained, so it usually falls
    mid-phonation; see para_synth/slots.py for the measurement.
  - **slot-constrained** (the default) — the model is offered only positions VAD found a
    real pause at, and answers with `{"slot": N, "tag": "..."}`. It never echoes the
    transcript; `slots.insert_tag_at()` writes it. Nothing needs verifying because nothing
    can be altered, and the position is a pause by construction.
"""
from __future__ import annotations

import base64
import json
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

SLOT_SYSTEM_PROMPT = """You choose where a paralinguistic sound belongs in a recording.

You are given a recording and its transcript. The transcript has numbered markers — <1>, \
<2>, … — at the points where the speaker actually pauses. These are the only positions \
available to you.

Listen to the delivery, then choose:
  - the ONE marker where a paralinguistic sound would most plausibly occur, and
  - the ONE tag that fits what the speaker is doing there, from this list: {tags}

The recording does NOT already contain the sound; you are deciding which one would belong.

Answer with JSON and nothing else: {{"slot": <marker number>, "tag": "<tag>"}}
If none of the offered positions would plausibly carry any of these sounds, answer \
{{"slot": null}}.
"""

# qwen3-omni-flash accepts up to 150s of audio; longer input is rejected by the API. Guard
# locally so an over-long file gives a clear error instead of an opaque API failure.
QWEN_OMNI_MAX_AUDIO_S = 150.0


class TaggingError(Exception):
    pass


class NoSuitableSlot(TaggingError):
    """The model was offered pause positions and rejected all of them.

    Distinct from the other tagging failures because it is an answer, not a malfunction:
    retrying will produce the same verdict, and the row should be skipped rather than
    re-asked.
    """


@dataclass
class TaggingResult:
    text: str
    tag: str
    # Which offered pause was chosen, in slot-constrained mode. None on the free path,
    # where the position came out of the model's own prose and no pause was involved.
    slot_i: int | None = None


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


def _qwen_omni_request(system_prompt: str, text: str, model: str, audio_path: str | Path) -> str:
    """One Qwen-Omni call with the recording attached, shared by both placement modes.

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
            {"role": "system", "content": system_prompt},
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


def _call_qwen_omni_audio(text: str, model: str, audio_path: str | Path) -> str:
    """Free placement with the recording attached, so tag choice/position can follow the
    actual delivery rather than the text alone."""
    return _qwen_omni_request(
        AUDIO_SYSTEM_PROMPT.format(tags=", ".join(VS_CLASSES)), text, model, audio_path
    )


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


def _parse_slot_choice(raw: str, n_slots: int) -> tuple[int, str]:
    """`{"slot": 3, "tag": "laughter"}` -> (3, "laughter").

    Tolerates the model wrapping its JSON in prose or a ```json fence, since that is the
    common failure and re-asking for it costs another call with the audio attached. Anything
    else — a slot outside the offered range, a tag outside the vocabulary — raises, and the
    caller's retry loop asks again.
    """
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m is None:
        raise TaggingError(f"no JSON object in the reply: {raw!r}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise TaggingError(f"reply is not valid JSON ({e}): {raw!r}") from e

    if obj.get("slot") is None:
        raise NoSuitableSlot("the model rejected every offered pause position")

    slot = obj["slot"]
    if not isinstance(slot, int) or not 1 <= slot <= n_slots:
        raise TaggingError(f"slot {slot!r} is not one of 1..{n_slots}")

    tag = str(obj.get("tag", "")).strip().lower().strip("[]")
    if tag not in VS_CLASSES:
        raise TaggingError(f"tag {tag!r} is not one of {VS_CLASSES}")
    return slot, tag


def choose_slot(
    text: str,
    slots: list,
    audio_path: str | Path,
    model: str | None = None,
    max_retries: int = 2,
) -> TaggingResult:
    """Ask the model to pick one of `slots` (para_synth.slots.Slot) and one tag for it, then
    build the tagged transcript from that choice.

    Always uses the audio backend: the whole point of constraining the position to a real
    pause is that the model can hear which pause is a breath worth filling and which is a
    hesitation. Raises NoSuitableSlot when it declines every position, and TaggingError when
    it fails to answer usably within `max_retries`.
    """
    from para_synth.slots import insert_tag_at, render_with_markers

    if not slots:
        raise NoSuitableSlot("no pause positions were offered")
    model = model or DEFAULT_MODELS["qwen_omni_audio"]
    marked = render_with_markers(text, slots)
    system_prompt = SLOT_SYSTEM_PROMPT.format(tags=", ".join(VS_CLASSES))

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = _qwen_omni_request(system_prompt, marked, model, audio_path)
            slot_i, tag = _parse_slot_choice(raw, len(slots))
            slot = slots[slot_i - 1]
            return TaggingResult(text=insert_tag_at(text, slot, tag), tag=f"[{tag}]", slot_i=slot_i)
        except NoSuitableSlot:
            raise
        except TaggingError as e:
            last_error = e
            print(f"⚠️  slot choice attempt {attempt + 1}/{max_retries + 1} failed: {e}")
    raise TaggingError(f"gave up after {max_retries + 1} attempts: {last_error}")

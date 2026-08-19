"""LLM transcript tagging: diagram box "Qwen/Gemini" + its callout —
"Add one of these tag (para tag exist in NV dataset) to the transcript without alter the
meaning".

New (not in the source notebook — cell 11's own markdown says this step already happened
upstream of the notebook; it only parsed `[tag]` back out of an already-tagged transcript
with a regex). This module is a real implementation of that upstream step, so the repo
covers the whole diagram end-to-end: given a plain transcript, ask an LLM to weave in
*one* paralinguistic tag from the VocalSound vocabulary without changing anything else, then
verify that promise mechanically rather than trusting the model's output blind.

Two backends behind one interface, matching the diagram's "Qwen/Gemini" label:
  - qwen:   DashScope's OpenAI-compatible endpoint. Requires DASHSCOPE_API_KEY.
  - gemini: Gemini API. Requires GEMINI_API_KEY.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

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


_BACKENDS = {"qwen": _call_qwen, "gemini": _call_gemini}


def insert_para_tag(text: str, backend: str = "qwen", model: str | None = None, max_retries: int = 2) -> TaggingResult:
    """Ask the LLM to insert one paralinguistic tag into `text`, retrying on a failed
    mechanical check (wrong tag count, tag outside the vocabulary, or any other text
    altered). Raises TaggingError if `max_retries` is exhausted."""
    if backend not in _BACKENDS:
        raise ValueError(f"unknown tagging backend {backend!r} — use one of {list(_BACKENDS)}")
    call = _BACKENDS[backend]
    default_model = {"qwen": "qwen-plus", "gemini": "gemini-2.5-flash"}[backend]
    model = model or default_model

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            candidate = call(text, model)
            tag = _extract_single_tag(text, candidate)
            return TaggingResult(text=candidate, tag=tag)
        except TaggingError as e:
            last_error = e
            print(f"⚠️  tagging attempt {attempt + 1}/{max_retries + 1} failed: {e}")
    raise TaggingError(f"gave up after {max_retries + 1} attempts: {last_error}")

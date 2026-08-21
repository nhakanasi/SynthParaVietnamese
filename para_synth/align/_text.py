"""Shared text-prep helpers for the alignment backends (mms.py, qwen3.py).

Ported from notebook45ee5319ae.ipynb cells 8, 10.
"""
from __future__ import annotations

import unicodedata
from typing import NamedTuple

from para_synth.dataset import TAG_RE


class Word(NamedTuple):
    """One aligned word and the span of audio it occupies."""

    text: str
    start: float
    end: float


class Token(NamedTuple):
    """One whitespace-separated token of a transcript, with where it sits in that string.

    `start`/`end` are character offsets into the original text, which is what lets a time
    chosen in the audio be turned back into a position in the transcript (para_synth.slots).
    """

    text: str
    start: int
    end: int


def token_spans(text: str) -> list[Token]:
    """Whitespace tokens of `text`, each carrying its character span."""
    tokens: list[Token] = []
    pos = 0
    for tok in text.split():
        start = text.index(tok, pos)
        tokens.append(Token(text=tok, start=start, end=start + len(tok)))
        pos = start + len(tok)
    return tokens


def deaccent_vi(w: str) -> str:
    """Dependency-free approximate romanization for when uroman isn't available: NFD-
    decompose (peels tone/circumflex/breve/horn marks off as separate combining
    characters), strip those, then handle đ/Đ by hand since Unicode gives it no canonical
    decomposition — it's a distinct base letter (U+0111), not "d" plus a combining stroke,
    so NFD alone won't touch it."""
    w = w.replace("đ", "d").replace("Đ", "D")
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", w)


def clean_word(w: str) -> str:
    return unicodedata.normalize("NFC", w).strip(",.!?;:\"'()…")


def words_around_tag(text: str) -> tuple[list[str] | None, list[str] | None]:
    """"a b [tag] c d" -> (["a","b"], ["c","d"]) — cleaned words before/after the tag.
    (None, None) if there's nothing on one side to bracket a boundary with."""
    tag_m = TAG_RE.search(text)
    if tag_m is None:
        return None, None
    before = [w for w in (clean_word(w) for w in text[: tag_m.start()].split()) if w]
    after = [w for w in (clean_word(w) for w in text[tag_m.end() :].split()) if w]
    if not before or not after:
        return None, None
    return before, after

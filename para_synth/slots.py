"""Candidate positions for a paralinguistic event: the pauses a speaker actually leaves,
expressed as positions in the transcript.

This inverts the order the pipeline used to work in. Previously an LLM inserted `[laughter]`
wherever the text suggested and forced alignment then reported what time that corresponded
to — which is a word boundary, and a word boundary in connected speech is usually continuous
phonation, not a pause. Measured on the first real 20-row batch, the aligned time landed on
speech nearly as loud as the surrounding segment (median junction activity 0.85), and moving
the cut afterwards is bounded by meaning rather than by acoustics: the transcript claims the
event sits between two specific words, so the audio cannot travel far from them. With the
nearest pause a median 2.02s away, 19 of 20 rows had nothing reachable (see
para_synth/vad.py and `vad.max_shift_s`).

So instead of finding a pause near a chosen position, this module offers only positions that
are already pauses, and the LLM picks one of them (para_synth.tagging). The tagged transcript
is then written by `insert_tag_at()` from that same choice, which makes "the text says the
event is here, and the audio splices it here" one fact rather than two estimates that have to
agree. Rows where VAD finds no qualifying pause produce no slots and are skipped — the event
is never forced into the middle of a word.

This is the ordering SynParaSpeech (arXiv:2509.14946) uses: segment on VAD first, then have
the LLM insert the tag at a segment boundary. It is also why that paper needs no crossfade
anywhere — with the cut in real silence, there is no seam to hide.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from para_synth.align._text import Token, Word, token_spans


@dataclass
class Slot:
    """One offered insertion position: a pause, and the point in the transcript it falls at.

    `char_offset` indexes the *raw* transcript and always sits at the start of the word that
    follows the pause, so inserting there reads naturally ("... nữa , [laughter] thì ...").
    """

    i: int  # 1-based, as shown to the model
    time_s: float  # pause midpoint — the furthest point from either speech edge
    pause_s: float
    char_offset: int
    before: str  # the word ending the pause's left side, for the prompt and for logs
    after: str


def _aligner_token(tok: str) -> str:
    """The form the Qwen3 forced aligner reduces a whitespace token to.

    Mirrors its `clean_token` (keep letters, digits and apostrophes), because the aligner
    tokenizes the text itself and drops whatever cleans to nothing — a standalone "," in
    these transcripts produces no aligned word at all. Reproducing the rule here is only an
    optimisation for matching: `map_words_to_tokens()` verifies every match and gives up on
    the row rather than guessing, so if the upstream rule ever changes the failure is a
    skipped row, never a tag placed at the wrong offset.
    """
    return "".join(
        ch for ch in tok
        if ch == "'" or unicodedata.category(ch)[0] in ("L", "N")
    )


def map_words_to_tokens(words: list[Word], tokens: list[Token]) -> list[int] | None:
    """For each aligned word, the index of the transcript token it came from.

    Walks both sequences in order, skipping tokens the aligner would have dropped. Returns
    None if the two disagree — the alignment covers different text than the transcript does,
    and no offset derived from it can be trusted.
    """
    mapping: list[int] = []
    t = 0
    for word in words:
        target = _aligner_token(word.text).casefold()
        while t < len(tokens) and _aligner_token(tokens[t].text).casefold() != target:
            if _aligner_token(tokens[t].text):
                return None  # a real word the aligner didn't report: the two have diverged
            t += 1
        if t >= len(tokens):
            return None
        mapping.append(t)
        t += 1
    return mapping


def candidate_slots(
    text: str,
    words: list[Word],
    pauses: list[tuple[float, float]],
    min_pause_s: float = 0.0,
) -> list[Slot]:
    """The pauses in `pauses` that fall in a gap between two aligned words, as transcript
    positions.

    A pause belongs to the gap between words *k-1* and *k* when it overlaps the silent
    interval between them; where several gaps overlap, the largest overlap wins. Pauses
    before the first word or after the last are dropped: splicing there puts the event
    outside the utterance, which is a different edit from the one being described.

    Two pauses can land in the same gap (VAD split by a breath the aligner didn't see); only
    the longer survives, since they denote the same insertion point.
    """
    tokens = token_spans(text)
    mapping = map_words_to_tokens(words, tokens)
    if mapping is None or len(words) < 2:
        return []

    best: dict[int, tuple[float, float]] = {}  # word index k -> (pause_s, midpoint)
    for lo, hi in pauses:
        if hi - lo < min_pause_s:
            continue
        overlaps = [
            (min(hi, words[k].start) - max(lo, words[k - 1].end), k)
            for k in range(1, len(words))
        ]
        overlap, k = max(overlaps)
        if overlap <= 0:
            continue
        if k not in best or hi - lo > best[k][0]:
            best[k] = (hi - lo, 0.5 * (lo + hi))

    slots = []
    for i, k in enumerate(sorted(best), start=1):
        pause_s, mid = best[k]
        slots.append(
            Slot(
                i=i,
                time_s=mid,
                pause_s=pause_s,
                char_offset=tokens[mapping[k]].start,
                before=words[k - 1].text,
                after=words[k].text,
            )
        )
    return slots


def render_with_markers(text: str, slots: list[Slot]) -> str:
    """`text` with `<1>`, `<2>` … at each candidate position — what the model is shown.

    Applied back-to-front so each insertion leaves the offsets of the ones before it intact.
    """
    out = text
    for slot in sorted(slots, key=lambda s: s.char_offset, reverse=True):
        out = f"{out[: slot.char_offset]}<{slot.i}> {out[slot.char_offset :]}"
    return out


def insert_tag_at(text: str, slot: Slot, tag: str) -> str:
    """`text` with `[tag]` inserted at `slot` — the only writer of tagged text on this path.

    The model is never asked to echo the transcript back, so there is nothing to verify:
    every word is preserved because none of them are ever rewritten.
    """
    return f"{text[: slot.char_offset]}[{tag}] {text[slot.char_offset :]}"


def slot_from_dict(raw: dict) -> Slot:
    """A Slot back out of `stages/slots.jsonl`."""
    return Slot(**raw)

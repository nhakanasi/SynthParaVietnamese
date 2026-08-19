"""Last-resort alignment: no acoustic info, just word-position fraction snapped to the
nearest real silence. Ported from notebook45ee5319ae.ipynb cell 8.
"""
from __future__ import annotations

from para_synth.audio_utils import split_nonsilent
from para_synth.dataset import TAG_RE


def estimate_tag_time_proportional(text: str, duration_s: float) -> float:
    """No acoustic info — assume ~constant speaking rate and place the tag at the same
    fraction through the audio as it sits through the words."""
    tag_m = TAG_RE.search(text)
    if tag_m is None:
        return duration_s / 2
    before_words = len(text[: tag_m.start()].split())
    total_words = len(TAG_RE.sub(" ", text).split())
    frac = before_words / max(total_words, 1)
    return frac * duration_s


def find_local_silence_near(speech, sr, target_s, search_s=0.5, top_db=35) -> float:
    """Snap an estimated insertion time to the nearest actual quiet moment within
    ±search_s of it. Only used for the proportional fallback — a real alignment result
    (qwen3/mfa/mms) is trusted as-is instead, since the tag is a post-hoc annotation and
    there's no guarantee of a genuine pause there."""
    target = int(target_s * sr)
    lo = max(0, target - int(search_s * sr))
    hi = min(len(speech), target + int(search_s * sr))
    iv = split_nonsilent(speech[lo:hi], top_db=top_db)
    if len(iv) < 2:
        return target_s  # no clear gaps nearby — use the estimate as-is
    local_target = target - lo
    candidates = [int((iv[i][1] + iv[i + 1][0]) / 2) for i in range(len(iv) - 1)]
    best = min(candidates, key=lambda c: abs(c - local_target))
    return (lo + best) / sr

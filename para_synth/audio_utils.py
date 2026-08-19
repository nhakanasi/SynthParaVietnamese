"""Shared, dependency-light audio DSP utilities.

Ported near-verbatim from notebook45ee5319ae.ipynb cell 9 ("Shared audio utilities").

NOTE: this deliberately avoids librosa/scipy. Seed-VC's requirements.txt downgrades numpy
to 1.x, but many environments (this one included, originally Kaggle) ship a scipy built
against numpy 2.x — importing librosa then dies with "No module named numpy.strings". So
`split_nonsilent` below is a from-scratch numpy port of librosa.effects.split, verified
sample-identical to librosa's implementation.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import torch


def _resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Best available resampler, no scipy/librosa required."""
    if orig_sr == target_sr:
        return x
    try:
        import soxr  # librosa's own backend

        return soxr.resample(x, orig_sr, target_sr, quality="HQ").astype(np.float32)
    except Exception:
        pass
    try:
        import torchaudio  # already present for seed-vc

        t = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0)
        return torchaudio.functional.resample(t, orig_sr, target_sr).squeeze(0).numpy()
    except Exception:
        pass
    # last resort: linear interpolation (fine for our short clips)
    n = int(round(len(x) * target_sr / orig_sr))
    return np.interp(
        np.linspace(0, len(x) - 1, n, dtype=np.float64),
        np.arange(len(x), dtype=np.float64),
        x,
    ).astype(np.float32)


def load_mono(path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1).astype(np.float32)
    if target_sr and sr != target_sr:
        x = _resample(x, sr, target_sr)
        sr = target_sr
    return x, sr


def split_nonsilent(x, top_db=30, frame_length=2048, hop_length=512) -> np.ndarray:
    """numpy port of librosa.effects.split — frame RMS thresholded at -top_db below the
    loudest frame. Verified sample-identical to librosa."""
    x = np.asarray(x, dtype=np.float32)
    xp = np.pad(x, frame_length // 2, mode="constant")
    n_frames = 1 + (len(xp) - frame_length) // hop_length
    if n_frames < 1:
        return np.array([[0, len(x)]], dtype=int)

    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    rms = np.sqrt(np.mean(xp[idx] ** 2, axis=1))
    ref = rms.max()
    if ref <= 0:
        return np.zeros((0, 2), dtype=int)

    db = 20 * np.log10(np.maximum(rms, 1e-10) / max(ref, 1e-10))
    nz = db > -top_db
    if not nz.any():
        return np.zeros((0, 2), dtype=int)

    edges = np.flatnonzero(np.diff(nz.astype(int)))
    starts = ([0] if nz[0] else []) + list(edges[nz[edges + 1]] + 1)
    ends = list(edges[~nz[edges + 1]] + 1) + ([len(nz)] if nz[-1] else [])
    iv = np.array(
        [[s * hop_length, min(e * hop_length, len(x))] for s, e in zip(starts, ends)],
        dtype=int,
    )
    return np.clip(iv, 0, len(x))


def trim_event(x, sr, top_db=30, pad_s=0.12) -> np.ndarray:
    """Keep first onset -> last offset (laughter comes in bursts, don't split them)."""
    iv = split_nonsilent(x, top_db=top_db)
    if len(iv) == 0:
        return x
    s = max(0, int(iv[0][0]) - int(pad_s * sr))
    e = min(len(x), int(iv[-1][1]) + int(pad_s * sr))
    return x[s:e]


def longest_gap(x, sr, top_db=35) -> int:
    """Midpoint of the widest silent stretch — usually a sentence boundary."""
    iv = split_nonsilent(x, top_db=top_db)
    if len(iv) < 2:
        return len(x)
    gaps = [
        (int(iv[i + 1][0] - iv[i][1]), int((iv[i][1] + iv[i + 1][0]) // 2))
        for i in range(len(iv) - 1)
    ]
    return max(gaps)[1]


def envelope(x, sr, fade_ms) -> np.ndarray:
    n = min(int(fade_ms / 1000 * sr), len(x) // 2)
    if n < 2:
        return x
    y = x.copy()
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def splice(speech, insert, sr, placement, gain_db, pad_ms, fade_ms, at_s=None):
    """`at_s`, when given, overrides `placement` with a specific cut time in seconds (an
    alignment estimate from para_synth.align) — 'gap'/'before'/'after' stay as a fallback
    for when that estimate isn't available."""
    ins = envelope(insert.copy(), sr, fade_ms)
    # level-match: put the insert gain_db below the speech peak
    tgt = np.max(np.abs(speech)) * (10 ** (gain_db / 20))
    ins = ins / (np.max(np.abs(ins)) + 1e-9) * tgt
    pad = np.zeros(int(pad_ms / 1000 * sr), dtype=np.float32)
    blk = np.concatenate([pad, ins, pad])
    if at_s is not None:
        cut = int(at_s * sr)
    elif placement == "before":
        cut = 0
    elif placement == "after":
        cut = len(speech)
    else:
        cut = longest_gap(speech, sr)
    cut = max(0, min(cut, len(speech)))
    out = np.concatenate([speech[:cut], blk, speech[cut:]])
    return out, cut / sr


def rms(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def cosine_fade_out(x, n: int):
    """Raised-cosine fade-out on the last `n` samples. Smoother roll-off than a linear
    ramp — a linear fade's slope changes abruptly at both ends, which can still leave an
    audible tick; the cosine curve's derivative goes to zero at the edges instead."""
    n = min(n, len(x))
    if n < 2:
        return x
    y = x.copy()
    curve = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, n, dtype=np.float32)))
    y[-n:] = y[-n:] * curve
    return y


def cosine_fade_in(x, n: int):
    n = min(n, len(x))
    if n < 2:
        return x
    y = x.copy()
    curve = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n, dtype=np.float32)))
    y[:n] = y[:n] * curve
    return y


def boundary_is_active(edge, overall_rms: float, ratio: float = 0.2) -> bool:
    """True if a short slice at a cut point still carries active phonation relative to
    its segment's own overall level — i.e. the cut lands mid-sound (a continuant, or a
    word forced-alignment placed right up against without a real pause) rather than in a
    natural quiet moment, and so needs damping / a gap rather than a bare concatenation."""
    if overall_rms <= 0:
        return False
    return rms(edge) > ratio * overall_rms


def matched_room_tone(n: int, reference, amplitude_ratio: float = 0.05):
    """A short low-level noise bridge whose amplitude tracks `reference`'s own RMS,
    instead of true digital silence — a hard zero-silence gap between two segments that
    both have real background noise reads as an artificial mute, especially on
    headphones. `reference` should be a quiet-ish slice near the boundary, not a loud one."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    amp = max(rms(reference) * amplitude_ratio, 1e-5)
    return np.random.normal(0, amp, n).astype(np.float32)


def adaptive_splice(speech, insert, sr, gain_db, min_pad_ms, max_gap_ms, fade_ms, at_s):
    """Two-junction splice (speech[:cut] -> insert -> speech[cut:]) that inspects each
    junction's boundary RMS before deciding what to do with it, instead of always paying a
    fixed fade+pad cost regardless of where the cut actually fell:

    - If a boundary is quiet already (`boundary_is_active` is False), it gets only a small
      `min_pad_ms` room-tone bridge — no fade needed, nothing was cut off.
    - If a boundary is "active" (real phonation right at the cut — common here since the
      cut point comes from forced alignment, not a silence heuristic, so it can land on a
      continuant), the adjacent segment gets a cosine fade at that edge, and the bridge
      widens to `max_gap_ms` so the damped tail has room to actually decay before the next
      sound starts.

    Same return contract as `splice()`: (waveform, cut_time_s).
    """
    fade_len = int(fade_ms / 1000 * sr)
    min_pad = int(min_pad_ms / 1000 * sr)
    max_gap = int(max_gap_ms / 1000 * sr)

    if at_s is None:
        raise ValueError("adaptive_splice requires at_s — an alignment estimate, not a placement heuristic")
    cut = max(0, min(int(at_s * sr), len(speech)))
    speech_before, speech_after = speech[:cut], speech[cut:]

    ins = insert.copy()
    tgt = np.max(np.abs(speech)) * (10 ** (gain_db / 20))
    ins = ins / (np.max(np.abs(ins)) + 1e-9) * tgt

    rms_before = rms(speech_before)
    rms_after = rms(speech_after)
    rms_ins = rms(ins)

    # Junction 1: end of speech_before <-> start of insert
    before_active = boundary_is_active(speech_before[-fade_len:], rms_before)
    if before_active:
        speech_before = cosine_fade_out(speech_before, fade_len)
    if boundary_is_active(ins[:fade_len], rms_ins):
        ins = cosine_fade_in(ins, fade_len)
    gap1 = matched_room_tone(max_gap if before_active else min_pad, speech_before[-fade_len:])

    # Junction 2: end of insert <-> start of speech_after
    ins_tail_active = boundary_is_active(ins[-fade_len:], rms_ins)
    if ins_tail_active:
        ins = cosine_fade_out(ins, fade_len)
    if boundary_is_active(speech_after[:fade_len], rms_after):
        speech_after = cosine_fade_in(speech_after, fade_len)
    gap2 = matched_room_tone(max_gap if ins_tail_active else min_pad, ins[-fade_len:])

    out = np.concatenate([speech_before, gap1, ins, gap2, speech_after])
    return out, cut / sr

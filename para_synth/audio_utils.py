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


def _exp_decay_curve(n: int, k: float = 5.0) -> np.ndarray:
    """Normalised exponential decay from exactly 1.0 to exactly 0.0 over `n` samples.

    `k` sets the curvature: the raw exp(-k·t) never reaches zero, so it's rescaled to hit
    0 at the end rather than stepping off a small residual (which would reintroduce the
    very click the fade exists to remove).
    """
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    e = np.exp(-k * t)
    return ((e - np.exp(-k)) / (1.0 - np.exp(-k))).astype(np.float32)


def exponential_fade_out(x, n: int, k: float = 5.0):
    """Exponential fade-out on the last `n` samples — drops fast at first, then tails off.

    Preferred over the cosine fade for damping a cut that landed mid-phonation: the bulk of
    the attenuation happens in the first few milliseconds, so the interrupted sound reads as
    "stopped" rather than "slowly turned down", which is what a speaker breaking off to
    laugh/cough actually does.
    """
    n = min(n, len(x))
    if n < 2:
        return x
    y = x.copy()
    y[-n:] = y[-n:] * _exp_decay_curve(n, k)
    return y


def exponential_fade_in(x, n: int, k: float = 5.0):
    """Exponential fade-in on the first `n` samples — the mirror of `exponential_fade_out`,
    i.e. rises fast and then flattens.

    Deliberately fast-rising: paralinguistic events (laughter, cough, sneeze) are
    attack-heavy, and a slow/symmetric fade-in on their onset audibly softens the transient
    and makes the event sound muffled. This preserves the attack while still removing the
    hard discontinuity at the join.
    """
    n = min(n, len(x))
    if n < 2:
        return x
    y = x.copy()
    y[:n] = y[:n] * (1.0 - _exp_decay_curve(n, k))
    return y


def speech_pause_s(speech, sr, top_db=30, lo=0.04, hi=0.40) -> float:
    """This speaker's own typical inter-utterance pause, in seconds — used to size the gap
    around a spliced-in event so it sits at the recording's own tempo.

    Measured as the median gap between voiced runs, which is self-calibrating in the same
    way `speaker_energy_score` is: a fixed pad (the notebook's `pad_ms: 10`) is far shorter
    than any real pause, so an event dropped in with one reads as spliced no matter how
    clean the fades are. Measured medians across real recordings here span ~116-488ms, i.e.
    the right value is per-speaker, not a constant.

    Clamped to [lo, hi]: a recording with no detectable pauses shouldn't collapse the gap to
    zero, and one long dramatic silence shouldn't stretch every event's gap to match it.
    """
    iv = split_nonsilent(speech, top_db=top_db)
    gaps = [(iv[i + 1][0] - iv[i][1]) / sr for i in range(len(iv) - 1)]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return lo
    return float(np.clip(np.median(gaps), lo, hi))


def quietest_window(x, sr, win_s: float = 0.1) -> np.ndarray:
    """The quietest `win_s` slice of `x` — the recording's own noise floor, used as the
    reference for `matched_room_tone`. Taken from the whole utterance rather than from the
    slice next to the cut, since the cut here usually lands *on* speech (forced alignment,
    not a silence heuristic), so the adjacent audio is a bad estimate of the noise floor."""
    win = max(int(win_s * sr), 1)
    if len(x) <= win:
        return x
    hop = max(win // 2, 1)
    starts = range(0, len(x) - win + 1, hop)
    best = min(starts, key=lambda i: rms(x[i : i + win]))
    return x[best : best + win]


def boundary_activity(edge, overall_rms: float) -> float:
    """How much energy a short slice at a cut point carries, as a fraction of its segment's
    own overall level. 0 = digital silence at the cut, ~1 = the cut lands on audio as loud
    as the segment's average, i.e. squarely mid-phonation.

    Relative rather than absolute for the same reason `speech_pause_s` is: a quiet speaker's
    mid-vowel is louder than a loud speaker's pause in absolute terms, so an absolute
    threshold would sort recordings by their gain staging instead of by where the cut landed.
    """
    if overall_rms <= 0:
        return 0.0
    return rms(edge) / overall_rms


def boundary_is_active(edge, overall_rms: float, ratio: float = 0.2) -> bool:
    """True if a short slice at a cut point still carries active phonation relative to
    its segment's own overall level — i.e. the cut lands mid-sound (a continuant, or a
    word forced-alignment placed right up against without a real pause) rather than in a
    natural quiet moment, and so needs damping / a gap rather than a bare concatenation."""
    return boundary_activity(edge, overall_rms) > ratio


def splice_boundary_activity(speech, sr, at_s: float, win_ms: float = 50.0) -> tuple[float, float]:
    """`boundary_activity` of both junctions a splice at `at_s` would create, as
    (before, after): the tail of the speech that precedes the event and the head of the
    speech that follows it.

    Computed from the *source* recording rather than the finished Para one on purpose. By
    the time `tempo_splice` is done, both junctions have been fade-damped and separated by a
    room-tone gap, so the finished file reads as clean whether the aligner picked a pause or
    the middle of a vowel. The badness being measured belongs to the insertion time, and the
    insertion time is a property of the source. That also means this is recoverable for
    already-rendered rows from `source_audio` + `splice_at_s` alone, with no re-synthesis.
    """
    cut = int(np.clip(at_s * sr, 0, len(speech)))
    before, after = speech[:cut], speech[cut:]
    win = max(int(win_ms * sr / 1000.0), 1)
    return (
        boundary_activity(before[-win:], rms(before)),
        boundary_activity(after[:win], rms(after)),
    )


def _loopable(ref: np.ndarray, xf: int) -> np.ndarray:
    """`ref` rearranged so that tiling it end-to-end has no seam: its tail is crossfaded
    onto its head, so the last sample of one repetition continues into the first sample of
    the next. Returns a unit of length `len(ref) - xf`."""
    fade = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    blended = ref[-xf:] * (1.0 - fade) + ref[:xf] * fade
    return np.concatenate([blended, ref[xf:-xf]])


def matched_room_tone(n: int, reference, amplitude_ratio: float = 1.0, rng=None):
    """A bridge of the recording's *own* background, tiled to length `n`, instead of true
    digital silence — a hard zero between two segments that both carry real background noise
    reads as an artificial mute, especially on headphones.

    Built from `reference`'s actual samples (pass `quietest_window()`'s output) rather than
    from synthesized noise, which is what this used to do. Gaussian noise can be matched to
    the floor's RMS but not to its *spectrum*, and real noise floors are strongly
    low-frequency tilted — worse, a denoised or band-limited corpus has almost no HF floor at
    all. Against either, a flat-white bridge is *more* audible than the silence it replaced:
    it reads as a short puff of hiss appearing exactly where the edit is. Tiling the real
    floor carries the true spectrum by construction, and needs no estimate of it.

    `_loopable` removes the seam at the tile boundary, and `rng` picks the starting phase so
    two gaps in the same recording aren't sample-identical. Pass a seeded `random.Random`
    (the pipeline threads `Config.rng(row.id)`) to keep a run reproducible; the previous
    implementation drew from numpy's *global* RNG, so `seed:` in the config never actually
    reproduced the bridge.

    Falls back to noise only when `reference` is too short to tile from.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    if len(ref) < 8:
        amp = max(rms(ref) * amplitude_ratio, 1e-5)
        draw = (rng.gauss if rng is not None else None)
        if draw is None:
            return np.random.normal(0, amp, n).astype(np.float32)
        return np.array([draw(0.0, amp) for _ in range(n)], dtype=np.float32)

    xf = max(len(ref) // 8, 1)
    unit = _loopable(ref, xf)
    tone = np.tile(unit, int(np.ceil(n / len(unit))) + 1)
    start = rng.randrange(len(unit)) if rng is not None else 0
    return (tone[start : start + n] * amplitude_ratio).astype(np.float32)


def existing_silence_s(speech, sr, at_s: float, top_db: float = 30.0,
                       win_s: float = 0.010) -> tuple[float, float]:
    """Seconds of already-silent audio immediately before and after `at_s`, as
    (before_s, after_s) — what a splice at that point gets for free.

    The gap `tempo_splice` inserts is a *target total*, not an addition, and this is the term
    that makes it one. Without it the two are additive, which is a real over-padding on the
    slot-constrained path: `slots.Slot.time_s` is by construction the midpoint of a VAD pause,
    so each side already carries `pause_s/2 + vad.edge_margin_s` of silence, and adding a
    full tempo gap on top puts the event inside up to ~0.9s of dead air where the speaker
    left 0.15s. That reads as a hole in the recording no matter how clean the junctions are.

    Measured here rather than read off `Slot.pause_s` on purpose: it also works on the paths
    where there is no slot (a pre-tagged manifest, or `vad.on_no_pause: keep` leaving the cut
    unsnapped), and it doesn't assume the VAD's pause midpoint is the *acoustic* midpoint —
    in general the two sides are not equal.

    Uses its own short-window envelope rather than `split_nonsilent`'s intervals, which was
    the first implementation and measured badly: that detector frames at 2048 samples and
    reports run edges padded by `frame_length // 2` (~46ms at 22.05kHz), so a cut 50ms inside
    a genuine 600ms pause came back as 0ms of silence on both sides. A 10ms window resolves
    the pause edge to within one frame. The threshold is the same `top_db`-below-peak rule
    `split_nonsilent` uses, so "silent" still means the same thing across the module.

    A cut in the middle of a voiced run returns (0, 0), and the full target gap is then
    added — the pre-slot behaviour, unchanged.
    """
    x = np.asarray(speech, dtype=np.float32)
    if len(x) == 0:
        return 0.0, 0.0
    win = max(int(win_s * sr), 1)
    hop = max(win // 2, 1)
    n_frames = 1 + max(len(x) - win, 0) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    idx = np.minimum(idx, len(x) - 1)
    env = np.sqrt(np.mean(x[idx].astype(np.float64) ** 2, axis=1))
    ref = env.max()
    if ref <= 0:
        return 0.0, 0.0
    quiet = env <= ref * (10 ** (-top_db / 20))

    cut_f = int(np.clip(at_s * sr, 0, len(x) - 1)) // hop
    cut_f = min(cut_f, n_frames - 1)
    if not quiet[cut_f]:
        return 0.0, 0.0

    lo = cut_f
    while lo > 0 and quiet[lo - 1]:
        lo -= 1
    hi = cut_f
    while hi < n_frames - 1 and quiet[hi + 1]:
        hi += 1
    cut = int(np.clip(at_s * sr, 0, len(x)))
    before = max(0.0, (cut - lo * hop) / sr)
    after = max(0.0, ((hi + 1) * hop - cut) / sr)
    return before, after


def level_insert(insert, speech, sr, gain_db: float, level_ref: str = "context_rms",
                 at_s: float | None = None, context_s: float = 2.5,
                 offset_db: float = 0.0, peak_headroom: float = 1.5) -> np.ndarray:
    """Scale the event to sit at the right level against the speech. Two references:

    - `"peak"` — the original: the event's *peak* is placed `gain_db` below the speech's
      peak over the whole utterance. The flaw is that a single plosive, mouth click or one
      clipped sample anywhere in a 20-second take sets the event's level, and it ignores
      where in the recording the event actually lands.
    - `"context_rms"` (default) — the event's *RMS* is placed `gain_db + offset_db` relative
      to the RMS of the `context_s` seconds of speech immediately before the cut. RMS-to-RMS
      because peak-to-peak compares a sustained laugh against a transient consonant, and
      local because a speaker's level varies across a take.

      This also removes a real inconsistency: `selection.speaker_energy_score` already picks
      the clip's *intensity* against a local `context_s` window, and the splice then levelled
      it against the global peak — the two stages were referenced to different things.

    `offset_db` is the per-class term (`vocalsound.LEVEL_OFFSET_DB`): a laugh genuinely sits
    above conversational RMS and a sniff genuinely sits below, so "match the speech level"
    is wrong for both in opposite directions.

    The result is capped near the recording's own loud peaks (`peak_headroom` x its 99.9th
    percentile) so the event can't run away with the file: `synthesize_row` peak-normalises
    the finished waveform, so an over-hot event doesn't clip — it quietly ducks the entire
    recording around itself. Headroom above 1.0 because a laugh is a high-crest sound and
    legitimately peaks above conversational speech; the cap is a runaway guard, not a target.
    """
    ins = np.asarray(insert, dtype=np.float32).copy()
    peak = float(np.max(np.abs(ins)))
    if peak <= 0:
        return ins

    if level_ref == "peak":
        tgt = float(np.max(np.abs(speech))) * (10 ** (gain_db / 20))
        return ins / (peak + 1e-9) * tgt
    if level_ref != "context_rms":
        raise ValueError(f"level_ref must be 'peak' or 'context_rms', got {level_ref!r}")

    end = len(speech) if at_s is None else int(np.clip(at_s * sr, 0, len(speech)))
    start = max(0, end - max(int(context_s * sr), 1))
    ref_rms = rms(speech[start:end]) or rms(speech)
    if ref_rms <= 0:
        return ins

    target_rms = ref_rms * (10 ** ((gain_db + offset_db) / 20))
    ins = ins * (target_rms / (rms(ins) + 1e-12))

    # Capped against a robust peak, not `max(|speech|)`. The whole point of this branch is
    # not to let one stray click set the event's level, and a cap read off the true maximum
    # smuggles exactly that back in through the other door — measured on a synthetic take, a
    # single 0.99 sample moved the event 8.4 dB by loosening the cap alone. The 99.9th
    # percentile tracks the recording's real loud peaks and ignores lone samples.
    cap = float(np.percentile(np.abs(speech), 99.9)) * peak_headroom
    hot = float(np.max(np.abs(ins)))
    if cap > 0 and hot > cap:
        ins = ins * (cap / hot)
    return ins


def adaptive_splice(speech, insert, sr, gain_db, min_pad_ms, max_gap_ms, fade_ms, at_s, rng=None):
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

    floor = quietest_window(speech, sr)
    rms_before = rms(speech_before)
    rms_after = rms(speech_after)
    rms_ins = rms(ins)

    # Junction 1: end of speech_before <-> start of insert
    before_active = boundary_is_active(speech_before[-fade_len:], rms_before)
    if before_active:
        speech_before = cosine_fade_out(speech_before, fade_len)
    if boundary_is_active(ins[:fade_len], rms_ins):
        ins = cosine_fade_in(ins, fade_len)
    # The bridge is built from the recording's own noise floor, not from the samples at
    # the boundary: `matched_room_tone` now tiles real audio, and the audio at a boundary
    # is speech, which tiled at any level is an audible ghost of the interrupted word.
    gap1 = matched_room_tone(max_gap if before_active else min_pad, floor, rng=rng)

    # Junction 2: end of insert <-> start of speech_after
    ins_tail_active = boundary_is_active(ins[-fade_len:], rms_ins)
    if ins_tail_active:
        ins = cosine_fade_out(ins, fade_len)
    if boundary_is_active(speech_after[:fade_len], rms_after):
        speech_after = cosine_fade_in(speech_after, fade_len)
    gap2 = matched_room_tone(max_gap if ins_tail_active else min_pad, floor, rng=rng)

    out = np.concatenate([speech_before, gap1, ins, gap2, speech_after])
    return out, cut / sr


def tempo_splice(speech, insert, sr, gain_db, fade_ms, at_s, gap_scale: float = 1.0,
                 fade_k: float = 5.0, pre_scale: float = 1.0, post_scale: float = 1.0,
                 level_ref: str = "context_rms", context_s: float = 2.5,
                 level_offset_db: float = 0.0, rng=None):
    """Two-junction splice that damps both cuts with a short exponential fade and separates
    them by a *total* silence sized to this speaker's own pause tempo and shaped by the
    event's respiratory profile.

    Motivated by a measured failure of `splice()` on real output: the forced-aligned cut
    landed mid-phonation on 20/20 rows of the first batch, and `splice()` concatenates the
    speech with no fade at all, so every row had a hard waveform discontinuity. Its
    `fade_ms` envelope applies only to the *insert*, and the Seed-VC insert arrives with
    ~140ms of leading/trailing silence, so that fade was spent entirely on silence while the
    event itself still began at full amplitude.

    The pieces, all of which need each other:

    1. **Trimmed insert** — the caller passes an insert whose dead air has been trimmed
       (see `trim_event`), so the fade lands on the actual event rather than on padding and
       the gap below is the only silence, at a length that was actually chosen.
    2. **Exponential fades at both junctions** — the interrupted speech is faded out and the
       event faded in (and symmetrically at the resume, since cutting mid-word means the
       resumed speech also starts mid-word). Exponential rather than cosine: it damps fast
       and preserves the event's attack — see `exponential_fade_out`/`_in`. Applied
       unconditionally; where the cut already fell in silence the fade is inaudible, so the
       conditional `adaptive_splice` uses only buys a way to be wrong.
    3. **A tempo-matched, respiration-shaped, budgeted gap** — three separate ideas:

       *Tempo*: `speech_pause_s` measures the speaker's median pause and that is the base, so
       the event sits in a silence the recording's own rhythm predicts instead of the fixed
       10ms `pad_ms`, which is shorter than any real pause.

       *Shape*: `pre_scale`/`post_scale` (from `vocalsound.GAP_SHAPE`) make the two sides
       asymmetric, because the respiratory state an event *ends* in is not the one it starts
       in. A laugh ends on depleted lungs and needs a long trailing beat plus an intake
       before speech resumes; a sniff ends ready to speak and needs almost none. A single
       symmetric gap gets at most one of the two sides right.

       *Budget*: the scaled targets are a total, and `existing_silence_s` measures what each
       side already has, so only the shortfall is inserted. Without this the gap is purely
       additive, which over-pads the slot-constrained path badly — see `existing_silence_s`.
       The shortfall is clamped at zero: silence already in the recording is the speaker's
       own audio, and this pipeline does not shorten that.

    Both bridges are `matched_room_tone`, not digital silence, for the reason documented
    there. `rng` seeds them; pass `Config.rng(row.id)` to keep a run reproducible.

    Same return contract as `splice()`: (waveform, cut_time_s).
    """
    if at_s is None:
        raise ValueError("tempo_splice requires at_s — an alignment estimate, not a placement heuristic")

    fade_len = int(fade_ms / 1000 * sr)
    cut = max(0, min(int(at_s * sr), len(speech)))
    speech_before, speech_after = speech[:cut], speech[cut:]

    ins = level_insert(insert, speech, sr, gain_db, level_ref=level_ref, at_s=at_s,
                       context_s=context_s, offset_db=level_offset_db)

    speech_before = exponential_fade_out(speech_before, fade_len, fade_k)
    ins = exponential_fade_in(ins, fade_len, fade_k)
    ins = exponential_fade_out(ins, fade_len, fade_k)
    speech_after = exponential_fade_in(speech_after, fade_len, fade_k)

    base = speech_pause_s(speech, sr) * gap_scale
    have_pre, have_post = existing_silence_s(speech, sr, at_s)
    gap_pre_n = max(0, int((base * pre_scale - have_pre) * sr))
    gap_post_n = max(0, int((base * post_scale - have_post) * sr))

    floor = quietest_window(speech, sr)
    gap1 = matched_room_tone(gap_pre_n, floor, amplitude_ratio=1.0, rng=rng)
    gap2 = matched_room_tone(gap_post_n, floor, amplitude_ratio=1.0, rng=rng)

    out = np.concatenate([speech_before, gap1, ins, gap2, speech_after])
    return out, cut / sr

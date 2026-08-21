"""Voice-activity detection, used to keep a splice out of the middle of an utterance.

Forced alignment answers "which two words does the `[tag]` sit between", and it answers it
well — but a word boundary is not the same thing as a *pause*. In connected speech the
boundary between two words is usually a point of continuous phonation, so the aligned time
lands mid-voice routinely: measured worst-junction activity on the first real 20-row batch
spanned 0.04 to 1.76 with a **median of 0.85** (see `quality.max_boundary_activity` in
configs/default.yaml), i.e. the typical cut fell on speech nearly as loud as the surrounding
segment average.

`quality.max_boundary_activity` catches that after the fact and rejects the row. This module
prevents it instead: take the aligned word boundary and move it to the nearest real pause,
so the event is inserted into silence between utterances rather than slicing a vowel in
half. Rows are only rejected when there is no pause to move to.

Two backends (`vad.backend`):
  - `silero` — Silero VAD, a real trained speech/non-speech classifier. Robust to
    background noise, breath, and room tone, which is exactly where an energy threshold
    misfires: a breath or a noisy floor reads as "not silent" to RMS but as "not speech" to
    a VAD, and the gap between utterances is precisely where breaths live. The pip package
    bundles its own weights, so this stays offline.
  - `energy` — thresholded frame RMS via `audio_utils.split_nonsilent`, the same detector
    the rest of the pipeline already uses. No extra dependency; the fallback when Silero
    isn't installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from para_synth.audio_utils import _resample, split_nonsilent

SILERO_SR = 16000  # Silero VAD accepts 8k or 16k only


@dataclass
class SnapResult:
    """Outcome of trying to move an aligned time onto a pause.

    `status` is one of:
      - `in_pause`  — the aligned time already fell inside a qualifying pause; kept (moved
                      to the pause's midpoint, which is the furthest point from either
                      speech edge).
      - `snapped`   — moved onto a nearby qualifying pause.
      - `no_pause`  — no qualifying pause within `max_shift_s`; `time_s` is unchanged and
                      the caller decides whether to keep or drop the row.
      - `disabled`  — VAD snapping is off.
    """

    time_s: float
    status: str
    shift_s: float = 0.0
    pause_s: float = 0.0


def _merge_close(intervals: list[tuple[float, float]], gap_s: float) -> list[tuple[float, float]]:
    """Join speech regions separated by less than `gap_s`. Silero emits a boundary at every
    brief stop-closure or glottal break; without merging, those register as "pauses" and are
    exactly the mid-utterance points this module exists to avoid."""
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] < gap_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def speech_intervals_energy(wav, sr, top_db: float = 30.0) -> list[tuple[float, float]]:
    iv = split_nonsilent(wav, top_db=top_db)
    return [(int(s) / sr, int(e) / sr) for s, e in iv]


def speech_intervals_silero(wav, sr, threshold: float = 0.5, min_speech_s: float = 0.05):
    import torch
    from silero_vad import get_speech_timestamps

    model = _load_silero()
    x = _resample(wav, sr, SILERO_SR) if sr != SILERO_SR else wav
    ts = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(x)).float(),
        model,
        sampling_rate=SILERO_SR,
        threshold=threshold,
        min_speech_duration_ms=int(min_speech_s * 1000),
        return_seconds=True,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


_SILERO_MODEL = None


def _load_silero():
    """Cached across rows — reloading the model per row dominates the cost of using it."""
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        from silero_vad import load_silero_vad

        print("🧠 Loading Silero VAD …")
        _SILERO_MODEL = load_silero_vad()
    return _SILERO_MODEL


def speech_intervals(wav, sr, backend: str = "silero", merge_gap_s: float = 0.10):
    """Speech regions as (start_s, end_s). Falls back to the energy detector if Silero is
    selected but not installed, rather than failing the run — the pipeline still works with
    the weaker detector, and the warning says what was lost."""
    if backend == "silero":
        try:
            iv = speech_intervals_silero(wav, sr)
        except ImportError:
            print("⚠️  silero-vad not installed — falling back to energy VAD "
                  "(pip install -e '.[vad]' for the real thing)")
            iv = speech_intervals_energy(wav, sr)
    elif backend == "energy":
        iv = speech_intervals_energy(wav, sr)
    else:
        raise ValueError(f"vad.backend must be 'silero' or 'energy', got {backend!r}")
    return _merge_close(iv, merge_gap_s)


def pause_intervals(wav, sr, backend: str = "silero", merge_gap_s: float = 0.10,
                    min_pause_s: float = 0.12, edge_margin_s: float = 0.05):
    """Gaps *between* speech regions, at least `min_pause_s` long.

    Leading and trailing silence are excluded: splicing there would put the event outside
    the utterance entirely, which is a different edit from the one the transcript's tag
    position describes. `edge_margin_s` trims each pause so the returned span excludes the
    immediate onset/offset of the neighbouring words, where energy is already rising.
    """
    speech = speech_intervals(wav, sr, backend=backend, merge_gap_s=merge_gap_s)
    pauses = []
    for (_, end_prev), (start_next, _) in zip(speech, speech[1:]):
        lo, hi = end_prev + edge_margin_s, start_next - edge_margin_s
        if hi - lo >= min_pause_s:
            pauses.append((lo, hi))
    return pauses


def snap_to_pause(t: float, pauses: list[tuple[float, float]], max_shift_s: float = 0.30) -> SnapResult:
    """Move an aligned time onto the nearest pause midpoint, within `max_shift_s`.

    `max_shift_s` is a semantic bound, not an acoustic one: the aligner placed the tag
    between two specific words, and moving the event far from there inserts it between
    *different* words than the transcript says. Keeping the shift small preserves the
    transcript's meaning; that is why the answer to "no pause nearby" is to reject the row
    rather than to search further afield.
    """
    if not pauses:
        return SnapResult(time_s=t, status="no_pause")

    for lo, hi in pauses:
        if lo <= t <= hi:
            mid = 0.5 * (lo + hi)
            return SnapResult(time_s=mid, status="in_pause", shift_s=mid - t, pause_s=hi - lo)

    lo, hi = min(pauses, key=lambda p: abs(0.5 * (p[0] + p[1]) - t))
    mid = 0.5 * (lo + hi)
    if abs(mid - t) > max_shift_s:
        return SnapResult(time_s=t, status="no_pause")
    return SnapResult(time_s=mid, status="snapped", shift_s=mid - t, pause_s=hi - lo)


def snap_insert_time(wav, sr, t: float, cfg) -> SnapResult:
    """Apply the configured VAD policy to one aligned insertion time. `cfg` is a VADConfig."""
    if not cfg.enabled:
        return SnapResult(time_s=t, status="disabled")
    pauses = pause_intervals(
        wav, sr,
        backend=cfg.backend,
        merge_gap_s=cfg.merge_gap_s,
        min_pause_s=cfg.min_pause_s,
        edge_margin_s=cfg.edge_margin_s,
    )
    return snap_to_pause(t, pauses, max_shift_s=cfg.max_shift_s)

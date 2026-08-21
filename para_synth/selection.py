"""Choosing *which* VocalSound clip a row gets, on the axes Seed-VC does not fix.

Seed-VC is a voice-conversion model: it takes the clip's content/timing and re-renders it
with a style vector extracted from the target speaker (CAM++ — see quality.py). So speaker
identity — timbre, formant structure, vocal-tract size, the things that make a laugh sound
like *that person's* laugh — is already handled downstream, and selecting for it here would
be wasted effort at best and double-counting at worst.

What conversion does *not* touch is everything that rides along in the source clip's own
content and channel. Measured on this repo's own 20-row run (scripts/measure_selection_axes.py
re-derives all of these):

  * **Intensity** — a boisterous belly laugh converted into a quiet speaker's timbre is
    still a boisterous laugh. Seed-VC changes who is laughing, never how hard. Crest factor
    passes through with median |log2(conv/src)| = 0.06 octave.
  * **Tempo** — with `seedvc.length_adjust: 1.0` the converted event keeps the source's
    timing: envelope rate passes through with median |log2(conv/src)| = 0.147 octave, so a
    laugh's burst rate is whatever the original contributor's was.
  * **Clarity / channel** — VocalSound is crowdsourced from whatever microphone each
    contributor had. Hiss and room noise are part of the signal Seed-VC reconstructs, so a
    noisy clip spliced into a clean recording reads as pasted-in no matter how good the
    conversion or the fades are. Widest measured spread of the three (clip SNR p10..p90 =
    33..60 dB against 42-55 dB for the speech recordings).

**Pitch is not on that list, and that is a measurement, not an oversight.** F0 is the most
obvious thing to want to match, but Seed-VC v1 regenerates it wholesale from the target
speaker: regressing the converted event's median F0 on both inputs gives
`log2(conv) = +0.04·log2(src) + 0.84·log2(speaker) + 1.17` — the source coefficient is
indistinguishable from zero, and dropping the source term entirely barely moves the residual
(0.229 -> 0.239 octave). Contour is regenerated too: the source/converted F0 shape
correlation is +0.14, i.e. none. Selecting clips by pitch would therefore spend weight on a
property the conversion is already responsible for, which is the same double-counting
mistake as selecting for timbre. This holds for `seedvc.f0_condition: false` (the default);
flipping it on makes Seed-VC track the source's pitch with RMVPE, at which point F0 *would*
become residual and this conclusion needs re-measuring.

Hence: match on those three, ignore identity and pitch. Each axis compares a clip measurement
against the same measurement taken on the *target recording*, and the axis distances are
summed with per-axis weights into a single sampling weight `exp(-distance)` — a bias, not
a filter, exactly as the energy-only version was (see "Clip selection" in docs/PIPELINE.md
for why every usable candidate must keep nonzero probability).

Two different comparison scales are used, on purpose:

  * Intensity is compared as **percentile rank** — crest factor and duration have no
    meaningful absolute scale shared between a laugh and a speech recording, so each side
    is ranked within its own reference population (clip: the candidate pool; speaker: all
    same-length windows of their own utterance).
  * Clarity and tempo are compared as **absolute dimensionless quantities** — dB SNR and
    envelope rate in Hz. These are already ratios/rates on perceptually meaningful scales
    and are invariant to recording gain, so a clip and a speech recording can be compared
    directly. Percentile-ranking them would actually be worse here: it needs a reference
    population, and the only one available for the speaker side is the other rows of the
    batch — which manufactures spread when every recording in a batch is equally clean.

Effective **bandwidth** was measured as a third clarity term and dropped, because neither
formulation carried channel information on this data. A 95%-energy spectral rolloff mostly
reports content (clip median 3953 Hz vs. speech median ~1500 Hz — voiced speech is
low-frequency-dominated, vocalisations are breathier, and that gap swamps any channel
difference), while a noise-floor-referenced "highest band above the floor" measure
saturates: clip p10..p90 = 7656..8000 Hz against 7127..7989 Hz for speech, i.e. both
corpora are simply full-band. See scripts/measure_selection_axes.py to re-check if either
corpus is replaced.

Tempo matching is restricted to `TEMPO_MATCHED_CLASSES`. Cough, sneeze, throat-clearing
and sniff are reflexive events whose duration and rate are set by airway mechanics rather
than by how fast the person was speaking, so there is nothing to match them against; see
"Deliberately not implemented" in docs/PIPELINE.md, which rejects *time-stretching* those
clips for the same reason. Selecting a clip whose native rate already fits is a different
operation from stretching one that doesn't, and does not touch the waveform.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from para_synth.audio_utils import load_mono, trim_event
from para_synth.vocalsound import list_clips

# Laughter is the only class whose rate is under voluntary motor control the way speaking
# rate is, so it's the only one where "match the speaker's tempo" means anything. A sigh is
# a single breath (no rate to speak of); the rest are reflexes.
TEMPO_MATCHED_CLASSES = {"laughter"}

# Tolerances that turn a physical difference into a ~1.0 distance unit, so the per-axis
# `*_weight` config values are comparable to each other and to the intensity axis (whose
# distance is a percentile difference, already 0..1).
# Measured spreads these are calibrated against (200-clip VocalSound sample + the 20-row
# sample_clean recordings; re-run scripts/measure_selection_axes.py if either corpus
# changes): clip SNR p10/med/p90 = 33/54/60 dB against speech 42-55 dB, so 12 dB is roughly
# a third of the usable clip spread. Clip envelope rate p10/med/p90 = 1.1/2.9/5.4 Hz
# (laughter alone 1.5/3.8/4.8) against speech 2.8-6.0 Hz, so an octave is again about a
# third of the spread. Both axes therefore produce distances that span ~0-2 across a
# candidate pool, i.e. weights that actually discriminate at a config weight near 1.
SNR_TOLERANCE_DB = 12.0
TEMPO_TOLERANCE_OCT = 1.0  # a 2x rate difference, e.g. a 3Hz chuckle vs a 6Hz giggle


def _rms(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def _percentile_ranks(values: list[float]) -> list[float]:
    """Map values to their rank within the list, scaled to 0..1 (ties share a rank)."""
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for position, i in enumerate(order):
        ranks[i] = position / (n - 1)
    return ranks


def _frame_rms(x, sr, win_s: float = 0.020, hop_s: float = 0.010) -> np.ndarray:
    """Short-time RMS envelope — the shared basis for the SNR and rate measures below."""
    x = np.asarray(x, dtype=np.float64)
    win = max(int(win_s * sr), 1)
    hop = max(int(hop_s * sr), 1)
    if len(x) < win:
        return np.array([_rms(x)])
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(x[idx] ** 2, axis=1))


# ── The three residual axes ──────────────────────────────────────────────────────────
# Every measure here is invariant to recording gain (they are all ratios or rates), which
# is what makes a clip and a speech recording comparable at all — absolute levels across
# these two would only tell you about VocalSound's crowdsourced mic gains.


def clip_features(trimmed, sr) -> tuple[float, float]:
    """Scale-invariant "how big is this vocalisation": (crest factor, duration).

    Deliberately avoids absolute loudness: `_finalize_clip` peak-normalises every clip and
    the splice re-levels it against the speaker anyway, so a clip's raw amplitude tells you
    about VocalSound's crowdsourced recording gain, not about how big the laugh was. What
    *does* survive normalisation is shape — a big, sustained belly laugh has both a higher
    crest factor (RMS close to its own peak, i.e. energy sustained rather than one
    transient spike) and a longer duration than a short polite chuckle.

    Returned raw and unscaled on purpose: these become percentile ranks within the actual
    candidate pool (`_percentile_ranks`) rather than being squashed through fixed
    constants. An earlier version hardcoded reference values here and turned out badly
    miscalibrated against real VocalSound data — measured crest factors cluster around 0.11
    (not the assumed ~0.5) and the median clip runs ~2.7s (saturating an assumed 2s
    ceiling), so nearly every clip scored ~0.58 and the weighting had nothing to
    discriminate on. Percentile ranks are self-calibrating: whatever the real distribution
    turns out to be, the ranks span 0..1 by construction.
    """
    peak = float(np.max(np.abs(trimmed))) + 1e-9
    return _rms(trimmed) / peak, len(trimmed) / sr


def snr_db(x, sr) -> float:
    """Crude but gain-invariant SNR: the loud percentile of the frame envelope over the
    quiet one, in dB.

    Percentiles rather than a voiced/unvoiced decision so this needs no VAD and can't be
    thrown off by a single click. Measure this on the *untrimmed* clip — `trim_event`
    removes exactly the leading/trailing silence the noise-floor estimate comes from.
    """
    env = _frame_rms(x, sr)
    if len(env) < 4:
        return 60.0
    floor = float(np.percentile(env, 10))
    active = float(np.percentile(env, 90))
    if active <= 0:
        return 0.0
    return float(np.clip(20 * np.log10(active / max(floor, 1e-9)), 0.0, 60.0))


def clipping_fraction(x) -> float:
    """Fraction of samples at or beyond digital full scale — the one axis that is bad in
    only one direction, so it gets a hard gate rather than a match. A clipped source is
    distortion Seed-VC will faithfully reconstruct; there is no target recording it
    "matches"."""
    x = np.asarray(x)
    if len(x) == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= 0.999))


def envelope_rate_hz(x, sr, min_separation_s: float = 0.06) -> float:
    """Rate of amplitude events per second — laugh bursts on a clip, syllables on speech.

    The *same* estimator has to run on both sides for the comparison to mean anything, so
    this stays a generic "how often does the envelope peak" measure rather than anything
    laughter- or speech-specific. Peak counting rather than autocorrelation because clips
    are short (~2.7s median) and often carry only 5-15 bursts, which is too few periods for
    a reliable autocorrelation peak.
    """
    env = _frame_rms(x, sr, win_s=0.030, hop_s=0.010)
    duration = len(x) / sr
    if len(env) < 3 or duration <= 0:
        return 0.0
    env = env / (env.max() + 1e-9)
    threshold = 0.25
    min_gap = max(int(min_separation_s / 0.010), 1)

    peaks = 0
    last = -min_gap
    for i in range(1, len(env) - 1):
        if env[i] >= threshold and env[i] >= env[i - 1] and env[i] > env[i + 1] and i - last >= min_gap:
            peaks += 1
            last = i
    return peaks / duration


def speaker_energy_score(speech, sr, at_s: float, context_s: float) -> float:
    """How loud is this speaker, in the moments before the splice point, *relative to
    their own range across this recording*? 0..1, where ~0.9 means "one of the louder
    moments for this speaker" and ~0.1 "one of their quieter moments".

    Self-relative on purpose: absolute RMS across recordings mostly reflects mic and
    recording gain, so comparing one speaker's raw level against another's — or against a
    VocalSound clip's — would be meaningless.

    Implemented as a percentile rank of the pre-splice window's RMS among all same-length
    windows in the utterance, rather than a plain ratio against the utterance mean: a
    ratio's spread depends on how dynamic the recording happens to be, and empirically
    clustered near the middle for most speech, which left the sampling weight with almost
    no signal. A percentile rank spans 0..1 by construction.
    """
    win = max(int(context_s * sr), 1)
    end = int(at_s * sr)
    start = max(0, end - win)
    context = speech[start:end]
    if len(context) == 0 or len(speech) < win:
        return 0.5

    hop = max(win // 4, 1)
    window_rms = [_rms(speech[i : i + win]) for i in range(0, len(speech) - win + 1, hop)]
    if not window_rms:
        return 0.5
    ctx_rms = _rms(context)
    # Fraction of windows quieter than the context — the context window isn't necessarily
    # one of the sampled windows (it isn't aligned to the hop grid), so normalise by the
    # window count, not count-1, to keep this in [0, 1].
    below = sum(1 for r in window_rms if r < ctx_rms)
    return below / len(window_rms)


# ── Target-side profile ──────────────────────────────────────────────────────────────


@dataclass
class SpeakerProfile:
    """The target recording measured on the same axes the candidate clips are.

    `energy_rank` is a percentile within this recording; `snr` and `rate` are absolute
    gain-invariant quantities directly comparable to a clip's. Identity/timbre is
    deliberately absent — that is Seed-VC's job, via the CAM++ style vector.
    """

    energy_rank: float
    snr: float
    rate: float


def profile_speaker(speech, sr, at_s: float, context_s: float) -> SpeakerProfile:
    """Measure the target recording where the event will land.

    Energy and rate come from the `context_s` window immediately before the splice point —
    a speaker's loudness and pace vary across a recording, and what an interrupting event
    has to sit next to is the moment it interrupts. Channel quality (SNR) is
    measured over the *whole* recording instead: it's a property of the microphone and the
    room, constant for the take, and a short window has too little silence in it to
    estimate a noise floor from.
    """
    end = int(at_s * sr)
    start = max(0, end - max(int(context_s * sr), 1))
    context = speech[start:end] if end > start else speech
    return SpeakerProfile(
        energy_rank=speaker_energy_score(speech, sr, at_s, context_s),
        snr=snr_db(speech, sr),
        rate=envelope_rate_hz(context, sr),
    )


# ── Candidate scoring ────────────────────────────────────────────────────────────────


@dataclass
class _Candidate:
    name: str
    trimmed: np.ndarray
    sr: int
    crest: float
    duration: float
    snr: float
    rate: float


@dataclass
class ClipPick:
    """The chosen clip plus the audit trail for why it was chosen."""

    name: str
    audio: np.ndarray
    sr: int
    intensity: float
    # {axis: {"target": …, "clip": …, "distance": …}} — lets a finished run be checked for
    # whether each axis actually tracked the target, the way the energy-only version's
    # speaker_score/clip_intensity pair did.
    axes: dict = field(default_factory=dict)


def _octave_distance(a: float, b: float, tolerance_oct: float) -> float:
    """|log2(a/b)| in tolerance units. Undefined-at-zero cases (a silent clip, a rate
    estimator that found no peaks) return a full unit of distance rather than infinity:
    unmeasurable is a reason to deprioritise a candidate, not to forbid it."""
    if a <= 0 or b <= 0:
        return 1.0
    return abs(np.log2(a / b)) / tolerance_oct


def _axis_distances(cand: _Candidate, profile: SpeakerProfile, intensity: float, vs_class: str):
    """Per-axis distances in comparable units. Intensity is a percentile difference (0..1);
    the others are physical differences scaled by their tolerance constant."""
    # Symmetric: a clip noticeably cleaner than the recording is a mismatch too — the
    # background drops out for the duration of the event, which reads as a punched-in edit
    # the same way excess hiss does (less badly, but the splice's matched_room_tone only
    # fills the *gap*, not the event itself).
    clarity = abs(cand.snr - profile.snr) / SNR_TOLERANCE_DB
    axes = {
        "intensity": {"target": profile.energy_rank, "clip": intensity,
                      "distance": abs(intensity - profile.energy_rank)},
        "clarity": {"target_snr_db": profile.snr, "clip_snr_db": cand.snr,
                    "distance": clarity},
    }
    if vs_class in TEMPO_MATCHED_CLASSES:
        axes["tempo"] = {
            "target_hz": profile.rate, "clip_hz": cand.rate,
            "distance": _octave_distance(cand.rate, profile.rate, TEMPO_TOLERANCE_OCT),
        }
    return axes


def _usable_clip(trimmed, sr) -> bool:
    return np.max(np.abs(trimmed)) > 0.05 and len(trimmed) / sr > 0.35


def _finalize_clip(trimmed, sr):
    """Normalise + pad to >=1s for Seed-VC's chunker."""
    out = trimmed / (np.max(np.abs(trimmed)) + 1e-9) * 0.95
    if len(out) / sr < 1.0:
        pad = int((1.0 - len(out) / sr) * sr / 2)
        out = np.pad(out, (pad, pad))
    return out


def pick_vocalsound_clip(
    vs_dir: Path,
    vs_class: str,
    rng: random.Random,
    cfg,  # SelectionConfig
    profile: SpeakerProfile | None = None,
    max_tries: int = 12,
) -> ClipPick:
    """Choose a VocalSound clip of `vs_class`, energy-trim it, normalise, and pad.

    With every axis weight at 0 (or no `profile`), this is uniform random over the class —
    every clip equally likely, which is what the pipeline did before any matching existed.

    Otherwise it loads `cfg.candidate_pool` random clips of the class, measures each on the
    axes Seed-VC won't fix (see the module docstring), and samples with weight
    `exp(-Σ weight_axis · distance_axis)`. Closer matches are *more likely* but never
    exclusive: every usable candidate keeps nonzero probability, so a big laugh from a
    soft-spoken speaker or a slightly noisier clip than the recording still happens — real
    recordings are not internally consistent either, and hard filtering would delete
    exactly the acoustic variance a downstream Para-TTS model benefits from.

    The one hard gate is `cfg.max_clipping`: a clipped source is distortion, not a
    mismatch, and Seed-VC reconstructs it faithfully. The gate is dropped rather than
    allowed to empty the pool, so it can never fail a row on its own.
    """
    pool = list_clips(vs_dir, vs_class)
    if not pool:
        raise RuntimeError(f"No '{vs_class}' clips found in {vs_dir}")

    weights_off = profile is None or max(
        cfg.energy_weight, cfg.clarity_weight, cfg.tempo_weight
    ) <= 0
    if weights_off:
        for _ in range(max_tries):
            pick = rng.choice(pool)
            raw, raw_sr = load_mono(vs_dir / pick)
            cand = trim_event(raw, raw_sr)
            if _usable_clip(cand, raw_sr):
                # 0.5 = unranked; there's no candidate set to rank this against.
                return ClipPick(pick, _finalize_clip(cand, raw_sr), raw_sr, 0.5)
        raise RuntimeError(f"{max_tries} picks of '{vs_class}' were all too quiet/short")

    names = rng.sample(pool, min(cfg.candidate_pool, len(pool)))
    candidates: list[_Candidate] = []
    clipped: list[_Candidate] = []
    for name in names:
        raw, raw_sr = load_mono(vs_dir / name)
        trimmed = trim_event(raw, raw_sr)
        if not _usable_clip(trimmed, raw_sr):
            continue
        crest, duration = clip_features(trimmed, raw_sr)
        cand = _Candidate(
            name=name,
            trimmed=trimmed,
            sr=raw_sr,
            crest=crest,
            duration=duration,
            # Channel measures come from the untrimmed clip — trim_event removes the
            # leading/trailing silence the noise floor is estimated from.
            snr=snr_db(raw, raw_sr),
            rate=envelope_rate_hz(trimmed, raw_sr),
        )
        (clipped if clipping_fraction(raw) > cfg.max_clipping else candidates).append(cand)

    if not candidates:
        # Every survivor was clipped: prefer a distorted clip over failing the row.
        candidates = clipped
    if not candidates:
        raise RuntimeError(f"none of {len(names)} sampled '{vs_class}' clips were usable")

    crest_ranks = _percentile_ranks([c.crest for c in candidates])
    duration_ranks = _percentile_ranks([c.duration for c in candidates])
    intensities = [(c + d) / 2 for c, d in zip(crest_ranks, duration_ranks)]

    axis_weights = {
        "intensity": cfg.energy_weight,
        "clarity": cfg.clarity_weight,
        "tempo": cfg.tempo_weight,
    }
    per_candidate = [
        _axis_distances(cand, profile, intensity, vs_class)
        for cand, intensity in zip(candidates, intensities)
    ]
    sampling_weights = [
        float(np.exp(-sum(axis_weights[a] * d["distance"] for a, d in axes.items())))
        for axes in per_candidate
    ]

    idx = rng.choices(range(len(candidates)), weights=sampling_weights, k=1)[0]
    chosen = candidates[idx]
    return ClipPick(
        name=chosen.name,
        audio=_finalize_clip(chosen.trimmed, chosen.sr),
        sr=chosen.sr,
        intensity=intensities[idx],
        axes=per_candidate[idx],
    )

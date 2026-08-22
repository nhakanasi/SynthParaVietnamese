#!/usr/bin/env python
"""Re-measure the evidence behind para_synth/selection.py's axis choices.

Two questions, both of which have to be answered with numbers rather than assumed — an
earlier version of the selector was miscalibrated for exactly that reason (see
`clip_features`' docstring):

  1. **Does the axis have spread?** A feature whose values all cluster gives the sampling
     weight nothing to discriminate on. Prints p10/median/p90 for every candidate axis over
     a VocalSound sample and over the speech recordings.

  2. **Is the axis residual?** Selection should only spend weight on properties Seed-VC
     does *not* regenerate — anything it re-synthesizes from the target speaker is already
     handled, and matching on it would double-count. Needs a finished run: compares each
     source clip against its own converted output to see what survived conversion.

Run after replacing either corpus, changing `seedvc.f0_condition`, or adding an axis:

    python scripts/measure_selection_axes.py [--run data/output/default_run]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from para_synth.audio_utils import load_mono, trim_event  # noqa: E402
from para_synth.config import load_config  # noqa: E402
from para_synth.selection import (  # noqa: E402
    clip_features,
    envelope_rate_hz,
    envelope_tilt_db,
    snr_db,
)


def _f0_track(x, sr, fmin=60.0, fmax=500.0, win_s=0.040, hop_s=0.010, voicing=0.3):
    """Autocorrelation pitch track. Lives here rather than in selection.py because it is
    only used to *prove* F0 is not a usable selection axis — see the module docstring of
    selection.py and "Deliberately not implemented" in docs/PIPELINE.md."""
    x = np.asarray(x, dtype=np.float64)
    win, hop = int(win_s * sr), int(hop_s * sr)
    if len(x) < win:
        return np.zeros(0), np.zeros(0)
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    spec = np.fft.rfft(frames, nfft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, nfft, axis=1)[:, :win]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = max(int(sr / fmax), 1), min(int(sr / fmin), win - 1)
    seg = ac[:, lo : hi + 1]
    best = np.argmax(seg, axis=1)
    strength = seg[np.arange(len(seg)), best]
    keep = strength > voicing
    return (np.arange(len(best)) * hop_s)[keep], (sr / (best + lo))[keep]


def _median_f0(x, sr) -> float:
    _, f = _f0_track(x, sr)
    return float(np.median(f)) if len(f) else 0.0


def _quantiles(label: str, values: list[float], fmt: str = "8.1f") -> None:
    usable = [v for v in values if v is not None]
    if not usable:
        print(f"  {label:22s} (no data)")
        return
    p10, p50, p90 = np.percentile(usable, [10, 50, 90])
    print(f"  {label:22s} p10={p10:{fmt}} med={p50:{fmt}} p90={p90:{fmt}}  n={len(usable)}")


def spread(cfg, n_clips: int, seed: int) -> None:
    print("── axis spread ─────────────────────────────────────────────────────────────")
    vs_dir = cfg.paths.vocalsound_dir
    names = sorted(p.name for p in vs_dir.iterdir() if p.suffix == ".wav")
    print(f"VocalSound clips available: {len(names)}")
    rng = random.Random(seed)
    per_axis: dict[str, list[float]] = {"snr_db": [], "rate_hz": [], "crest": [],
                                        "duration_s": [], "tilt_db": [], "f0_hz": []}
    for name in rng.sample(names, min(n_clips, len(names))):
        raw, sr = load_mono(vs_dir / name)
        trimmed = trim_event(raw, sr)
        if np.max(np.abs(trimmed)) <= 0.05 or len(trimmed) / sr <= 0.35:
            continue
        crest, duration = clip_features(trimmed, sr)
        per_axis["snr_db"].append(snr_db(raw, sr))  # untrimmed: trim removes the noise floor
        per_axis["rate_hz"].append(envelope_rate_hz(trimmed, sr))
        # tilt_db is the one axis whose tolerance (selection.TILT_TOLERANCE_DB) is still a
        # prior rather than a measurement — set it from this p10/p90 before raising
        # selection.tilt_weight above 0.
        per_axis["tilt_db"].append(envelope_tilt_db(trimmed, sr))
        per_axis["crest"].append(crest)
        per_axis["duration_s"].append(duration)
        per_axis["f0_hz"].append(_median_f0(trimmed, sr))
    print("clips:")
    for axis, values in per_axis.items():
        _quantiles(axis, values,
                   "8.2f" if axis in ("crest", "rate_hz", "duration_s") else "8.1f")

    speech_files = sorted(cfg.paths.raw_audio_dir.glob("*.wav"))
    if not speech_files:
        print("speech: (no recordings in paths.raw_audio_dir)")
        return
    speech: dict[str, list[float]] = {"snr_db": [], "rate_hz": [], "tilt_db": [], "f0_hz": []}
    for path in speech_files:
        wav, sr = load_mono(path, cfg.sample_rate)
        speech["snr_db"].append(snr_db(wav, sr))
        speech["rate_hz"].append(envelope_rate_hz(wav, sr))
        speech["tilt_db"].append(envelope_tilt_db(wav, sr))
        speech["f0_hz"].append(_median_f0(wav, sr))
    print("speech:")
    for axis, values in speech.items():
        _quantiles(axis, values, "8.2f" if axis == "rate_hz" else "8.1f")


def residual(run_dir: Path, sample_rate: int) -> None:
    """What survives Seed-VC? Anything that doesn't is not worth selecting on."""
    print("\n── residual vs. regenerated (needs a finished run) ──────────────────────────")
    meta_path = run_dir / "metadata_synth.jsonl"
    if not meta_path.is_file():
        print(f"  no {meta_path} — run the pipeline first")
        return
    rows = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()]

    src_f0, conv_f0, spk_f0 = [], [], []
    rate_ratio, energy_ratio, contour = [], [], []
    for row in rows:
        try:
            src, src_sr = load_mono(row["vs_source_audio"])
            conv, conv_sr = load_mono(row["converted_audio"])
            speech, speech_sr = load_mono(row["source_audio"], sample_rate)
        except Exception as exc:  # a work-dir that's been cleaned since the run
            print(f"  skipped {row['id']}: {exc}")
            continue
        src, conv = trim_event(src, src_sr), trim_event(conv, conv_sr)

        a, b, c = _median_f0(src, src_sr), _median_f0(conv, conv_sr), _median_f0(speech, speech_sr)
        if min(a, b, c) > 0:
            src_f0.append(np.log2(a))
            conv_f0.append(np.log2(b))
            spk_f0.append(np.log2(c))

        rs, rc = envelope_rate_hz(src, src_sr), envelope_rate_hz(conv, conv_sr)
        if rs > 0 and rc > 0:
            rate_ratio.append(abs(np.log2(rc / rs)))
        cs, _ = clip_features(src, src_sr)
        cc, _ = clip_features(conv, conv_sr)
        if cs > 0 and cc > 0:
            energy_ratio.append(abs(np.log2(cc / cs)))

        ts, fs = _f0_track(src, src_sr)
        tc, fc = _f0_track(conv, conv_sr)
        if len(fs) >= 5 and len(fc) >= 5:
            n = min(len(fs), len(fc), 40)
            gs = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(fs)), np.log2(fs))
            gc = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(fc)), np.log2(fc))
            gs, gc = gs - gs.mean(), gc - gc.mean()
            denom = gs.std() * gc.std()
            if denom > 0:
                contour.append(float(np.mean(gs * gc) / denom))

    if src_f0:
        # log2(conv) ~ a·log2(src) + b·log2(speaker) + c. A source coefficient near zero
        # means Seed-VC re-pitches from the target and the source's F0 is discarded.
        design = np.column_stack([src_f0, spk_f0, np.ones(len(src_f0))])
        coef, *_ = np.linalg.lstsq(design, np.array(conv_f0), rcond=None)
        print(f"  F0 model (n={len(src_f0)}): log2(conv) = {coef[0]:+.2f}·log2(src) "
              f"{coef[1]:+.2f}·log2(speaker) {coef[2]:+.2f}")
        print(f"    residual std {np.std(np.array(conv_f0) - design @ coef):.3f} oct;"
              f" ignoring src entirely {np.std(np.array(conv_f0) - np.array(spk_f0)):.3f} oct")
    if contour:
        print(f"  F0 contour shape corr(src, conv): median {np.median(contour):+.2f} (n={len(contour)})")
    if rate_ratio:
        print(f"  envelope rate |log2(conv/src)|:   median {np.median(rate_ratio):.3f} oct (n={len(rate_ratio)})")
    if energy_ratio:
        print(f"  crest factor |log2(conv/src)|:    median {np.median(energy_ratio):.3f} oct (n={len(energy_ratio)})")
    print("\n  Read as: near-zero source coefficient / near-zero correlation = Seed-VC")
    print("  regenerates that property, so selecting on it double-counts. A small ratio =")
    print("  the property passes through unchanged, so selection is the only place to fix it.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run", default=None, help="output dir of a finished run")
    parser.add_argument("--clips", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    spread(cfg, args.clips, args.seed)
    residual(Path(args.run) if args.run else cfg.paths.output_dir, cfg.sample_rate)


if __name__ == "__main__":
    main()

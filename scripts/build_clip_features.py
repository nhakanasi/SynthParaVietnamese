#!/usr/bin/env python
"""Precompute per-clip acoustic features for the whole VocalSound corpus, once.

`pick_vocalsound_clip()` measures candidates by decoding them, which costs a file read per
candidate and therefore caps `selection.candidate_pool` at a few dozen. That means the
selector picks the best of ~48 out of a class of ~3,500, across three or four axes at the
same time — and best-of-48 on a joint match is a long way from best-available. It also makes
the intensity percentile ranks *draw-wide*: ranked inside 48 resampled clips, the same clip
scores differently on different rows, and in-pool rank 1.0 only means "loudest of those 48".

With a manifest, the candidate set is the whole class, ranks are corpus-wide, and exactly
one clip per row is ever decoded. Point `selection.feature_manifest` at the output:

    python scripts/build_clip_features.py
    # -> data/vocalsound/clip_features.json

WHAT IS STORED, AND WHY IT ISN'T A COMMITMENT: raw physical measurements only — crest,
duration, SNR in dB, envelope rate in Hz, envelope tilt in dB, clipping fraction. No ranks,
no distances, no scores. Ranks depend on the population and distances depend on the target
recording and the axis weights, so storing either would make re-weighting an axis, re-scaling
a tolerance, or rewriting `_axis_distances` require a full re-extraction. As stored, all of
those are free; only changing an extractor in selection.py costs a rebuild, and
`selection.FEATURE_VERSION` makes a stale manifest a warning-and-fallback rather than a
silently wrong run. This is the same cache-the-measurement / re-tune-the-threshold policy
the NISQA stage already uses.

Clips `_usable_clip()` would reject are omitted entirely, so they never occupy a pool slot.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from para_synth.audio_utils import load_mono  # noqa: E402
from para_synth.config import load_config  # noqa: E402
from para_synth.selection import FEATURE_VERSION, measure_clip  # noqa: E402
from para_synth.vocalsound import VS_CLASSES, VS_PAT  # noqa: E402


def build(vs_dir: Path, out_path: Path, only_class: str | None = None) -> None:
    names = sorted(
        f.name for f in vs_dir.iterdir()
        if (m := VS_PAT.match(f.name)) and (only_class is None or m.group(4) == only_class)
    )
    if not names:
        raise SystemExit(f"no VocalSound clips found in {vs_dir}")

    print(f"📐 measuring {len(names)} clips in {vs_dir}")
    clips: dict[str, dict] = {}
    skipped = 0
    started = time.time()
    for i, name in enumerate(names, 1):
        try:
            raw, sr = load_mono(vs_dir / name)
        except Exception as exc:  # a corrupt clip shouldn't lose the whole corpus
            print(f"   ⚠️  {name}: {exc}")
            skipped += 1
            continue
        m = measure_clip(raw, sr)
        if m is None:  # too quiet or too short for _usable_clip
            skipped += 1
            continue
        clips[name] = {k: round(float(v), 6) for k, v in m.items()}
        if i % 500 == 0 or i == len(names):
            rate = i / max(time.time() - started, 1e-9)
            print(f"   {i}/{len(names)}  ({rate:.0f} clips/s, {skipped} skipped)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "feature_version": FEATURE_VERSION,
        "source_dir": str(vs_dir),
        "n_clips": len(clips),
        "n_skipped": skipped,
        "clips": clips,
    }))
    size_mb = out_path.stat().st_size / 1e6
    print(f"✅ {len(clips)} clips ({skipped} skipped) -> {out_path} ({size_mb:.1f} MB) "
          f"in {time.time() - started:.0f}s")
    print(f"   set `selection.feature_manifest: {out_path}` in your config to use it")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="config yaml (default: configs/default.yaml)")
    ap.add_argument("--out", default=None,
                    help="output json (default: <vocalsound_dir>/../clip_features.json)")
    ap.add_argument("--class", dest="only_class", default=None, choices=VS_CLASSES,
                    help="measure only one class (default: all)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    vs_dir = cfg.paths.vocalsound_dir
    out = Path(args.out) if args.out else vs_dir.parent / "clip_features.json"
    build(vs_dir, out, args.only_class)


if __name__ == "__main__":
    main()

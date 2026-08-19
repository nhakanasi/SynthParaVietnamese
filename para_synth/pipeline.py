"""Orchestrates the full diagram: pick a VocalSound clip -> Seed-VC it into the speaker's
voice -> score the conversion -> align the transcript's [tag] in time -> splice -> write the
new "Para recording + tagged transcript" dataset row.

Ported/generalized from notebook45ee5319ae.ipynb cell 25's batch loop.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from para_synth.align import AlignmentPipeline
from para_synth.audio_utils import adaptive_splice, load_mono, splice, trim_event
from para_synth.config import Config
from para_synth.dataset import ManifestRow, extract_tag
from para_synth.quality import SpeakerSimilarity
from para_synth.seedvc import run_seedvc
from para_synth.vocalsound import list_clips, match_vs_class


@dataclass
class SynthesisResult:
    id: str
    source_audio: str
    para_audio: str
    vs_source_audio: str
    converted_audio: str
    vs_class: str
    nv_tag: str
    transcript: str
    splice_at_s: float
    insert_stage: str
    sim_converted: float
    sim_raw_baseline: float
    # Selection audit trail — lets a run be checked for whether energy weighting actually
    # tracked the speaker, and how much intensity spread survived in the dataset.
    vs_clip: str
    speaker_score: float
    clip_intensity: float


CANDIDATE_POOL_SIZE = 24  # clips actually loaded+scored per row when weighting is on


def _usable_clip(trimmed, sr) -> bool:
    return np.max(np.abs(trimmed)) > 0.05 and len(trimmed) / sr > 0.35


def _finalize_clip(trimmed, sr):
    """Normalise + pad to >=1s for Seed-VC's chunker."""
    out = trimmed / (np.max(np.abs(trimmed)) + 1e-9) * 0.95
    if len(out) / sr < 1.0:
        pad = int((1.0 - len(out) / sr) * sr / 2)
        out = np.pad(out, (pad, pad))
    return out


def _rms(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def clip_features(trimmed, sr) -> tuple[float, float]:
    """Raw scale-invariant "how big is this vocalisation" features: (crest factor, duration).

    Deliberately avoids absolute loudness: `_finalize_clip` peak-normalises every clip and
    `splice` re-levels it against the speaker anyway, so a clip's raw amplitude tells you
    about VocalSound's crowdsourced recording gain, not about how big the laugh was. What
    *does* survive normalisation is shape — a big, sustained belly laugh has both a higher
    crest factor (RMS close to its own peak, i.e. energy sustained rather than one
    transient spike) and a longer duration than a short polite chuckle.

    Returned raw and unscaled on purpose: these get converted to percentile ranks within
    the actual candidate pool (see `_percentile_ranks`) rather than squashed through fixed
    constants. An earlier version hardcoded reference values here and turned out to be
    badly miscalibrated against real VocalSound data — measured crest factors cluster
    around 0.11 (not the assumed ~0.5) and the median clip runs ~2.7s (saturating an
    assumed 2s ceiling), so nearly every clip scored ~0.58 and the weighting below had
    nothing to discriminate on. Percentile ranks are self-calibrating: whatever the real
    distribution turns out to be, the ranks span 0..1 by construction.
    """
    peak = float(np.max(np.abs(trimmed))) + 1e-9
    return _rms(trimmed) / peak, len(trimmed) / sr


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
    clustered near the middle for most speech, which left the sampling weight below with
    almost no signal. A percentile rank spans 0..1 by construction.
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


def pick_vocalsound_clip(
    vs_dir: Path,
    vs_class: str,
    rng: random.Random,
    max_tries: int = 12,
    speaker_score: float | None = None,
    energy_weight: float = 0.0,
):
    """Choose a VocalSound clip of `vs_class`, energy-trim it, normalise, and pad.

    With `energy_weight == 0` (or no `speaker_score`), this is uniform random over the
    class — every clip equally likely.

    With `energy_weight > 0`, it loads a random candidate subset, ranks each candidate's
    intensity *within that subset* (percentile rank of crest factor and duration, averaged
    — see `clip_features`), and samples with weight `exp(-energy_weight * |clip_rank -
    speaker_score|)` so clips whose intensity is closer to the speaker's own current level
    are *more likely* but never exclusive. Every usable candidate keeps nonzero
    probability: a big laugh from a soft-spoken speaker still happens (real people do
    that), just less often than uniform sampling would produce it. This preserves the
    acoustic variance a downstream Para-TTS model benefits from while making the typical
    example less jarring — it is explicitly not a hard "soft speakers only get soft laughs"
    filter, which would encode a stereotype that is often false.

    Returns (name, finalized_clip, sr, intensity_rank). For the uniform path the returned
    rank is 0.5 (unranked — there's no candidate set to rank against).
    """
    pool = list_clips(vs_dir, vs_class)
    if not pool:
        raise RuntimeError(f"No '{vs_class}' clips found in {vs_dir}")

    if energy_weight <= 0 or speaker_score is None:
        for _ in range(max_tries):
            pick = rng.choice(pool)
            raw, raw_sr = load_mono(vs_dir / pick)
            cand = trim_event(raw, raw_sr)
            if _usable_clip(cand, raw_sr):
                return pick, _finalize_clip(cand, raw_sr), raw_sr, 0.5
        raise RuntimeError(f"{max_tries} picks of '{vs_class}' were all too quiet/short")

    candidates = rng.sample(pool, min(CANDIDATE_POOL_SIZE, len(pool)))
    scored = []
    for name in candidates:
        raw, raw_sr = load_mono(vs_dir / name)
        cand = trim_event(raw, raw_sr)
        if _usable_clip(cand, raw_sr):
            scored.append((name, cand, raw_sr, clip_features(cand, raw_sr)))
    if not scored:
        raise RuntimeError(f"none of {len(candidates)} sampled '{vs_class}' clips were usable")

    crest_ranks = _percentile_ranks([f[0] for _, _, _, f in scored])
    duration_ranks = _percentile_ranks([f[1] for _, _, _, f in scored])
    intensities = [(c + d) / 2 for c, d in zip(crest_ranks, duration_ranks)]

    weights = [np.exp(-energy_weight * abs(i - speaker_score)) for i in intensities]
    idx = rng.choices(range(len(scored)), weights=weights, k=1)[0]
    name, cand, raw_sr, _ = scored[idx]
    return name, _finalize_clip(cand, raw_sr), raw_sr, intensities[idx]


def synthesize_row(
    row: ManifestRow,
    cfg: Config,
    speaker_sim: SpeakerSimilarity,
    insert_at_s: float,
    insert_stage: str,
) -> SynthesisResult:
    rng = cfg.rng()
    nv_tag = extract_tag(row.text)
    if nv_tag is None:
        raise ValueError(f"row {row.id!r} has no [tag] in its transcript — run tagging first")
    vs_class = match_vs_class(nv_tag, rng)

    speech_arr, speech_sr = load_mono(row.audio_filepath, cfg.sample_rate)
    speaker_score = speaker_energy_score(speech_arr, speech_sr, insert_at_s, cfg.selection.context_s)

    vs_name, vs_clip, vs_sr, vs_intensity = pick_vocalsound_clip(
        cfg.paths.vocalsound_dir, vs_class, rng,
        speaker_score=speaker_score, energy_weight=cfg.selection.energy_weight,
    )
    vs_src_path = cfg.paths.work_dir / f"vs_{row.id}.wav"
    sf.write(vs_src_path, vs_clip, vs_sr, subtype="PCM_16")

    converted = run_seedvc(vs_src_path, row.audio_filepath, cfg.paths.work_dir / "output_vc", cfg.seedvc)

    conv_arr, conv_sr = load_mono(converted, cfg.sample_rate)

    sim_converted = speaker_sim.similarity(speech_arr, speech_sr, conv_arr, conv_sr)
    sim_raw_baseline = speaker_sim.similarity(speech_arr, speech_sr, vs_clip, vs_sr)
    if sim_converted <= sim_raw_baseline:
        print(f"   ⚠️  {row.id}: conversion didn't clearly beat the raw-clip baseline "
              f"({sim_converted:.3f} <= {sim_raw_baseline:.3f})")

    if cfg.splice.adaptive:
        final, at = adaptive_splice(
            speech_arr, conv_arr, cfg.sample_rate, cfg.splice.para_gain_db,
            cfg.splice.min_pad_ms, cfg.splice.max_gap_ms, cfg.splice.fade_ms,
            at_s=insert_at_s,
        )
    else:
        final, at = splice(
            speech_arr, conv_arr, cfg.sample_rate,
            cfg.splice.placement, cfg.splice.para_gain_db, cfg.splice.pad_ms, cfg.splice.fade_ms,
            at_s=insert_at_s,
        )
    final = final / (np.max(np.abs(final)) + 1e-9) * 0.95

    para_path = cfg.paths.output_dir / f"para_{row.id}.wav"
    sf.write(para_path, final, cfg.sample_rate, subtype="PCM_16")

    return SynthesisResult(
        id=row.id,
        source_audio=str(row.audio_filepath),
        para_audio=str(para_path),
        vs_source_audio=str(vs_src_path),
        converted_audio=str(converted),
        vs_class=vs_class,
        nv_tag=nv_tag,
        transcript=row.text,
        splice_at_s=at,
        insert_stage=insert_stage,
        sim_converted=sim_converted,
        sim_raw_baseline=sim_raw_baseline,
        vs_clip=vs_name,
        speaker_score=speaker_score,
        clip_intensity=vs_intensity,
    )


def synthesize_batch(rows: list[ManifestRow], cfg: Config, language: str | None = "Vietnamese") -> list[SynthesisResult]:
    """The full pipeline over a batch of already-tagged rows (see dataset.tagged_rows).
    Writes `metadata_synth.jsonl` into cfg.paths.output_dir — the diagram's "new dataset"."""
    cfg.ensure_dirs()
    align_pipeline = AlignmentPipeline(cfg.alignment, cfg.models)
    align_pipeline.prepare_mfa_batch(rows, cfg.paths.work_dir / "mfa_corpus", cfg.paths.work_dir / "mfa_aligned")
    speaker_sim = SpeakerSimilarity(cfg.models.speaker_id_source())

    results: list[SynthesisResult] = []
    for i, row in enumerate(rows, 1):
        print(f"\n[{i}/{len(rows)}] {row.id}")
        try:
            wav, sr = load_mono(row.audio_filepath)
            insert_at_s, insert_stage = align_pipeline.find_insert_time(row, wav, sr, language=language)
            print(f"   🧭 insertion time estimate: {insert_at_s:.2f}s (stage={insert_stage})")
            result = synthesize_row(row, cfg, speaker_sim, insert_at_s, insert_stage)
            results.append(result)
            print(f"   ✅ sim_converted={result.sim_converted:.3f} sim_raw_baseline={result.sim_raw_baseline:.3f} "
                  f"spliced@{result.splice_at_s:.2f}s")
        except Exception as e:
            print(f"   ⚠️  skipped {row.id}: {type(e).__name__}: {e}")

    out_path = cfg.paths.output_dir / "metadata_synth.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print(f"\n📦 {len(results)}/{len(rows)} rows synthesized -> {cfg.paths.output_dir}")
    print(f"   metadata: {out_path}")
    return results

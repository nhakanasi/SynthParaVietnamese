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
from para_synth.audio_utils import load_mono, splice, trim_event
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


def pick_vocalsound_clip(vs_dir: Path, vs_class: str, rng: random.Random, max_tries: int = 12):
    """Pool filter -> energy-trim (audio_utils.trim_event) -> normalise -> pad to >=1s for
    Seed-VC's chunker."""
    pool = list_clips(vs_dir, vs_class)
    if not pool:
        raise RuntimeError(f"No '{vs_class}' clips found in {vs_dir}")
    for _ in range(max_tries):
        pick = rng.choice(pool)
        raw, raw_sr = load_mono(vs_dir / pick)
        cand = trim_event(raw, raw_sr)
        if np.max(np.abs(cand)) > 0.05 and len(cand) / raw_sr > 0.35:
            cand = cand / (np.max(np.abs(cand)) + 1e-9) * 0.95
            if len(cand) / raw_sr < 1.0:
                pad = int((1.0 - len(cand) / raw_sr) * raw_sr / 2)
                cand = np.pad(cand, (pad, pad))
            return pick, cand, raw_sr
    raise RuntimeError(f"{max_tries} picks of '{vs_class}' were all too quiet/short")


def synthesize_row(
    row: ManifestRow,
    cfg: Config,
    align_pipeline: AlignmentPipeline,
    speaker_sim: SpeakerSimilarity,
    insert_at_s: float,
    insert_stage: str,
) -> SynthesisResult:
    rng = cfg.rng()
    nv_tag = extract_tag(row.text)
    if nv_tag is None:
        raise ValueError(f"row {row.id!r} has no [tag] in its transcript — run tagging first")
    vs_class = match_vs_class(nv_tag, rng)

    vs_name, vs_clip, vs_sr = pick_vocalsound_clip(cfg.paths.vocalsound_dir, vs_class, rng)
    vs_src_path = cfg.paths.work_dir / f"vs_{row.id}.wav"
    sf.write(vs_src_path, vs_clip, vs_sr, subtype="PCM_16")

    converted = run_seedvc(vs_src_path, row.audio_filepath, cfg.paths.work_dir / "output_vc", cfg.seedvc)

    speech_arr, speech_sr = load_mono(row.audio_filepath, cfg.sample_rate)
    conv_arr, conv_sr = load_mono(converted, cfg.sample_rate)

    sim_converted = speaker_sim.similarity(speech_arr, speech_sr, conv_arr, conv_sr)
    sim_raw_baseline = speaker_sim.similarity(speech_arr, speech_sr, vs_clip, vs_sr)
    if sim_converted <= sim_raw_baseline:
        print(f"   ⚠️  {row.id}: conversion didn't clearly beat the raw-clip baseline "
              f"({sim_converted:.3f} <= {sim_raw_baseline:.3f})")

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
            result = synthesize_row(row, cfg, align_pipeline, speaker_sim, insert_at_s, insert_stage)
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

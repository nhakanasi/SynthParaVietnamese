"""Orchestrates the full diagram: pick a VocalSound clip -> Seed-VC it into the speaker's
voice -> score the conversion -> align the transcript's [tag] in time -> splice -> filter on
predicted quality -> write the new "Para recording + tagged transcript" dataset row.

Ported/generalized from notebook45ee5319ae.ipynb cell 25's batch loop, then split into
stages: `align_batch()` -> `synthesize_batch()` -> `filter_batch()`, each persisting what it
produced and skipping rows it has already done. The split exists because the three stages
differ in cost by orders of magnitude — alignment is a forward pass, synthesis is 50 steps of
diffusion per row, filtering is about a second per row — so re-tuning a splice or a quality
threshold shouldn't have to re-pay for the stage before it. See "Staged execution" in
docs/PIPELINE.md for the artifact contracts.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from para_synth.align import AlignmentPipeline
from para_synth.audio_utils import (
    adaptive_splice,
    load_mono,
    splice,
    splice_boundary_activity,
    tempo_splice,
    trim_event,
)
from para_synth.config import Config
from para_synth.dataset import ManifestRow, extract_tag, merge_by_id, read_jsonl, write_jsonl
from para_synth.quality import SpeakerSimilarity, build_speaker_similarity
from para_synth.seedvc import run_seedvc
from para_synth.selection import pick_vocalsound_clip, profile_speaker
from para_synth.vad import snap_insert_time
from para_synth.vocalsound import gap_shape, level_offset_db, match_vs_class


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
    # Selection audit trail — lets a finished run be checked for whether each selection
    # axis actually tracked the target recording, and how much spread survived in the
    # dataset. `selection_axes` is {axis: {target…, clip…, distance}} from
    # selection.pick_vocalsound_clip(); empty when all axis weights are off (uniform pick).
    vs_clip: str
    speaker_score: float
    clip_intensity: float
    selection_axes: dict



def synthesize_row(
    row: ManifestRow,
    cfg: Config,
    speaker_sim: SpeakerSimilarity,
    insert_at_s: float,
    insert_stage: str,
) -> SynthesisResult:
    rng = cfg.rng(row.id)  # per-row keys — see Config.rng(); a bare rng() repeats per row
    nv_tag = extract_tag(row.text)
    if nv_tag is None:
        raise ValueError(f"row {row.id!r} has no [tag] in its transcript — run tagging first")
    vs_class = match_vs_class(nv_tag, rng)

    speech_arr, speech_sr = load_mono(row.audio_filepath, cfg.sample_rate)
    # Measure the target on the axes Seed-VC won't fix (intensity, channel clarity, tempo)
    # — not on identity/timbre, which the CAM++ style vector already handles downstream.
    profile = profile_speaker(speech_arr, speech_sr, insert_at_s, cfg.selection.context_s)

    pick = pick_vocalsound_clip(
        cfg.paths.vocalsound_dir, vs_class, rng, cfg.selection, profile=profile,
    )
    vs_name, vs_clip, vs_sr = pick.name, pick.audio, pick.sr
    vs_src_path = cfg.paths.work_dir / f"vs_{row.id}.wav"
    sf.write(vs_src_path, vs_clip, vs_sr, subtype="PCM_16")

    converted = run_seedvc(vs_src_path, row.audio_filepath, cfg.paths.work_dir / "output_vc", cfg.seedvc)

    conv_arr, conv_sr = load_mono(converted, cfg.sample_rate)
    # Re-trim after conversion, not just before it: `_finalize_clip` zero-pads short clips
    # to >=1s for Seed-VC's chunker, and Seed-VC returns that padding, so the converted
    # event arrives with dead air at both ends (~140ms measured). Splicing it in unchanged
    # means the fade at each junction is spent on padding while the event itself still
    # starts at full amplitude — the silence around the event has to be one the splice
    # chose, not one inherited from the chunker.
    conv_arr = trim_event(conv_arr, conv_sr, pad_s=0.0)

    sim_converted = speaker_sim.similarity(speech_arr, speech_sr, conv_arr, conv_sr)
    sim_raw_baseline = speaker_sim.similarity(speech_arr, speech_sr, vs_clip, vs_sr)
    if sim_converted <= sim_raw_baseline:
        print(f"   ⚠️  {row.id}: conversion didn't clearly beat the raw-clip baseline "
              f"({sim_converted:.3f} <= {sim_raw_baseline:.3f})")

    if cfg.splice.mode == "tempo":
        # The event class sets how the silence is shaped and how loud the event sits, both
        # as relative terms on top of per-speaker measurements — see vocalsound.GAP_SHAPE
        # and LEVEL_OFFSET_DB. `rng` seeds the room-tone bridges so a run reproduces.
        pre_scale, post_scale = gap_shape(vs_class, cfg.splice.gap_shape)
        final, at = tempo_splice(
            speech_arr, conv_arr, cfg.sample_rate, cfg.splice.gain_db(),
            cfg.splice.fade_ms, at_s=insert_at_s,
            gap_scale=cfg.splice.gap_scale, fade_k=cfg.splice.fade_k,
            pre_scale=pre_scale, post_scale=post_scale,
            level_ref=cfg.splice.level_ref, context_s=cfg.selection.context_s,
            level_offset_db=level_offset_db(vs_class, cfg.splice.level_offsets_db),
            rng=rng,
        )
    elif cfg.splice.mode == "adaptive":
        final, at = adaptive_splice(
            speech_arr, conv_arr, cfg.sample_rate, cfg.splice.para_gain_db,
            cfg.splice.min_pad_ms, cfg.splice.max_gap_ms, cfg.splice.fade_ms,
            at_s=insert_at_s, rng=rng,
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
        speaker_score=profile.energy_rank,
        clip_intensity=pick.intensity,
        selection_axes=pick.axes,
    )


def _file_stamp(path) -> list | None:
    """[size, mtime_ns] for a file, or None if it's gone (or unrecorded) — how a cached
    stage artifact tells whether the audio it describes is still the audio on disk."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    st = p.stat()
    return [st.st_size, st.st_mtime_ns]


def _measurement_is_current(cached: dict, row: dict, need_nisqa: bool) -> bool:
    """Whether a cached `filter`-stage measurement still describes this synthesized row.

    Two independent halves, because the two measures depend on different things:

    * the splice-boundary activity is a function of the source recording and the insertion
      time, so it is current while the cached `splice_at_s` matches the row's;
    * the NISQA scores describe one particular rendering of `para_*.wav`, which
      `synth --force` overwrites in place — the id alone can't tell them apart, so the
      size+mtime stamp is what makes a re-synthesized row get re-scored rather than inherit
      the previous take's numbers. A cached entry with no stamp at all is stale by
      definition, not a match against a missing file's `None`.
    """
    if cached.get("splice_at_s") != row.get("splice_at_s"):
        return False
    if not need_nisqa:
        return True
    stamp = cached.get("para_stat")
    return stamp is not None and stamp == _file_stamp(cached.get("para_audio"))


def slots_path(cfg: Config) -> Path:
    return cfg.paths.stage_dir / "slots.jsonl"


def tagged_path(cfg: Config) -> Path:
    """Which pause each row's tag was placed in. Distinct from the tagged transcripts
    themselves (`data/tagged/transcripts/`, the human-readable artifact) because the time is
    what `align_batch()` needs and re-deriving it from the text would put us back to
    guessing at a position that was already decided."""
    return cfg.paths.stage_dir / "tagged.jsonl"


def align_path(cfg: Config) -> Path:
    return cfg.paths.stage_dir / "align.jsonl"


def synth_path(cfg: Config) -> Path:
    return cfg.paths.output_dir / "metadata_synth.jsonl"


def quality_path(cfg: Config) -> Path:
    """The `filter` stage's measurement cache. Named for the stage rather than for NISQA
    because it also holds the splice-boundary measure, which needs no model."""
    return cfg.paths.stage_dir / "quality.jsonl"


def filter_is_configured(cfg: Config) -> bool:
    """Whether the `filter` stage has any criterion to apply. Both of its measures are
    independently switchable, so with NISQA disabled and no boundary threshold set there is
    nothing to filter on and the stage would only copy its input."""
    return cfg.quality.nisqa.enabled or cfg.quality.max_boundary_activity is not None


def filtered_path(cfg: Config) -> Path:
    return cfg.paths.output_dir / "metadata_filtered.jsonl"


def slots_batch(
    rows: list[ManifestRow],
    cfg: Config,
    language: str | None = "Vietnamese",
    force: bool = False,
) -> list[dict]:
    """Stage 0a: the pauses each recording offers as insertion positions.

    Runs VAD and forced alignment over *untagged* transcripts and writes the intersection —
    every pause that falls in a gap between two aligned words, with the character offset in
    the transcript it corresponds to — to `stages/slots.jsonl`. See para_synth/slots.py for
    why the pipeline finds positions before choosing one rather than the other way round.

    A row with no qualifying pause gets an empty candidate list rather than being omitted:
    "measured, and there is nowhere to put it" is a different fact from "not measured yet",
    and only the first should survive a re-run without `--force`.
    """
    from para_synth.slots import candidate_slots
    from para_synth.vad import pause_intervals

    cfg.ensure_dirs()
    out_path = slots_path(cfg)
    cached = read_jsonl(out_path)
    done = {r["id"] for r in cached}
    todo = list(rows) if force else [r for r in rows if r.id not in done]

    if not todo:
        print(f"⏭️  all {len(rows)} rows already have candidate slots -> {out_path}")
        return cached

    align_pipeline = AlignmentPipeline(cfg.alignment, cfg.models)
    fresh: list[dict] = []
    n_empty = 0
    for i, row in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {row.id}")
        try:
            wav, sr = load_mono(row.audio_filepath)
            pauses = pause_intervals(
                wav, sr,
                backend=cfg.vad.backend,
                merge_gap_s=cfg.vad.merge_gap_s,
                min_pause_s=cfg.vad.min_pause_s,
                edge_margin_s=cfg.vad.edge_margin_s,
            )
            words = align_pipeline.word_times(row.audio_filepath, row.text, language=language)
            slots = candidate_slots(row.text, words, pauses) if words else []
            if not slots:
                n_empty += 1
                reason = "alignment unavailable" if not words else f"{len(pauses)} pause(s), none between two words"
                print(f"   🚫 no candidate positions ({reason})")
            else:
                print(f"   🔇 {len(slots)} candidate position(s): "
                      + ", ".join(f"<{s.i}> {s.before}|{s.after} @{s.time_s:.2f}s ({s.pause_s:.2f}s)"
                                  for s in slots[:6])
                      + (" …" if len(slots) > 6 else ""))
            fresh.append({"id": row.id, "candidates": [asdict(s) for s in slots]})
        except Exception as e:
            print(f"   ⚠️  skipped {row.id}: {type(e).__name__}: {e}")

    merged = merge_by_id(cached, fresh)
    write_jsonl(out_path, merged)
    n_slots = sum(len(r["candidates"]) for r in merged)
    print(f"\n🔇 {len(fresh)}/{len(todo)} rows measured, {n_slots} candidate positions "
          f"({len(merged)} rows total) -> {out_path}")
    if n_empty:
        print(f"   {n_empty} row(s) have no pause to insert into and will be skipped by "
              f"`tag-transcripts` — lower vad.min_pause_s to offer more, at the cost of "
              f"offering stop closures as if they were pauses")
    return merged


def tag_batch(rows: list[ManifestRow], cfg: Config, force: bool = False) -> list[dict]:
    """Stage 0b: choose one position and one tag per row, and write the tagged transcript.

    Slot-constrained (`tagging.slot_constrained`, the default): the model hears the recording,
    sees only the pause positions `slots_batch()` found, and answers with a slot number and a
    tag. The transcript is then written locally by `slots.insert_tag_at()`, so the text and
    the eventual splice time are the same decision rather than two that have to be
    reconciled. Rows with no candidate position are skipped.

    Free (`slot_constrained: false`): the historical path, where the model returns the
    transcript with a tag woven in wherever it chose. Kept because it needs no VAD, no
    aligner and no audio-capable model — but its position is unconstrained, which is what
    `quality.max_boundary_activity` then spends its time rejecting.
    """
    from para_synth.slots import slot_from_dict
    from para_synth.tagging import NoSuitableSlot, TaggingError, choose_slot, insert_para_tag

    cfg.ensure_dirs()
    cfg.paths.tagged_transcript_dir.mkdir(parents=True, exist_ok=True)
    out_path = tagged_path(cfg)
    cached = read_jsonl(out_path)
    done = {
        r["id"] for r in cached
        if (cfg.paths.tagged_transcript_dir / f"{r['id']}.txt").is_file()
    }
    todo = list(rows) if force else [r for r in rows if r.id not in done]
    if not todo:
        print(f"⏭️  all {len(rows)} rows already tagged -> {cfg.paths.tagged_transcript_dir}")
        return cached

    slots_by_id = {r["id"]: r["candidates"] for r in read_jsonl(slots_path(cfg))}
    if cfg.tagging.slot_constrained and not slots_by_id:
        raise FileNotFoundError(
            f"no candidate positions at {slots_path(cfg)} — run `para-synth slots` first, or "
            f"set tagging.slot_constrained: false to let the model place the tag freely"
        )

    fresh: list[dict] = []
    n_no_slot = 0
    for i, row in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {row.id}")
        try:
            if cfg.tagging.slot_constrained:
                slots = [slot_from_dict(s) for s in slots_by_id.get(row.id, [])]
                result = choose_slot(
                    row.text, slots, row.audio_filepath,
                    model=cfg.tagging.qwen_omni_audio_model,
                )
                slot = slots[result.slot_i - 1]
                entry = {
                    "id": row.id,
                    "tag": result.tag,
                    "slot_i": result.slot_i,
                    "insert_at_s": slot.time_s,
                    "pause_s": slot.pause_s,
                }
                print(f"   🏷️  {result.tag} at <{result.slot_i}> {slot.before}|{slot.after} "
                      f"@{slot.time_s:.2f}s (pause {slot.pause_s:.2f}s)")
            else:
                result = insert_para_tag(
                    row.text,
                    backend=cfg.tagging.backend,
                    model=cfg.tagging.model_for_backend(),
                    audio_path=row.audio_filepath,  # used only by audio backends
                )
                entry = {"id": row.id, "tag": result.tag, "slot_i": None, "insert_at_s": None}
                print(f"   🏷️  {result.tag} (free placement)")

            (cfg.paths.tagged_transcript_dir / f"{row.id}.txt").write_text(result.text, encoding="utf-8")
            fresh.append(entry)
        except NoSuitableSlot as e:
            n_no_slot += 1
            print(f"   🚫 skipped: {e}")
        except TaggingError as e:
            print(f"   ⚠️  tagging failed: {e}")

    merged = merge_by_id(cached, fresh)
    write_jsonl(out_path, merged)
    print(f"\n🏷️  {len(fresh)}/{len(todo)} rows tagged ({len(merged)} total) -> "
          f"{cfg.paths.tagged_transcript_dir}")
    if n_no_slot:
        print(f"   {n_no_slot} row(s) had no usable pause position and carry no tag — they "
              f"drop out of the dataset here rather than being spliced mid-utterance")
    return merged


def align_batch(
    rows: list[ManifestRow],
    cfg: Config,
    language: str | None = "Vietnamese",
    force: bool = False,
) -> list[dict]:
    """Stage 1: where in each recording the transcript's `[tag]` belongs.

    Writes `{id, insert_at_s, insert_stage}` per row to `stages/align.jsonl`. Rows already
    in that file are skipped unless `force`, which is what makes re-running the rest of the
    pipeline free of the aligner's model load.

    Two ways a row gets its time. If `tag_batch()` placed the tag into a pause it chose, that
    time is already known exactly and is used as-is — no model runs, and no search can move
    the splice away from the position the transcript describes. Otherwise the tag arrived
    from somewhere this pipeline didn't control (a pre-tagged JSONL manifest), so its time
    has to be recovered by forced alignment and then nudged toward a pause if one is close
    enough; that is the older path, and `vad.max_shift_s` / `vad.on_no_pause` govern only it.
    """
    cfg.ensure_dirs()
    out_path = align_path(cfg)
    cached = read_jsonl(out_path)
    done = {r["id"] for r in cached}
    todo = list(rows) if force else [r for r in rows if r.id not in done]

    if not todo:
        print(f"⏭️  all {len(rows)} rows already aligned -> {out_path}")
        return cached

    chosen = {
        r["id"]: r for r in read_jsonl(tagged_path(cfg))
        if r.get("insert_at_s") is not None
    }
    from_slots = [r for r in todo if r.id in chosen]
    todo = [r for r in todo if r.id not in chosen]
    prechosen = [
        {
            "id": row.id,
            "insert_at_s": chosen[row.id]["insert_at_s"],
            "insert_stage": "vad_slot",
            "insert_at_aligned_s": chosen[row.id]["insert_at_s"],
            "vad_status": "slot",
            "vad_shift_s": 0.0,
            "vad_pause_s": chosen[row.id].get("pause_s", 0.0),
        }
        for row in from_slots
    ]
    if prechosen:
        print(f"🔇 {len(prechosen)} row(s) were tagged into a pause chosen by "
              f"`tag-transcripts` — using those times directly, no alignment needed")
    if not todo:
        merged = merge_by_id(cached, prechosen)
        write_jsonl(out_path, merged)
        print(f"\n🧭 {len(prechosen)} rows resolved ({len(merged)} total) -> {out_path}")
        return merged

    align_pipeline = AlignmentPipeline(cfg.alignment, cfg.models)
    fresh: list[dict] = []
    vad_counts: dict[str, int] = {}
    dropped_no_pause = 0
    for i, row in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {row.id}")
        try:
            wav, sr = load_mono(row.audio_filepath)
            insert_at_s, insert_stage = align_pipeline.find_insert_time(row, wav, sr, language=language)
            print(f"   🧭 insertion time estimate: {insert_at_s:.2f}s (stage={insert_stage})")

            # A word boundary is not a pause — move the cut onto one, so the event lands in
            # silence between utterances instead of slicing a vowel. See para_synth/vad.py.
            snap = snap_insert_time(wav, sr, insert_at_s, cfg.vad)
            vad_counts[snap.status] = vad_counts.get(snap.status, 0) + 1
            if snap.status == "no_pause" and cfg.vad.enabled:
                if cfg.vad.on_no_pause == "skip":
                    print(f"   ⏭️  skipped: no VAD pause within {cfg.vad.max_shift_s:.2f}s of "
                          f"{insert_at_s:.2f}s (vad.on_no_pause: skip)")
                    dropped_no_pause += 1
                    continue
                print(f"   ⚠️  no VAD pause within {cfg.vad.max_shift_s:.2f}s — splicing at the "
                      f"unsnapped time (vad.on_no_pause: keep)")
            elif snap.status in ("in_pause", "snapped"):
                print(f"   🔇 {snap.status}: {insert_at_s:.2f}s -> {snap.time_s:.2f}s "
                      f"({snap.shift_s:+.2f}s, pause {snap.pause_s:.2f}s)")

            fresh.append({
                "id": row.id,
                "insert_at_s": snap.time_s,
                "insert_stage": insert_stage,
                # Keep the pre-snap time and the VAD verdict: they explain why a row moved
                # (or was dropped) without re-running the aligner to find out.
                "insert_at_aligned_s": insert_at_s,
                "vad_status": snap.status,
                "vad_shift_s": snap.shift_s,
                "vad_pause_s": snap.pause_s,
            })
        except Exception as e:
            print(f"   ⚠️  skipped {row.id}: {type(e).__name__}: {e}")

    merged = merge_by_id(cached, prechosen + fresh)
    write_jsonl(out_path, merged)
    print(f"\n🧭 {len(fresh)}/{len(todo)} rows aligned ({len(merged)} total) -> {out_path}")
    if cfg.vad.enabled and vad_counts:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(vad_counts.items()))
        print(f"   VAD pause snapping ({cfg.vad.backend}): {summary}")
        if dropped_no_pause:
            print(f"   {dropped_no_pause} row(s) dropped for having no pause within "
                  f"{cfg.vad.max_shift_s:.2f}s — raise vad.max_shift_s, lower vad.min_pause_s, "
                  f"or set vad.on_no_pause: keep to include them")
    return merged


def synthesize_batch(rows: list[ManifestRow], cfg: Config, force: bool = False) -> list[dict]:
    """Stage 2: convert, splice, and write one Para recording per row.

    Reads the insertion times from `stages/align.jsonl` (run `align_batch()` first — a row
    with no alignment is skipped rather than guessed at), and appends to
    `metadata_synth.jsonl` in cfg.paths.output_dir, the diagram's "new dataset".

    A row counts as done only when its metadata entry *and* its `para_*.wav` both exist, so
    deleting an output file is enough to make this stage rebuild it.
    """
    cfg.ensure_dirs()
    alignments = {r["id"]: r for r in read_jsonl(align_path(cfg))}
    if not alignments:
        raise FileNotFoundError(
            f"no alignments at {align_path(cfg)} — run `para-synth align` (or `para-synth run`) first"
        )

    out_path = synth_path(cfg)
    cached = read_jsonl(out_path)
    done = {r["id"] for r in cached if Path(r["para_audio"]).is_file()}

    todo, unaligned = [], []
    for row in rows:
        if row.id not in alignments:
            unaligned.append(row.id)
        elif force or row.id not in done:
            todo.append(row)
    if unaligned:
        print(f"⚠️  {len(unaligned)} row(s) have no alignment and were skipped: {', '.join(unaligned[:5])}"
              + (f" … +{len(unaligned) - 5} more" if len(unaligned) > 5 else ""))
    if not todo:
        print(f"⏭️  all {len(rows) - len(unaligned)} aligned rows already synthesized -> {out_path}")
        return cached

    speaker_sim = build_speaker_similarity(cfg)
    fresh: list[dict] = []
    for i, row in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {row.id}")
        align = alignments[row.id]
        try:
            result = synthesize_row(row, cfg, speaker_sim, align["insert_at_s"], align["insert_stage"])
            fresh.append(asdict(result))
            print(f"   ✅ sim_converted={result.sim_converted:.3f} sim_raw_baseline={result.sim_raw_baseline:.3f} "
                  f"spliced@{result.splice_at_s:.2f}s")
        except Exception as e:
            print(f"   ⚠️  skipped {row.id}: {type(e).__name__}: {e}")

    merged = merge_by_id(cached, fresh)
    write_jsonl(out_path, merged)
    print(f"\n📦 {len(fresh)}/{len(todo)} rows synthesized ({len(merged)} total) -> {cfg.paths.output_dir}")
    print(f"   metadata: {out_path}")
    return merged


def filter_batch(cfg: Config, force: bool = False) -> list[dict]:
    """Stage 3: measure every finished row and keep the ones that pass.

    Two measures, deliberately independent because they answer different questions:

    * **NISQA** scores the source recording *and* the Para recording, all five dimensions,
      unconditionally — thresholds are applied afterwards. It hears hiss, coloration and
      audible seams in the delivered file.
    * **Splice-boundary activity** measures the source recording at the insertion time. It
      is what NISQA structurally cannot see: by the time the file is written both junctions
      have been fade-damped and separated by a room-tone gap, so a cut taken from the middle
      of a vowel still scores as a clean signal even though the event interrupts the speaker
      mid-word. Costs no model — it is read off audio the stage already loads.

    Everything measured lands in `stages/quality.jsonl`; the passing subset of
    `metadata_synth.jsonl` goes to `metadata_filtered.jsonl` carrying its measurements.
    Nothing is deleted and no row is measured twice, so re-tuning a threshold re-reads the
    cache and never loads the model. See para_synth/nisqa.py.
    """
    from para_synth.nisqa import NisqaScorer, failed_criteria

    if not filter_is_configured(cfg):
        raise RuntimeError(
            "the filter stage has no criteria: quality.nisqa.enabled is false and "
            "quality.max_boundary_activity is null, so every row would pass. Enable one of "
            "them, or drop the stage from your run."
        )

    cfg.ensure_dirs()
    rows = read_jsonl(synth_path(cfg))
    if not rows:
        raise FileNotFoundError(
            f"no synthesized rows at {synth_path(cfg)} — run `para-synth synth` (or `para-synth run`) first"
        )

    need_nisqa = cfg.quality.nisqa.enabled
    cache_path = quality_path(cfg)
    cached = read_jsonl(cache_path)
    by_id = {r["id"]: r for r in cached}
    measured = (
        {}
        if force
        else {
            row["id"]: by_id[row["id"]]
            for row in rows
            if row["id"] in by_id and _measurement_is_current(by_id[row["id"]], row, need_nisqa)
        }
    )
    todo = [r for r in rows if r["id"] not in measured]

    if todo:
        # Only pay for the model when something actually needs scoring — a threshold change
        # alone leaves `todo` empty and this whole branch is skipped.
        scorer = NisqaScorer() if need_nisqa else None
        for i, row in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {row['id']}", end=" ")
            try:
                source_wav, source_sr = load_mono(Path(row["source_audio"]), cfg.sample_rate)
                before, after = splice_boundary_activity(source_wav, source_sr, row["splice_at_s"])
                entry = {
                    "id": row["id"],
                    "para_audio": row["para_audio"],
                    "para_stat": _file_stamp(row["para_audio"]),
                    "splice_at_s": row["splice_at_s"],
                    "boundary_activity": {"before": before, "after": after},
                }
                report = f"boundary {before:.2f}/{after:.2f}"
                if scorer is not None:
                    para_wav, para_sr = load_mono(Path(row["para_audio"]), cfg.sample_rate)
                    entry["source"] = scorer.score(source_wav, source_sr)
                    entry["para"] = scorer.score(para_wav, para_sr)
                    report = (f"mos {entry['source']['mos']:.2f} -> {entry['para']['mos']:.2f} "
                              f"(drop {entry['source']['mos'] - entry['para']['mos']:+.2f}) " + report)
                prior = by_id.get(row["id"])
                if scorer is None and prior and prior.get("para_stat") == entry["para_stat"]:
                    # Re-measuring with NISQA switched off would otherwise drop scores that
                    # are still valid for this exact file — expensive output, thrown away by
                    # a run that simply wasn't asking about it.
                    entry |= {k: prior[k] for k in ("source", "para") if k in prior}
                measured[row["id"]] = entry
                print(report)
            except Exception as e:
                print(f"⚠️  measuring failed: {type(e).__name__}: {e}")
        write_jsonl(cache_path, merge_by_id(cached, list(measured.values())))
    else:
        print(f"⏭️  reusing cached quality measurements for {len(rows)} rows -> {cache_path}")

    limit = cfg.quality.max_boundary_activity
    kept, rejected = [], []
    for row in rows:
        entry = measured.get(row["id"])
        if entry is None:  # measuring failed for this row — it can't be judged, so keep it
            kept.append(row)
            continue

        failures = failed_criteria(entry, cfg.quality.nisqa) if need_nisqa else []
        boundary = entry["boundary_activity"]
        # Judged on the louder junction: one clean side doesn't excuse an event that starts
        # mid-syllable, and the two junctions are independent places for that to happen.
        worst = max(boundary["before"], boundary["after"])
        if limit is not None and worst > limit:
            side = "before" if boundary["before"] >= boundary["after"] else "after"
            failures.append(f"boundary_activity {worst:.2f} ({side}) > {limit}")

        enriched = {**row, "boundary_activity": boundary}
        if need_nisqa:
            enriched |= {"nisqa_source": entry["source"], "nisqa_para": entry["para"]}
        if failures:
            rejected.append((row["id"], failures))
        else:
            kept.append(enriched)

    out_path = filtered_path(cfg)
    write_jsonl(out_path, kept)
    print(f"\n🔎 quality filter: {len(kept)}/{len(rows)} rows kept -> {out_path}")
    for row_id, failures in rejected:
        print(f"   ❌ {row_id}: {'; '.join(failures)}")
    return kept

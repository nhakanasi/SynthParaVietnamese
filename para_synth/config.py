"""Typed config loaded from configs/default.yaml (or a user-supplied override).

Replaces the notebook's module-level globals (cell 6) with a plain dataclass that gets
threaded through every function instead of relied on as ambient state.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SeedVCConfig:
    repo_dir: Path
    diffusion_steps: int = 50
    length_adjust: float = 1.0
    cfg_rate: float = 0.7
    f0_condition: bool = False


@dataclass
class SpliceConfig:
    placement: str = "gap"  # "gap" | "before" | "after"
    para_gain_db: float = -3.0
    pad_ms: int = 10
    fade_ms: int = 50
    # Which splice implementation in para_synth.audio_utils joins the event to the speech:
    #   "tempo"    — tempo_splice(): exponential fades at both junctions + a room-tone gap
    #                 sized to the speaker's own median pause. The default, because the
    #                 forced-aligned cut lands mid-phonation in practice (20/20 rows of the
    #                 first real batch) and the other two modes leave that audible.
    #   "adaptive" — adaptive_splice(): fades/widens a junction only when its boundary RMS
    #                 shows the cut landed on active phonation.
    #   "fixed"    — splice(): the notebook's original, fixed fade+pad regardless of the cut.
    mode: str = "tempo"
    min_pad_ms: int = 10  # adaptive only: bridge width at a boundary that was already quiet
    max_gap_ms: int = 70  # adaptive only: bridge width at a boundary that needed damping
    # tempo only. gap_scale multiplies the measured median pause (1.0 = exactly the
    # speaker's own tempo). fade_k is the exponential fade's curvature — higher damps faster.
    gap_scale: float = 1.0
    fade_k: float = 5.0
    # tempo only. The gap is a *budget*, not an addition: `audio_utils.existing_silence_s`
    # measures the silence each side of the cut already has and only the shortfall is
    # inserted. Without this the tempo gap stacks on top of the pause the slot path
    # deliberately cut into, which over-pads badly — see that function's docstring.
    # `gap_shape` overrides vocalsound.GAP_SHAPE per class as {class: [pre, post]}, the
    # asymmetric pre/post multipliers on the speaker's measured median pause.
    gap_shape: dict = field(default_factory=dict)
    # How the event's level is referenced to the speech (see audio_utils.level_insert):
    #   "context_rms" — event RMS vs. the RMS of the `selection.context_s` seconds before
    #                   the cut, plus the per-class term below. The default.
    #   "peak"        — the original: event peak `para_gain_db` under the utterance's peak.
    level_ref: str = "context_rms"
    # The gain that goes with level_ref: "context_rms". Kept separate from `para_gain_db`
    # rather than reusing it, because the two are not the same quantity — one is dB against
    # the local speech RMS, the other dB under the whole utterance's peak — and a single
    # field meaning either depending on a neighbouring field is a trap. 0.0 means "the event
    # sits at the speaker's local speech RMS", with vocalsound.LEVEL_OFFSET_DB then moving
    # each class off that baseline in the direction its physics implies.
    context_gain_db: float = 0.0
    # Overrides vocalsound.LEVEL_OFFSET_DB per class, {class: dB}. Only used by
    # level_ref: "context_rms" — a laugh sits above conversational RMS and a sniff below,
    # so a single target level is wrong for both in opposite directions.
    level_offsets_db: dict = field(default_factory=dict)

    def __post_init__(self):
        valid = {"tempo", "adaptive", "fixed"}
        if self.mode not in valid:
            raise ValueError(f"splice.mode must be one of {sorted(valid)}, got {self.mode!r}")
        if self.level_ref not in {"context_rms", "peak"}:
            raise ValueError(
                f"splice.level_ref must be 'context_rms' or 'peak', got {self.level_ref!r}"
            )

    def gain_db(self) -> float:
        """The gain field that belongs to the configured `level_ref`."""
        return self.context_gain_db if self.level_ref == "context_rms" else self.para_gain_db


@dataclass
class SelectionConfig:
    """How pick_vocalsound_clip() (para_synth/selection.py) chooses a VocalSound clip.

    Matching runs only on the axes Seed-VC does *not* normalise away. Speaker identity and
    timbre are excluded on purpose — conversion handles those via the CAM++ style vector,
    so selecting for them would double-count. What survives conversion untouched is the
    clip's intensity, its recording channel, and its timing; those get one weight each.

    All three are *soft* preferences, never hard filters: the axis distances are summed and
    turned into a sampling weight `exp(-distance)`, so every usable candidate keeps nonzero
    probability and the dataset keeps the acoustic variance a Para-TTS model benefits from.
    All weights at 0.0 is pure-random (every clip in the class equally likely). ~1-2 is a
    moderate bias per axis. See docs/PIPELINE.md.
    """

    # Intensity: crest factor + duration percentile vs. the speaker's own energy percentile
    # at the splice point. Seed-VC changes who is laughing, never how hard.
    energy_weight: float = 0.0
    # Channel: SNR in dB and effective bandwidth in octaves. VocalSound is crowdsourced
    # from whatever microphone each contributor had, and hiss/band-limiting is part of the
    # signal Seed-VC reconstructs — the axis a clean recording most audibly rejects.
    clarity_weight: float = 0.0
    # Timing: envelope rate in octaves. Only applies to selection.TEMPO_MATCHED_CLASSES
    # (laughter) — cough/sneeze/throat-clearing/sniff are reflexes with no link to
    # speaking rate. Ignored for every other class regardless of this value.
    tempo_weight: float = 0.0
    # Trend: how much the event's loudness rises or falls across its own span, in dB,
    # against the same measure on the speech running into the splice point. Independent of
    # tempo_weight (rate says how often the envelope peaks, this says which way it is
    # going), and it is the selection-side answer to "let the event decay into the trailing
    # pause" — pick a clip that already decays instead of DSP-shaping one that doesn't.
    # UNCALIBRATED: selection.TILT_TOLERANCE_DB is a prior, not a measurement from this
    # corpus. Keep this modest until scripts/measure_selection_axes.py has reported the
    # tilt_db spread on both corpora.
    tilt_weight: float = 0.0
    # Seconds of speech immediately before the splice point used as the speaker's
    # reference level and pace — local context, not the whole utterance, since a speaker's
    # energy varies across a recording and what matters is the moment the event interrupts.
    # (Channel quality is measured over the whole recording instead — it's a property of
    # the mic and the room, and a short window has too little silence to estimate it from.)
    context_s: float = 2.5
    # Clips loaded and measured per row when any weight is on. Larger = better odds the
    # pool contains a genuinely close match on all axes at once, at ~a few ms per clip
    # against Seed-VC's ~16s of conversion, so this is cheap to raise.
    candidate_pool: int = 48
    # Hard gate, not a match: fraction of samples at full scale above which a candidate is
    # dropped. Clipping is distortion Seed-VC faithfully reconstructs; unlike noise or
    # bandwidth there's no target recording it can be "close to". Dropped rather than
    # allowed to empty the pool, so it can never fail a row on its own.
    max_clipping: float = 0.01
    # Standard deviation of the per-row intensity *target*, in percentile-rank units.
    # 0.0 aims every row at the speaker's own energy rank, which assumes a person's laugh
    # scales with how loudly they were speaking; a quiet, slow speaker with a loud laugh is
    # a real and common person that assumption cannot produce except through the sampling
    # weight's tail. Above zero, the target is drawn per row around the speaker's rank, so
    # the correlation survives on average while genuine mismatches occur at a rate that was
    # chosen. This is NOT the same as lowering energy_weight, which only makes the axis
    # sloppy in both directions. ~0.2 keeps most rows within a fifth of the speaker's rank.
    intensity_spread: float = 0.0
    # Optional path to a precomputed per-clip feature manifest
    # (scripts/build_clip_features.py). When present, the candidate set is the *whole* class
    # rather than `candidate_pool` random decodes, only the winning clip is decoded, and the
    # intensity percentile ranks become corpus-wide instead of draw-wide. `candidate_pool`
    # is then unused. It is a cache, not a commitment: it stores raw measurements only, so
    # re-weighting or re-scaling any axis needs no rebuild — only changing an extractor
    # does, which selection.FEATURE_VERSION detects and warns about.
    feature_manifest: Optional[str] = None


@dataclass
class AlignmentConfig:
    use_qwen3: bool = True


@dataclass
class VADConfig:
    """Keeps the splice out of the middle of an utterance — see para_synth/vad.py."""

    enabled: bool = True
    backend: str = "silero"  # "silero" | "energy"
    min_pause_s: float = 0.12
    max_shift_s: float = 0.5
    merge_gap_s: float = 0.10
    edge_margin_s: float = 0.05
    on_no_pause: str = "keep"  # "keep" | "skip" — see configs/default.yaml for the measurement


@dataclass
class ModelsConfig:
    """third_party/models/ is laid out by pipeline purpose (asr/, aligner/, conversion/,
    speaker-id/, splicing/) — see third_party/models/README.md. Each `*_source()` helper
    returns the local snapshot dir if it's actually been downloaded, else the Hub id
    (triggers an on-demand download inside transformers/qwen_asr instead of failing)."""

    qwen3_asr_dir: Path
    qwen3_asr_hub_id: str = "Qwen/Qwen3-ASR-0.6B"
    qwen3_forced_aligner_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "third_party/models/aligner/qwen3-forcedaligner-0.6b"
    )
    qwen3_forced_aligner_hub_id: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    speaker_id_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "third_party/models/speaker-id/wavlm-base-plus-sv"
    )
    speaker_id_hub_id: str = "microsoft/wavlm-base-plus-sv"

    def qwen3_asr_source(self) -> str:
        return str(self.qwen3_asr_dir) if self.qwen3_asr_dir.is_dir() else self.qwen3_asr_hub_id

    def qwen3_forced_aligner_source(self) -> str:
        return (
            str(self.qwen3_forced_aligner_dir)
            if self.qwen3_forced_aligner_dir.is_dir()
            else self.qwen3_forced_aligner_hub_id
        )

    def speaker_id_source(self) -> str:
        return str(self.speaker_id_dir) if self.speaker_id_dir.is_dir() else self.speaker_id_hub_id


@dataclass
class ASRConfig:
    language: Optional[str] = "Vietnamese"  # None -> auto-detect


@dataclass
class NisqaConfig:
    """Thresholds for the `filter` stage (para_synth/nisqa.py, pipeline.filter_batch()).

    NISQA v2.0 predicts a MOS plus four degradation dimensions from a single recording, with
    no clean reference to compare against. Every threshold here is optional: `None` means
    the criterion isn't checked, and a row passes when every criterion that *is* set passes.
    Rows that fail are recorded and left out of `metadata_filtered.jsonl`; nothing is
    deleted, and every score is cached, so re-tuning a threshold costs no model time.

    `max_mos_drop` is the primary criterion, and the reason the stage scores the source
    recording as well as the finished one. An absolute floor mostly answers "was this corpus
    clean", which is a property of the recordings the pipeline was handed rather than of
    anything it did; the drop answers "did this pipeline degrade this recording", which is
    the question the filter exists to ask, and it stays meaningful on a corpus that is
    uniformly noisy or uniformly clean. Same self-calibrating-relative-measure reasoning as
    selection.py's axis distances — see docs/PIPELINE.md.

    The weights torchmetrics downloads are CC BY-NC-SA 4.0 (non-commercial). Setting
    `enabled: false` means they are never fetched.
    """

    enabled: bool = True
    # Each dimension has the same pair of criteria as the MOS: an absolute floor on the
    # finished recording, and a cap on how far it fell below the source recording. All five
    # values are 1-5 and higher-is-better, so a floor is a minimum and a drop is a maximum.
    # `loudness` is deliberately absent: synthesize_row() peak-normalises its output, so a
    # loudness difference measures that normalisation rather than anything the splice did.
    min_mos: Optional[float] = None
    max_mos_drop: Optional[float] = 0.5
    min_noisiness: Optional[float] = None
    max_noisiness_drop: Optional[float] = None
    # `discontinuity` scores isolated interruptions, so its drop is the dimension that
    # answers "did the splice leave an audible seam" more specifically than the MOS does.
    min_discontinuity: Optional[float] = None
    max_discontinuity_drop: Optional[float] = 0.4
    min_coloration: Optional[float] = None
    max_coloration_drop: Optional[float] = None


@dataclass
class QualityConfig:
    """Which speaker encoder scores the conversions (para_synth/quality.py).

    "campplus" is the same CAM++ encoder Seed-VC v1 conditions the conversion on, so it
    measures the conversion in the space it was actually aimed at; it needs the seed-vc
    checkout present. "wavlm" is an independent verifier that needs nothing but
    transformers, which is why it stays the fallback default here.
    """

    speaker_embedder: str = "wavlm"  # "campplus" | "wavlm"

    # Splice-junction check applied by the same `filter` stage, and the one criterion in
    # here that needs no model at all. NISQA scores the *finished* recording, where the
    # junction has already been faded and gapped, so a cut that landed in the middle of a
    # vowel can still read as clean; this measures the source recording at the cut point
    # instead, as the ratio between the energy right at the cut and the surrounding
    # segment's own level (audio_utils.splice_boundary_activity). Higher = the aligner put
    # the event further into active phonation. None = not checked; a row is judged on the
    # louder of its two junctions. adaptive_splice() calls the same measure "active" above
    # 0.2, but that is a threshold for *damping* a junction, not for rejecting a row.
    max_boundary_activity: Optional[float] = None
    nisqa: NisqaConfig = field(default_factory=NisqaConfig)

    def __post_init__(self):
        valid = {"campplus", "wavlm"}
        if self.speaker_embedder not in valid:
            raise ValueError(
                f"quality.speaker_embedder must be one of {sorted(valid)}, "
                f"got {self.speaker_embedder!r}"
            )


@dataclass
class PathsConfig:
    raw_audio_dir: Path
    raw_transcript_dir: Path
    tagged_transcript_dir: Path
    vocalsound_dir: Path
    work_dir: Path
    output_dir: Path
    # Per-stage artifacts (align.jsonl, quality.jsonl) that let each pipeline stage resume
    # instead of recomputing. Scratch, same status as work_dir's converted clips — the
    # dataset itself is what lands in output_dir.
    stage_dir: Path = field(default_factory=lambda: REPO_ROOT / "data/work/stages")
    # A JSONL manifest ({audio_filepath, text, ...}) to take rows from, instead of pairing
    # raw_audio_dir with a transcript directory. None = use the directory layout. Such a
    # manifest normally arrives with its `[tag]` already inline, so `transcribe` and
    # `tag-transcripts` have nothing to do and the run is align -> synth -> filter.
    manifest: Optional[Path] = None


@dataclass
class TaggingConfig:
    # "qwen" / "gemini" send the transcript only. "qwen_omni_audio" also sends the
    # recording so tag choice and placement can follow the actual delivery — richer, but
    # costs more per row and caps audio length, so it stays opt-in.
    backend: str = "qwen"
    qwen_model: str = "qwen-plus"
    gemini_model: str = "gemini-2.5-flash"
    qwen_omni_audio_model: str = "qwen3-omni-flash"
    # Offer the model only positions VAD found a real pause at (para_synth/slots.py), rather
    # than letting it place the tag anywhere in the text. Always uses the audio backend, and
    # requires `para-synth slots` to have run first.
    slot_constrained: bool = True

    def model_for_backend(self) -> str:
        return getattr(self, f"{self.backend}_model")


@dataclass
class Config:
    seed: Optional[int]
    seedvc: SeedVCConfig
    splice: SpliceConfig
    selection: SelectionConfig
    alignment: AlignmentConfig
    vad: VADConfig
    models: ModelsConfig
    asr: ASRConfig
    quality: QualityConfig
    paths: PathsConfig
    tagging: TaggingConfig
    sample_rate: int = 22050

    def rng(self, *keys) -> random.Random:
        """A random generator seeded from the base seed mixed with `keys`.

        ALWAYS pass keys that identify what you're drawing for (e.g. a row id, a variant
        index). A bare `Config.rng()` returns the *same* sequence every call, so calling it
        per-row hands every row identical draws — which silently made every row of a given
        tag class pick the exact same VocalSound clip before this took keys. Mixing the
        keys in keeps runs reproducible (same seed -> same dataset) while making different
        rows and variants genuinely independent.
        """
        if self.seed is None:
            return random.Random()  # explicitly non-reproducible: fresh entropy per call
        if not keys:
            return random.Random(self.seed)
        digest = hashlib.sha256("\x00".join(str(k) for k in keys).encode("utf-8")).hexdigest()
        return random.Random(self.seed ^ int(digest[:16], 16))

    def ensure_dirs(self) -> None:
        for d in (self.paths.work_dir, self.paths.output_dir, self.paths.stage_dir):
            d.mkdir(parents=True, exist_ok=True)


def _quality_config(raw: dict) -> QualityConfig:
    """`quality` is the one block with a nested sub-block (`nisqa`), so it can't go through
    the plain `QualityConfig(**raw)` the other blocks use."""
    raw = dict(raw)
    return QualityConfig(nisqa=NisqaConfig(**raw.pop("nisqa", {})), **raw)


def _selection_config(raw: dict, root: Path) -> SelectionConfig:
    """`feature_manifest` is the one selection field that names a file, so it gets the same
    repo-root-relative treatment every path in `paths:` does."""
    raw = dict(raw)
    manifest = raw.get("feature_manifest")
    if manifest:
        raw["feature_manifest"] = str(_resolve(root, manifest))
    return SelectionConfig(**raw)


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def load_config(path: str | Path = REPO_ROOT / "configs/default.yaml") -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent  # configs/default.yaml -> repo root

    paths = raw["paths"]
    return Config(
        seed=raw.get("seed"),
        seedvc=SeedVCConfig(
            repo_dir=_resolve(root, raw["seedvc"]["repo_dir"]),
            diffusion_steps=raw["seedvc"]["diffusion_steps"],
            length_adjust=raw["seedvc"]["length_adjust"],
            cfg_rate=raw["seedvc"]["cfg_rate"],
            f0_condition=raw["seedvc"]["f0_condition"],
        ),
        splice=SpliceConfig(**raw["splice"]),
        selection=_selection_config(raw.get("selection", {}), root),
        alignment=AlignmentConfig(use_qwen3=raw["alignment"].get("use_qwen3", True)),
        vad=VADConfig(**(raw.get("vad") or {})),
        models=ModelsConfig(
            qwen3_asr_dir=_resolve(root, raw["models"]["qwen3_asr_dir"]),
            qwen3_asr_hub_id=raw["models"]["qwen3_asr_hub_id"],
            qwen3_forced_aligner_dir=_resolve(root, raw["models"]["qwen3_forced_aligner_dir"]),
            qwen3_forced_aligner_hub_id=raw["models"]["qwen3_forced_aligner_hub_id"],
            speaker_id_dir=_resolve(root, raw["models"]["speaker_id_dir"]),
            speaker_id_hub_id=raw["models"]["speaker_id_hub_id"],
        ),
        asr=ASRConfig(language=raw.get("asr", {}).get("language", "Vietnamese")),
        quality=_quality_config(raw.get("quality", {})),
        paths=PathsConfig(
            raw_audio_dir=_resolve(root, paths["raw_audio_dir"]),
            raw_transcript_dir=_resolve(root, paths["raw_transcript_dir"]),
            tagged_transcript_dir=_resolve(root, paths["tagged_transcript_dir"]),
            vocalsound_dir=_resolve(root, paths["vocalsound_dir"]),
            work_dir=_resolve(root, paths["work_dir"]),
            output_dir=_resolve(root, paths["output_dir"]),
            stage_dir=_resolve(root, paths.get("stage_dir", "data/work/stages")),
            manifest=_resolve(root, paths["manifest"]) if paths.get("manifest") else None,
        ),
        tagging=TaggingConfig(**raw["tagging"]),
        sample_rate=raw.get("sample_rate", 22050),
    )

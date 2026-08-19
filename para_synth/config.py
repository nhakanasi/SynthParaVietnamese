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
    # adaptive_splice() (para_synth.audio_utils) instead of splice(): inspects each
    # junction's boundary RMS and only fades/widens the gap where the cut actually lands
    # on active phonation, instead of always applying a fixed fade+pad. See audio_utils.py.
    adaptive: bool = False
    min_pad_ms: int = 10  # bridge width at a boundary that was already quiet
    max_gap_ms: int = 70  # bridge width at a boundary that needed damping


@dataclass
class SelectionConfig:
    """How pick_vocalsound_clip() chooses which VocalSound clip to use for a row.

    `energy_weight` = 0.0 is pure-random (every clip in the class equally likely). Higher
    values bias sampling toward clips whose loudness is closer to the target speaker's own
    recent level — a *soft* preference, never a hard filter: every clip keeps nonzero
    probability, so genuinely loud-laugh-from-a-soft-speaker pairings still occur, just
    less often than jarringly-mismatched ones otherwise would. See docs/PIPELINE.md.
    """

    energy_weight: float = 0.0
    # Seconds of speech immediately before the splice point used as the speaker's
    # reference level — local context, not the whole utterance, since a speaker's energy
    # varies across a recording and what matters is the moment the event interrupts.
    context_s: float = 2.5


@dataclass
class AlignmentConfig:
    use_qwen3: bool = True
    use_mfa: bool = True
    mfa_env_name: str = "aligner"
    mfa_acoustic_model: str = "vietnamese_mfa"
    mfa_dictionary: str = "vietnamese_mfa"
    conda_dir: Path = field(default_factory=lambda: REPO_ROOT / "third_party/miniforge3")


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
class PathsConfig:
    raw_audio_dir: Path
    raw_transcript_dir: Path
    tagged_transcript_dir: Path
    vocalsound_dir: Path
    work_dir: Path
    output_dir: Path


@dataclass
class TaggingConfig:
    backend: str = "qwen"  # "qwen" | "gemini"
    qwen_model: str = "qwen-plus"
    gemini_model: str = "gemini-2.5-flash"


@dataclass
class Config:
    seed: Optional[int]
    seedvc: SeedVCConfig
    splice: SpliceConfig
    selection: SelectionConfig
    alignment: AlignmentConfig
    models: ModelsConfig
    asr: ASRConfig
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
        for d in (self.paths.work_dir, self.paths.output_dir):
            d.mkdir(parents=True, exist_ok=True)


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
        selection=SelectionConfig(**raw.get("selection", {})),
        alignment=AlignmentConfig(
            use_qwen3=raw["alignment"].get("use_qwen3", True),
            use_mfa=raw["alignment"]["use_mfa"],
            mfa_env_name=raw["alignment"]["mfa_env_name"],
            mfa_acoustic_model=raw["alignment"]["mfa_acoustic_model"],
            mfa_dictionary=raw["alignment"]["mfa_dictionary"],
            conda_dir=_resolve(root, raw["alignment"]["conda_dir"]),
        ),
        models=ModelsConfig(
            qwen3_asr_dir=_resolve(root, raw["models"]["qwen3_asr_dir"]),
            qwen3_asr_hub_id=raw["models"]["qwen3_asr_hub_id"],
            qwen3_forced_aligner_dir=_resolve(root, raw["models"]["qwen3_forced_aligner_dir"]),
            qwen3_forced_aligner_hub_id=raw["models"]["qwen3_forced_aligner_hub_id"],
            speaker_id_dir=_resolve(root, raw["models"]["speaker_id_dir"]),
            speaker_id_hub_id=raw["models"]["speaker_id_hub_id"],
        ),
        asr=ASRConfig(language=raw.get("asr", {}).get("language", "Vietnamese")),
        paths=PathsConfig(
            raw_audio_dir=_resolve(root, paths["raw_audio_dir"]),
            raw_transcript_dir=_resolve(root, paths["raw_transcript_dir"]),
            tagged_transcript_dir=_resolve(root, paths["tagged_transcript_dir"]),
            vocalsound_dir=_resolve(root, paths["vocalsound_dir"]),
            work_dir=_resolve(root, paths["work_dir"]),
            output_dir=_resolve(root, paths["output_dir"]),
        ),
        tagging=TaggingConfig(**raw["tagging"]),
        sample_rate=raw.get("sample_rate", 22050),
    )

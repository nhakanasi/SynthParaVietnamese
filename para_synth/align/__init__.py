"""Where does the transcript's [tag] actually belong in time?

Diagram: the "Splice" arrow needs a real timestamp, not a silence heuristic — for
"à [laughter] thì", this locates real word-level timestamps for "à" and "thì" against the
actual audio and inserts the vocalisation exactly between them.

Stage order, first success wins (see configs/default.yaml: alignment.*):
  1. qwen3        — Qwen3-ForcedAligner (para_synth.align.qwen3), no dictionary needed
  2. mfa          — Montreal Forced Aligner (para_synth.align.mfa), batch-computed upfront
  3. mms          — torchaudio MMS_FA (para_synth.align.mms)
  4. proportional — word-fraction estimate snapped to nearest silence (para_synth.align.proportional)

A real alignment result (stages 1-3) is used as-is, no silence-snapping, since the tag is a
post-hoc annotation and there's no guarantee of a genuine pause there. Only the proportional
fallback (stage 4), which has no acoustic grounding at all, benefits from snapping to a
nearby real quiet spot.
"""
from __future__ import annotations

from pathlib import Path

from para_synth.align.mfa import MFAAligner
from para_synth.align.proportional import estimate_tag_time_proportional, find_local_silence_near
from para_synth.config import AlignmentConfig, ModelsConfig
from para_synth.dataset import ManifestRow

__all__ = ["AlignmentPipeline", "MFAAligner"]


class AlignmentPipeline:
    def __init__(self, alignment_cfg: AlignmentConfig, models_cfg: ModelsConfig):
        self.cfg = alignment_cfg
        self.models_cfg = models_cfg
        self._qwen3 = None
        self._qwen3_failed = False
        self._mms = None
        self._mms_failed = False
        self._mfa_results: dict[str, float | None] = {}

    def prepare_mfa_batch(self, rows: list[ManifestRow], corpus_dir: Path, out_dir: Path) -> None:
        """MFA needs the full batch upfront to build its corpus (one `mfa align` call for
        everything), unlike the other stages which run per-row — so this is a separate
        phase, called once before find_insert_time() for any row."""
        if not self.cfg.use_mfa:
            return
        mfa = MFAAligner(self.cfg)
        if mfa.setup():
            self._mfa_results = mfa.align_batch(rows, corpus_dir, out_dir)

    def _get_qwen3(self):
        if self._qwen3 is None and not self._qwen3_failed and self.cfg.use_qwen3:
            try:
                from para_synth.align.qwen3 import Qwen3Aligner

                self._qwen3 = Qwen3Aligner(self.models_cfg.qwen3_forced_aligner_source())
            except Exception as e:
                print(f"⚠️  Qwen3 aligner unavailable ({type(e).__name__}: {e}) — skipping this stage")
                self._qwen3_failed = True
        return self._qwen3

    def _get_mms(self):
        if self._mms is None and not self._mms_failed:
            try:
                from para_synth.align.mms import MMSAligner

                self._mms = MMSAligner()
            except Exception as e:
                print(f"⚠️  MMS aligner unavailable ({type(e).__name__}: {e}) — skipping this stage")
                self._mms_failed = True
        return self._mms

    def find_insert_time(self, row: ManifestRow, wav, sr, language: str | None = "Vietnamese") -> tuple[float, str]:
        qwen3 = self._get_qwen3()
        if qwen3 is not None:
            t = qwen3.estimate_tag_time(row.audio_filepath, row.text, language=language)
            if t is not None:
                return t, "qwen3"

        t = self._mfa_results.get(row.id)
        if t is not None:
            return t, "mfa"

        mms = self._get_mms()
        if mms is not None:
            t = mms.estimate_tag_time(wav, sr, row.text)
            if t is not None:
                return t, "mms"

        t = estimate_tag_time_proportional(row.text, len(wav) / sr)
        return find_local_silence_near(wav, sr, t), "proportional"

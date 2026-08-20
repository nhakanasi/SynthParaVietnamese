"""Where does the transcript's [tag] actually belong in time?

Diagram: the "Splice" arrow needs a real timestamp, not a silence heuristic — for
"à [laughter] thì", this locates real word-level timestamps for "à" and "thì" against the
actual audio and inserts the vocalisation exactly between them.

Stage order, first success wins (see configs/default.yaml: alignment.*):
  1. qwen3        — Qwen3-ForcedAligner (para_synth.align.qwen3), no dictionary needed
  2. mms          — torchaudio MMS_FA (para_synth.align.mms)
  3. proportional — word-fraction estimate snapped to nearest silence (para_synth.align.proportional)

A real alignment result (stages 1-2) is used as-is, no silence-snapping, since the tag is a
post-hoc annotation and there's no guarantee of a genuine pause there. Only the proportional
fallback (stage 3), which has no acoustic grounding at all, benefits from snapping to a
nearby real quiet spot.

Montreal Forced Aligner used to sit between qwen3 and mms; it was removed. It needed a conda
env with Kaldi that nothing else in this repo uses, its `vietnamese_mfa` dictionary marks the
colloquial Vietnamese particles this pipeline splices around as out-of-vocabulary (`spn`) —
the exact failure the source notebook's cell 22 was diagnosing — and qwen3 resolved every row
of every real batch, so the stage never actually ran. MMS_FA stays as the acoustic backstop:
it ships inside torchaudio and costs nothing to keep.
"""
from __future__ import annotations

from para_synth.align.proportional import estimate_tag_time_proportional, find_local_silence_near
from para_synth.config import AlignmentConfig, ModelsConfig
from para_synth.dataset import ManifestRow

__all__ = ["AlignmentPipeline"]


class AlignmentPipeline:
    def __init__(self, alignment_cfg: AlignmentConfig, models_cfg: ModelsConfig):
        self.cfg = alignment_cfg
        self.models_cfg = models_cfg
        self._qwen3 = None
        self._qwen3_failed = False
        self._mms = None
        self._mms_failed = False

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

        mms = self._get_mms()
        if mms is not None:
            t = mms.estimate_tag_time(wav, sr, row.text)
            if t is not None:
                return t, "mms"

        t = estimate_tag_time_proportional(row.text, len(wav) / sr)
        return find_local_silence_near(wav, sr, t), "proportional"

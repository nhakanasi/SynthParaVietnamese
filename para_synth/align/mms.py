"""torchaudio MMS_FA forced alignment — the acoustic backstop in the alignment chain (after
qwen3), ahead of the proportional last resort. Ported from notebook45ee5319ae.ipynb
cell 8's "stage 1".

Kept as a dependency-light fallback for when Qwen3-ForcedAligner isn't set up or fails on a
row: torchaudio ships MMS_FA, so this needs no extra download beyond what seed-vc's own
requirements.txt already pulls in — which is also why it survived the removal of the
Montreal Forced Aligner stage (see docs/PIPELINE.md: "Alignment stage order").
"""
from __future__ import annotations

import numpy as np
import torch

from para_synth.align._text import deaccent_vi, words_around_tag
from para_synth.audio_utils import _resample


class MMSAligner:
    def __init__(self):
        import torchaudio
        from torchaudio.pipelines import MMS_FA as fa_bundle

        self.model = fa_bundle.get_model(with_star=False)
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.tokenizer = fa_bundle.get_tokenizer()
        self.aligner = fa_bundle.get_aligner()
        self.sr = fa_bundle.sample_rate
        self.device = next(self.model.parameters()).device

        try:
            import uroman as ur

            self._uroman = ur.Uroman()
            print("   ✅ uroman loaded for Vietnamese romanization")
        except Exception as e:
            self._uroman = None
            print(f"   ⓘ uroman unavailable ({type(e).__name__}) — using manual de-accenting instead")

    def _romanize_word(self, w: str) -> str:
        if self._uroman is not None:
            return self._uroman.romanize_string(w).lower().strip()
        return deaccent_vi(w).lower().strip()

    @torch.no_grad()
    def estimate_tag_time(self, wav, sr, text: str) -> float | None:
        """Force-align the real words around [tag] against `wav`; return the time
        boundary between the word groups right before/after it, or None if the tag has
        nothing to bracket or alignment fails."""
        before_words, after_words = words_around_tag(text)
        if before_words is None:
            return None

        try:
            wav_fa = _resample(wav, sr, self.sr) if sr != self.sr else wav
            waveform = torch.from_numpy(np.ascontiguousarray(wav_fa)).float().unsqueeze(0).to(self.device)
            emission, _ = self.model(waveform)

            romanized = [self._romanize_word(w) for w in (before_words + after_words)]
            romanized = [w for w in romanized if w]
            tokens = self.tokenizer(romanized)
            token_spans = self.aligner(emission[0], tokens)

            n_before = len(before_words)
            if n_before >= len(token_spans):
                return None
            boundary_frame = token_spans[n_before - 1][-1].end
            frames_per_s = emission.size(1) / (waveform.size(1) / self.sr)
            return boundary_frame / frames_per_s
        except Exception as e:
            print(f"      ⚠️  MMS forced-alignment failed for this row ({type(e).__name__}: {e})")
            return None

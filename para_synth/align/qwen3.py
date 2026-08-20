"""Qwen3-ForcedAligner backend — first choice in the alignment chain.

New (not in the source notebook, which only had MMS_FA + MFA + proportional). Needs no
pronunciation dictionary and handles accented/multilingual speech natively (Vietnamese is
one of its supported languages), which sidesteps the "spn" out-of-vocabulary problem that
made the dictionary-based Montreal Forced Aligner stage unusable for colloquial Vietnamese
fillers/particles — and, since this stage resolves every row in practice, is why that stage
was removed outright (see docs/PIPELINE.md: "Alignment stage order"). Model: Qwen/Qwen3-ForcedAligner-0.6B — see
third_party/models/README.md for where the offline weights live.

Requires the `qwen-asr` package (`pip install qwen-asr`).
"""
from __future__ import annotations

from pathlib import Path

from para_synth.align._text import words_around_tag


class Qwen3Aligner:
    def __init__(self, model_source: str, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from qwen_asr import Qwen3ForcedAligner

        torch_dtype = getattr(torch, dtype)
        device = device if torch.cuda.is_available() else "cpu"
        print(f"🧠 Loading Qwen3-ForcedAligner from {model_source} ({device}, {dtype}) …")
        self.model = Qwen3ForcedAligner.from_pretrained(
            model_source, device_map=device, dtype=torch_dtype
        )

    def estimate_tag_time(self, audio_path: str | Path, text: str, language: str | None = "Vietnamese") -> float | None:
        """Align only the real words around [tag] (not the bracket annotation itself)
        against the audio at `audio_path`, and return the boundary between the word groups
        right before/after it — same "before/after word groups" contract as
        align.mms, so the backends are interchangeable in align.find_insert_time().

        Takes a file path rather than a pre-loaded (wav, sr) array: the documented
        qwen_asr API accepts a path/URL/base64/ndarray, but only a path lets the model's
        own preprocessing pick the right resampling — an already-decoded array carries no
        sample rate the API can recover, so passing one risks silently misinterpreting the
        audio's duration.
        """
        before_words, after_words = words_around_tag(text)
        if before_words is None:
            return None

        try:
            transcript = " ".join(before_words + after_words)
            results = self.model.align(audio=str(audio_path), text=transcript, language=language)
            items = results[0].items if hasattr(results[0], "items") else results[0]

            n_before = len(before_words)
            if n_before >= len(items):
                return None
            return items[n_before - 1].end_time
        except Exception as e:
            print(f"      ⚠️  Qwen3 forced-alignment failed for this row ({type(e).__name__}: {e})")
            return None

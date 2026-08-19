"""Speaker-similarity quality check: does the Seed-VC output actually sound like the
original speaker? Ported from notebook45ee5319ae.ipynb cell 17.

Seed-VC's job in this pipeline is to re-voice a VocalSound clip into each speaker's own
timbre. This loads a speaker-verification model (`microsoft/wavlm-base-plus-sv`, pure
transformers/torch — no librosa, so it doesn't hit the numpy-version landmine noted in
audio_utils.py) and scores each conversion with cosine similarity between x-vector
embeddings:

* `sim_converted` — original speech vs. the Seed-VC output. High = the conversion actually
  sounds like that speaker.
* `sim_raw_baseline` — original speech vs. the *unconverted* VocalSound clip. Should score
  noticeably lower; if `sim_converted` isn't clearly above it, Seed-VC didn't do its job.

This is a relative signal for triaging conversions, not a certified same/different-speaker
verdict — treat a low score as "listen to this one," not an automatic reject.

third_party/models/speaker-id/ is reserved for this model's offline cache (see
third_party/models/README.md) and for a future standalone speaker-ID extraction step,
separate from this pairwise similarity check.
"""
from __future__ import annotations

import os
import sys

import torch

from para_synth.audio_utils import _resample


def _quiet_transformers_tf_probe() -> None:
    """`Wav2Vec2FeatureExtractor.pad()` can probe is_tf_tensor(), which imports
    tensorflow — and many environments' tensorflow needs a newer protobuf than seed-vc's
    own pin, so that import can die with "cannot import name 'runtime_version' from
    'google.protobuf'". USE_TF=0 makes transformers skip the tensorflow probe entirely.
    Must be set before the first `import transformers` in this process (seed-vc's own
    inference.py runs as a separate subprocess and isn't affected either way)."""
    os.environ["USE_FLAX"] = "0"
    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"
    for mod in [m for m in sys.modules if m.split(".")[0] in ("transformers", "jax", "jaxlib", "flax", "tensorflow")]:
        del sys.modules[mod]


class SpeakerSimilarity:
    def __init__(self, model_source: str = "microsoft/wavlm-base-plus-sv"):
        _quiet_transformers_tf_probe()
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        print(f"🧠 Loading {model_source} for speaker-similarity scoring …")
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_source)
        self.model = WavLMForXVector.from_pretrained(model_source).eval()
        self.sr = self.extractor.sampling_rate  # 16000

    @torch.no_grad()
    def embedding(self, wav, sr):
        if sr != self.sr:
            wav = _resample(wav, sr, self.sr)
        inputs = self.extractor([wav], sampling_rate=self.sr, return_tensors="pt")
        return self.model(**inputs).embeddings[0]

    def similarity(self, wav_a, sr_a, wav_b, sr_b) -> float:
        """Cosine similarity between x-vector embeddings — a relative quality signal, not
        a calibrated same/different-speaker verdict."""
        ea = self.embedding(wav_a, sr_a)
        eb = self.embedding(wav_b, sr_b)
        return torch.nn.functional.cosine_similarity(ea, eb, dim=0).item()

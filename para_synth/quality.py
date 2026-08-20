"""Speaker-similarity quality check: does the Seed-VC output actually sound like the
original speaker? Ported from notebook45ee5319ae.ipynb cell 17.

Seed-VC's job in this pipeline is to re-voice a VocalSound clip into each speaker's own
timbre. This loads a speaker-verification model and scores each conversion with cosine
similarity between speaker embeddings. Two backends, chosen by `quality.speaker_embedder`:

* `campplus` — the *same* CAM++ speaker encoder Seed-VC v1's `inference.py` uses to build
  the style vector it conditions the conversion on (`funasr/campplus`,
  `campplus_cn_common.bin`, via seed-vc's vendored `modules.campplus.DTDNN`). Scoring in
  the embedding space the model actually cloned in answers "did the conversion land where
  it was aimed", which is the question this check exists to ask.
* `wavlm` — `microsoft/wavlm-base-plus-sv`, an independent verifier. Needs no seed-vc
  checkout and is the fallback when one isn't available, but it judges the conversion from
  outside the space it was optimised for, so a conversion can be a faithful clone by CAM++
  and still score modestly here.

Both are pure transformers/torch — no librosa, so neither hits the numpy-version landmine
noted in audio_utils.py. Both report:

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
from pathlib import Path

import numpy as np
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


class CampPlusSimilarity:
    """CAM++ speaker embeddings, loaded from seed-vc's own vendored implementation.

    Deliberately imports `modules.campplus.DTDNN` out of the seed-vc checkout rather than
    reimplementing or pip-installing another copy: the point of this backend is that the
    embedding is bit-for-bit the one `inference.py` conditioned the conversion on, and a
    second implementation (different fbank defaults, different checkpoint revision) would
    quietly stop being that. The cost is that this backend needs `seedvc.repo_dir` present
    — hence `wavlm` staying available as the standalone option.

    Feature extraction mirrors inference.py exactly: 80-bin kaldi fbank at 16k, dither=0,
    mean-normalised over time.
    """

    sr = 16000

    def __init__(self, seedvc_repo_dir):
        import torch
        import torchaudio  # noqa: F401  — imported here so a missing torchaudio fails loudly

        repo = str(Path(seedvc_repo_dir))
        if not (Path(repo) / "modules" / "campplus" / "DTDNN.py").is_file():
            raise FileNotFoundError(
                f"CAM++ speaker embedding needs seed-vc's vendored modules, not found under "
                f"{repo}. Run `para-synth setup-seedvc`, or set quality.speaker_embedder: wavlm."
            )
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from modules.campplus.DTDNN import CAMPPlus

        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download("funasr/campplus", "campplus_cn_common.bin")
        print(f"🧠 Loading CAM++ ({ckpt}) for speaker-similarity scoring …")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = self.model.eval().to(self.device)

    @torch.no_grad()
    def embedding(self, wav, sr):
        import torchaudio

        if sr != self.sr:
            wav = _resample(wav, sr, self.sr)
        x = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0)
        feat = torchaudio.compliance.kaldi.fbank(
            x, num_mel_bins=80, dither=0, sample_frequency=self.sr
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        return self.model(feat.unsqueeze(0).to(self.device))[0]

    def similarity(self, wav_a, sr_a, wav_b, sr_b) -> float:
        ea = self.embedding(wav_a, sr_a)
        eb = self.embedding(wav_b, sr_b)
        return torch.nn.functional.cosine_similarity(ea, eb, dim=0).item()


def build_speaker_similarity(cfg):
    """Pick the speaker-embedding backend named by `quality.speaker_embedder`."""
    backend = cfg.quality.speaker_embedder
    if backend == "campplus":
        return CampPlusSimilarity(cfg.seedvc.repo_dir)
    if backend == "wavlm":
        return SpeakerSimilarity(cfg.models.speaker_id_source())
    raise ValueError(
        f"quality.speaker_embedder must be 'campplus' or 'wavlm', got {backend!r}"
    )

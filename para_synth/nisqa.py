"""NISQA v2.0 speech-quality scoring — the model behind the pipeline's `filter` stage.

`quality.py` answers "does the converted event sound like this speaker". Nothing answered
"does the finished Para recording sound acceptable at all" — hiss picked up from the
VocalSound clip's microphone, an audible step at a splice junction, or the timbre shift
Seed-VC leaves behind are all invisible to a speaker-similarity cosine. NISQA is a
no-reference (single-ended) predictor: it takes one recording and predicts what a listening
panel would have scored it, with no clean reference signal to compare against — which is the
only kind of measure available here, since a Para recording has no "correct" version.

It returns five values in this order (torchmetrics' order, which follows the original code
rather than the paper's text — the paper lists coloration before discontinuity):

* `mos`           — overall quality, 1-5.
* `noisiness`     — hiss/background noise. The dimension a mismatched VocalSound microphone
                    moves; `selection.clarity_weight` is what tries to prevent it upstream.
* `discontinuity` — isolated interruptions/artefacts. The dimension a bad splice junction
                    moves, and the most specific signal in here for this pipeline.
* `coloration`    — timbral distortion/band-limiting, i.e. the Seed-VC conversion's own
                    footprint.
* `loudness`      — level problems.

All five are "higher is better" (5 = no audible degradation of that kind), so every
threshold in `NisqaConfig` is a floor.

**Licence**: torchmetrics downloads the upstream NISQA weights (`nisqa.tar` from
github.com/gabrielmittag/NISQA), which are CC BY-NC-SA 4.0 — **non-commercial**. The
torchmetrics code is Apache-2.0, but the weights are not: this stage is fine for research
and internal dataset triage and must not ship inside a commercial product. Set
`quality.nisqa.enabled: false` to run the pipeline without ever downloading them.

Runs on CPU regardless of GPU: torchmetrics' NISQA computes its mel spectrogram with librosa
and never moves the model off CPU, so there is nothing to configure. It costs roughly a
second per recording — three orders of magnitude below Seed-VC's diffusion, which is why the
filter is worth having as its own resumable stage rather than folded into synthesis.
"""
from __future__ import annotations

import numpy as np

# torchmetrics' output order — see the module docstring.
DIMENSIONS = ("mos", "noisiness", "discontinuity", "coloration", "loudness")

# The subset `failed_criteria` thresholds. `loudness` is scored and cached like the rest but
# never judged: synthesize_row() peak-normalises every finished recording, so its loudness
# score describes that normalisation rather than anything the pipeline did to the speech.
CRITERION_DIMENSIONS = ("mos", "noisiness", "discontinuity", "coloration")

_MISSING_DEP_HINT = (
    "The NISQA quality filter needs `torchmetrics`, `librosa` and `requests`. Install them "
    'with `pip install "torchmetrics>=1.5" librosa requests` (or `pip install -e ".[nisqa]"`), '
    "or set quality.nisqa.enabled: false in your config to skip the filter stage."
)


class NisqaScorer:
    """One loaded NISQA model, reused across a batch.

    Heavy imports happen in __init__ rather than at module import, so `para-synth run` with
    the filter disabled never pulls torchmetrics in — the same lazy pattern
    `quality.CampPlusSimilarity` uses, and for the same reason: an optional dependency
    shouldn't be able to break an unrelated stage.
    """

    def __init__(self) -> None:
        try:
            import torch  # noqa: F401
            from torchmetrics.functional.audio.nisqa import (
                non_intrusive_speech_quality_assessment,
            )
        except ImportError as e:  # noqa: BLE001 — re-raised with the config key to flip
            raise ImportError(f"{_MISSING_DEP_HINT} (import failed: {e})") from e

        self._torch = torch
        self._nisqa = non_intrusive_speech_quality_assessment
        print("🧠 Loading NISQA v2.0 (CC BY-NC-SA 4.0 weights) for quality scoring …")
        # Force the download/load now instead of on the first score(), so a missing weight
        # file surfaces before a batch has been half-processed. lru_cache'd upstream, so
        # this is the only load in the process.
        from torchmetrics.functional.audio.nisqa import _load_nisqa_model

        _load_nisqa_model()

    def score(self, wav: np.ndarray, sr: int) -> dict[str, float]:
        """Score one mono recording. Raises RuntimeError (from torchmetrics) if the audio
        is too short to form a single mel window, or too long for the model's segment cap."""
        x = np.asarray(wav, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=tuple(range(1, x.ndim)))
        out = self._nisqa(self._torch.from_numpy(np.ascontiguousarray(x)), int(sr))
        return dict(zip(DIMENSIONS, (float(v) for v in out.tolist())))


def failed_criteria(scores: dict, cfg) -> list[str]:
    """Which of `cfg` (a NisqaConfig)'s thresholds this row fails. Empty list = it passes.

    `scores` is `{"source": {dim: value}, "para": {dim: value}}` — the source recording as
    it arrived and the finished Para recording built from it. Each dimension carries the
    same two criteria: an absolute floor (`min_<dim>`) on the finished recording, and a cap
    on how far it fell below its own source (`max_<dim>_drop`). Anything left at null is not
    checked, so which questions get asked is entirely the config's decision.
    """
    para, source = scores.get("para") or {}, scores.get("source") or {}
    failed: list[str] = []

    for dim in CRITERION_DIMENSIONS:
        value = para.get(dim)
        if value is None:  # this dimension wasn't scored — nothing to judge it against
            continue

        floor = getattr(cfg, f"min_{dim}")
        if floor is not None and value < floor:
            failed.append(f"{dim} {value:.2f} < {floor}")

        max_drop = getattr(cfg, f"max_{dim}_drop")
        if max_drop is not None and source.get(dim) is not None:
            drop = source[dim] - value
            if drop > max_drop:
                failed.append(f"{dim}_drop {drop:.2f} > {max_drop}")

    return failed

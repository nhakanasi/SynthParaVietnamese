# Pipeline deep-dive

This is the detail layer behind README.md: what each diagram node actually does, which
module implements it, where its code came from, and every non-obvious gotcha worth knowing
before you touch it. See `synthesis.drawio.png` for the diagram itself.

## Diagram → code map

```
Non-vocal sound (VocalSound)  ──┐
                                 ├─► SeedVC v1 ──► Splice ──► Para recording ──┐
Speech recording ───────────────┘                                              ├─► new dataset
Transcription ──► Qwen/Gemini (add tag) ──────────────────────► new transcript ┘
                        ▲
     "random matching tag" feedback arrow (Para recording → Non-vocal sound)
```

| Diagram node | Module | Notes |
|---|---|---|
| Non-vocal sound (VocalSound dataset) | `para_synth/vocalsound.py` | download, class taxonomy, tag→class mapping |
| Speech recording | `para_synth/dataset.py` | `data/raw/audio/{id}.wav` |
| Transcription | `para_synth/dataset.py`, `para_synth/asr.py` | `data/raw/transcripts/{id}.txt`; `asr.py` fills gaps for audio-only input |
| Qwen/Gemini (add tag) | `para_synth/tagging.py` | writes `data/tagged/transcripts/{id}.txt` |
| SeedVC v1 | `para_synth/seedvc.py` | re-voices the VocalSound clip into the speaker's timbre |
| (alignment — implicit, needed for "Splice" to be more than a guess) | `para_synth/align/` | qwen3 → mfa → mms → proportional, first success wins |
| Splice | `para_synth/audio_utils.py: splice()` | inserts the re-voiced clip at the aligned timestamp |
| Para recording + new transcript → new dataset | `para_synth/pipeline.py` | writes `metadata_synth.jsonl` |
| "random matching tag" feedback arrow | `para_synth/vocalsound.py: match_vs_class()` | random `VS_CLASSES` pick when a tag can't be matched to a class |

## Notebook provenance

`reference/notebook45ee5319ae.ipynb` is the original Kaggle notebook this repo was built
from. Its own cell-by-cell markdown is worth reading directly for the reasoning behind each
piece; this table is just the map from cell to module:

| Notebook cell(s) | Module |
|---|---|
| 2 (clone seed-vc, pip install) | `seedvc.py: setup_seedvc()` |
| 3 (protobuf/transformers/peft/uroman fixes) | `env_check.py`, gotchas below |
| 5–7 (config, VS_PAT, VS_DIR discovery) | `config.py`, `vocalsound.py` |
| 9 (shared audio utils) | `audio_utils.py` — verbatim port |
| 11–13 (dataset discovery, metadata load, tag extraction) | `dataset.py` — generalized to local `audio/`+`transcripts/` folders instead of one Kaggle `metadata (1).jsonl` |
| 15 (tag → VocalSound class mapping) | `vocalsound.py: match_vs_class()` |
| 17 (WavLM speaker-similarity QC) | `quality.py` |
| 8, 19 (MMS_FA + proportional fallback) | `align/mms.py`, `align/proportional.py` |
| 9, 21 (MFA conda bootstrap) | `align/mfa.py` |
| 10, 23 (MFA batch align + TextGrid parsing) | `align/mfa.py` |
| 25 (pick clip, run seed-vc, splice, write dataset) | `seedvc.py: run_seedvc()`, `pipeline.py` |
| 27 (listen & inspect) | `cli.py: cmd_inspect` |
| *not in the notebook* — the diagram's Qwen/Gemini tagging box | `tagging.py` — new code; the notebook's own cell 11 markdown says tagging happened upstream of it |
| *not in the notebook* — audio-only input | `asr.py` — new code, Qwen3-ASR |
| *not in the notebook* — Qwen3-ForcedAligner | `align/qwen3.py` — new code, added as the preferred alignment stage |
| *not in the notebook* — adaptive splice boundary conditioning | `audio_utils.py: adaptive_splice()` — new code, opt-in via `splice.adaptive` |

## Splicing: fixed vs. adaptive

`splice()` (the notebook's original, still the default) always applies the same fixed
`fade_ms` envelope to the insert and the same fixed `pad_ms` silence on both sides,
regardless of what's actually at the cut point. Since the cut point comes from forced
alignment rather than a silence heuristic, it can land on active phonation (a continuant
running right up to the next word) rather than a natural pause — a fixed small pad doesn't
guarantee a clean edge there.

`adaptive_splice()` (`splice.adaptive: true` in config) inspects each of the *two* junctions
(speech→insert, insert→speech) independently: it measures the boundary's RMS against that
segment's own overall level (`boundary_is_active()`), and only pays for a cosine fade +
a wider `max_gap_ms` bridge where the boundary actually needs it — an already-quiet boundary
gets the same small `min_pad_ms` room-tone bridge `splice()` would apply everywhere. The
bridge itself is low-level matched noise (`matched_room_tone()`), not true digital silence,
since a hard zero between two segments that both have real background noise reads as an
artificial mute. Verified against synthetic quiet-boundary and active-boundary cases (see
git history) — a quiet cut gets the minimal bridge, an active cut gets the full damped
bridge.

Deliberately *not* implemented: full mel-spectrogram + neural-vocoder (e.g. HiFi-GAN)
resynthesis of the stitched waveform. That approach regenerates the *entire* recording's
phase through a vocoder, not just the two junctions — this pipeline's whole purpose is
augmenting real recordings with a real re-voiced event while leaving the rest of the speaker's
audio untouched, so full resynthesis works against that goal (timbre/naturalness drift across
audio that should stay byte-for-byte original) for a problem the two-junction adaptive fade
already solves without an extra model dependency.

## Alignment stage order

`para_synth/align/__init__.py: AlignmentPipeline.find_insert_time()` tries, first success
wins:

1. **qwen3** (`align/qwen3.py`) — Qwen3-ForcedAligner-0.6B. No pronunciation dictionary
   needed, handles Vietnamese natively. Preferred because it sidesteps gotcha #6 below.
2. **mfa** (`align/mfa.py`) — Montreal Forced Aligner + `vietnamese_mfa`. Computed once per
   batch upfront (MFA needs the whole corpus, not per-utterance calls), so
   `AlignmentPipeline.prepare_mfa_batch()` must run before any `find_insert_time()` call.
3. **mms** (`align/mms.py`) — torchaudio's built-in `MMS_FA` bundle. No extra download.
4. **proportional** (`align/proportional.py`) — no acoustic info at all: assumes constant
   speaking rate, places the tag at the same word-fraction through the audio, then snaps to
   the nearest real silence. Last resort only.

A real alignment result (stages 1–3) is used **as-is** — no silence-snapping — since the tag
is a post-hoc LLM annotation and there's no guarantee of a genuine pause where it belongs.
Only stage 4 benefits from snapping to a nearby quiet spot, since it has no acoustic grounding
to begin with.

## Gotchas (from the source notebook — still apply here)

1. **numpy 1.x vs 2.x**: Seed-VC's `requirements.txt` downgrades numpy to 1.x, but many ML
   base images ship a scipy built against numpy 2.x — importing librosa then dies with
   `No module named numpy.strings`. `audio_utils.py` deliberately avoids
   librosa/scipy entirely: `split_nonsilent()` is a from-scratch numpy port of
   `librosa.effects.split`, verified sample-identical.

2. **transformers probing for tensorflow**: `Wav2Vec2FeatureExtractor.pad()` (used inside
   `quality.py`) can probe `is_tf_tensor()`, which imports tensorflow — and tensorflow often
   needs a newer protobuf than Seed-VC's `<4.0.0` pin, crashing with `cannot import name
   'runtime_version' from 'google.protobuf'`. `quality.py` sets `USE_TF=0` (and clears any
   already-imported `transformers`/`tensorflow` modules) before its first `transformers`
   import to skip the probe. Must happen before *any* `import transformers` in that process —
   Seed-VC's own `inference.py` runs as a separate subprocess, so it isn't affected either way.

3. **peft crashing on import even though nothing uses it**: WavLM's TDNN layer
   conditionally imports `peft` at call time via `is_peft_available()`, which can crash on an
   old `peft` even though `quality.py` doesn't use LoRA/peft at all. If speaker-similarity
   scoring fails on import, `pip uninstall peft` — nothing here needs it. `env_check.py`
   flags this.

4. **uroman for Vietnamese romanization**: the MMS_FA fallback aligner (`align/mms.py`) was
   trained on romanized text. `uroman` isn't always installable; `align/_text.py:
   deaccent_vi()` is a dependency-free approximate fallback (NFD-decompose to peel off
   diacritics, handle `đ`/`Đ` by hand since Unicode gives it no canonical decomposition).

5. **MFA needs conda/Kaldi, not just pip**: MFA's Kaldi binaries are only distributed via
   conda-forge — plain `pip install montreal-forced-aligner` can't work. `align/mfa.py:
   MFAAligner.setup()` bootstraps Miniforge if no conda exists at all, creates an `aligner`
   env, and downloads the `vietnamese_mfa` acoustic model + dictionary. One-time,
   multi-minute cost; non-fatal on failure (falls through to the next alignment stage).

6. **MFA's `spn` (out-of-vocabulary) placeholder**: MFA marks a word `spn` ("speech noise")
   when it isn't in `vietnamese_mfa`'s dictionary — common for Vietnamese colloquial
   fillers/particles (à, ha, thì, …). An OOV word still gets *an* interval, so a naive word-
   count check won't catch it, but its timing is a rough guess, not real alignment.
   `align/mfa.py` rejects any boundary touching an `spn` word and falls through to the next
   stage instead of trusting it. This is the exact class of bug the source notebook's cell 22
   was diagnosing (splices landing mid-word); Qwen3-ForcedAligner (stage 1) avoids the whole
   problem by not needing a dictionary in the first place.

7. **`mfa` must run via `conda run -n <env>`, not a bare absolute path**: `mfa` shells out
   internally to other conda-installed tools (openfst's `fstcompile`, for alignment
   specifically) via `PATH` lookup. A bare absolute-path call never adds the env's `bin/` to
   `PATH`, and fails with `ThirdpartyError: Could not find 'fstcompile'` even though `mfa`
   itself ran fine.

8. **Seed-VC output detection uses mtime, not a before/after set-difference**: Seed-VC's
   output filenames are deterministic (`vc_{source_stem}_{target_stem}_..._.wav`). If a
   same-named file from an earlier run already exists in the output dir, a naive
   before/after set-difference can't see this run's (correctly rewritten) file and reports
   "no output" even though Seed-VC succeeded. `seedvc.py: run_seedvc()` instead looks for
   files with `mtime >= call_start_time`.

9. **"without alter the meaning" is checked mechanically, not trusted**: `tagging.py` verifies
   an LLM's tagged output has *exactly one* bracketed tag, that the tag is in `VS_CLASSES`,
   and that stripping the tag back out reproduces the original transcript exactly — retrying
   on any of those three failing rather than accepting a paraphrased or otherwise-altered
   transcript.

## Model taxonomy

See `third_party/models/README.md` — weights are organized by pipeline purpose (asr,
aligner, conversion, speaker-id, splicing), not by vendor.

## Deliberately not implemented

A few DSP/selection ideas were considered and rejected — noted here so they don't get
re-proposed or "fixed in" without re-litigating the reasoning:

- **Mel-spectrogram + neural-vocoder (HiFi-GAN) resynthesis of the stitched waveform** at
  the splice step — see "Splicing: fixed vs. adaptive" above. Regenerates the entire
  recording's phase, not just the two junctions; works against the goal of leaving the
  speaker's real audio untouched outside the inserted event.
- **Time-stretching the VocalSound clip to match the speaker's speech tempo** — cough,
  sneeze, throat-clearing, and sniff are reflexive/mechanical events whose duration is set
  by airway mechanics, not speaking rate, so there's no real perceptual link to stretch
  toward. They're also broadband/transient/noise-like, and phase-vocoder/WSOLA time-stretch
  algorithms (built for quasi-periodic content) tend to smear or add metallic artifacts on
  that kind of signal. If a class's duration ever needs adjusting, `seedvc.length_adjust`
  (already wired through `run_seedvc()`) lets Seed-VC's own conversion generate a
  naturally longer/shorter take instead of stretching a fixed recording after the fact.
- **Hard-filtering VocalSound clips by the speaker's vocal register** (e.g. a soft-spoken
  speaker may *only* receive soft laughs) — a real person's laugh/cough doesn't reliably
  scale with how loud or soft they speak, so a hard filter encodes a stereotype that's
  often false and deletes exactly the acoustic variance a downstream Para-TTS model
  benefits from seeing. The *soft* version of this idea is implemented instead — see
  "Clip selection" below.

## Tagging: text-only vs. audio-conditioned

`tagging.py` has two kinds of backend for the diagram's "Qwen/Gemini" box:

- **Text-only** (`qwen`, `gemini`) — the default. Sends just the transcript; the model
  infers which paralinguistic event fits from wording alone.
- **Audio-conditioned** (`qwen_omni_audio`) — opt-in. Also sends the recording, so tag
  choice *and* placement can follow the actual delivery: a sigh after a slow, breathy
  phrase, positioned where the speaker really draws breath rather than wherever the text
  suggests. Uses `qwen3-omni-flash` over DashScope's OpenAI-compatible endpoint.

Two DashScope quirks the audio path has to honour, both required for the Omni models:
the request **must be streamed** (a non-streamed call is rejected), and `modalities=["text"]`
stops the model also synthesizing a spoken reply that would just be discarded. There's also
a hard 150s audio cap on `qwen3-omni-flash`, guarded locally (`QWEN_OMNI_MAX_AUDIO_S`) so an
over-long file fails with a clear message instead of an opaque API error.

Audio conditioning costs meaningfully more per row than the text path, which is why it is
not the default. Gemini's audio input is deliberately *not* wired up: Google's audio docs
state Gemini "can only infer responses to English-language speech," which was not verified
either way for Vietnamese — so only the Qwen path is implemented for now.

Whichever backend runs, the output goes through the same mechanical check
(`_extract_single_tag`): exactly one bracketed tag, drawn from `VS_CLASSES`, and stripping
it back out must reproduce the input transcript exactly. Audio conditioning changes *which*
tag is chosen, never the guarantee that nothing else was altered.

## Clip selection: energy-weighted, not filtered

`pick_vocalsound_clip()` (`pipeline.py`) biases — never restricts — which clip a row gets.
With `selection.energy_weight: 0` it's uniform random over the matched class. Above zero,
it loads a random candidate subset, ranks each candidate's intensity within that subset,
and samples with weight `exp(-energy_weight * |clip_rank - speaker_score|)`. Closer
matches get more probability; every usable candidate keeps nonzero probability, so a big
laugh from a soft-spoken speaker still occurs — just less often than uniform sampling
would produce it.

Both sides of that comparison are **relative, self-calibrating measures**, which matters:

- `clip_features()` returns raw crest factor and duration, *not* absolute loudness —
  clips get peak-normalised by `_finalize_clip()` and re-levelled by `splice()` anyway, so
  raw amplitude only tells you about VocalSound's crowdsourced recording gain. Those raw
  features are converted to percentile ranks *within the candidate pool*.
- `speaker_energy_score()` is the percentile rank of the pre-splice window's RMS among all
  same-length windows in that utterance — i.e. "is this a loud or quiet moment *for this
  speaker*", which is invariant to mic and recording gain.

The percentile framing is load-bearing, not stylistic. An earlier version of this used
fixed reference constants (crest ÷ 0.5, duration ÷ 2s) and was badly miscalibrated against
real VocalSound audio: measured crest factors cluster near 0.11 and the median clip runs
~2.7s, saturating the duration term, so nearly every clip scored ~0.58 and weighting had
no signal to act on — measured mean picked intensity was identical (0.568 vs 0.563) for
soft vs. loud speakers. With percentile ranks the same test tracks the speaker properly
(0.41 / 0.48 / 0.58 mean rank for speaker scores 0.15 / 0.50 / 0.85) while picks still
span nearly the full intensity range. If you change these features, re-run that check
rather than assuming a new proxy has usable spread.

Note that `splice.para_gain_db` and `seedvc.length_adjust` are *not* substitutes here:
gain only rescales the mix level and Seed-VC converts timbre, so neither turns an
acoustically boisterous laugh into a gentle one. Selection is the layer where that
distinction is actually made.

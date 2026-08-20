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
| (which clip of that class — implicit) | `para_synth/selection.py` | matches the axes Seed-VC does *not* regenerate; see "Clip selection" below |
| Speech recording | `para_synth/dataset.py` | `data/raw/audio/{id}.wav`, or a row of the JSONL manifest `paths.manifest` names |
| Transcription | `para_synth/dataset.py`, `para_synth/asr.py` | `data/raw/transcripts/{id}.txt`; `asr.py` fills gaps for audio-only input |
| Qwen/Gemini (add tag) | `para_synth/tagging.py` | writes `data/tagged/transcripts/{id}.txt` |
| SeedVC v1 | `para_synth/seedvc.py` | re-voices the VocalSound clip into the speaker's timbre |
| (alignment — implicit, needed for "Splice" to be more than a guess) | `para_synth/align/` | qwen3 → mms → proportional, first success wins |
| Splice | `para_synth/audio_utils.py: splice()` | inserts the re-voiced clip at the aligned timestamp |
| (quality filtering — implicit) | `para_synth/nisqa.py`, `audio_utils.py: splice_boundary_activity()` | NISQA MOS on the finished recording, plus how far into speech the event was inserted; see "Quality filtering" below |
| Para recording + new transcript → new dataset | `para_synth/pipeline.py` | writes `metadata_synth.jsonl`, then `metadata_filtered.jsonl` |
| "random matching tag" feedback arrow | `para_synth/vocalsound.py: match_vs_class()` | random `VS_CLASSES` pick when a tag can't be matched to a class |

## Staged execution

`para-synth run` is a chain of independently runnable stages, not one monolithic pass. Each
stage persists what it produced, keyed by row id, and on the next invocation skips rows that
artifact already covers — so re-tuning something cheap never re-pays for something expensive
upstream. That matters here because the stages differ in cost by orders of magnitude:
alignment is one forward pass per row, synthesis is 50 diffusion steps per row (~30s on a
5070 Ti, ~9 min on CPU), and filtering is under a second per row.

| stage | command | reads | writes |
|---|---|---|---|
| asr *(optional)* | `para-synth transcribe` | `data/raw/audio/` | `data/raw/transcripts/{id}.txt` |
| tag | `para-synth tag-transcripts` | raw transcripts | `data/tagged/transcripts/{id}.txt` |
| align | `para-synth align` | tagged manifest | `data/work/stages/align.jsonl` — `{id, insert_at_s, insert_stage}` |
| synth | `para-synth synth` | `align.jsonl` | `para_{id}.wav` + `metadata_synth.jsonl` |
| filter | `para-synth filter` | `metadata_synth.jsonl` | `stages/quality.jsonl` + `metadata_filtered.jsonl` |
| export | `para-synth export` | `metadata_filtered.jsonl` | `manifest.jsonl` — the passing rows in the caller's own JSONL shape |

`para-synth run` runs align → synth → filter, skipping filter when it has no criteria
configured at all (`quality.nisqa.enabled: false` *and* `quality.max_boundary_activity:
null`). The first two stages take `--limit N`; all of them take `--force` to redo rows
they've already done.

Resume is keyed on evidence, not just on a row id being present:

- **synth** treats a row as done only when its metadata entry *and* its `para_*.wav` both
  exist, so deleting an output file is enough to make it rebuild exactly that row.
- **filter** caches its two measures against different evidence, because they depend on
  different things. The NISQA scores are stored against the size+mtime of the file they
  scored, so a row that `synth --force` re-rendered gets re-scored while everything else
  stays cached. (Seed-VC's diffusion is stochastic and this repo doesn't seed its
  subprocess, so a forced re-render really is a different take, not a byte-identical one.)
  The splice-boundary measure is stored against the row's `splice_at_s`, since that and the
  source recording are all it depends on.
- Stage artifacts accumulate rather than being truncated: running a stage with `--limit 5`
  after a full run leaves the other 15 rows' entries in place.

The tagged transcripts and the raw audio are the pipeline's inputs, so `align` and `synth`
both re-read the manifest; nothing caches the manifest itself.

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
| 11–13 (dataset discovery, metadata load, tag extraction) | `dataset.py` — generalized to local `audio/`+`transcripts/` folders, with the notebook's `metadata (1).jsonl` shape kept as an alternative input (`paths.manifest`) and as the export format |
| 15 (tag → VocalSound class mapping) | `vocalsound.py: match_vs_class()` |
| 17 (speaker-similarity QC) | `quality.py` — WavLM in the notebook; CAM++ added, see below |
| 8, 19 (MMS_FA + proportional fallback) | `align/mms.py`, `align/proportional.py` |
| 9-10, 21, 23 (MFA conda bootstrap, batch align, TextGrid parsing) | *not ported — see "Alignment stage order"* |
| 25 (pick clip, run seed-vc, splice, write dataset) | `seedvc.py: run_seedvc()`, `pipeline.py` |
| 27 (listen & inspect) | `cli.py: cmd_inspect` |
| *not in the notebook* — the diagram's Qwen/Gemini tagging box | `tagging.py` — new code; the notebook's own cell 11 markdown says tagging happened upstream of it |
| *not in the notebook* — audio-only input | `asr.py` — new code, Qwen3-ASR |
| *not in the notebook* — Qwen3-ForcedAligner | `align/qwen3.py` — new code, added as the preferred alignment stage |
| *not in the notebook* — adaptive splice boundary conditioning | `audio_utils.py: adaptive_splice()` — new code, `splice.mode: adaptive` |
| *not in the notebook* — tempo-matched splice | `audio_utils.py: tempo_splice()` — new code, `splice.mode: tempo` (the default) |

## Splicing: fixed vs. adaptive vs. tempo

Selected by `splice.mode` in config. **`tempo` is the default** — see the measurements below
for why the other two aren't.

`splice()` (`mode: fixed`, the notebook's original) always applies the same fixed `fade_ms`
envelope to the insert and the same fixed `pad_ms` silence on both sides, regardless of
what's actually at the cut point. Two things went wrong with it on the first real batch:

- It **never fades the speech at all** — it concatenates `speech[:cut]` directly. Since the
  cut comes from forced alignment rather than a silence heuristic, it lands mid-phonation:
  measured at **20/20 rows** of the first batch, every one a hard waveform discontinuity.
- Its `fade_ms` envelope applies only to the insert, and the insert arrives from Seed-VC
  with **~140ms of leading/trailing silence** (`_finalize_clip` zero-pads short clips to
  ≥1s for Seed-VC's chunker, and Seed-VC returns that padding). So the fade was spent
  entirely on silence while the event still began at full amplitude. `pipeline.py` now
  re-trims the *converted* audio (`trim_event(..., pad_s=0.0)`) before any mode splices it —
  the silence around an event has to be one the splice chose, not one inherited from the
  chunker.

`adaptive_splice()` (`mode: adaptive`) inspects each of the *two* junctions (speech→insert,
insert→speech) independently: it measures the boundary's RMS against that segment's own
overall level (`boundary_is_active()`), and only pays for a cosine fade + a wider
`max_gap_ms` bridge where the boundary actually needs it. Sound in principle, but the
conditional turned out to buy nothing in practice — the boundary is active essentially
always, so the test only adds a way to be wrong.

`tempo_splice()` (`mode: tempo`, the default) drops the conditional and fixes the gap
length, which neither other mode addresses:

- **Exponential fades at both junctions**, unconditionally — speech faded out into the
  event, event faded in, and symmetrically at the resume (cutting mid-word means the
  resumed speech also *starts* mid-word). Exponential rather than cosine because it damps
  fast and, mirrored, reaches 50% in the first 14% of the fade — preserving the attack of
  laughter/cough/sneeze, which a symmetric fade audibly muffles.
- **A gap sized to the speaker's own tempo.** `speech_pause_s()` takes the median gap
  between voiced runs in that utterance. Measured across the first batch these span
  **40–400ms per speaker** (median ~130ms), against the fixed `pad_ms: 10` — shorter than
  any real pause, so an event dropped in with one reads as spliced no matter how clean the
  fades are. Self-calibrating for the same reason `speaker_energy_score` is, and clamped so
  a pause-free recording can't collapse the gap and one dramatic silence can't stretch it.

Measured A/B over all 20 rows (max sample-to-sample step at the junction, lower = less
audible as a cut): mean **0.04065 → 0.00067, a 61x reduction**, improving on every row.

Both bridges are `matched_room_tone()` — low-level noise at the recording's own floor
(`quietest_window()`), not true digital silence, since a hard zero between two segments that
both have real background noise reads as an artificial mute.

Deliberately *not* implemented: full mel-spectrogram + neural-vocoder (e.g. HiFi-GAN)
resynthesis of the stitched waveform. That approach regenerates the *entire* recording's
phase through a vocoder, not just the two junctions — this pipeline's whole purpose is
augmenting real recordings with a real re-voiced event while leaving the rest of the speaker's
audio untouched, so full resynthesis works against that goal (timbre/naturalness drift across
audio that should stay byte-for-byte original) for a problem the two-junction fade +
tempo-matched gap already solves without an extra model dependency.

## Alignment stage order

`para_synth/align/__init__.py: AlignmentPipeline.find_insert_time()` tries, first success
wins:

1. **qwen3** (`align/qwen3.py`) — Qwen3-ForcedAligner-0.6B. No pronunciation dictionary
   needed, handles Vietnamese natively. Resolves every row in practice (20/20 on each real
   batch run so far), so the stages below are backstops, not a rotation.
2. **mms** (`align/mms.py`) — torchaudio's built-in `MMS_FA` bundle. No extra download.
3. **proportional** (`align/proportional.py`) — no acoustic info at all: assumes constant
   speaking rate, places the tag at the same word-fraction through the audio, then snaps to
   the nearest real silence. Last resort only.

A real alignment result (stages 1–2) is used **as-is** — no silence-snapping — since the tag
is a post-hoc LLM annotation and there's no guarantee of a genuine pause where it belongs.
Only stage 3 benefits from snapping to a nearby quiet spot, since it has no acoustic grounding
to begin with.

### Why Montreal Forced Aligner was removed

MFA sat between qwen3 and mms (ported from notebook cells 9-10, 21, 23) and was deleted along
with `align/mfa.py`, `para-synth setup-mfa`, `prepare.sh --with-mfa`, and the
`alignment.use_mfa` / `mfa_*` / `conda_dir` config keys. Three reasons, in order of weight:

1. **It never ran.** qwen3 is tried first and returned a timestamp for every row of every
   real batch, so MFA's result was computed and then discarded — while still paying its full
   setup cost up front, since `prepare_mfa_batch()` had to run before the first row.
2. **Its dictionary is wrong for this task specifically.** `vietnamese_mfa` marks
   out-of-vocabulary words `spn` ("speech noise"), which covers exactly the colloquial
   fillers and particles (à, ha, thì, …) that a paralinguistic tag gets spliced next to. An
   OOV word still gets *an* interval, so a naive word-count check won't catch it, but its
   timing is a guess rather than an alignment — this is the bug the source notebook's cell 22
   was diagnosing (splices landing mid-word). `align/mfa.py` handled it by rejecting any
   boundary touching an `spn` word and falling through, i.e. by not being used.
3. **It was the single heaviest dependency in the repo.** MFA's Kaldi binaries only ship via
   conda-forge, so `MFAAligner.setup()` had to bootstrap Miniforge (a `curl` of a *Linux*
   installer, which 404s on Windows on every single run), create a separate `aligner` conda
   env, and download an acoustic model plus dictionary — multi-minute, and nothing else in
   the pipeline uses conda at all.

MMS_FA stays as the acoustic backstop: it ships inside torchaudio, needs no dictionary, and
costs nothing to keep. If qwen3 ever has to be dropped, `align/mfa.py` is recoverable from
git history — but re-read point 2 before reviving it.


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

5. **Seed-VC output detection uses mtime, not a before/after set-difference**: Seed-VC's
   output filenames are deterministic (`vc_{source_stem}_{target_stem}_..._.wav`). If a
   same-named file from an earlier run already exists in the output dir, a naive
   before/after set-difference can't see this run's (correctly rewritten) file and reports
   "no output" even though Seed-VC succeeded. `seedvc.py: run_seedvc()` instead looks for
   files with `mtime >= call_start_time`.

6. **"without alter the meaning" is checked mechanically, not trusted**: `tagging.py` verifies
   an LLM's tagged output has *exactly one* bracketed tag, that the tag is in `VS_CLASSES`,
   and that stripping the tag back out reproduces the original transcript exactly — retrying
   on any of those three failing rather than accepting a paraphrased or otherwise-altered
   transcript.

7. **The three pip installs must land in this order: para-synth, Seed-VC, `qwen-asr`.**
   Every one of them pins `transformers` and pip is last-write-wins. `qwen-asr` wants
   `4.57.6` and actually needs it — Qwen3-ForcedAligner-0.6B's `config.json` declares
   `model_type: qwen3_asr`, which older releases can't build (`from_pretrained` fails
   outright, so the align chain silently drops to MMS). Seed-VC works on 4.57.6, so the
   aligner's newer pin wins. `scripts/prepare.sh` steps 2, 3, 3b encode this order.

8. **WavLM speaker-sim needs safetensors, because of gotcha 7.** `microsoft/wavlm-base-plus-sv`
   ships only `pytorch_model.bin`, transformers ≥4.56 refuses `torch.load` unless
   torch ≥2.6 (CVE-2025-32434), and Seed-VC pins torch 2.4 — so `quality.py` dies at model
   load with "we now require users to upgrade torch to at least v2.6". safetensors is
   exempt from that check, so `scripts/download_speaker_id.sh` fetches the `.bin` once and
   converts it into `third_party/models/speaker-id/wavlm-base-plus-sv/model.safetensors`,
   which `ModelsConfig.speaker_id_source()` then prefers over the Hub id.

9. **A truncated weight download looks like a successful one.** `download_qwen3_models.sh`
   skips any file that already exists, so an interrupted `curl` left a short
   `model.safetensors` that every later run happily "found" — surfacing much later, and far
   from the cause, as `SafetensorError: incomplete metadata, file not fully covered`. The
   script now downloads to `$f.part` and renames only on success. If you hit that error on
   a pre-existing checkout, delete the file and re-fetch; compare against the size in the
   Hub's file metadata rather than eyeballing it.
## Model taxonomy

See `third_party/models/README.md` — weights are organized by pipeline purpose (asr,
aligner, conversion, speaker-id, splicing), not by vendor.

### Speaker-similarity QC: CAM++ vs. WavLM

`quality.speaker_embedder` picks which encoder scores each conversion.

**`campplus`** (the default) is not a second opinion — it is *the same* encoder Seed-VC v1's
`inference.py` runs the reference audio through to build the style vector it conditions the
conversion on (`funasr/campplus`, `campplus_cn_common.bin`). `CampPlusSimilarity` imports
seed-vc's own vendored `modules.campplus.DTDNN` and mirrors its feature extraction exactly
(80-bin kaldi fbank at 16k, `dither=0`, mean-normalised over time) rather than
reimplementing it, precisely so the two can't drift apart; the price is that this backend
needs `seedvc.repo_dir` on disk.

**`wavlm`** (`microsoft/wavlm-base-plus-sv`, the notebook's original) needs no seed-vc
checkout, so it stays the dataclass default and the fallback. But it judges the conversion
from outside the space the conversion was optimised for, and on this data it produced
actively misleading verdicts: on the first real batch it flagged several rows as failures
because `sim_converted <= sim_raw_baseline` — i.e. it scored an *unconverted* VocalSound
laugh as more like the speaker than the re-voiced one. Same row, both encoders:

| encoder | `sim_converted` | `sim_raw_baseline` |
|---|---|---|
| wavlm | 0.410 | 0.539 ❌ |
| campplus | 0.314 | 0.114 ✅ |

Across all 20 rows CAM++ gives a median `sim_raw_baseline` of 0.030 (correctly: a stranger's
laughter is unrelated to this speaker) against a median `sim_converted` of 0.400, and zero
rows where the conversion fails to beat its baseline. Read the absolute numbers as modest
either way — these are 1-3s non-speech events, not utterances — and keep treating a low
score as "listen to this one", not an automatic reject.

### Quality filtering: what came out, and where it went in

`sim_converted` answers "does this event sound like this speaker". It says nothing about
whether the delivered recording is pleasant to listen to — hiss inherited from the
VocalSound contributor's microphone, an audible step at a splice junction, and Seed-VC's own
timbral footprint are all invisible to a cosine between speaker embeddings. The `filter`
stage (`para_synth/nisqa.py`) closes that gap with NISQA v2.0, a no-reference model: it
predicts a listening panel's MOS from one recording alone, which is the only kind of measure
available here, since a Para recording has no "correct" version to diff against.

It returns five values, all 1-5 and all higher-is-better. Four of them carry a pair of
thresholds — an absolute floor `min_<dim>` and a cap on the fall from source to Para,
`max_<dim>_drop`: `mos` overall, plus `noisiness` (the dimension a mismatched microphone moves — what
`selection.clarity_weight` tries to prevent upstream), `discontinuity` (a bad splice
junction, the most specific signal in here for this pipeline), `coloration` (the conversion's
own footprint) and `loudness`. `loudness` has no threshold at all: `synthesize_row()`
peak-normalises every finished recording, so a loudness difference measures that
normalisation rather than anything the pipeline did to the speech.

**The load-bearing threshold is `max_mos_drop`, not `min_mos`,** which is why the stage
scores the source recording as well as the finished one. An absolute floor mostly measures
how clean the input corpus was — a property of the recordings the pipeline was handed, not
of anything it did — and its right value moves whenever the corpus does. The drop measures
what this pipeline did to this recording, and stays meaningful on a corpus that is uniformly
noisy or uniformly clean. Same self-calibrating-relative-measure reasoning as the selection
axes below, and the same reason the earlier fixed-constant selector had to be replaced.

Measured on the 20-row batch (source MOS 4.17-4.92, Para MOS 3.99-4.83): the drop ranges
from **-0.11 to +0.59**, median +0.15. Three rows come out *better* than their source, which
is real — the spliced event is clean, converted audio and it dilutes whatever the source's
own worst stretch was. The default `max_mos_drop: 0.5` rejects 1 row of 20; `0.3` rejects 5.
Absolute MOS reads high across the board (nothing near a 2.5 floor), so `min_mos` ships as
`null`: on this corpus it would only ever be dead config.

`max_discontinuity_drop` is the same idea aimed at the splice specifically, and ships
much tighter, at **0.22**. Measured discontinuity drops span -0.28 to +0.43, median +0.11,
so unlike the MOS threshold this one cuts well into the distribution rather than trimming an
outlier — deliberately, because this is the dimension that scores a seam at the splice, so
it is the one worth being strict on, and 0.22 is where the batch stopped being acceptable by
ear. It rejects 8 rows of 20 on its own.

Note it lands on a knife edge: `tXO46Ys-7Qc_00028` measures +0.2231 and the next row down is
+0.1681, so anything in 0.17-0.22 filters identically, and 0.23 keeps that row. The
remaining floors and drops ship as `null`.

**The NISQA drops are not reproducible across runs, and the spread is comparable to the
threshold.** A full from-scratch re-run of the same 20 rows, same seed, reproduced every
alignment time, clip pick and splice time exactly — and moved the measured discontinuity
drops by up to 0.25 (`Rk_d1RT4KHs_00087` +0.43 to +0.20, `Dxra61hzE20_00017` +0.32 to +0.10,
`ALPlmChny4A_00101` +0.23 to +0.48), changing the kept count from 10/20 to 13/20. That is
Seed-VC's stochastic diffusion, which `seed` does not control (it seeds clip selection, not
the conversion subprocess), so each re-render is genuinely a different take. Treat a NISQA
rejection as "this take was bad", not "this row is bad": `synth --force` on a rejected row
draws again. The boundary measure has no such variance — it is computed from the source
recording, and reproduced to the last decimal on every row.

#### What NISQA structurally cannot see: `max_boundary_activity`

NISQA scores the *finished* recording. By the time that file is written, `tempo_splice` has
already fade-damped both junctions and separated them with a room-tone gap — so an event
that interrupts the speaker in the middle of a vowel still reads as a clean signal, because
by then it *is* one. The badness is not in the waveform; it is in where the event was put,
and NISQA has no opinion about that. Nor does anything else in the pipeline: alignment
reports a time, not a judgement about it.

`quality.max_boundary_activity` measures that directly, from the source recording rather
than the Para one. `audio_utils.splice_boundary_activity()` takes a 50ms slice at the cut on
each side and divides its RMS by that side's own overall RMS: 0 means the cut fell in
silence, ~1 means it fell on speech as loud as the segment's average, i.e. squarely
mid-phonation. A row is judged on the louder of its two junctions. It is relative for the
same reason `speech_pause_s` is — a quiet speaker's mid-vowel is louder in absolute terms
than a loud speaker's pause, so an absolute threshold would sort recordings by their gain
staging instead of by where the cut landed.

This reuses the measure `adaptive_splice` already had (`boundary_is_active`, "active" above
0.2), but for a different decision: 0.2 is the point at which a junction is worth *damping*,
which is far below the point at which a row is worth rejecting. Measured worst-junction
activity on the 20-row batch spans **0.04 to 1.76**, median **0.85** — forced alignment
lands mid-phonation as the normal case, not the exception, which is exactly why
`tempo_splice` is the default splice mode. A threshold of 1.2 rejects 3 rows of 20, 1.0
rejects 7, 0.8 rejects 11. It ships at **1.2**, picked by listening to the batch grouped
into those bands rather than derived from the distribution — there is no outlier to trim
here, since landing mid-phonation is the normal case, so the threshold is a judgement about
what is acceptable and has to be re-made on a different corpus.

The two criteria are genuinely independent: of the 3 rows 1.2 rejects, 2 pass every NISQA
threshold, and the row `max_mos_drop` rejects has a boundary activity of 0.29. Together the
shipped thresholds keep **10 of 20** rows — 8 rejected on `max_discontinuity_drop`, 3 on
`max_boundary_activity`, 1 on `max_mos_drop`, overlapping.

It costs no model: the filter stage already loads the source recording, and the measure is
an RMS ratio on top of it. So it works with `quality.nisqa.enabled: false`, and the filter
stage runs whenever *either* criterion is configured.

#### Caching

Nothing is deleted. Failing rows stay in `metadata_synth.jsonl` and their audio stays on
disk; `metadata_filtered.jsonl` is the passing subset, each row carrying its
`boundary_activity` and (when NISQA ran) its `nisqa_source` and `nisqa_para` scores.
Everything measured is cached in `data/work/stages/quality.jsonl` — named for the stage
rather than for NISQA, since it also holds the model-free boundary measure — so re-tuning
any threshold is a ~2s re-read with no model load. Measure first, pick the threshold from
the distribution you actually got.

**Licence**: torchmetrics downloads the upstream NISQA weights (`nisqa.tar` from
`gabrielmittag/NISQA`), which are **CC BY-NC-SA 4.0 — non-commercial**. The torchmetrics
code itself is Apache-2.0; the weights are not. Fine for research and internal dataset
triage, not for shipping in a product. `quality.nisqa.enabled: false` means they are never
downloaded, and `torchmetrics` stays an optional extra (`pip install -e ".[nisqa]"`) rather
than a core dependency for exactly that reason.

## Deliberately not implemented

A few DSP/selection ideas were considered and rejected — noted here so they don't get
re-proposed or "fixed in" without re-litigating the reasoning:

- **Mel-spectrogram + neural-vocoder (HiFi-GAN) resynthesis of the stitched waveform** at
  the splice step — see "Splicing: fixed vs. adaptive vs. tempo" above. Regenerates the entire
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
  Note this rejects *stretching* a clip, not `selection.tempo_weight`, which picks a clip
  whose native rate already fits and never touches a waveform — and which is restricted to
  laughter for exactly the reflex argument above (`selection.TEMPO_MATCHED_CLASSES`).
- **Selecting clips by pitch (F0), or by F0 contour shape** — measured and rejected: with
  `seedvc.f0_condition: false` Seed-VC v1 regenerates the event's pitch from the target
  speaker, so the source clip's F0 contributes essentially nothing (regression coefficient
  +0.04, contour correlation +0.14). See the table in "Clip selection" below. Re-measure with
  `scripts/measure_selection_axes.py` before reviving this if `f0_condition` is turned on.
- **Effective bandwidth as a second clarity term** — measured and dropped, because neither
  formulation carries channel information on this data. A 95%-energy spectral rolloff mostly
  reports *content* (clip median 3953 Hz vs. speech median ~1500 Hz: voiced speech is
  low-frequency-dominated, vocalisations are breathier, and that gap swamps any channel
  difference), while a noise-floor-referenced "highest band above the floor" measure
  saturates (clips p10..p90 = 7656..8000 Hz, speech 7127..7989 Hz — both corpora are simply
  full-band). Clarity is dB SNR alone.
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

## Clip selection: match what Seed-VC leaves behind

`pick_vocalsound_clip()` (`para_synth/selection.py`) biases — never restricts — which clip a
row gets. With every `selection.*_weight` at 0 it's uniform random over the matched class.
Above zero it loads `selection.candidate_pool` clips of the class, measures each, and samples
with weight `exp(-Σ weight_axis · distance_axis)`. Closer matches get more probability; every
usable candidate keeps nonzero probability, so a big laugh from a soft-spoken speaker still
occurs — just less often than uniform sampling would produce it.

The design question is *which* axes deserve weight. Seed-VC is a conversion model: whatever
it regenerates from the target speaker is already handled, and selecting on it double-counts.
So the axes are chosen by measurement — `scripts/measure_selection_axes.py` compares each
source clip against its own converted output on a finished run, and an axis earns a weight
only if the property survives conversion:

| property | survives Seed-VC? | measured | axis? |
|---|---|---|---|
| speaker identity / timbre | no — rebuilt from the CAM++ style vector | (that *is* the conversion) | ❌ handled downstream |
| **pitch (F0 level)** | **no** | `log2(conv) = +0.04·log2(src) + 0.84·log2(speaker) + 1.17` — source coefficient ≈ 0; dropping the source term moves the residual 0.229 → 0.239 oct | ❌ regenerated |
| **pitch contour** | **no** | source/converted F0 shape correlation **+0.14** | ❌ regenerated |
| **channel / clarity** | **yes** | source hiss is reconstructed faithfully; clip SNR spans p10..p90 = 33..60 dB vs 42-55 dB for the recordings | ✅ `clarity_weight` |
| **tempo** | **yes** | envelope rate median &#124;log2(conv/src)&#124; = **0.147 oct** at `length_adjust: 1.0` | ✅ `tempo_weight` (laughter only) |
| **intensity** | **yes** | crest factor median &#124;log2(conv/src)&#124; = **0.06 oct** | ✅ `energy_weight` |

Pitch is the one that surprises people, and it is the reason this section exists: matching a
laugh's F0 to the speaker's is the most intuitive idea available and it is measurably
pointless here — Seed-VC v1 re-pitches the event from the target speaker regardless of what
the source clip's pitch was. **This holds for `seedvc.f0_condition: false` only.** Turning
that on makes Seed-VC track the source's pitch with RMVPE, at which point F0 becomes residual
and the table above needs re-measuring before it can be trusted.

Two different comparison scales are used, on purpose:

- **Intensity is a percentile rank.** `clip_features()` returns raw crest factor and
  duration, *not* absolute loudness — clips get peak-normalised by `_finalize_clip()` and
  re-levelled by the splice anyway, so raw amplitude only tells you about VocalSound's
  crowdsourced recording gain. Those raw features become percentile ranks *within the
  candidate pool*, and `speaker_energy_score()` is the matching percentile rank of the
  pre-splice window's RMS among all same-length windows of that utterance — i.e. "is this a
  loud or quiet moment *for this speaker*", invariant to mic and recording gain.
- **Clarity and tempo are absolute.** dB SNR and envelope rate in Hz are already
  ratios/rates on perceptually meaningful scales, so a clip and a recording can be compared
  directly, scaled by `SNR_TOLERANCE_DB` / `TEMPO_TOLERANCE_OCT` into the same distance
  units as the percentile axis. Percentile-ranking them would be *worse*: it needs a
  reference population, and the only one available for the speaker side is the other rows of
  the batch, which manufactures spread when every recording in the batch is equally clean.

The percentile framing on the intensity axis is load-bearing, not stylistic. An earlier
version used fixed reference constants (crest ÷ 0.5, duration ÷ 2s) and was badly
miscalibrated against real VocalSound audio: measured crest factors cluster near 0.11 and the
median clip runs ~2.7s, saturating the duration term, so nearly every clip scored ~0.58 and
weighting had no signal to act on — measured mean picked intensity was identical (0.568 vs
0.563) for soft vs. loud speakers. With percentile ranks the same test tracks the speaker
properly (0.41 / 0.48 / 0.58 mean rank for speaker scores 0.15 / 0.50 / 0.85).

Measured effect of the whole thing, 400 draws per row over the 20-row sample batch, weighted
sampling vs. uniform (`selection` defaults, mean distance of the chosen clip):

| axis | weighted | uniform | reduction |
|---|---|---|---|
| clarity | 0.577 | 0.926 | **-38%** |
| tempo | 0.495 | 0.766 | **-35%** |
| intensity | 0.247 | 0.260 | -5% |

Intensity moves least because it had least room: its uniform-baseline mismatch is already
small in these units, so the exponential spends its weight where the real spread is. That is
the intended behaviour, not a bug — but if you change a feature, re-run
`scripts/measure_selection_axes.py` rather than assuming a new proxy has usable spread.

`selection.max_clipping` is the one hard gate, because clipping is bad in only one direction:
a clipped source is distortion Seed-VC reconstructs faithfully, not a mismatch that some
recording would be "close to". It is dropped rather than allowed to empty the pool, so it can
never fail a row on its own. (On the 200-clip sample measured here, 0% of VocalSound clips
tripped it — cheap insurance, not an active filter.)

Note that `splice.para_gain_db` and `seedvc.length_adjust` are *not* substitutes here: gain
only rescales the mix level and Seed-VC converts timbre, so neither turns an acoustically
boisterous laugh into a gentle one. Selection is the layer where that distinction is made.

# para-synth

Synthesizes "Para recording + tagged transcript" training data: take a Vietnamese speech
recording, take a paralinguistic clip from the [VocalSound](https://github.com/YuanGongND/vocalsound)
dataset (laughter / sigh / cough / throat-clearing / sneeze / sniff), re-voice the clip into
that speaker's own timbre with [Seed-VC](https://github.com/Plachtaa/seed-vc), splice it back
into the recording at the point the transcript's `[tag]` actually denotes (via forced
alignment, not a silence heuristic), and carry a tagged transcript through — producing a new
dataset row per `docs/synthesis.drawio.png`.

Extracted and generalized from `reference/notebook45ee5319ae.ipynb` (a Kaggle notebook) into
a modular, locally-runnable package. See **`docs/PIPELINE.md`** for the full diagram-to-code
map, notebook provenance, and every non-obvious environment gotcha — read that before making
changes.

```
VocalSound clip  ──►  source  ─┐
                                ├─►  Seed-VC  ──►  event in speaker's voice ──┐
Speech recording ──►  target  ─┘                                              ├─► Para recording
       └──────────────────────────────────────────────────────────────────────┘

Transcript ──► Qwen/Gemini (insert one [tag]) ──► tagged transcript ──────────► new dataset row
```

---

## 1. Install

One command does the whole setup — dependencies, Seed-VC, model weights, VocalSound, and
the `.gitignore`/`.env` scaffolding:

```bash
./scripts/prepare.sh
```

Requires Python ≥ 3.10 and downloads ~6.5GB. Every step is idempotent and independently
skippable, so it's safe to re-run after a partial or failed setup:

```bash
./scripts/prepare.sh --venv            # create and use ./.venv
./scripts/prepare.sh --skip-dataset    # skip the ~2.9GB VocalSound download
./scripts/prepare.sh --help            # all flags
```

Then check the environment:

```bash
para-synth doctor      # checks for the dependency landmines listed in docs/PIPELINE.md
```

<details>
<summary>Or run the setup steps individually</summary>

```bash
pip install -e ".[tagging-qwen,audio-extra,mms,nisqa]"
./scripts/download_vocalsound.sh       # ~2.9GB, VocalSound audio
./scripts/download_qwen3_models.sh     # ~3.6GB, Qwen3-ASR-0.6B + Qwen3-ForcedAligner-0.6B
./scripts/download_speaker_id.sh       # ~390MB, wavlm-base-plus-sv for the speaker-sim QC
para-synth setup-seedvc                # git clone + pip install seed-vc
pip install -e ".[qwen3]"              # AFTER seed-vc — see the install-order note below
para-synth doctor
```

Install order matters, and pip resolves last-write-wins:

- Seed-VC's `requirements.txt` pins `numpy<2` and an older `transformers`, so install it
  *after* this package.
- `qwen-asr` (the `[qwen3]` extra) pins `transformers==4.57.6`, and it genuinely needs
  something that new — Qwen3-ForcedAligner-0.6B's config declares `model_type: qwen3_asr`,
  an architecture older releases can't build. So it goes *after* Seed-VC, which runs fine
  on 4.57.6.
- That combination is also why `download_speaker_id.sh` converts the WavLM speaker-sim
  weights to safetensors: upstream ships only `pytorch_model.bin`, and transformers ≥4.56
  refuses `torch.load` unless torch ≥2.6 (CVE-2025-32434) while Seed-VC pins torch 2.4.

`prepare.sh` already does all of that in the right order.
</details>

`prepare.sh` creates `.env` from `.env.example` if it's missing — fill in
`DASHSCOPE_API_KEY` or `GEMINI_API_KEY` (whichever `configs/default.yaml: tagging.backend`
needs) before running `tag-transcripts`.

GPU is optional but strongly recommended — Seed-VC, Qwen3-ASR/ForcedAligner, and the WavLM
speaker-similarity check are all much faster on one. NISQA and the MMS fallback aligner run
on CPU regardless.

---

## 2. Put your data in

```
data/
├── raw/
│   ├── audio/            put your raw speech recordings here: {id}.wav
│   └── transcripts/      matching plain transcripts: {id}.txt (no [tag] yet)
├── tagged/
│   └── transcripts/      output of `para-synth tag-transcripts`: {id}.txt (has [tag])
├── vocalsound/           offline VocalSound copy (from download_vocalsound.sh)
├── work/                 scratch: intermediate Seed-VC outputs, converted clips
│   └── stages/           per-stage resume artifacts: align.jsonl, quality.jsonl
└── output/<run_name>/    para_{id}.wav + metadata_synth.jsonl (every row, with
                          diagnostics), metadata_filtered.jsonl (the rows that passed the
                          quality gate) and manifest.jsonl (those rows as a deliverable)
```

Audio and transcript are paired by **filename stem** — `data/raw/audio/{id}.wav` with
`data/raw/transcripts/{id}.txt` (see `para_synth/dataset.py: build_manifest()`). Only the
two `data/raw/` directories are yours to fill; everything else is generated. If you have
audio but no transcripts, `para-synth transcribe` writes them for you with Qwen3-ASR.

### Or: start from a JSONL manifest

If your corpus already ships a manifest — one JSON object per line, audio paths relative to
the manifest's own directory — point `paths.manifest` at it and the directory layout above
is bypassed entirely:

```json
{"audio_filepath": "wavs/Dxra61hzE20_00017.wav", "text": "... phương pháp tự nhiên là [laughter] ít có dùng thuốc ...", "duration": 15.0025, "lang": "vi", "dataset_name": "SpeechCake_PODCAST"}
```

Only `audio_filepath` and `text` are read; every other column is carried through untouched
and written back out by `para-synth export`. A manifest like this normally already has its
`[tag]` inline, in which case `transcribe` and `tag-transcripts` have nothing to do and the
run is just align → synth → filter → export.

---

## 3. Run it

```bash
para-synth transcribe                 # optional: fill data/raw/transcripts/ via Qwen3-ASR
para-synth tag-transcripts            # LLM: insert one paralinguistic [tag] per transcript
para-synth build-manifest --tagged    # sanity-check the audio <-> transcript pairing
para-synth run --limit 5              # synthesize a small batch first
para-synth inspect                    # review the worst-scoring rows of that batch
para-synth run                        # drop --limit once the small batch looks right
```

`scripts/run_pipeline.sh` is the same sequence as a script.

`run` chains three stages, which also run individually:

```bash
para-synth align      # stage 1: where in time each [tag] belongs
para-synth synth      # stage 2: pick a clip, Seed-VC it, splice it -> para_*.wav
para-synth filter     # stage 3: quality gate -> metadata_filtered.jsonl
```

| stage | reads | writes |
|---|---|---|
| `align` | the tagged manifest | `data/work/stages/align.jsonl` |
| `synth` | `align.jsonl` | `para_{id}.wav` + `metadata_synth.jsonl` in the output dir |
| `filter` | `metadata_synth.jsonl` | `data/work/stages/quality.jsonl` + `metadata_filtered.jsonl` |
| `export` | `metadata_filtered.jsonl` | `manifest.jsonl` — the deliverable |

Each stage **skips rows it has already done**, so changing a splice or quality setting never
re-pays for the stage before it — re-splicing costs no alignment, re-filtering costs no
diffusion. Add `--force` to redo rows anyway; `align`, `synth` and `run` also take
`--limit N`. The artifact contracts and resume rules are in "Staged execution" in
`docs/PIPELINE.md`.

Nothing is ever deleted. Rows that fail the quality gate stay in `metadata_synth.jsonl` with
their audio on disk; `metadata_filtered.jsonl` is just the passing subset.

Finally, to hand the result on:

```bash
para-synth export                     # -> <output_dir>/manifest.jsonl
para-synth export --out somewhere/manifest.jsonl
```

That writes the rows that **passed** the filter, in the same JSONL shape described in
section 2, so the result drops straight back into whatever consumes the input corpus.
`audio_filepath` points at the Para recording and is relative to the manifest's own
directory, so the folder stays portable; `duration` is remeasured, because splicing an event
in is exactly what changed it; extra columns from the input manifest (`lang`,
`dataset_name`, …) are carried through. Use this rather than `metadata_synth.jsonl`, which
is an internal artifact holding absolute paths and per-row diagnostics.

Every subcommand takes `--config path/to.yaml` to use a config other than
`configs/default.yaml`. Copying the default and editing the copy is the usual way to try a
setting without touching the shipped one.

---

## 4. Configuring: `configs/default.yaml`

Every tunable lives in one file. The comments in it are the authoritative version of what
follows; this section is the map. Values quoted below are the shipped defaults.

### `seed`

```yaml
seed: 0        # null -> a different random clip pick every run
```

Fixes which VocalSound clip each row draws. With a seed set, re-running the pipeline
reproduces the same picks and splice times, which is what makes a from-scratch re-run
comparable to the previous one. Note it does **not** make Seed-VC deterministic: its
diffusion is stochastic and this repo does not seed its subprocess, so a forced re-render is
genuinely a different take.

### `seedvc` — the voice conversion

```yaml
seedvc:
  repo_dir: third_party/seed-vc
  diffusion_steps: 50      # 30-50 = best quality; lower is faster and grainier
  length_adjust: 1.0       # 1.0 keeps the event's original timing
  cfg_rate: 0.7            # classifier-free guidance strength
  f0_condition: false      # RMVPE pitch conditioning
```

`f0_condition` is the one to be careful with. RMVPE pitch tracking is built for sung or
spoken pitch and produces garbage on coughs, sniffs and throat-clearing — only turn it on
for a laughter-only run.

### `splice` — how the event is joined to the speech

```yaml
splice:
  mode: tempo              # tempo | adaptive | fixed
  placement: gap           # legacy fallback: gap | before | after
  para_gain_db: -3.0       # event level relative to the speech peak
  fade_ms: 50              # fade length at the junctions
  pad_ms: 10               # "fixed" only: silence either side of the insert
  min_pad_ms: 10           # "adaptive" only: bridge at an already-quiet boundary
  max_gap_ms: 70           # "adaptive" only: bridge at a boundary that needed damping
  gap_scale: 1.0           # "tempo" only: multiplier on the speaker's median pause
  fade_k: 5.0              # "tempo" only: fade curvature; higher damps faster
```

The three `mode`s differ in what they assume about where the cut landed:

- **`tempo`** (default) — exponential fades at both junctions plus a room-tone gap sized to
  this speaker's *own* median pause. The default because forced alignment lands mid-phonation
  in practice (20 rows of 20 in the first real batch), which the other modes handle worse.
- **`adaptive`** — measures the boundary RMS and only fades/widens a junction where the cut
  actually landed on active phonation.
- **`fixed`** — the original notebook behaviour: a fixed fade on the insert and a fixed pad,
  regardless of context. It never fades the *speech*, so a mid-vowel cut stays a hard step.

`placement` is only consulted if alignment fails completely — both the aligner and the
proportional fallback returning nothing.

### `selection` — which VocalSound clip a row gets

```yaml
selection:
  energy_weight: 1.0       # intensity match
  clarity_weight: 1.5      # channel match (SNR + bandwidth)
  tempo_weight: 1.0        # envelope rate — LAUGHTER ONLY
  context_s: 2.5           # seconds of speech before the splice used as the reference
  candidate_pool: 48       # clips loaded and measured per row
  max_clipping: 0.01       # hard gate: drop candidates this clipped
```

Within the VocalSound class the tag maps to, clips are matched on the axes Seed-VC does
*not* normalise away. Speaker timbre is deliberately excluded — conversion already handles
that. Each axis distance is scaled so ~1.0 is a clearly audible mismatch, and the weighted
sum becomes a sampling weight `exp(-distance)`.

This is a **soft bias, never a filter**: every usable clip keeps nonzero probability, so a
loud laugh from a soft-spoken speaker still happens sometimes — real people do that, and the
dataset keeps acoustic variance a Para-TTS model benefits from. Set all three weights to
`0.0` for pure random selection.

`clarity_weight` is highest because VocalSound's crowdsourced microphones vary far more than
its speakers' energy does, and Seed-VC reconstructs source hiss and band-limiting faithfully.
`tempo_weight` is ignored for cough/sneeze/throat-clearing/sniff, which are reflexes whose
rate has no link to speaking rate.

### `sample_rate`

```yaml
sample_rate: 22050       # Seed-VC's base vocoder rate; everything is resampled to this
```

### `alignment` — where the `[tag]` goes in time

```yaml
alignment:
  use_qwen3: true
```

Backends are tried in order and the first success wins: **qwen3**
(Qwen3-ForcedAligner-0.6B — needs no pronunciation dictionary, so it avoids the OOV problem
Vietnamese colloquial fillers cause; resolves every row in practice), then **mms**
(torchaudio MMS_FA, an acoustic backstop), then **proportional** (word-fraction estimate
snapped to the nearest silence). Setting `use_qwen3: false` starts at MMS. Montreal Forced
Aligner was removed — see "Alignment stage order" in `docs/PIPELINE.md`.

### `models` — where weights live

```yaml
models:
  qwen3_asr_dir: third_party/models/asr/qwen3-asr-0.6b
  qwen3_asr_hub_id: Qwen/Qwen3-ASR-0.6B
  qwen3_forced_aligner_dir: third_party/models/aligner/qwen3-forcedaligner-0.6b
  qwen3_forced_aligner_hub_id: Qwen/Qwen3-ForcedAligner-0.6B
  speaker_id_dir: third_party/models/speaker-id/wavlm-base-plus-sv
  speaker_id_hub_id: microsoft/wavlm-base-plus-sv
```

`third_party/models/` is laid out by pipeline purpose rather than by vendor — see
`third_party/models/README.md`. Each `_dir` falls back to downloading its `_hub_id` at run
time if the local snapshot is missing, so the download scripts are an offline convenience,
not a requirement. Seed-VC manages its own checkpoints inside its own checkout, so it has no
path here.

### `quality` — the speaker check and the `filter` stage

```yaml
quality:
  speaker_embedder: campplus       # campplus | wavlm
  max_boundary_activity: 1.2
  nisqa:
    enabled: true
    min_mos: null
    max_mos_drop: 0.5
    min_noisiness: null
    max_noisiness_drop: null
    min_discontinuity: null
    max_discontinuity_drop: 0.22
    min_coloration: null
    max_coloration_drop: null
```

**`speaker_embedder`** picks which encoder scores each conversion into the row's
`sim_converted`. `campplus` is the same CAM++ encoder Seed-VC builds its style vector from,
so it measures the conversion in the space it was actually conditioned on (and needs
`seedvc.repo_dir` to exist). `wavlm` is an independent verifier that needs no Seed-VC
checkout, but judges the clone from outside the space it was optimised for, so a faithful
conversion can still score modestly. This is a score, not a gate — the `filter` stage does
not threshold it.

The rest of the block is the `filter` stage, which applies **two independent kinds of
check**:

**`max_boundary_activity`** asks *where the event was put*. It measures the source recording
at the insertion time: the energy right at the cut divided by the surrounding speech's own
level, where 0 means the cut fell in silence and ~1 means it fell on speech as loud as the
segment average. A row is judged on the louder of its two junctions. This is the check NISQA
structurally cannot make — by the time the finished file exists, the splice has already
faded both junctions and separated them with a room-tone gap, so an event that interrupts
the speaker mid-vowel reads as clean audio, because by then it is. Needs no model.

**`nisqa`** asks *how the audio came out*. NISQA v2.0 predicts what a listening panel would
score the finished recording with no reference signal, returning five values, all 1–5 and
all higher-is-better: `mos` overall, `noisiness` (hiss — a mismatched VocalSound microphone),
`discontinuity` (isolated interruptions — the dimension a bad splice moves), `coloration`
(timbral distortion — Seed-VC's own footprint), and `loudness`. Four of them carry two
optional criteria each:

- `min_<dim>` — an absolute floor on the finished recording.
- `max_<dim>_drop` — a cap on how far it fell below the *source* recording it was built from.

A row passes when every criterion that is set passes; `null` means not checked. `loudness`
has no threshold at all, because the pipeline peak-normalises its output, so a loudness
difference measures that normalisation rather than the splice.

**The drops are the load-bearing criteria, and that is the design.** An absolute floor mostly
measures how clean your input corpus was; the drop measures what this pipeline did to it,
which is the question worth asking and the one that survives a uniformly noisy or uniformly
clean corpus. Same relative-measure reasoning as the `selection` axes above.

**The shipped thresholds are calibrated on 20 rows, and two of the three were picked by ear.**
Measured on that batch: MOS drop −0.11 to +0.59 (median +0.15), discontinuity drop −0.28 to
+0.43 (median +0.11), worst-junction boundary activity 0.04 to 1.76 (median 0.85). Together
they keep 10 rows of 20. `max_mos_drop: 0.5` trims a clear outlier and anything from 0.40 to
0.55 filters identically. `max_discontinuity_drop: 0.22` deliberately cuts *into* the
distribution instead, because discontinuity is the dimension that scores a seam at the splice
specifically. `max_boundary_activity: 1.2` has no outlier to trim — landing mid-phonation is
the normal case — so it is purely a judgement about what was acceptable by ear; 1.0 would
reject 7 rows and 0.8 would reject 11. **Re-listen before reusing any of them on a different
corpus.**

One more caveat, and it is the important one: **the NISQA drops are not reproducible.**
Seed-VC's diffusion is stochastic and `seed` does not reach its subprocess, so re-rendering a
row gives a genuinely different take. A from-scratch re-run of the same 20 rows reproduced
every insert time, clip pick and splice time exactly, and every boundary-activity value — and
still moved individual discontinuity drops by up to 0.25, taking the kept count from 10/20 to
13/20. So read a NISQA rejection as "this take was bad", not "this row is bad" — `para-synth
synth --force` on a rejected row draws again — and don't tighten these thresholds on the
evidence of a single batch.

Re-tuning is cheap by design: everything measured is cached in
`data/work/stages/quality.jsonl`, so changing a threshold and re-running `para-synth filter`
is a seconds-long re-read that never loads a model. Measure first, then pick thresholds from
the distribution you actually got.

> **Licence:** torchmetrics downloads the upstream NISQA weights, which are
> **CC BY-NC-SA 4.0 — non-commercial**. Fine for research and internal dataset triage; do not
> ship them in a product. `enabled: false` means they are never downloaded, and the boundary
> check still runs without them. If *both* `nisqa.enabled: false` and
> `max_boundary_activity: null`, the filter stage has no criteria at all: `para-synth run`
> skips it and `para-synth filter` refuses rather than passing every row silently.

### `paths`

```yaml
paths:
  manifest: null                  # a JSONL manifest to read rows from, or null
  raw_audio_dir: data/raw/audio
  raw_transcript_dir: data/raw/transcripts
  tagged_transcript_dir: data/tagged/transcripts
  vocalsound_dir: data/vocalsound/audio_16k
  work_dir: data/work
  stage_dir: data/work/stages
  output_dir: data/output/default_run
```

`manifest` selects the input source: set it and rows come from that JSONL file, leave it
`null` and they come from pairing `raw_audio_dir` with `tagged_transcript_dir`. The two are
mutually exclusive, and `id` is the audio filename stem either way — so a corpus can be
switched between them without invalidating any stage cache.

`output_dir` is the one to change per experiment — point it at
`data/output/<something>` and that run's Para recordings and metadata land there, leaving
previous runs intact. `work_dir` and `stage_dir` are scratch and safe to delete; deleting
`stage_dir` just means the next run re-aligns and re-measures.

### `asr`

```yaml
asr:
  language: Vietnamese     # null -> auto-detect
```

Used only by `para-synth transcribe`, for recordings with no transcript yet.

### `tagging` — the LLM that inserts the `[tag]`

```yaml
tagging:
  backend: qwen                        # qwen | gemini | qwen_omni_audio
  qwen_model: qwen-plus
  gemini_model: gemini-2.5-flash
  qwen_omni_audio_model: qwen3-omni-flash
```

`qwen` and `gemini` send the transcript text only — cheap, and the default.
`qwen_omni_audio` also sends the recording, so the model picks a tag and a position that suit
the actual delivery (a sigh after a slow breathy phrase, placed where the speaker pauses)
rather than inferring from text alone; it costs more per row and caps audio at 150s. The
backend you choose determines which API key `.env` needs: `DASHSCOPE_API_KEY` for the Qwen
backends, `GEMINI_API_KEY` for Gemini.

---

## Repo layout

```
para_synth/            installable package — see docs/PIPELINE.md for what each module does
  align/                 alignment backends: qwen3, mms, proportional
  audio_utils.py         shared DSP (load/resample/split/splice/envelope) + the splice-
                         boundary measure the filter stage thresholds
  asr.py                 Qwen3-ASR wrapper (audio-only input)
  cli.py                 `para-synth` subcommands
  config.py              typed config loaded from configs/*.yaml
  dataset.py             row sources (directory pairing or JSONL manifest), manifest
                         export, and JSONL artifact I/O
  env_check.py           dependency-landmine doctor
  nisqa.py               NISQA quality scoring for the `filter` stage
  pipeline.py            orchestrates the stages (align -> synth -> filter)
  quality.py             WavLM/CAM++ speaker-similarity QC
  seedvc.py              Seed-VC setup + inference wrapper
  selection.py           which VocalSound clip each row gets
  tagging.py             LLM transcript tagging (Qwen/Gemini)
  vocalsound.py          VocalSound download, indexing, tag->class mapping
configs/default.yaml     all tunables (section 4 above)
docs/                    PIPELINE.md (deep-dive) + synthesis.drawio.png (diagram)
reference/               original Kaggle notebook, kept for provenance
scripts/prepare.sh       one-shot environment setup (deps, models, dataset, gitignore)
scripts/                 offline-download and example-run shell scripts
third_party/             gitignored: cloned seed-vc, downloaded model weights (see
                         third_party/models/README.md for the purpose-based layout)
data/                    gitignored: see section 2 above
```

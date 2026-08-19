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

## Pipeline

```
VocalSound clip  ──►  source  ─┐
                                ├─►  Seed-VC  ──►  event in speaker's voice ──┐
Speech recording ──►  target  ─┘                                              ├─► Para recording
       └──────────────────────────────────────────────────────────────────────┘

Transcript ──► Qwen/Gemini (insert one [tag]) ──► tagged transcript ──────────► new dataset row
```

## Install

```bash
pip install -e .
# plus whichever optional extras you need:
pip install -e ".[qwen3,tagging-qwen,audio-extra,mms]"
```

Then, one-time setup (each is idempotent — safe to re-run):

```bash
./scripts/download_vocalsound.sh       # ~2.9GB, VocalSound audio (offline, no Python needed)
./scripts/download_qwen3_models.sh     # ~3.6GB, Qwen3-ASR-0.6B + Qwen3-ForcedAligner-0.6B
para-synth setup-seedvc                # git clone + pip install seed-vc
para-synth setup-mfa                   # optional — qwen3 is tried first in the align chain
para-synth doctor                      # checks for the dependency landmines in docs/PIPELINE.md
```

Copy `.env.example` to `.env` and fill in `DASHSCOPE_API_KEY` or `GEMINI_API_KEY` (whichever
`configs/default.yaml: tagging.backend` is set to) before running `tag-transcripts`.

GPU is optional but strongly recommended — Seed-VC, Qwen3-ASR/ForcedAligner, and the WavLM
speaker-similarity check are all much faster on one. MFA's alignment (if you enable it) runs
on CPU regardless.

## Directory layout

```
data/
├── raw/
│   ├── audio/            put your raw speech recordings here: {id}.wav
│   └── transcripts/      matching plain transcripts: {id}.txt (no [tag] yet)
├── tagged/
│   └── transcripts/      output of `para-synth tag-transcripts`: {id}.txt (has [tag])
├── vocalsound/            offline VocalSound copy (from download_vocalsound.sh)
├── work/                  scratch: intermediate Seed-VC outputs, MFA corpora
└── output/<run_name>/     Para recordings + metadata_synth.jsonl ("new dataset")
```

`data/raw/audio/{id}.wav` and `data/raw/transcripts/{id}.txt` are paired by filename stem —
see `para_synth/dataset.py: build_manifest()`. If you only have audio and no transcript yet,
`para-synth transcribe` fills in `data/raw/transcripts/` via Qwen3-ASR first.

## Quickstart

```bash
para-synth transcribe            # only if data/raw/audio has clips missing a transcript
para-synth tag-transcripts       # LLM: insert a paralinguistic [tag] into each transcript
para-synth build-manifest --tagged   # sanity-check the audio<->transcript pairing
para-synth run --limit 5         # synthesize a small batch first
para-synth inspect               # review the worst-scoring rows
para-synth run                   # drop --limit once a small batch looks right
```

Or see `scripts/run_pipeline.sh` for the same sequence as a script.

## Configuration

All tunables live in `configs/default.yaml` (diffusion steps, splice gain/padding/fade,
alignment stage toggles, model paths, tagging backend/model). Pass `--config path/to.yaml` to
any `para-synth` subcommand to override it.

## Repo layout

```
para_synth/            installable package — see docs/PIPELINE.md for what each module does
  align/                 alignment backends: qwen3, mfa, mms, proportional
  audio_utils.py          shared DSP (load/resample/split/splice/envelope)
  asr.py                  Qwen3-ASR wrapper (audio-only input)
  cli.py                  `para-synth` subcommands
  config.py               typed config loaded from configs/*.yaml
  dataset.py              local audio/transcript manifest builder
  env_check.py            dependency-landmine doctor
  pipeline.py             orchestrates the full synthesis batch
  quality.py              WavLM speaker-similarity QC
  seedvc.py                Seed-VC setup + inference wrapper
  tagging.py              LLM transcript tagging (Qwen/Gemini)
  vocalsound.py            VocalSound download, indexing, tag->class mapping
configs/default.yaml     all tunables
docs/                    PIPELINE.md (deep-dive) + synthesis.drawio.png (diagram)
reference/               original Kaggle notebook, kept for provenance
scripts/                 offline-download and example-run shell scripts
third_party/             gitignored: cloned seed-vc, downloaded model weights (see
                          third_party/models/README.md for the purpose-based layout)
data/                    gitignored: see "Directory layout" above
```

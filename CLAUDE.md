# CLAUDE.md

Orientation for a future Claude Code session picking this repo up cold.

## What this is

`para-synth` synthesizes "Para recording + tagged transcript" training data by re-voicing
VocalSound paralinguistic clips (laughter/sigh/cough/throat-clearing/sneeze/sniff) into a
speaker's own timbre with Seed-VC, then splicing them into that speaker's recording at a
forced-aligned timestamp. Read **README.md** first, then **docs/PIPELINE.md** (diagram-to-code
map, notebook provenance, and a numbered list of real environment gotchas — several of the
modules here exist specifically to work around one of those) before editing anything.

`reference/notebook45ee5319ae.ipynb` is the original Kaggle notebook this repo was extracted
from — most modules are near-verbatim ports of specific cells; `docs/PIPELINE.md` has the
cell→module table. When in doubt about *why* a piece of DSP/alignment code does something
non-obvious, check the corresponding notebook cell's markdown before assuming it's arbitrary.

## Current state (as of this repo's creation)

- ✅ `data/vocalsound/audio_16k/` — downloaded (Zenodo mirror; see `vocalsound.py` for why
  not the official Dropbox link).
- ✅ `third_party/models/asr/qwen3-asr-0.6b/`, `third_party/models/aligner/qwen3-forcedaligner-0.6b/` — downloaded.
- ✅ `third_party/seed-vc/` — cloned, **not** `pip install -r requirements.txt`'d yet (run
  `para-synth setup-seedvc` or do it manually before the first real `para-synth run`).
- ⬜ MFA — not bootstrapped (`para-synth setup-mfa`); not required, since `align/qwen3.py` is
  tried first in the alignment chain and doesn't need it.
- ⬜ No `DASHSCOPE_API_KEY`/`GEMINI_API_KEY` configured — `tagging.py` (the LLM tag-insertion
  step) has not been executed or tested. Copy `.env.example` to `.env` before using it.
- ⬜ `data/raw/audio/` and `data/raw/transcripts/` are empty — no actual speech recordings
  are checked into this repo. Populate those before running anything past `setup-*`.
- ⬜ Nothing in `para_synth/` has been run end-to-end (no GPU / API keys available when this
  repo was built) — modules were written carefully against the notebook's working reference
  implementation, but treat a first real run as the first real test.

## Module map (see docs/PIPELINE.md for the full diagram mapping)

- `para_synth/audio_utils.py`, `quality.py`, `align/mfa.py`, `align/mms.py`,
  `align/proportional.py` — near-verbatim ports of already-working notebook code. Don't
  "simplify" these without checking the notebook cell's markdown first; several odd-looking
  lines (numpy/scipy avoidance, `USE_TF=0`, the `spn` OOV rejection, mtime-based output
  detection) are workarounds for specific real bugs, not accidental complexity.
- `para_synth/tagging.py`, `para_synth/asr.py`, `para_synth/align/qwen3.py` — genuinely new
  code (not in the source notebook). These implement diagram nodes the notebook didn't cover;
  no notebook cell to check them against, so double-check them more carefully than the ports.
- `para_synth/pipeline.py`, `dataset.py`, `vocalsound.py`, `cli.py` — new integration code
  gluing the above together around a local `data/raw/audio` + `data/raw/transcripts` layout
  instead of the notebook's Kaggle-mount sniffing.

## Conventions

- Config is threaded explicitly via `para_synth.config.Config` (loaded from
  `configs/default.yaml`), not module-level globals — the notebook used the latter (cell 6)
  since notebook cells share state implicitly; don't reintroduce that pattern here.
- `third_party/` and `data/` are gitignored (large binaries) — don't try to commit anything
  under them. `third_party/models/README.md` explains the purpose-based weight layout.
- The alignment chain order (qwen3 → mfa → mms → proportional) and why is documented in both
  `configs/default.yaml`'s comments and `docs/PIPELINE.md` — keep both in sync if it changes.

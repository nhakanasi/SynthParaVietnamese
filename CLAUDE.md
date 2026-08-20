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

## Setting up

`./scripts/prepare.sh` does the whole environment setup in one go (deps, Seed-VC, Qwen3
weights, VocalSound, `.gitignore`/`.env`). Idempotent and independently skippable per step
— prefer it over running the individual `scripts/download_*.sh` / `para-synth setup-*`
commands, and prefer *fixing* it over working around it if setup breaks.

## Current state (as of this repo's creation, on the machine it was built on)

The repo was scaffolded on a machine with **Python 3.9 and no GPU**, while the pipeline
targets Python ≥3.10 with a GPU — so most of it has never executed. Specifically:

- ✅ **The pipeline has now run end-to-end**, 20/20 rows, on Windows/CPU with no GPU. It
  needs a Python ≥3.10 env with Seed-VC's own `requirements.txt` installed *after*
  para-synth's, then `qwen-asr` last of all (last-write-wins — see gotcha 7 in
  docs/PIPELINE.md) — and a conda env, not system Python: installing Seed-VC's deps into
  system Python downgrades protobuf globally and breaks unrelated packages. Budget ~3h for
  20 rows on CPU (Seed-VC 50-step diffusion).
- ✅ **The `filter` stage has run on the full 20-row batch** (`para-synth filter`, ~15s
  including the NISQA model load, ~2.5s on cached measurements). `torchmetrics` 1.9.0
  installs into the conda env without moving numpy/torch/transformers. Measured across the
  batch: MOS drop -0.11 to +0.59 (median +0.15), discontinuity drop -0.28 to +0.43 (median
  +0.11), worst-junction boundary activity 0.04 to 1.76 (median 0.85). The shipped
  thresholds — `max_mos_drop: 0.5`, `max_discontinuity_drop: 0.22`,
  `max_boundary_activity: 1.2` — keep **10 of 20** rows. Only the MOS one trims an outlier;
  the other two were picked by listening to the batch grouped into score bands, so they are
  judgements about this corpus and have to be re-made on another one. `0.22` in particular
  is a knife edge: the nearest rows are +0.2231 and +0.1681.
- ⚠️ **NISQA drops move between runs by about as much as the thresholds are worth.** A clean
  from-scratch re-run (`rm -rf data/work data/output/default_run && para-synth run`) exactly
  reproduced align.jsonl and every deterministic field of metadata_synth.jsonl — insert
  times, clip picks, splice times — and identical boundary-activity values, but discontinuity
  drops shifted by up to 0.25 per row and the kept count went 10/20 to 13/20. Cause: Seed-VC
  diffusion is stochastic and `seed` doesn't reach its subprocess. So a NISQA rejection is a
  verdict on that take, not on that row, and any threshold calibration on a single batch is
  measuring one sample of a distribution. Don't tighten these thresholds further without
  re-rendering the same rows a few times first.
- ⚠️ No LLM API call has ever been made — `tagging.py`'s request shapes (both the text
  backends and `qwen_omni_audio`) are built from documented formats, unverified live.
  Expect small adjustments on first contact. (The example data ships pre-tagged, so the
  end-to-end run above did not exercise `tagging.py`.)
- 🟡 `data/vocalsound/audio_16k/` — **partial** (~458 of 21,024 clips). The download was
  interrupted; re-run `./scripts/download_vocalsound.sh`, which resumes.
- ✅ `third_party/models/asr/qwen3-asr-0.6b/`, `.../aligner/qwen3-forcedaligner-0.6b/`, and
  `.../speaker-id/wavlm-base-plus-sv/` — downloaded and verified loadable. The aligner's
  `model.safetensors` was silently truncated at first (see gotcha 9); if any model fails to
  load, check its file size against the Hub's before debugging anything else.
- ✅ **qwen3 is the alignment stage that actually runs** (`insert_stage: qwen3` in the output
  metadata). Note that the Qwen3-ForcedAligner card lists 11 supported languages and
  Vietnamese is not among them — but `language` only selects the tokenizer inside `qwen-asr`
  (Japanese/Korean get special cases, everything else splits on spaces, which is right for
  Vietnamese), there's no validation to trip over, and on a spot-check its timestamps land
  within ~70ms of MMS's. Treat it as working-but-out-of-spec, not as officially supported.
- ✅ `third_party/seed-vc/` — cloned, requirements installed, checkpoints fetched on first run.
- ⬜ No `DASHSCOPE_API_KEY`/`GEMINI_API_KEY` set.
- 🟡 `data/raw/audio/` and `data/raw/transcripts/` — no speech recordings ship with this
  repo, but `data/raw/sample_clean/` has a 20-row example (audio + pre-tagged JSONL) that
  can be unpacked into that layout. Populate one or the other before running past setup.

## Module map (see docs/PIPELINE.md for the full diagram mapping)

- `para_synth/audio_utils.py`, `quality.py`, `align/mms.py`,
  `align/proportional.py` — near-verbatim ports of already-working notebook code. Don't
  "simplify" these without checking the notebook cell's markdown first; several odd-looking
  lines (numpy/scipy avoidance, `USE_TF=0`, the `spn` OOV rejection, mtime-based output
  detection) are workarounds for specific real bugs, not accidental complexity.
- `para_synth/tagging.py`, `para_synth/asr.py`, `para_synth/align/qwen3.py`,
  `audio_utils.py: adaptive_splice()` and `tempo_splice()` — genuinely new code (not in the
  source notebook). These implement diagram nodes (or, for the two splice variants, a
  refinement) the notebook didn't cover; no notebook cell to check them against, so
  double-check them more carefully than the ports. Which one runs is `splice.mode`
  (`tempo` | `adaptive` | `fixed`); `tempo_splice()` is the default because the notebook's
  `splice()` (`fixed`) never fades the speech, and the forced-aligned cut lands
  mid-phonation on 20/20 rows of a real batch. See "Splicing" in docs/PIPELINE.md.
- `para_synth/dataset.py` — two input sources producing the same `ManifestRow`:
  `build_manifest()` pairs `data/raw/audio/{id}.wav` with a transcript of the same stem, and
  `read_manifest_jsonl()` reads a `{audio_filepath, text, ...}` JSONL manifest when
  `paths.manifest` is set. `id` is the audio stem in both, so switching a corpus between
  them keeps every stage cache valid. Columns the pipeline doesn't interpret ride along in
  `ManifestRow.extra` and are written back by `write_manifest_jsonl()` — the caller's dataset
  should not lose columns just because this pipeline reads two of them. Keep
  `metadata_synth.jsonl` (internal, absolute paths, diagnostics) distinct from the export
  (deliverable, relative paths, caller's schema); they are not the same artifact.
- `para_synth/selection.py` — which VocalSound clip a row gets. Only matches on properties
  Seed-VC does *not* regenerate (intensity, channel SNR, tempo); identity/timbre **and
  pitch** are deliberately excluded, both measured rather than assumed — Seed-VC v1 with
  `f0_condition: false` re-pitches the event from the target speaker, so selecting on F0
  double-counts. `scripts/measure_selection_axes.py` re-derives the evidence; re-run it
  before adding an axis or flipping `seedvc.f0_condition`. See "Clip selection" in
  docs/PIPELINE.md.
- `para_synth/nisqa.py` — half of the `filter` stage: NISQA v2.0 predicts a listening-panel
  MOS plus four degradation dimensions for each finished Para recording, catching
  hiss/splice-steps/coloration that speaker similarity is blind to. The thresholds that
  matter are the `max_<dim>_drop` pair (finished vs. the speaker's own source recording), not
  the absolute floors — same self-calibrating-relative reasoning as `selection.py`. Its
  weights are CC BY-NC-SA 4.0, **non-commercial**; keep that in mind before wiring it into
  anything shipped.
  The other half needs no model and lives in `audio_utils.splice_boundary_activity()`:
  NISQA scores the *finished* file, where both junctions have already been faded and gapped,
  so it cannot tell an event dropped into a pause from one dropped into the middle of a
  vowel. That check measures the *source* recording at the insertion time instead, which is
  also why it retrofits to already-rendered rows with no re-synthesis. Don't fold the two
  together — they answer different questions and are separately switchable. See "Quality
  filtering" in docs/PIPELINE.md.
- `para_synth/pipeline.py`, `dataset.py`, `vocalsound.py`, `cli.py` — new integration code
  gluing the above together around a local `data/raw/audio` + `data/raw/transcripts` layout
  instead of the notebook's Kaggle-mount sniffing. `pipeline.py` is three resumable stages
  (`align_batch` -> `synthesize_batch` -> `filter_batch`), each persisting a JSONL artifact
  keyed by row id and skipping rows it already did — don't collapse them back into one pass;
  the point is that a splice or threshold change costs no diffusion. See "Staged execution"
  in docs/PIPELINE.md.

## Conventions

- Config is threaded explicitly via `para_synth.config.Config` (loaded from
  `configs/default.yaml`), not module-level globals — the notebook used the latter (cell 6)
  since notebook cells share state implicitly; don't reintroduce that pattern here.
- `third_party/` and `data/` are gitignored (large binaries) — don't try to commit anything
  under them. `third_party/models/README.md` explains the purpose-based weight layout.
- The alignment chain order (qwen3 → mms → proportional) and why is documented in both
  `configs/default.yaml`'s comments and `docs/PIPELINE.md` — keep both in sync if it changes.
  Montreal Forced Aligner was removed from this chain — don't reintroduce it without reading
  the rationale in `docs/PIPELINE.md: Alignment stage order` first.

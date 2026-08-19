#!/usr/bin/env bash
# Example end-to-end run. See README.md for what each stage needs and when to skip it.
set -euo pipefail
cd "$(dirname "$0")/.."

para-synth doctor

# One-time setup (skip any already done):
# ./scripts/download_vocalsound.sh
# ./scripts/download_qwen3_models.sh
# para-synth setup-seedvc
# para-synth setup-mfa          # optional — qwen3 is tried first, see configs/default.yaml

# Only needed if data/raw/audio/ has clips with no data/raw/transcripts/{id}.txt yet:
# para-synth transcribe

# Insert a paralinguistic [tag] into each transcript (needs DASHSCOPE_API_KEY or
# GEMINI_API_KEY in .env — see configs/default.yaml: tagging.backend):
para-synth tag-transcripts

# Sanity-check the pairing before committing to a full run:
para-synth build-manifest --tagged

# Full synthesis. Drop --limit once a small batch looks right.
para-synth run --limit 5

para-synth inspect

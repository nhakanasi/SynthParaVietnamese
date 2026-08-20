#!/usr/bin/env bash
#
# One-shot environment setup: python deps, Seed-VC, model weights, VocalSound, gitignore.
#
# Every step is idempotent and independently skippable, so this is safe to re-run after a
# partial or failed setup — it detects what's already in place and moves on.
#
#   ./scripts/prepare.sh                 # everything
#   ./scripts/prepare.sh --venv          # create+use ./.venv instead of the ambient python
#   ./scripts/prepare.sh --skip-dataset  # skip the ~2.9GB VocalSound download
#
# Run --help for the full flag list.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

USE_VENV=0
SKIP_DEPS=0
SKIP_SEEDVC=0
SKIP_MODELS=0
SKIP_DATASET=0

usage() {
  sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Flags:
  --venv           create and use ./.venv rather than the active python
  --skip-deps      don't pip install
  --skip-seedvc    don't clone/install Seed-VC
  --skip-models    don't download Qwen3 ASR/aligner weights (~3.6GB)
  --skip-dataset   don't download VocalSound (~2.9GB)
  -h, --help       show this
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) USE_VENV=1 ;;
    --skip-deps) SKIP_DEPS=1 ;;
    --skip-seedvc) SKIP_SEEDVC=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --skip-dataset) SKIP_DATASET=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33m⚠️  %s\033[0m\n' "$1"; }
ok()   { printf '\033[32m✅ %s\033[0m\n' "$1"; }

# Track what actually succeeded so the closing summary reflects reality rather than
# assuming every step worked.
SUMMARY=""
record() { SUMMARY="${SUMMARY}\n  $1"; }

# ── 0 · Interpreter ──────────────────────────────────────────────────────────
step "Checking Python"
command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }
PY=python3
python3 - <<'EOF' || { echo "Python >= 3.10 required (pyproject requires-python)" >&2; exit 1; }
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
EOF
ok "$($PY --version)"

if [ "$USE_VENV" = 1 ]; then
  step "Creating ./.venv"
  [ -d .venv ] || $PY -m venv .venv
  PY="$REPO_ROOT/.venv/bin/python"
  ok "using $PY"
  record "activate the venv before running para-synth:  source .venv/bin/activate"
fi

# ── 1 · .gitignore ───────────────────────────────────────────────────────────
# Appended rather than overwritten: the checked-in .gitignore already has finer-grained
# data/ rules (it keeps the raw/audio + raw/transcripts folder skeleton via .gitkeep), so
# this only fills in entries that are genuinely missing.
step "Ensuring .gitignore covers generated/downloaded paths"
touch .gitignore
added=0
for pattern in "third_party/" "data/" ".env" "__pycache__/" "*.egg-info/" ".venv/"; do
  if ! grep -qxF "$pattern" .gitignore; then
    # data/ is already covered by the more specific data/* rules — don't add a blanket
    # entry that would also ignore the .gitkeep skeleton.
    if [ "$pattern" = "data/" ] && grep -q '^data/\*' .gitignore; then continue; fi
    printf '%s\n' "$pattern" >> .gitignore
    echo "   + $pattern"
    added=$((added + 1))
  fi
done
[ "$added" -eq 0 ] && ok ".gitignore already complete" || ok "added $added entry/entries"

# ── 2 · Python dependencies ──────────────────────────────────────────────────
if [ "$SKIP_DEPS" = 0 ]; then
  step "Installing para-synth + extras"
  $PY -m pip install --upgrade pip -q
  $PY -m pip install -e ".[tagging-qwen,audio-extra,mms,nisqa]"
  ok "package installed (editable)"
  record "para-synth CLI installed"
else
  warn "skipping dependency install"
fi

# ── 3 · Seed-VC ──────────────────────────────────────────────────────────────
# Installed AFTER our own deps on purpose: seed-vc's requirements.txt pins numpy<2 and an
# older transformers, and pip resolves last-write-wins. Doing it in this order leaves the
# environment in the state seed-vc's own inference.py expects. `para-synth doctor` at the
# end reports any remaining conflict rather than letting it surface mid-run.
if [ "$SKIP_SEEDVC" = 0 ]; then
  step "Setting up Seed-VC"
  if [ -d third_party/seed-vc/.git ]; then
    ok "third_party/seed-vc already cloned"
  else
    git clone --depth 1 https://github.com/Plachtaa/seed-vc.git third_party/seed-vc
  fi
  if [ -f third_party/seed-vc/requirements.txt ]; then
    $PY -m pip install -r third_party/seed-vc/requirements.txt
    ok "Seed-VC requirements installed"
    record "Seed-VC ready (checkpoints download on first conversion)"
  else
    warn "third_party/seed-vc/requirements.txt missing — clone may be incomplete"
  fi
else
  warn "skipping Seed-VC setup"
fi

# ── 3a · CUDA torch ──────────────────────────────────────────────────────────
# seed-vc's requirements.txt asks for nightly cu126 wheels in its first three lines and then
# pins bare `torch==2.4.0` further down; pip is last-write-wins, so step 3 always leaves a
# CPU-only torch behind. Silent, not loud — the pipeline still runs, ~32x slower for Seed-VC
# (measured: 530s vs 16.5s per row). torch 2.4 is also too old for anything newer than
# Hopper, so a current card gets no GPU at all rather than an error. Override the wheel
# index with TORCH_CUDA_INDEX if cu128 isn't right for your driver.
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
if [ "$SKIP_DEPS" = 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
  step "Reinstalling torch with CUDA support"
  if $PY -m pip install --index-url "$TORCH_CUDA_INDEX" torch torchaudio torchvision; then
    ok "CUDA torch installed"
    record "torch built for CUDA (para-synth doctor verifies it can see the GPU)"
  else
    warn "CUDA torch install failed — the pipeline will run on CPU and be very slow"
  fi
fi

# ── 3b · Qwen3 forced aligner ────────────────────────────────────────────────
# Deliberately the LAST install, after Seed-VC rather than alongside our own deps in step 2:
# qwen-asr pins transformers==4.57.6, and Qwen3-ForcedAligner-0.6B's config declares
# model_type "qwen3_asr" — an architecture older transformers releases don't know, so if
# Seed-VC's requirements land afterwards and pull transformers back down, the aligner can't
# load at all. Seed-VC runs fine on 4.57.6 (verified end-to-end), so the newer pin wins.
if [ "$SKIP_DEPS" = 0 ]; then
  step "Installing the Qwen3 forced aligner (qwen-asr)"
  if $PY -m pip install -e ".[qwen3]"; then
    ok "qwen-asr installed"
    record "Qwen3 forced aligner available (first stage of the align chain)"
  else
    warn "qwen-asr install failed — the align chain falls back to MMS"
  fi
fi

# ── 4 · Qwen3 model weights ──────────────────────────────────────────────────
if [ "$SKIP_MODELS" = 0 ]; then
  step "Downloading Qwen3 ASR + forced-aligner weights (~3.6GB)"
  ./scripts/download_qwen3_models.sh
  record "Qwen3 ASR + ForcedAligner weights in third_party/models/"
  step "Downloading the speaker-similarity model (~390MB)"
  ./scripts/download_speaker_id.sh
  record "wavlm-base-plus-sv in third_party/models/speaker-id/ (converted to safetensors)"
else
  warn "skipping model download"
fi

# ── 5 · VocalSound ───────────────────────────────────────────────────────────
if [ "$SKIP_DATASET" = 0 ]; then
  step "Downloading VocalSound (~2.9GB)"
  if ./scripts/download_vocalsound.sh; then
    record "VocalSound in data/vocalsound/audio_16k/"
  else
    warn "VocalSound download incomplete — re-run ./scripts/download_vocalsound.sh (it resumes)"
  fi
else
  warn "skipping VocalSound download"
fi

# ── 6 · .env ─────────────────────────────────────────────────────────────────
step "Checking API key file"
if [ -f .env ]; then
  ok ".env exists"
else
  cp .env.example .env
  warn "created .env from .env.example — add DASHSCOPE_API_KEY or GEMINI_API_KEY before"
  warn "running 'para-synth tag-transcripts'"
fi

# ── 7 · Verify ───────────────────────────────────────────────────────────────
step "Environment check"
$PY -m para_synth.cli doctor || warn "doctor reported issues (see docs/PIPELINE.md 'Gotchas')"

step "Done"
printf 'Ready:%b\n' "${SUMMARY:-\n  (nothing installed — all steps skipped)}"
cat <<'EOF'

Next:
  1. Put recordings in   data/raw/audio/{id}.wav
     and transcripts in  data/raw/transcripts/{id}.txt
     (audio with no transcript? run `para-synth transcribe` to fill them in)
  2. para-synth tag-transcripts     # needs an API key in .env
  3. para-synth run --limit 5       # small batch first
  4. para-synth inspect             # check the worst-scoring rows

See README.md for the full workflow and docs/PIPELINE.md for design detail.
EOF

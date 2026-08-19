#!/usr/bin/env bash
# Download the Qwen3-ASR-0.6B and Qwen3-ForcedAligner-0.6B weights for offline use.
# No Python required — plain curl against the Hugging Face resolve API, same approach as
# scripts/download_vocalsound.sh.
#
# Usage: scripts/download_qwen3_models.sh [--asr-only|--aligner-only]
set -euo pipefail
cd "$(dirname "$0")/.."

FILES=(README.md chat_template.json config.json generation_config.json merges.txt
       preprocessor_config.json tokenizer_config.json vocab.json model.safetensors)

fetch() {
  local repo="$1" dest="$2"
  mkdir -p "$dest"
  for f in "${FILES[@]}"; do
    if [ -f "$dest/$f" ]; then
      echo "✅ $dest/$f already present"
      continue
    fi
    echo "📥 $repo/$f"
    curl -fL --retry 3 -o "$dest/$f" "https://huggingface.co/$repo/resolve/main/$f"
  done
}

WANT_ASR=1
WANT_ALIGNER=1
case "${1:-}" in
  --asr-only) WANT_ALIGNER=0 ;;
  --aligner-only) WANT_ASR=0 ;;
esac

[ "$WANT_ASR" = 1 ] && fetch "Qwen/Qwen3-ASR-0.6B" "third_party/models/asr/qwen3-asr-0.6b"
[ "$WANT_ALIGNER" = 1 ] && fetch "Qwen/Qwen3-ForcedAligner-0.6B" "third_party/models/aligner/qwen3-forcedaligner-0.6b"

echo "✅ done"

#!/usr/bin/env bash
# Download microsoft/wavlm-base-plus-sv (speaker-similarity QC, para_synth/quality.py) for
# offline use, and convert its weights to safetensors.
#
# The conversion is not cosmetic: upstream ships only pytorch_model.bin, and transformers
# >= 4.56 refuses torch.load unless torch >= 2.6 (CVE-2025-32434) — but Seed-VC pins
# torch 2.4, so from_pretrained() on the .bin dies with "we now require users to upgrade
# torch to at least v2.6". safetensors is exempt from that check, so converting once here
# is what lets the two pins coexist in one env.
#
# Usage: scripts/download_speaker_id.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=microsoft/wavlm-base-plus-sv
DEST=third_party/models/speaker-id/wavlm-base-plus-sv
mkdir -p "$DEST"

for f in config.json preprocessor_config.json pytorch_model.bin; do
  if [ -f "$DEST/$f" ]; then
    echo "✅ $DEST/$f already present"
    continue
  fi
  echo "📥 $REPO/$f"
  curl -fL --retry 3 -o "$DEST/$f.part" "https://huggingface.co/$REPO/resolve/main/$f"
  mv "$DEST/$f.part" "$DEST/$f"
done

if [ -f "$DEST/model.safetensors" ]; then
  echo "✅ $DEST/model.safetensors already present"
else
  echo "🔄 converting pytorch_model.bin -> model.safetensors"
  python - "$DEST" <<'PY'
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

dest = Path(sys.argv[1])
sd = torch.load(dest / "pytorch_model.bin", map_location="cpu", weights_only=True)
# Shared storages are legal in a .bin but rejected by safetensors — clone so each tensor
# owns its buffer. contiguous() for the same reason (a view can't be serialised as-is).
sd = {k: v.detach().clone().contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
save_file(sd, dest / "model.safetensors", metadata={"format": "pt"})
print("wrote", dest / "model.safetensors")
PY
fi

echo "✅ done"

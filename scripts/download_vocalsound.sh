#!/usr/bin/env bash
# Download VocalSound (21,024 16kHz clips) for offline use, no Python required.
#
# Source: Zenodo record 14650192, a WebDataset re-package of github.com/YuanGongND/vocalsound
# (same audio, same f0003_0_laughter.wav-style filenames) on stable, DOI-backed hosting — the
# official Dropbox-hosted release is frequently rate-limited ("Link Temporarily Disabled")
# since this is a popular research dataset; confirmed dead when this repo was built.
#
# `para_synth.vocalsound.download()` is the Python equivalent (used by
# `para-synth download-vocalsound`) if you'd rather call it from code.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="data/vocalsound"
AUDIO_DIR="$DEST/audio_16k"
EXPECTED_COUNT=21024

mkdir -p "$DEST/_shards" "$AUDIO_DIR"

existing=$(find "$AUDIO_DIR" -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
if [ "$existing" -ge "$EXPECTED_COUNT" ]; then
  echo "✅ VocalSound already present: $AUDIO_DIR ($existing clips)"
  exit 0
fi

urls_file=$(mktemp)
curl -s "https://zenodo.org/api/records/14650192/files" \
  | jq -r '.entries[].links.content' > "$urls_file"
total=$(wc -l < "$urls_file")
echo "📥 Downloading $total VocalSound shards from Zenodo (~2.9GB total)"

i=0
while IFS= read -r url; do
  i=$((i + 1))
  fname=$(basename "$(dirname "$url")")
  echo "[$i/$total] $fname"
  curl -fL --retry 4 --max-time 300 -o "$DEST/_shards/$fname" "$url"
  tar -xf "$DEST/_shards/$fname" -C "$AUDIO_DIR" || true
  rm -f "$DEST/_shards/$fname"
done < "$urls_file"
rm -f "$urls_file"
rmdir "$DEST/_shards" 2>/dev/null || true

find "$AUDIO_DIR" -name '*.json' -delete

n=$(find "$AUDIO_DIR" -name '*.wav' | wc -l | tr -d ' ')
if [ "$n" -lt "$EXPECTED_COUNT" ]; then
  echo "❌ only extracted $n/$EXPECTED_COUNT expected clips into $AUDIO_DIR"
  exit 1
fi
echo "✅ VocalSound ready: $AUDIO_DIR ($n clips)"

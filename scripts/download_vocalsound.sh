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
# Zenodo's API occasionally 504s under load, so retry the listing before giving up.
listing=""
for attempt in 1 2 3 4 5; do
  listing=$(curl -s --max-time 60 "https://zenodo.org/api/records/14650192/files" || true)
  if printf '%s' "$listing" | grep -q '"entries"'; then break; fi
  echo "  Zenodo listing attempt $attempt failed, retrying…" >&2
  sleep 3
done
if ! printf '%s' "$listing" | grep -q '"entries"'; then
  echo "❌ could not fetch the Zenodo file listing" >&2
  exit 1
fi

# Parse with jq when available, else python3 — jq isn't installed by default on most
# machines, and python3 is guaranteed present for this project anyway.
if command -v jq >/dev/null 2>&1; then
  printf '%s' "$listing" | jq -r '.entries[].links.content' > "$urls_file"
else
  printf '%s' "$listing" | python3 -c \
    'import json,sys; [print(e["links"]["content"]) for e in json.load(sys.stdin)["entries"]]' > "$urls_file"
fi
total=$(wc -l < "$urls_file")
echo "📥 Downloading $total VocalSound shards from Zenodo (~2.9GB total)"

i=0
failed=""
while IFS= read -r url; do
  i=$((i + 1))
  fname=$(basename "$(dirname "$url")")
  echo "[$i/$total] $fname"
  # Zenodo drops long connections fairly often (observed mid-transfer resets), so retry
  # each shard a few times before recording it as failed rather than aborting the run.
  ok=0
  for attempt in 1 2 3 4 5; do
    if curl -fL --retry 6 --retry-all-errors --max-time 600 --connect-timeout 30 \
         -o "$DEST/_shards/$fname" "$url"; then
      ok=1
      break
    fi
    echo "    retry $attempt for $fname" >&2
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    echo "    ⚠️  giving up on $fname" >&2
    failed="$failed $fname"
    continue
  fi
  tar -xf "$DEST/_shards/$fname" -C "$AUDIO_DIR" || true
  rm -f "$DEST/_shards/$fname"
done < "$urls_file"
rm -f "$urls_file"
rmdir "$DEST/_shards" 2>/dev/null || true

if [ -n "$failed" ]; then
  echo "⚠️  shards that never downloaded:$failed"
  echo "   re-run this script — completed clips are kept, so it resumes where it left off."
fi

find "$AUDIO_DIR" -name '*.json' -delete

n=$(find "$AUDIO_DIR" -name '*.wav' | wc -l | tr -d ' ')
if [ "$n" -lt "$EXPECTED_COUNT" ]; then
  echo "❌ only extracted $n/$EXPECTED_COUNT expected clips into $AUDIO_DIR"
  exit 1
fi
echo "✅ VocalSound ready: $AUDIO_DIR ($n clips)"

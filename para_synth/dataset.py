"""Local raw-dataset manifest: pairs `audio/{id}.wav` with `transcripts/{id}.txt`.

Diagram nodes: "Speech recording" + "Transcription". Replaces the notebook's
`find_sample_dataset()`/metadata-jsonl loading (cells 11-13, built around one specific
Kaggle dataset's `metadata (1).jsonl` + auto-detected columns) with a plain local layout the
user controls directly: `data/raw/audio/{id}.wav` next to `data/raw/transcripts/{id}.txt`.

Each row produced here has the same shape (`id`, `audio_filepath`, `text`) that
para_synth.vocalsound / para_synth.align / para_synth.pipeline consume, so the rest of the
pipeline doesn't care whether `text` came from `data/raw/transcripts` (pre-tag) or
`data/tagged/transcripts` (post-tag, has `[tag]` inserted by para_synth.tagging).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")

# A transcript's inline para tag, e.g. "... [cười]" or "... [laughter]".
TAG_RE = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class ManifestRow:
    id: str
    audio_filepath: Path
    text: str


def build_manifest(audio_dir: Path, transcript_dir: Path) -> list[ManifestRow]:
    """Pair every audio file in `audio_dir` with the transcript of the same stem in
    `transcript_dir`. Audio files with no matching transcript are skipped (and reported)
    rather than silently dropped."""
    audio_dir, transcript_dir = Path(audio_dir), Path(transcript_dir)
    audio_files = sorted(
        f for f in audio_dir.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS
    )

    rows: list[ManifestRow] = []
    missing: list[str] = []
    for audio_path in audio_files:
        transcript_path = transcript_dir / f"{audio_path.stem}.txt"
        if not transcript_path.is_file():
            missing.append(audio_path.name)
            continue
        text = transcript_path.read_text(encoding="utf-8").strip()
        rows.append(ManifestRow(id=audio_path.stem, audio_filepath=audio_path, text=text))

    if missing:
        print(f"⚠️  {len(missing)} audio file(s) skipped — no matching transcript in {transcript_dir}:")
        for name in missing[:10]:
            print(f"     {name}")
        if len(missing) > 10:
            print(f"     … +{len(missing) - 10} more")

    print(f"✅ {len(rows)} paired rows from {audio_dir} + {transcript_dir}")
    return rows


def extract_tag(text: str) -> str | None:
    """Pull the inline `[tag]` back out of a tagged transcript, or None if it has none."""
    m = TAG_RE.search(str(text))
    return f"[{m.group(1)}]" if m else None


def tagged_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    """Rows whose `text` carries an inline tag — the ones the pipeline can actually
    synthesize a Para recording for."""
    return [r for r in rows if extract_tag(r.text) is not None]

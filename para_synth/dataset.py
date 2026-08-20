"""Where the pipeline's rows come from, and what the finished ones are written back as.

Diagram nodes: "Speech recording" + "Transcription". Two input sources, both producing the
same `ManifestRow`:

- `build_manifest()` pairs `data/raw/audio/{id}.wav` with `data/raw/transcripts/{id}.txt`
  by filename stem — a plain local layout the user controls directly, and the default.
- `read_manifest_jsonl()` reads a JSONL manifest (`{audio_filepath, text, ...}`), the shape
  the notebook's `find_sample_dataset()` loading expected (cells 11-13, built around one
  specific Kaggle dataset's `metadata (1).jsonl`). Set `paths.manifest` to use it. This is
  the format a corpus normally arrives in and the format `write_manifest_jsonl()` exports
  the passing rows back to, so the pipeline is a transformation of the caller's dataset
  rather than a different dataset that happens to be derived from it.

Each row produced here has the same shape (`id`, `audio_filepath`, `text`) that
para_synth.vocalsound / para_synth.align / para_synth.pipeline consume, so the rest of the
pipeline doesn't care whether `text` came from `data/raw/transcripts` (pre-tag) or
`data/tagged/transcripts` (post-tag, has `[tag]` inserted by para_synth.tagging).
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")

# The two keys this pipeline reads out of a JSONL manifest row, and writes back into one.
# Everything else in the row is carried through untouched — see ManifestRow.extra.
MANIFEST_AUDIO_KEY = "audio_filepath"
MANIFEST_TEXT_KEY = "text"

# A transcript's inline para tag, e.g. "... [cười]" or "... [laughter]".
TAG_RE = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class ManifestRow:
    id: str
    audio_filepath: Path
    text: str
    # Columns a JSONL manifest carried that this pipeline doesn't interpret — `lang`,
    # `dataset_name`, anything else the caller's schema has. Kept verbatim so
    # `write_manifest_jsonl` can hand them back on the way out: a dataset that arrives with
    # five columns should not leave with three because the pipeline only reads two of them.
    extra: dict = field(default_factory=dict)


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


def read_manifest_jsonl(path: Path) -> list[ManifestRow]:
    """Rows of a JSONL manifest: one object per line carrying at least `audio_filepath`
    and `text`, the shape the notebook's original `metadata (1).jsonl` had.

    `audio_filepath` is resolved relative to the manifest's own directory when it isn't
    absolute, which is what makes such a manifest portable — the same file works on any
    machine the dataset is copied to. Row `id` is the audio file's stem, exactly as in
    `build_manifest()`, so every stage artifact keys the same way whichever source the rows
    came from and a corpus can be switched between the two without invalidating a cache.

    A manifest like this usually arrives already tagged (its `text` has the `[laughter]`
    inline), in which case `transcribe` and `tag-transcripts` have nothing left to do.
    """
    path = Path(path)
    rows: list[ManifestRow] = []
    missing: list[str] = []
    for raw in read_jsonl(path):
        audio = Path(raw[MANIFEST_AUDIO_KEY])
        if not audio.is_absolute():
            audio = path.parent / audio
        if not audio.is_file():
            missing.append(raw[MANIFEST_AUDIO_KEY])
            continue
        rows.append(
            ManifestRow(
                id=audio.stem,
                audio_filepath=audio,
                text=raw[MANIFEST_TEXT_KEY],
                extra={
                    k: v for k, v in raw.items()
                    if k not in (MANIFEST_AUDIO_KEY, MANIFEST_TEXT_KEY)
                },
            )
        )

    if missing:
        print(f"⚠️  {len(missing)} row(s) skipped — audio not found under {path.parent}:")
        for name in missing[:10]:
            print(f"     {name}")
        if len(missing) > 10:
            print(f"     … +{len(missing) - 10} more")

    print(f"✅ {len(rows)} rows from {path}")
    return rows


def write_manifest_jsonl(
    path: Path, rows: list[dict], source_rows: dict[str, ManifestRow] | None = None
) -> None:
    """Write finished Para recordings out as a JSONL manifest in the same shape
    `read_manifest_jsonl` reads, so the result drops straight back into whatever consumes
    the input corpus.

    `rows` are `metadata_filtered.jsonl` rows — the ones that passed the quality gate.
    `source_rows` maps id to the input `ManifestRow` when the run started from a manifest,
    and its `extra` columns are carried through.

    Two fields are deliberately not copied from the input: `audio_filepath` points at the
    Para recording and is written relative to this manifest's own directory rather than
    absolute, since this file is the deliverable and has to survive being moved; and
    `duration` is remeasured, because splicing an event in is precisely what changed it.
    """
    import soundfile as sf

    path = Path(path)
    source_rows = source_rows or {}

    out = []
    for row in rows:
        para = Path(row["para_audio"])
        try:
            audio_path = para.relative_to(path.parent)
        except ValueError:
            # The recording doesn't live under the manifest — an absolute path is then the
            # only one that still resolves, at the cost of portability for that row.
            audio_path = para
        entry = {
            MANIFEST_AUDIO_KEY: audio_path.as_posix(),
            MANIFEST_TEXT_KEY: row["transcript"],
            "duration": round(sf.info(para).duration, 4),
        }
        source = source_rows.get(row["id"])
        if source is not None:
            entry |= {k: v for k, v in source.extra.items() if k not in entry}
        out.append(entry)

    write_jsonl(path, out)
    print(f"📄 {len(out)} rows -> {path}")


def extract_tag(text: str) -> str | None:
    """Pull the inline `[tag]` back out of a tagged transcript, or None if it has none."""
    m = TAG_RE.search(str(text))
    return f"[{m.group(1)}]" if m else None


def tagged_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    """Rows whose `text` carries an inline tag — the ones the pipeline can actually
    synthesize a Para recording for."""
    return [r for r in rows if extract_tag(r.text) is not None]


# ── Stage artifacts ──────────────────────────────────────────────────────────────────
# Every pipeline stage (align / synth / filter) persists what it produced as a JSONL file
# keyed by row id, then reads it back on the next invocation to skip work it already did.
# That's what makes the stages independently re-runnable: see "Staged execution" in
# docs/PIPELINE.md.


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a stage artifact, or [] if it hasn't been produced yet — a missing artifact
    means "nothing done yet", never an error."""
    path = Path(path)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_by_id(cached: list[dict], fresh: list[dict]) -> list[dict]:
    """Freshly computed rows layered over cached ones, sorted by id.

    Rows the current invocation didn't touch (a `--limit` subset, or ids that failed this
    time) are kept from the cache rather than dropped, so a stage artifact accumulates
    across partial runs instead of being truncated to whatever the last one covered.
    """
    by_id = {row["id"]: row for row in cached}
    by_id.update({row["id"]: row for row in fresh})
    return [by_id[key] for key in sorted(by_id)]

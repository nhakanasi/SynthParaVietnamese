"""VocalSound dataset: offline download, local indexing, and tag->class mapping.

Diagram node: "Non vocal sound (VocalSound dataset)" + the "random matching tag" feedback
arrow from Para recording back up to it.

Source dataset: Gong, Yu & Glass, "VocalSound: A Dataset for Improving Human Vocal Sounds
Recognition" (ICASSP 2022) — https://github.com/YuanGongND/vocalsound. The notebook consumed
this via a Kaggle mount (`currenredman/vocal-sound`) and had to sniff out the mount path
(`find_vocalsound_dir` in cell 7) since Kaggle's mount convention varies. Locally there's no
mount-path guessing needed — `download()` puts it at a known path and everything else just
reads `Config.paths.vocalsound_dir` directly.
"""
from __future__ import annotations

import io
import random
import re
import tarfile
from pathlib import Path
from typing import Optional

import requests

VS_CLASSES = ["laughter", "sigh", "cough", "throatclearing", "sneeze", "sniff"]

# VocalSound filenames look like: f0003_0_laughter.wav  (gender, speaker id, take, class)
VS_PAT = re.compile(r"^([fm])(\d+)_(\d+)_(" + "|".join(VS_CLASSES) + r")\.wav$")

# The official release (github.com/YuanGongND/vocalsound) is a Dropbox-hosted zip that's
# frequently rate-limited ("Link Temporarily Disabled") since this is a popular research
# dataset — confirmed dead when this repo was built. The Zenodo mirror below re-packages the
# same 16kHz audio (same filenames, same VS_PAT convention) as WebDataset tar shards on
# stable, DOI-backed hosting, so it's the default source here.
VOCALSOUND_ZENODO_RECORD = "14650192"
VOCALSOUND_ZENODO_FILES_API = f"https://zenodo.org/api/records/{VOCALSOUND_ZENODO_RECORD}/files"
# Legacy official source, kept as an alternative in case Dropbox's rate limit has cleared —
# see `download(..., source="dropbox")`.
VOCALSOUND_16K_DROPBOX_URL = "https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1"
EXPECTED_CLIP_COUNT = 21024

# Tag vocabulary a downstream transcript's inline `[tag]` gets mapped onto. Keyword lists
# cover common Vietnamese + English spellings, matched against the tag with its brackets
# stripped (tags look like "[cười]", "[ho]", etc). Extend these tuples if a new dataset's
# tag value_counts show vocabulary not covered here.
TAG_TO_VS = {
    "laughter": ["cười", "cuoi", "laugh"],
    "sigh": ["thở dài", "tho dai", "thở phào", "sigh"],
    "cough": ["ho", "cough"],
    "throatclearing": ["hắng giọng", "hang giong", "đằng hắng", "dang hang", "clear"],
    "sneeze": ["hắt hơi", "hat hoi", "sneeze"],
    "sniff": ["hít mũi", "hit mui", "khịt mũi", "khit mui", "sniff"],
}

# word-boundary match, not substring — plain "in" would match "ho" (cough) inside unrelated
# words like "hoa" or "học", since Vietnamese doesn't use ASCII \b cues.
_TAG_PATTERNS = {
    vs_cls: [re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)", re.UNICODE) for k in keys]
    for vs_cls, keys in TAG_TO_VS.items()
}


def _download_stream(url: str, dest: Path, timeout: int = 60) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r   {written / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.1f}%)", end="", flush=True)
    print()


def _download_from_zenodo(audio_dir: Path) -> None:
    """Download every WebDataset tar shard from the Zenodo mirror (train/valid/test, ~2.9GB
    total) and extract just the .wav members (original VS_PAT filenames preserved) into
    `audio_dir`. Each shard also carries a .json per clip (just `{"label": ...}`) — redundant
    with the filename's own class field, so those aren't extracted."""
    resp = requests.get(VOCALSOUND_ZENODO_FILES_API, timeout=30)
    resp.raise_for_status()
    entries = resp.json()["entries"]
    print(f"📥 Downloading VocalSound from Zenodo record {VOCALSOUND_ZENODO_RECORD} "
          f"({len(entries)} shards, ~{sum(e['size'] for e in entries) / 1e9:.1f}GB)")

    audio_dir.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(entries, 1):
        shard_url = entry["links"]["content"]
        print(f"[{i}/{len(entries)}] {entry['key']}")
        buf = io.BytesIO()
        with requests.get(shard_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 20):
                buf.write(chunk)
        buf.seek(0)
        with tarfile.open(fileobj=buf) as tf:
            for member in tf.getmembers():
                if member.name.endswith(".wav"):
                    member.name = Path(member.name).name  # flatten any internal path
                    tf.extract(member, path=audio_dir)


def _download_from_dropbox(dest_dir: Path, audio_dir: Path, url: str, keep_zip: bool) -> None:
    """The official release zip. Frequently rate-limited ("Link Temporarily Disabled") —
    kept as an opt-in alternative to the Zenodo mirror, not the default."""
    import shutil
    import zipfile

    zip_path = dest_dir / "vs_release_16k.zip"
    print(f"📥 Downloading VocalSound 16kHz release (~1.7GB) from {url}")
    _download_stream(url, zip_path)
    print("✅ Download complete, extracting…")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    if not audio_dir.is_dir():
        nested = next(dest_dir.glob("**/audio_16k"), None)
        if nested is None:
            raise RuntimeError(f"Extracted {zip_path} but found no audio_16k/ folder under {dest_dir}")
        if nested.parent != dest_dir:
            for item in nested.parent.iterdir():
                shutil.move(str(item), str(dest_dir / item.name))
            shutil.rmtree(nested.parent, ignore_errors=True)

    if not keep_zip:
        zip_path.unlink(missing_ok=True)


def download(
    dest_dir: Path,
    source: str = "zenodo",
    dropbox_url: str = VOCALSOUND_16K_DROPBOX_URL,
    keep_zip: bool = False,
) -> Path:
    """Download + extract VocalSound into `dest_dir`. `source`: "zenodo" (default, stable
    mirror) or "dropbox" (official release, often rate-limited).

    Idempotent: skips the download if `dest_dir/audio_16k` already has the expected file
    count. Returns the path to the extracted `audio_16k` directory.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = dest_dir / "audio_16k"

    if audio_dir.is_dir():
        n = sum(1 for f in audio_dir.iterdir() if VS_PAT.match(f.name))
        if n >= EXPECTED_CLIP_COUNT:
            print(f"✅ VocalSound already present: {audio_dir} ({n} clips)")
            return audio_dir
        print(f"⚠️  {audio_dir} exists but only has {n}/{EXPECTED_CLIP_COUNT} clips — re-downloading")

    if source == "zenodo":
        _download_from_zenodo(audio_dir)
    elif source == "dropbox":
        _download_from_dropbox(dest_dir, audio_dir, dropbox_url, keep_zip)
    else:
        raise ValueError(f"unknown source {source!r} — use 'zenodo' or 'dropbox'")

    n = sum(1 for f in audio_dir.iterdir() if VS_PAT.match(f.name))
    if n < EXPECTED_CLIP_COUNT:
        raise RuntimeError(f"Extracted only {n}/{EXPECTED_CLIP_COUNT} expected clips into {audio_dir}")
    print(f"✅ VocalSound ready: {audio_dir} ({n} clips)")

    return audio_dir


def _strip_tag_brackets(tag_value: str) -> str:
    """ "[cười]" -> "cười" — tags in the source transcripts are wrapped in brackets."""
    return str(tag_value).strip().strip("[]").strip().lower()


def match_vs_class(tag_value: str, rng: Optional[random.Random] = None) -> str:
    """Exact match against VS_CLASSES first (a transcript's tag may already BE the English
    class name — "[laughter]", "[sniff]", etc — which the word-boundary keyword matcher
    below can't catch: "laugh" as a keyword can't match inside "laughter" without failing
    its own trailing boundary check). Falls through to fuzzy Vietnamese/English keyword
    matching, then — the diagram's "random matching tag" feedback arrow — a random class.
    """
    rng = rng or random
    t = _strip_tag_brackets(tag_value)
    if t in VS_CLASSES:
        return t
    for vs_cls, patterns in _TAG_PATTERNS.items():
        if any(p.search(t) for p in patterns):
            return vs_cls
    return rng.choice(VS_CLASSES)


def list_clips(vs_dir: Path, vs_class: str) -> list[str]:
    vs_dir = Path(vs_dir)
    return [f.name for f in vs_dir.iterdir() if (m := VS_PAT.match(f.name)) and m.group(4) == vs_class]

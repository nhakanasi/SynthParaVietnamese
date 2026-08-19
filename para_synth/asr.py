"""Audio-only input handling: transcribe raw speech recordings that have no pre-existing
transcript, via Qwen3-ASR.

Diagram: the "pre-exist ASR dataset / Omnivoice TTS (more emotion/voice)" box normally
supplies a transcript alongside the recording. When a `data/raw/audio/{id}.wav` has no
matching `data/raw/transcripts/{id}.txt`, `transcribe_missing()` fills that gap so the rest
of the pipeline (para_synth.tagging, para_synth.align, ...) has something to work with —
audio-only input becomes text+audio input before it ever reaches those stages.

New (not in the source notebook, which always started from a pre-tagged `metadata (1).jsonl`
that already had transcripts). Model: Qwen/Qwen3-ASR-0.6B — see third_party/models/README.md
for where the offline weights live.

Requires the `qwen-asr` package (`pip install qwen-asr`).
"""
from __future__ import annotations

from pathlib import Path

from para_synth.dataset import AUDIO_EXTS


class Qwen3ASR:
    def __init__(self, model_source: str, device: str = "cuda", dtype: str = "bfloat16", max_new_tokens: int = 256):
        import torch
        from qwen_asr import Qwen3ASRModel

        torch_dtype = getattr(torch, dtype)
        device = device if torch.cuda.is_available() else "cpu"
        print(f"🧠 Loading Qwen3-ASR from {model_source} ({device}, {dtype}) …")
        self.model = Qwen3ASRModel.from_pretrained(
            model_source, dtype=torch_dtype, device_map=device, max_new_tokens=max_new_tokens
        )

    def transcribe(self, audio_path: str | Path, language: str | None = "Vietnamese") -> str:
        results = self.model.transcribe(audio=str(audio_path), language=language)
        return results[0].text.strip()


def transcribe_missing(
    audio_dir: Path,
    transcript_dir: Path,
    asr: Qwen3ASR,
    language: str | None = "Vietnamese",
    overwrite: bool = False,
) -> list[str]:
    """Run ASR on every audio file in `audio_dir` that has no matching .txt in
    `transcript_dir` yet (or all of them, if `overwrite`), writing plain (untagged)
    transcripts. Returns the list of ids transcribed. Run this *before*
    para_synth.tagging.insert_para_tag() — ASR output has no [tag] in it yet.
    """
    audio_dir, transcript_dir = Path(audio_dir), Path(transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(f for f in audio_dir.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
    done: list[str] = []
    for audio_path in audio_files:
        out_path = transcript_dir / f"{audio_path.stem}.txt"
        if out_path.is_file() and not overwrite:
            continue
        text = asr.transcribe(audio_path, language=language)
        out_path.write_text(text, encoding="utf-8")
        print(f"📝 {audio_path.name} -> {out_path.name}: {text!r}")
        done.append(audio_path.stem)

    print(f"✅ transcribed {len(done)}/{len(audio_files)} audio file(s) missing a transcript")
    return done

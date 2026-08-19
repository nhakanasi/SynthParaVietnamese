"""`para-synth` command-line entry points. One subcommand per pipeline stage — see
README.md for the recommended run order and docs/PIPELINE.md for what each stage does.
"""
from __future__ import annotations

import argparse
import sys

from para_synth.config import Config, load_config


def cmd_doctor(args, cfg: Config) -> None:
    from para_synth.env_check import run_doctor

    run_doctor()


def cmd_download_vocalsound(args, cfg: Config) -> None:
    from para_synth.vocalsound import download

    download(cfg.paths.vocalsound_dir.parent, source=args.source)


def cmd_setup_seedvc(args, cfg: Config) -> None:
    from para_synth.seedvc import setup_seedvc

    setup_seedvc(cfg.seedvc.repo_dir)


def cmd_setup_mfa(args, cfg: Config) -> None:
    from para_synth.align.mfa import MFAAligner

    MFAAligner(cfg.alignment).setup()


def cmd_transcribe(args, cfg: Config) -> None:
    from para_synth.asr import Qwen3ASR, transcribe_missing

    asr = Qwen3ASR(cfg.models.qwen3_asr_source())
    transcribe_missing(cfg.paths.raw_audio_dir, cfg.paths.raw_transcript_dir, asr,
                        language=cfg.asr.language, overwrite=args.overwrite)


def cmd_tag_transcripts(args, cfg: Config) -> None:
    from para_synth.dataset import build_manifest
    from para_synth.tagging import TaggingError, insert_para_tag

    rows = build_manifest(cfg.paths.raw_audio_dir, cfg.paths.raw_transcript_dir)
    cfg.paths.tagged_transcript_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for row in rows:
        out_path = cfg.paths.tagged_transcript_dir / f"{row.id}.txt"
        if out_path.is_file() and not args.overwrite:
            continue
        try:
            result = insert_para_tag(
                row.text,
                backend=cfg.tagging.backend,
                model=cfg.tagging.model_for_backend(),
                audio_path=row.audio_filepath,  # used only by audio backends
            )
            out_path.write_text(result.text, encoding="utf-8")
            print(f"🏷️  {row.id}: {result.tag} -> {out_path.name}")
            n_ok += 1
        except TaggingError as e:
            print(f"⚠️  {row.id}: tagging failed: {e}")
    print(f"✅ tagged {n_ok}/{len(rows)} transcripts -> {cfg.paths.tagged_transcript_dir}")


def cmd_build_manifest(args, cfg: Config) -> None:
    from para_synth.dataset import build_manifest, tagged_rows

    transcript_dir = cfg.paths.tagged_transcript_dir if args.tagged else cfg.paths.raw_transcript_dir
    rows = build_manifest(cfg.paths.raw_audio_dir, transcript_dir)
    if args.tagged:
        rows = tagged_rows(rows)
        print(f"🏷️  {len(rows)} rows carry an inline tag")


def cmd_run(args, cfg: Config) -> None:
    from para_synth.dataset import build_manifest, tagged_rows
    from para_synth.pipeline import synthesize_batch

    rows = build_manifest(cfg.paths.raw_audio_dir, cfg.paths.tagged_transcript_dir)
    rows = tagged_rows(rows)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("❌ no tagged rows to synthesize — run `para-synth tag-transcripts` first")
        sys.exit(1)
    synthesize_batch(rows, cfg, language=cfg.asr.language)


def cmd_inspect(args, cfg: Config) -> None:
    import json

    meta_path = cfg.paths.output_dir / "metadata_synth.jsonl"
    if not meta_path.is_file():
        print(f"❌ {meta_path} not found — run `para-synth run` first")
        sys.exit(1)
    rows = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda r: r["sim_converted"])
    for r in rows[: args.limit]:
        print(f"🇻🇳 {r['transcript']}")
        print(f"🏷️  tag={r['nv_tag']!r} vs_class={r['vs_class']!r} inserted@{r['splice_at_s']:.2f}s "
              f"({r['insert_stage']})")
        print(f"   sim_converted={r['sim_converted']:.3f} sim_raw_baseline={r['sim_raw_baseline']:.3f}")
        print(f"   source: {r['source_audio']}")
        print(f"   para:   {r['para_audio']}")
        print("—" * 60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="para-synth", description=__doc__)
    p.add_argument("--config", default=None, help="path to a config YAML (default: configs/default.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check for known dependency landmines").set_defaults(func=cmd_doctor)

    dl = sub.add_parser("download-vocalsound", help="download VocalSound for offline use")
    dl.add_argument("--source", choices=["zenodo", "dropbox"], default="zenodo")
    dl.set_defaults(func=cmd_download_vocalsound)

    sub.add_parser("setup-seedvc", help="clone + install Seed-VC").set_defaults(func=cmd_setup_seedvc)
    sub.add_parser("setup-mfa", help="bootstrap Montreal Forced Aligner + vietnamese_mfa").set_defaults(func=cmd_setup_mfa)

    tr = sub.add_parser("transcribe", help="Qwen3-ASR any raw audio missing a transcript")
    tr.add_argument("--overwrite", action="store_true")
    tr.set_defaults(func=cmd_transcribe)

    tag = sub.add_parser("tag-transcripts", help="LLM: insert a paralinguistic [tag] into each transcript")
    tag.add_argument("--overwrite", action="store_true")
    tag.set_defaults(func=cmd_tag_transcripts)

    mf = sub.add_parser("build-manifest", help="preview the audio<->transcript pairing")
    mf.add_argument("--tagged", action="store_true", help="use data/tagged/transcripts instead of data/raw/transcripts")
    mf.set_defaults(func=cmd_build_manifest)

    run = sub.add_parser("run", help="run the full synthesis pipeline over tagged rows")
    run.add_argument("--limit", type=int, default=None, help="only process the first N rows (sanity-check)")
    run.set_defaults(func=cmd_run)

    insp = sub.add_parser("inspect", help="print the worst-scoring rows from the last run")
    insp.add_argument("--limit", type=int, default=5)
    insp.set_defaults(func=cmd_inspect)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    args.func(args, cfg)


if __name__ == "__main__":
    main()

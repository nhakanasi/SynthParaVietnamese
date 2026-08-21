"""`para-synth` command-line entry points. One subcommand per pipeline stage — see
README.md for the recommended run order and docs/PIPELINE.md for what each stage does.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from para_synth.config import REPO_ROOT, Config, load_config


def cmd_doctor(args, cfg: Config) -> None:
    from para_synth.env_check import run_doctor

    run_doctor()


def cmd_download_vocalsound(args, cfg: Config) -> None:
    from para_synth.vocalsound import download

    download(cfg.paths.vocalsound_dir.parent, source=args.source)


def cmd_setup_seedvc(args, cfg: Config) -> None:
    from para_synth.seedvc import setup_seedvc

    setup_seedvc(cfg.seedvc.repo_dir)


def cmd_transcribe(args, cfg: Config) -> None:
    from para_synth.asr import Qwen3ASR, transcribe_missing

    asr = Qwen3ASR(cfg.models.qwen3_asr_source())
    transcribe_missing(cfg.paths.raw_audio_dir, cfg.paths.raw_transcript_dir, asr,
                        language=cfg.asr.language, overwrite=args.overwrite)


def _untagged_rows(args, cfg: Config):
    """The plain, untagged rows the pre-synthesis stages work on, honouring --limit.

    Always the raw transcript directory: `slots` and `tag-transcripts` exist to *produce* a
    tagged transcript, so a JSONL manifest — which arrives with its tag already inline — has
    nothing for them to do and goes straight to `align`.
    """
    from para_synth.dataset import build_manifest

    rows = build_manifest(cfg.paths.raw_audio_dir, cfg.paths.raw_transcript_dir)
    if getattr(args, "limit", None):
        rows = rows[: args.limit]
    return rows


def cmd_slots(args, cfg: Config) -> None:
    from para_synth.pipeline import slots_batch

    slots_batch(_untagged_rows(args, cfg), cfg, language=cfg.asr.language, force=args.force)


def cmd_tag_transcripts(args, cfg: Config) -> None:
    from para_synth.pipeline import tag_batch

    tag_batch(_untagged_rows(args, cfg), cfg, force=args.overwrite)


def _manifest_rows(cfg: Config, tagged: bool = True):
    """Every row the pipeline can see, from whichever input source the config selects.

    `paths.manifest` wins when it's set: a corpus that ships its own JSONL manifest carries
    the transcript inside it, so there is no separate transcript directory to choose
    between, and `tagged` doesn't apply.
    """
    from para_synth.dataset import build_manifest, read_manifest_jsonl

    if cfg.paths.manifest:
        return read_manifest_jsonl(cfg.paths.manifest)
    transcript_dir = cfg.paths.tagged_transcript_dir if tagged else cfg.paths.raw_transcript_dir
    return build_manifest(cfg.paths.raw_audio_dir, transcript_dir)


def cmd_build_manifest(args, cfg: Config) -> None:
    from para_synth.dataset import tagged_rows

    rows = _manifest_rows(cfg, tagged=args.tagged)
    if args.tagged or cfg.paths.manifest:
        rows = tagged_rows(rows)
        print(f"🏷️  {len(rows)} rows carry an inline tag")


def cmd_export(args, cfg: Config) -> None:
    from para_synth.dataset import read_jsonl, read_manifest_jsonl, write_manifest_jsonl

    src = cfg.paths.output_dir / "metadata_filtered.jsonl"
    rows = read_jsonl(src)
    if not rows:
        print(f"❌ no rows in {src} — run `para-synth filter` first")
        sys.exit(1)

    # Only consulted for the columns this pipeline doesn't interpret (`lang`,
    # `dataset_name`, …); without an input manifest there are none to carry through.
    source_rows = (
        {r.id: r for r in read_manifest_jsonl(cfg.paths.manifest)} if cfg.paths.manifest else {}
    )
    out = Path(args.out) if args.out else cfg.paths.output_dir / "manifest.jsonl"
    write_manifest_jsonl(out, rows, source_rows)


def _tagged_manifest(args, cfg: Config):
    """The tagged rows the synthesis stages operate on, honouring --limit."""
    from para_synth.dataset import tagged_rows

    rows = tagged_rows(_manifest_rows(cfg))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        if cfg.paths.manifest:
            print(f"❌ no rows in {cfg.paths.manifest} carry an inline [tag] in their "
                  "`text` — nothing to synthesize.")
        else:
            print("❌ no tagged rows to synthesize — run `para-synth tag-transcripts` first")
        sys.exit(1)
    return rows


def cmd_align(args, cfg: Config) -> None:
    from para_synth.pipeline import align_batch

    align_batch(_tagged_manifest(args, cfg), cfg, language=cfg.asr.language, force=args.force)


def cmd_synth(args, cfg: Config) -> None:
    from para_synth.pipeline import synthesize_batch

    synthesize_batch(_tagged_manifest(args, cfg), cfg, force=args.force)


def cmd_filter(args, cfg: Config) -> None:
    from para_synth.pipeline import filter_batch

    filter_batch(cfg, force=args.force)


def cmd_run(args, cfg: Config) -> None:
    from para_synth.pipeline import align_batch, filter_batch, filter_is_configured, synthesize_batch

    rows = _tagged_manifest(args, cfg)
    align_batch(rows, cfg, language=cfg.asr.language, force=args.force)
    synthesize_batch(rows, cfg, force=args.force)
    if filter_is_configured(cfg):
        filter_batch(cfg, force=args.force)
    else:
        print("⏭️  skipping the filter stage: quality.nisqa.enabled is false and "
              "quality.max_boundary_activity is null, so there is nothing to filter on")


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

    tr = sub.add_parser("transcribe", help="Qwen3-ASR any raw audio missing a transcript")
    tr.add_argument("--overwrite", action="store_true")
    tr.set_defaults(func=cmd_transcribe)

    sl = sub.add_parser("slots", help="stage 0a: find the pauses each recording offers -> stages/slots.jsonl")
    sl.add_argument("--limit", type=int, default=None, help="only process the first N rows (sanity-check)")
    sl.add_argument("--force", action="store_true", help="redo rows this stage has already done")
    sl.set_defaults(func=cmd_slots)

    tag = sub.add_parser("tag-transcripts", help="stage 0b: LLM picks one pause and one [tag] per row")
    tag.add_argument("--limit", type=int, default=None, help="only process the first N rows (sanity-check)")
    tag.add_argument("--overwrite", action="store_true", help="re-tag rows already tagged")
    tag.set_defaults(func=cmd_tag_transcripts)

    mf = sub.add_parser("build-manifest", help="preview the audio<->transcript pairing")
    mf.add_argument("--tagged", action="store_true", help="use data/tagged/transcripts instead of data/raw/transcripts")
    mf.set_defaults(func=cmd_build_manifest)

    ex = sub.add_parser("export", help="write the rows that passed the filter as a JSONL manifest")
    ex.add_argument("--out", default=None, help="output path (default: <output_dir>/manifest.jsonl)")
    ex.set_defaults(func=cmd_export)

    # The synthesis stages, in order. Each one persists what it produced and skips rows it
    # has already done, so they can be run individually and re-run cheaply; `run` chains
    # all three. See "Staged execution" in docs/PIPELINE.md.
    for name, help_text, func, takes_limit in (
        ("align", "stage 1: find each transcript tag's insertion time -> stages/align.jsonl", cmd_align, True),
        ("synth", "stage 2: convert + splice each row -> metadata_synth.jsonl", cmd_synth, True),
        # `filter` reads metadata_synth.jsonl rather than the manifest, so it has no --limit:
        # what it covers is whatever `synth` has produced so far.
        ("filter", "stage 3: quality-check the finished recordings -> metadata_filtered.jsonl", cmd_filter, False),
        ("run", "run all three synthesis stages over tagged rows", cmd_run, True),
    ):
        sp = sub.add_parser(name, help=help_text)
        if takes_limit:
            sp.add_argument("--limit", type=int, default=None, help="only process the first N rows (sanity-check)")
        sp.add_argument("--force", action="store_true", help="redo rows this stage has already done")
        sp.set_defaults(func=func)

    insp = sub.add_parser("inspect", help="print the worst-scoring rows from the last run")
    insp.add_argument("--limit", type=int, default=5)
    insp.set_defaults(func=cmd_inspect)

    return p


def _load_dotenv() -> None:
    """Read repo-root .env into the environment, without adding a dependency.

    `scripts/prepare.sh` creates .env from .env.example and both the README and
    .env.example tell you to put DASHSCOPE_API_KEY / GEMINI_API_KEY there — but nothing
    ever read the file, so `tag-transcripts` failed with "DASHSCOPE_API_KEY is not set"
    even after you followed the instructions. Values already in the environment win, so an
    explicitly exported key still overrides the file.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    args.func(args, cfg)


if __name__ == "__main__":
    main()

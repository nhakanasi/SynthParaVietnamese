"""Seed-VC: re-voice the VocalSound clip into the speaker's own timbre.

Diagram node: "SeedVC v1". Ported from notebook45ee5319ae.ipynb cells 2 (setup) and 25
(`run_seedvc`, the inference subprocess call).

Seed-VC (github.com/Plachtaa/seed-vc) manages its own checkpoint download the first time
inference.py runs — see third_party/models/README.md for why no weights are stored under
third_party/models/ for this stage.
"""
from __future__ import annotations

import glob
import subprocess
import time
from pathlib import Path

from para_synth.config import SeedVCConfig

SEEDVC_GIT_URL = "https://github.com/Plachtaa/seed-vc.git"


def setup_seedvc(repo_dir: Path, git_url: str = SEEDVC_GIT_URL) -> None:
    """Clone Seed-VC and install its pinned requirements. Idempotent."""
    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        print(f"📥 Cloning Seed-VC into {repo_dir}")
        subprocess.run(["git", "clone", git_url, str(repo_dir)], check=True)
    else:
        print(f"✅ Seed-VC already present at {repo_dir}")

    print("📦 Installing Seed-VC's pinned requirements (~2-3 min)")
    subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], cwd=repo_dir, check=True)
    print("✅ Seed-VC environment ready")


def _tail(s: str | None, n: int = 1200) -> str:
    return (s or "(empty)")[-n:]


def run_seedvc(source_path: str | Path, target_path: str | Path, out_dir: Path, cfg: SeedVCConfig) -> Path:
    """Invoke seed-vc's inference.py: `source_path` is the VocalSound clip (the sound to
    re-voice), `target_path` is the speaker recording whose timbre it should take on.

    Detects its output by mtime >= when this call started, not "wasn't in a before-
    snapshot" — seed-vc's output filenames are deterministic
    (vc_{source_stem}_{target_stem}_..._.wav), so a same-named file from an earlier run may
    already exist in out_dir before this call starts; a before/after set-difference then
    can't see this run's (correctly rewritten) file at all and reports "no output" even
    though seed-vc succeeded.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    import torch

    cmd = [
        "python", "inference.py",
        "--source", str(source_path), "--target", str(target_path), "--output", str(out_dir),
        "--diffusion-steps", str(cfg.diffusion_steps), "--length-adjust", str(cfg.length_adjust),
        "--inference-cfg-rate", str(cfg.cfg_rate), "--f0-condition", str(cfg.f0_condition),
        "--auto-f0-adjust", "False", "--semi-tone-shift", "0",
        "--fp16", str(torch.cuda.is_available()),
    ]
    result = subprocess.run(cmd, cwd=cfg.repo_dir, capture_output=True, text=True)
    written = sorted(
        (f for f in glob.glob(str(out_dir / "*.wav")) if Path(f).stat().st_mtime >= start - 1),
        key=lambda f: Path(f).stat().st_mtime,
    )
    if result.returncode != 0 or not written:
        raise RuntimeError(
            f"Seed-VC produced no output for {Path(target_path).name} (exit code {result.returncode}):\n"
            f"--- stdout (tail) ---\n{_tail(result.stdout)}\n"
            f"--- stderr (tail) ---\n{_tail(result.stderr, 500)}"
        )
    return Path(written[-1])

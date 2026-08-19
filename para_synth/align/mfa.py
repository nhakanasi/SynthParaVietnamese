"""Montreal Forced Aligner (MFA) backend: conda/Kaldi bootstrap + batch corpus alignment.

Ported from notebook45ee5319ae.ipynb cells 9 (setup) and 10 (batch align + TextGrid
parsing). Second choice in the alignment chain, behind Qwen3-ForcedAligner — MFA needs a
pronunciation dictionary, so words outside `vietnamese_mfa`'s vocabulary (common for
Vietnamese colloquial fillers/particles: à, ha, thì, …) get marked "spn" (speech-noise) with
untrustworthy timing even though the word count still lines up. This module rejects any
boundary touching an `spn` word rather than trusting it — see `mfa_insert_time` below —
which is exactly the class of bug the notebook was diagnosing in its own cell 22 markdown.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from para_synth.align._text import clean_word, words_around_tag
from para_synth.config import AlignmentConfig
from para_synth.dataset import ManifestRow


def _find_conda_bin(conda_dir: Path) -> str | None:
    which = shutil.which("conda")
    if which:
        return which
    for root in [Path("/opt/conda"), conda_dir, Path.home() / "miniconda3", Path.home() / "anaconda3"]:
        candidate = root / "bin" / "conda"
        if candidate.is_file():
            return str(candidate)
    return None


def _find_mfa_bin(env_name: str, conda_bin: str | None, conda_dir: Path) -> str | None:
    roots = [Path(conda_bin).parent.parent] if conda_bin else []
    roots += [Path("/opt/conda"), conda_dir, Path.home() / "miniconda3", Path.home() / "anaconda3"]
    for conda_root in roots:
        candidate = conda_root / "envs" / env_name / "bin" / "mfa"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("mfa")


def parse_textgrid_words(path: Path) -> list[tuple[float, float, str]]:
    """Minimal Praat long-format TextGrid parser — pulls just the 'words' IntervalTier's
    (xmin, xmax, text) triples, in order. Self-contained on purpose: avoids adding another
    pip dependency (textgrid/praatio) for a format this simple and this well-specified."""
    content = Path(path).read_text(encoding="utf-8")
    for block in re.split(r"item \[\d+\]:", content)[1:]:
        name_m = re.search(r'name\s*=\s*"([^"]*)"', block)
        if name_m and name_m.group(1) == "words":
            return [
                (float(m.group(1)), float(m.group(2)), m.group(3))
                for m in re.finditer(
                    r'xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*"([^"]*)"',
                    block,
                )
            ]
    return []


class MFAAligner:
    def __init__(self, cfg: AlignmentConfig):
        self.cfg = cfg
        self.ready = False
        self.conda_bin: str | None = None
        self.mfa_bin: str | None = None

    def _mfa_cmd(self, *args: str) -> list[str]:
        """Run mfa via `conda run -n <env>`, not a bare absolute path — mfa shells out
        internally to other conda-installed tools (openfst's fstcompile, for alignment
        specifically) via PATH lookup, and a bare absolute-path call never adds the env's
        bin/ to PATH. Confirmed by the source notebook: a bare-path 'align' call failed
        with ThirdpartyError: Could not find 'fstcompile' even though mfa itself ran."""
        return [self.conda_bin, "run", "-n", self.cfg.mfa_env_name, "mfa", *args]

    def setup(self) -> bool:
        """Bootstrap Miniforge (if no conda exists at all) -> create the `aligner` conda
        env with montreal-forced-aligner -> download the acoustic model + dictionary.
        One-time, multi-minute cost. Returns False (non-fatal) on any failure so callers
        fall through to the next alignment stage instead of crashing."""
        try:
            self.conda_bin = _find_conda_bin(self.cfg.conda_dir)
            if self.conda_bin is None:
                print("📦 No conda found — bootstrapping Miniforge …")
                installer = self.cfg.conda_dir.parent / "miniforge.sh"
                installer.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        "curl", "-fsSL", "-o", str(installer),
                        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-"
                        + ("MacOSX" if shutil.os.uname().sysname == "Darwin" else "Linux")
                        + "-"
                        + shutil.os.uname().machine
                        + ".sh",
                    ],
                    check=True,
                )
                subprocess.run(["bash", str(installer), "-b", "-p", str(self.cfg.conda_dir)], check=True)
                self.conda_bin = str(self.cfg.conda_dir / "bin" / "conda")
                assert Path(self.conda_bin).is_file(), "Miniforge installer ran but its conda binary is still missing"
                print(f"✅ Miniforge installed: {self.conda_bin}")
            else:
                print(f"✅ conda found: {self.conda_bin}")

            self.mfa_bin = _find_mfa_bin(self.cfg.mfa_env_name, self.conda_bin, self.cfg.conda_dir)
            if self.mfa_bin is None:
                print(f"📦 Creating conda env '{self.cfg.mfa_env_name}' with Montreal Forced Aligner "
                      f"(this can take several minutes — Kaldi is a large conda-forge package)…")
                subprocess.run(
                    [self.conda_bin, "create", "-n", self.cfg.mfa_env_name, "-c", "conda-forge",
                     "montreal-forced-aligner", "-y", "-q"],
                    check=True,
                )
                self.mfa_bin = _find_mfa_bin(self.cfg.mfa_env_name, self.conda_bin, self.cfg.conda_dir)
                assert self.mfa_bin, f"conda env '{self.cfg.mfa_env_name}' created but no mfa binary found in it"
            else:
                print(f"✅ '{self.cfg.mfa_env_name}' conda env already has mfa at {self.mfa_bin}")

            print(f"📥 Downloading {self.cfg.mfa_acoustic_model} acoustic model + {self.cfg.mfa_dictionary} dictionary…")
            subprocess.run(self._mfa_cmd("model", "download", "acoustic", self.cfg.mfa_acoustic_model), check=True)
            subprocess.run(self._mfa_cmd("model", "download", "dictionary", self.cfg.mfa_dictionary), check=True)

            self.ready = True
            print(f"✅ MFA ready: {self.mfa_bin}")
        except Exception as e:
            print(f"⚠️  MFA setup failed ({type(e).__name__}: {e})")
            print("   → alignment falls back to the next stage for every row")
        return self.ready

    def align_batch(
        self, rows: list[ManifestRow], corpus_dir: Path, out_dir: Path
    ) -> dict[str, float | None]:
        """Build a one-shot MFA corpus for `rows`, run `mfa align` once, and return a
        {row.id: insert_time_s | None} map. None means "no usable MFA boundary for this
        row" (missing alignment, word-count mismatch, or a boundary touching an OOV `spn`
        word) — the caller should fall through to the next alignment stage for those rows.
        """
        corpus_dir, out_dir = Path(corpus_dir), Path(out_dir)
        corpus_dir.mkdir(parents=True, exist_ok=True)

        row_words: dict[str, tuple[list[str], list[str]]] = {}
        for row in rows:
            before, after = words_around_tag(row.text)
            if before is None:
                continue
            row_words[row.id] = (before, after)
            shutil.copy(row.audio_filepath, corpus_dir / f"{row.id}.wav")
            (corpus_dir / f"{row.id}.lab").write_text(" ".join(before + after), encoding="utf-8")

        print(f"📁 corpus: {len(row_words)}/{len(rows)} rows written to {corpus_dir}")
        if not row_words:
            return {}

        print("⏳ running mfa align (loads the acoustic model once for the whole batch)…")
        try:
            subprocess.run(
                self._mfa_cmd("align", str(corpus_dir), self.cfg.mfa_acoustic_model,
                               self.cfg.mfa_dictionary, str(out_dir), "--clean", "-j", "2"),
                check=True, capture_output=True, text=True,
            )
            print("✅ mfa align complete")
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or str(e))[-800:]
            print(f"⚠️  mfa align failed — every row falls back to the next stage:\n{tail}")
            return {}

        results: dict[str, float | None] = {}
        for row_id, (before, after) in row_words.items():
            tg_path = out_dir / f"{row_id}.TextGrid"
            if not tg_path.is_file():
                results[row_id] = None
                continue
            intervals = [(xmin, xmax, t) for xmin, xmax, t in parse_textgrid_words(tg_path) if t.strip()]
            n_before = len(before)
            if len(intervals) != n_before + len(after) or not (0 < n_before < len(intervals)):
                results[row_id] = None
                continue
            if intervals[n_before - 1][2] == "spn" or intervals[n_before][2] == "spn":
                results[row_id] = None  # OOV word touching the boundary — timing untrustworthy
                continue
            results[row_id] = intervals[n_before - 1][1]  # xmax of the last "before" word
        return results

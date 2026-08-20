"""Environment doctor: non-fatal checks for the dependency landmines notebook45ee5319ae.ipynb
cell 3 had to work around. Run via `para-synth doctor` before a first real pipeline run.

These are Kaggle-base-image quirks originally, but the underlying conflicts (numpy version
skew between seed-vc and scipy/librosa, transformers probing for tensorflow, peft's optional
import chain crashing without peft actually being used) can show up in any environment that
mixes seed-vc's pinned requirements with a general ML base image — hence checking for them
here instead of assuming they're Kaggle-only.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import shutil


def _check_import(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, "importable"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _version_tuple(v: str) -> tuple[int, ...]:
    """"4.10.2" -> (4, 10, 2). Plain string comparison ("10.0" >= "4.0") gives wrong
    answers once a version's leading component crosses a digit-count boundary, so
    version-gated checks below compare parsed tuples instead."""
    parts = []
    for p in v.split(".")[:3]:
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _check_torch_cuda() -> list[str]:
    """Warn when torch can't actually use the GPU this machine has.

    Worth its own check because the failure is silent and expensive rather than loud:
    everything still runs, just on CPU, ~32x slower for Seed-VC (measured: 530s vs 16.5s
    per row). seed-vc's own requirements.txt causes it — the file asks for nightly cu126
    wheels in its first three lines and then pins bare `torch==2.4.0` further down, so
    pip's last-write-wins resolution installs the CPU build over the CUDA one.

    Two distinct failures, hence two branches: a CPU-only build on a machine that has a
    driver, and a CUDA build too old to have kernels for the GPU's architecture (torch
    2.4 has nothing for a Blackwell sm_120 card, so it reports no GPU at all rather than
    erroring).
    """
    out: list[str] = []
    try:
        import torch
    except ImportError:
        return ["torch is not installed"]

    build = torch.version.cuda or "cpu-only"
    print(f"torch: {torch.__version__} (cuda {build})")

    driver_present = shutil.which("nvidia-smi") is not None
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        arch = f"sm_{cap[0]}{cap[1]}"
        print(f"gpu: {name} ({arch})")
        if arch not in torch.cuda.get_arch_list():
            out.append(
                f"torch has no kernels for {arch} (built for "
                f"{', '.join(torch.cuda.get_arch_list())}) — install a build new enough for "
                f"this GPU or every CUDA op will fail at launch."
            )
    elif driver_present:
        out.append(
            "an NVIDIA driver is present but torch reports no usable GPU"
            + (" — this is a CPU-only torch build. " if not torch.version.cuda else " — ")
            + "Reinstall from the CUDA index matching your card, e.g. "
            "`pip install --index-url https://download.pytorch.org/whl/cu128 "
            "torch torchaudio torchvision`. Seed-VC is ~32x slower without it."
        )
    else:
        print("gpu: none detected (CPU-only run; Seed-VC will be very slow)")
    return out


def run_doctor() -> list[str]:
    """Print + return a list of warning strings (empty if everything looks fine)."""
    warnings: list[str] = []

    try:
        import numpy

        print(f"numpy: {numpy.__version__}")
        if _version_tuple(numpy.__version__)[0] >= 2:
            warnings.append(
                "numpy is 2.x — seed-vc's requirements.txt expects numpy 1.x; a scipy/librosa "
                "built against numpy 2.x can then fail with \"No module named numpy.strings\". "
                "audio_utils.py avoids librosa/scipy entirely for this reason, but seed-vc's "
                "own inference.py subprocess may still hit it."
            )
    except ImportError:
        warnings.append("numpy is not installed")

    try:
        protobuf_version = md.version("protobuf")
        print(f"protobuf: {protobuf_version}")
        if _version_tuple(protobuf_version) >= (4, 0, 0):
            warnings.append(
                "protobuf >= 4.0.0 — seed-vc's requirements.txt pins protobuf<4.0.0; a mismatch "
                "can surface as \"cannot import name 'runtime_version' from 'google.protobuf'\" "
                "when transformers probes for tensorflow (see quality.py's USE_TF=0 workaround)."
            )
    except md.PackageNotFoundError:
        pass

    try:
        md.version("peft")
        warnings.append(
            "peft is installed — WavLM's TDNN layer (used by quality.py's speaker-similarity "
            "check) conditionally imports peft at call time via is_peft_available(), which can "
            "crash on an old peft even though nothing here actually uses LoRA/peft. If "
            "quality.py fails on import, `pip uninstall peft` (nothing in this repo needs it)."
        )
    except md.PackageNotFoundError:
        pass

    warnings.extend(_check_torch_cuda())

    ok, detail = _check_import("uroman")
    if ok:
        print("uroman: importable")
    else:
        print(f"uroman: not available ({detail}) — para_synth.align.mms will fall back to "
              f"manual de-accenting for Vietnamese romanization")

    for w in warnings:
        print(f"⚠️  {w}")
    if not warnings:
        print("✅ no known environment landmines detected")
    return warnings

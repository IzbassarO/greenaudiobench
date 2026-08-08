"""Shared utilities: hashing, provenance metadata, official-run guards, seeding."""

from __future__ import annotations

import datetime
import hashlib
import platform
import random
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Provenance columns that every official results CSV must carry.
PROVENANCE_COLUMNS = (
    "git_commit", "git_dirty", "timestamp", "python_version",
    "torch_version", "cuda_version", "gpu_name", "driver_version",
)


class ProvenanceError(RuntimeError):
    """Raised when an official run would be recorded with broken provenance."""


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed (archives/checkpoints can be several GB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def utc_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def git_commit() -> str:
    try:
        return _git("rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


#: Experiment-owned output dirs: the pipeline's own results must not make the
#: tree look dirty (M3 writes results/accuracy.csv -> M4/M5 must still start).
#: ONLY these are excluded — source/config/test changes still count as dirty.
_DIRTY_EXCLUDES = (":(exclude)results", ":(exclude)figures")


def git_dirty() -> bool:
    """True when the working tree differs from HEAD (untracked files count),
    ignoring only the experiment-owned output directories results/ and
    figures/."""
    try:
        return bool(_git("status", "--porcelain", "--", ".", *_DIRTY_EXCLUDES))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True  # cannot prove the tree is clean -> treat as dirty


def run_metadata() -> dict:
    """Provenance for results CSVs. Keys are exactly PROVENANCE_COLUMNS."""
    torch_version = "not_installed"
    cuda_version = "none"
    gpu_name = "none"
    driver_version = "none"
    try:
        import torch  # noqa: PLC0415 — heavyweight, keep optional

        torch_version = torch.__version__
        if torch.cuda.is_available():
            cuda_version = str(torch.version.cuda)
            gpu_name = torch.cuda.get_device_name(0)
            try:
                import pynvml  # noqa: PLC0415

                pynvml.nvmlInit()
                driver_version = str(pynvml.nvmlSystemGetDriverVersion())
                pynvml.nvmlShutdown()
            except Exception:  # NVML optional; absence must not kill a run
                driver_version = "unavailable"
    except ImportError:
        pass
    return {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "timestamp": utc_timestamp(),
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
    }


def assert_official_run(require_cuda: bool = True, allow_dirty: bool = False) -> dict:
    """Gate for scripts that write official results/*.csv.

    - refuses a dirty working tree unless allow_dirty=True (an explicit
      --allow-dirty debug override; git_dirty is still recorded either way);
    - refuses to run without CUDA when require_cuda=True, so CPU smoke runs
      can never masquerade as official GPU results.
    Returns run_metadata() on success.
    """
    meta = run_metadata()
    if meta["git_dirty"] and not allow_dirty:
        raise ProvenanceError(
            "working tree is dirty — commit your changes or pass an explicit "
            "--allow-dirty debug override (results will be marked git_dirty=True)"
        )
    if require_cuda and meta["gpu_name"] == "none":
        raise ProvenanceError(
            "official run requires a CUDA GPU (Colab T4); "
            "CPU smoke tests must use --smoke and never write to results/"
        )
    return meta


def smoke_output_dir() -> Path:
    """Where --smoke runs write their outputs. NEVER results/ (official only)."""
    out = REPO_ROOT / "data" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int) -> None:
    """Seed all RNGs in use (torch included once installed)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

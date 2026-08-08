"""Shared utilities: hashing, run metadata for results CSVs, seeding."""

from __future__ import annotations

import datetime
import hashlib
import random
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed (archives can be several GB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def utc_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_metadata() -> dict[str, str]:
    """Mandatory provenance columns for every results CSV (hard rule #4)."""
    torch_version = "not_installed"
    gpu_name = "none"
    try:
        import torch  # noqa: PLC0415 — optional until M2

        torch_version = torch.__version__
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "git_commit": git_commit(),
        "gpu_name": gpu_name,
        "torch_version": torch_version,
        "timestamp": utc_timestamp(),
    }


def set_seed(seed: int) -> None:
    """Seed all RNGs in use. Torch is seeded too once it is installed (M2+)."""
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

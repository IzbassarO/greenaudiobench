"""Dataset specs, file resolution and statistics for ESC-50 and UrbanSound8K."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import soundfile as sf

from .folds import ESC50_FOLDS, US8K_FOLDS, load_esc50_meta, load_us8k_meta


@dataclass(frozen=True)
class DatasetSpec:
    key: str                 # short id used in paths ("esc50", "urbansound8k")
    pretty_name: str
    root_subdir: str         # top-level dir inside the extracted archive
    meta_relpath: str        # metadata CSV, relative to root_subdir
    filename_column: str
    label_column: str
    expected_clips: int      # from the official dataset description
    expected_classes: int
    folds: tuple[int, ...]


ESC50 = DatasetSpec(
    key="esc50",
    pretty_name="ESC-50",
    root_subdir="ESC-50-master",
    meta_relpath="meta/esc50.csv",
    filename_column="filename",
    label_column="category",
    expected_clips=2000,
    expected_classes=50,
    folds=ESC50_FOLDS,
)

US8K = DatasetSpec(
    key="urbansound8k",
    pretty_name="UrbanSound8K",
    root_subdir="UrbanSound8K",
    meta_relpath="metadata/UrbanSound8K.csv",
    filename_column="slice_file_name",
    label_column="class",
    expected_clips=8732,
    expected_classes=10,
    folds=US8K_FOLDS,
)

SPECS: dict[str, DatasetSpec] = {ESC50.key: ESC50, US8K.key: US8K}


def dataset_root(spec: DatasetSpec, data_dir: str | Path) -> Path:
    """Root of the extracted dataset, e.g. data/raw/esc50/ESC-50-master."""
    return Path(data_dir) / "raw" / spec.key / spec.root_subdir


def load_meta(spec: DatasetSpec, data_dir: str | Path) -> pd.DataFrame:
    csv_path = dataset_root(spec, data_dir) / spec.meta_relpath
    if spec.key == "esc50":
        return load_esc50_meta(csv_path)
    if spec.key == "urbansound8k":
        return load_us8k_meta(csv_path)
    raise ValueError(f"unknown dataset {spec.key}")


def audio_path(spec: DatasetSpec, root: Path, row: pd.Series) -> Path:
    """Absolute path of one clip given its metadata row."""
    if spec.key == "esc50":
        return root / "audio" / row["filename"]
    if spec.key == "urbansound8k":
        return root / "audio" / f"fold{row['fold']}" / row["slice_file_name"]
    raise ValueError(f"unknown dataset {spec.key}")


def compute_stats(spec: DatasetSpec, data_dir: str | Path) -> dict:
    """Scan every clip referenced by the metadata; report counts/durations/rates.

    Reads only WAV headers (soundfile.info), so this is fast even for US8K.
    """
    root = dataset_root(spec, data_dir)
    meta = load_meta(spec, data_dir)

    durations: list[float] = []
    samplerates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    missing: list[str] = []
    unreadable: list[str] = []
    for _, row in meta.iterrows():
        path = audio_path(spec, root, row)
        if not path.is_file():
            missing.append(str(path.relative_to(root)))
            continue
        try:
            info = sf.info(str(path))
        except sf.LibsndfileError:
            unreadable.append(str(path.relative_to(root)))
            continue
        durations.append(info.frames / info.samplerate)
        samplerates[info.samplerate] += 1
        channels[info.channels] += 1

    per_fold = meta["fold"].value_counts().sort_index()
    return {
        "dataset": spec.pretty_name,
        "n_meta_rows": int(len(meta)),
        "n_files_found": len(durations),
        "n_missing": len(missing),
        "missing_examples": missing[:5],
        "n_unreadable": len(unreadable),
        "unreadable_examples": unreadable[:5],
        "n_classes": int(meta[spec.label_column].nunique()),
        "folds": {int(k): int(v) for k, v in per_fold.items()},
        "total_duration_sec": round(sum(durations), 1),
        "total_duration_hours": round(sum(durations) / 3600, 3),
        "min_duration_sec": round(min(durations), 3) if durations else None,
        "max_duration_sec": round(max(durations), 3) if durations else None,
        "samplerates": {int(k): int(v) for k, v in samplerates.most_common()},
        "channels": {int(k): int(v) for k, v in channels.most_common()},
    }


def cross_check(spec: DatasetSpec, stats: dict) -> list[str]:
    """Compare scanned stats against the official dataset description.

    Returns a list of failure messages; empty list == all checks passed.
    """
    failures = []
    if stats["n_meta_rows"] != spec.expected_clips:
        failures.append(
            f"metadata rows {stats['n_meta_rows']} != expected {spec.expected_clips}"
        )
    if stats["n_files_found"] != stats["n_meta_rows"]:
        failures.append(
            f"files found {stats['n_files_found']} != metadata rows {stats['n_meta_rows']}"
            f" ({stats['n_missing']} missing, {stats['n_unreadable']} unreadable)"
        )
    if stats["n_classes"] != spec.expected_classes:
        failures.append(f"classes {stats['n_classes']} != expected {spec.expected_classes}")
    if set(stats["folds"]) != set(spec.folds):
        failures.append(f"folds {sorted(stats['folds'])} != official {list(spec.folds)}")
    return failures

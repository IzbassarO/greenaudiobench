#!/usr/bin/env python3
"""Download GreenAudioBench datasets from FIRST-PARTY sources only.

  esc50        : GitHub karolpiczak/ESC-50 (archive/refs/heads/master.zip)
  urbansound8k : Zenodo, DOI 10.5281/zenodo.1203745

Hard rules honored:
  - never Kaggle / mirrors;
  - resume + retry on network failure;
  - SHA-256 recorded in data/CHECKSUMS.txt from the ACTUAL downloaded file;
  - official folds untouched; clip counts cross-checked against metadata.

Idempotent: verified archives are not re-downloaded, extracted datasets are
not re-extracted. Use --force to redo everything for a dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gab.datasets import SPECS, compute_stats, cross_check, dataset_root  # noqa: E402
from gab.utils import sha256_file, utc_timestamp  # noqa: E402

# ESC-50 is pinned to the exact commit we downloaded and verified on
# 2026-08-08 (the commit sha is embedded as the zip archive comment of the
# downloaded esc50-master.zip and matches upstream master of that date).
ESC50_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"

SOURCES = {
    "esc50": {
        "url": f"https://github.com/karolpiczak/ESC-50/archive/{ESC50_COMMIT}.zip",
        "archive": "esc50-master.zip",
        "kind": "zip",
        # a pinned-commit archive extracts to ESC-50-<sha>/ — rename to the
        # canonical root so datasets.py sees a stable path
        "rename_root": (f"ESC-50-{ESC50_COMMIT}", "ESC-50-master"),
        "src_revision": ESC50_COMMIT,
    },
    "urbansound8k": {
        "url": "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1",
        "archive": "UrbanSound8K.tar.gz",
        "kind": "tar",
        "src_revision": "zenodo DOI 10.5281/zenodo.1203745",
    },
}

CHUNK = 1 << 20  # 1 MiB


def log(msg: str) -> None:
    print(f"[download_data] {msg}", flush=True)


def download_with_resume(url: str, dest: Path, max_retries: int = 8) -> None:
    """Stream url to dest, resuming a partial .part file when the server allows."""
    if dest.exists():
        log(f"archive already present, skipping download: {dest.name}")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, max_retries + 1):
        pos = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        try:
            with requests.get(url, stream=True, headers=headers,
                              timeout=(30, 300), allow_redirects=True) as r:
                if pos and r.status_code == 200:
                    # server ignored the Range header — restart from scratch
                    log("server does not support resume, restarting download")
                    pos = 0
                elif r.status_code not in (200, 206):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                total = r.headers.get("Content-Length")
                total_h = f"{(pos + int(total)) / 1e9:.2f} GB" if total else "unknown size"
                log(f"downloading {dest.name} ({total_h}), attempt {attempt}"
                    + (f", resuming at {pos / 1e9:.2f} GB" if pos else ""))
                mode = "ab" if pos else "wb"
                done = pos
                last_report = time.monotonic()
                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        if time.monotonic() - last_report > 30:
                            log(f"  ... {done / 1e9:.2f} GB")
                            last_report = time.monotonic()
            part.rename(dest)
            log(f"downloaded {dest.name}: {dest.stat().st_size / 1e9:.2f} GB")
            return
        except (requests.RequestException, OSError) as exc:
            wait = min(10 * attempt, 60)
            log(f"download interrupted ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url} after {max_retries} attempts")


def extract_archive(archive: Path, kind: str, target_dir: Path, marker: Path,
                    spec_rename_root: tuple[str, str] | None = None) -> None:
    if marker.exists():
        log(f"already extracted, skipping: {target_dir}")
        return
    log(f"extracting {archive.name} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "zip":
            with zipfile.ZipFile(archive) as zf:
                bad = zf.testzip()  # CRC check of every member
                if bad is not None:
                    raise zipfile.BadZipFile(f"CRC failure in member {bad}")
                zf.extractall(target_dir)
        elif kind == "tar":
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(target_dir, filter="data")
        else:
            raise ValueError(f"unknown archive kind {kind}")
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as exc:
        # corrupted archive: remove it so the next run re-downloads
        log(f"ARCHIVE CORRUPTED ({exc}); deleting {archive.name} — rerun to re-download")
        archive.unlink(missing_ok=True)
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    rename = spec_rename_root
    if rename is not None:
        src, dst = target_dir / rename[0], target_dir / rename[1]
        if src.is_dir() and not dst.exists():
            src.rename(dst)
    if not marker.exists():
        raise RuntimeError(f"extraction finished but expected path missing: {marker}")


class ChecksumMismatch(SystemExit):
    pass


def _parse_checksums(checksums_path: Path) -> dict[str, tuple[str, str]]:
    """rel_name -> (comment line, checksum line)."""
    entries: dict[str, tuple[str, str]] = {}
    if checksums_path.exists():
        comment = ""
        for line in checksums_path.read_text().splitlines():
            if line.startswith("# file:"):
                comment = line
            elif line and not line.startswith("#"):
                name = line.split(maxsplit=1)[1].strip()
                entries[name] = (comment, line)
                comment = ""
    return entries


def recorded_sha(checksums_path: Path, rel_name: str) -> str | None:
    entry = _parse_checksums(checksums_path).get(rel_name)
    return entry[1].split()[0] if entry else None


def update_checksums(checksums_path: Path, rel_name: str, sha: str,
                     size: int, when: str, src_revision: str = "",
                     force: bool = False) -> None:
    """VERIFY against the committed record; never silently overwrite it.

    - entry exists and hash matches -> keep the original line untouched
      (including the original downloaded_utc — no churn in a committed file);
    - entry exists and hash DIFFERS -> abort loudly (the on-disk archive does
      not match the committed provenance record); --force accepts the new
      value explicitly;
    - no entry -> record it (first actual download).
    File format stays `shasum -a 256 -c` compatible; written atomically.
    """
    header = (
        "# GreenAudioBench dataset archive checksums.\n"
        "# SHA-256 recorded from the ACTUAL downloaded files (never copied).\n"
        "# Verify with: cd data && shasum -a 256 -c CHECKSUMS.txt\n"
    )
    entries = _parse_checksums(checksums_path)
    existing = entries.get(rel_name)
    if existing is not None:
        old_sha = existing[1].split()[0]
        if old_sha == sha and not force:
            log(f"checksum VERIFIED against committed record: {rel_name}")
            return
        if old_sha != sha and not force:
            raise ChecksumMismatch(
                f"CHECKSUM MISMATCH for {rel_name}:\n"
                f"  committed : {old_sha}\n"
                f"  on disk   : {sha}\n"
                "The archive does not match the committed provenance record. "
                "Delete the archive to re-download, or re-run with --force to "
                "accept the new hash (this rewrites committed provenance)."
            )
        log(f"--force: replacing recorded checksum for {rel_name}")
    rev = f"  src_revision={src_revision}" if src_revision else ""
    entries[rel_name] = (
        f"# file: {rel_name}  size_bytes={size}  downloaded_utc={when}{rev}",
        f"{sha}  {rel_name}",
    )
    lines = [header]
    for name in sorted(entries):
        comment, checksum_line = entries[name]
        lines.append(comment + "\n" + checksum_line + "\n")
    tmp = checksums_path.with_suffix(".txt.tmp")
    tmp.write_text("\n".join(lines))
    tmp.replace(checksums_path)


def process_dataset(key: str, data_dir: Path, force: bool) -> dict:
    spec = SPECS[key]
    src = SOURCES[key]
    archive_dir = data_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / src["archive"]
    root = dataset_root(spec, data_dir)          # e.g. data/raw/esc50/ESC-50-master
    extract_target = root.parent                 # e.g. data/raw/esc50
    marker = root / spec.meta_relpath            # metadata CSV == extraction done

    if force:
        log(f"--force: removing {archive} and {extract_target}")
        archive.unlink(missing_ok=True)
        shutil.rmtree(extract_target, ignore_errors=True)

    log(f"=== {spec.pretty_name} ===")
    download_with_resume(src["url"], archive)

    log(f"computing SHA-256 of {archive.name} ...")
    sha = sha256_file(archive)
    update_checksums(data_dir / "CHECKSUMS.txt", f"archives/{src['archive']}",
                     sha, archive.stat().st_size, utc_timestamp(),
                     src_revision=src.get("src_revision", ""), force=force)
    log(f"sha256({archive.name}) = {sha}")

    extract_archive(archive, src["kind"], extract_target, marker,
                    spec_rename_root=src.get("rename_root"))

    log("scanning clips (headers only) and cross-checking against metadata ...")
    stats = compute_stats(spec, data_dir)
    failures = cross_check(spec, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    if failures:
        for f in failures:
            log(f"CROSS-CHECK FAILED [{key}]: {f}")
        raise SystemExit(f"{key}: dataset failed cross-checks — see messages above")
    log(f"{spec.pretty_name}: all cross-checks passed "
        f"({stats['n_files_found']} clips, {stats['total_duration_hours']} h)")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(SOURCES),
                        default=["esc50", "urbansound8k"])
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--force", action="store_true",
                        help="re-download and re-extract even if cached")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_stats = {}
    for key in args.datasets:
        all_stats[key] = process_dataset(key, data_dir, args.force)

    stats_path = data_dir / "raw" / "dataset_stats.json"
    existing = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    existing.update(all_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    log(f"stats written to {stats_path}")
    log("done.")


if __name__ == "__main__":
    main()

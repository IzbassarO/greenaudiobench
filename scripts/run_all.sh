#!/usr/bin/env bash
# GreenAudioBench end-to-end pipeline. Idempotent: every stage skips
# already-cached artifacts. GPU stages require CUDA (Colab T4).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PY" ]; then PY=python3; fi
echo "run_all: using $PY"

# --- M1: data download + checksums + stats ---------------------------------
"$PY" scripts/download_data.py --datasets esc50 urbansound8k

# --- M2: embedding extraction (GPU; Colab T4) ------------------------------
# "$PY" scripts/extract_embeddings.py            # TODO(M2)

# --- M3/M6: linear probes ---------------------------------------------------
# "$PY" scripts/run_probes.py                    # TODO(M3)

# --- M4: zero-shot CLAP -----------------------------------------------------
# "$PY" scripts/run_zeroshot.py                  # TODO(M4)

# --- M5: efficiency measurements (GPU; Colab T4) ----------------------------
# "$PY" scripts/measure_efficiency.py            # TODO(M5)

# --- M7: tables + figures ---------------------------------------------------
# "$PY" scripts/make_tables.py                   # TODO(M7)

echo "run_all: done (stages beyond M1 are not implemented yet)"

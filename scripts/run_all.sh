#!/usr/bin/env bash
# GreenAudioBench end-to-end pipeline. Idempotent: every stage skips
# already-cached artifacts. GPU stages require CUDA (official env: Colab T4).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then PY=python3; fi
echo "run_all: using $PY"

# --- M1: data download + checksum verification + stats ----------------------
"$PY" scripts/download_data.py --datasets esc50 urbansound8k

# --- unit tests (CPU, no checkpoints) ---------------------------------------
"$PY" -m pytest -q

# --- GPU stages (official: Colab T4 only) -----------------------------------
if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  "$PY" scripts/extract_embeddings.py --dataset esc50 --batch-size 16   # M2
  "$PY" scripts/run_probes.py --dataset esc50                          # M3
  "$PY" scripts/run_zeroshot.py --dataset esc50                        # M4
  "$PY" scripts/measure_efficiency.py                                  # M5
else
  echo "run_all: no CUDA device — GPU stages (M2/M4/M5) skipped; they are"
  echo "run_all: Colab-only (see notebooks/colab_runner.ipynb). M3 runs once"
  echo "run_all: embedding caches exist."
fi

# --- M7: tables + figures ---------------------------------------------------
# "$PY" scripts/make_tables.py                   # TODO(M7)

echo "run_all: done"

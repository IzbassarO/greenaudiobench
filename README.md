# GreenAudioBench

A reproducible accuracy–energy–latency benchmark of audio foundation models
(PANNs CNN14, AST, BEATs, LAION-CLAP, MS-CLAP) for environmental sound
classification on ESC-50 and UrbanSound8K, runnable end-to-end on a free
Colab T4 GPU.

Two evaluation protocols:

- **P1 — zero-shot** (CLAP family): cosine similarity between audio and text
  embeddings of class-name prompts.
- **P2 — frozen linear probe** (all models): cached frozen embeddings +
  scikit-learn logistic regression, official cross-validation folds, 5 seeds.

Efficiency metrics per model (on NVIDIA T4): parameters, GMACs, latency
(ms/clip, batch 1 and 32) and measured energy (J/clip via NVML power
sampling, cross-checked with CodeCarbon), fp32 and fp16.

## Setup

```bash
uv venv --python 3.13 .venv          # or: python3.13 -m venv .venv
uv pip install -r env/requirements.txt
```

Exact package versions used in experiments: `env/requirements-lock.txt`.

## Usage

```bash
bash scripts/run_all.sh              # idempotent, skips cached artifacts
```

Individual stages:

```bash
.venv/bin/python scripts/download_data.py --datasets esc50 urbansound8k
.venv/bin/python -m pytest           # unit tests (no audio/checkpoints needed)
# GPU stages (official env: Colab T4 — notebooks/colab_runner.ipynb):
.venv/bin/python scripts/extract_embeddings.py --dataset esc50   # M2
.venv/bin/python scripts/run_probes.py --dataset esc50           # M3 (CPU ok)
.venv/bin/python scripts/run_zeroshot.py --dataset esc50         # M4
.venv/bin/python scripts/measure_efficiency.py                   # M5
# every GPU script supports --smoke for a tiny CPU sanity run that can
# never write into results/ (outputs go to data/smoke/)
```

Model checkpoints are pinned (revision + SHA-256) in
`src/gab/models/registry.py` and documented in `env/MODELS.md`; official runs
refuse dirty working trees and unpinned checkpoints.

## Data

Datasets are downloaded **only from first-party sources** (never mirrors):

| Dataset | Source | License |
|---------|--------|---------|
| ESC-50 | github.com/karolpiczak/ESC-50 | CC BY-NC 3.0 |
| UrbanSound8K | Zenodo, DOI 10.5281/zenodo.1203745 | CC BY-NC 3.0 |

SHA-256 checksums of the downloaded archives are recorded in
`data/CHECKSUMS.txt` (committed). Official cross-validation folds are used
as published — never re-split, never merged. Raw audio is not committed and
must not be redistributed.

If you use these datasets, cite the original papers:

- K. J. Piczak, "ESC: Dataset for Environmental Sound Classification,"
  *Proc. ACM Multimedia*, 2015.
- J. Salamon, C. Jacoby, and J. P. Bello, "A Dataset and Taxonomy for Urban
  Sound Research," *Proc. ACM Multimedia*, 2014.

## Status

- [x] **M0** — repo skeleton, env files, fold-parsing unit tests
- [x] **M1** — ESC-50 + UrbanSound8K downloaded, checksums, loaders with
      official folds, dataset stats cross-checked against metadata
- [x] **M2** — embedding extraction implemented (5 adapters, ESC-50);
      official extraction runs on Colab T4
- [x] **M3** — linear probe pipeline implemented (deterministic, fold-faithful)
- [x] **M4** — zero-shot CLAP implemented
- [x] **M5** — efficiency measurement implemented (NVML energy protocol);
      official numbers T4-only, not yet run
- [ ] M6 — UrbanSound8K (repeat M2–M5)
- [ ] M7 — tables + Pareto figures
- [ ] M8 — TAU 2020 Mobile (optional)

## License

Code: MIT (see `LICENSE`). Datasets keep their own non-commercial licenses.

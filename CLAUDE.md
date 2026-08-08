# CLAUDE.md — GreenAudioBench

## Project
**Paper:** "GreenAudioBench: A Reproducible Accuracy–Energy–Latency Benchmark of Audio Foundation Models for Environmental Sound Classification"
**Target venue:** IEEE UEMCON 2026 (regular paper, ≤7 pages, IEEE two-column, double-blind, submission via EDAS). **Fallback venue:** IEEE INDICON 2026 (deadline Aug 31, 2026). Same IEEE template for both.
**Roles:** The human author writes ALL paper prose himself. You (the agent) build code, download data, run experiments, and produce tables/figures/CSVs. **Never generate paper text. Never invent numbers.**

## One-paragraph pitch
Audio foundation models (PANNs, AST, BEATs, CLAP family) are compared almost exclusively on accuracy. Real deployment decisions need energy and latency too. MLPerf Tiny treats energy as a first-class metric but requires physical MCU rigs. GreenAudioBench delivers the first measured-energy, cross-model accuracy × energy × latency comparison of audio foundation models on standard environmental-sound benchmarks (ESC-50, UrbanSound8K; TAU 2020 Mobile as phase 2), under two protocols — zero-shot (CLAP-style) and frozen-embedding linear probe — fully reproducible on a free Colab T4 GPU. Deliverable artifact: open code + cached embeddings + full results CSVs.

## Hard rules
1. **Data only from the first-party sources listed below.** NEVER download datasets from Kaggle or any mirror, even if faster.
2. **Official cross-validation folds only.** Never re-split, never merge folds.
3. **No backbone fine-tuning.** Frozen embeddings + linear probes only, plus CLAP zero-shot.
4. **Reproducibility:** seeds {0,1,2,3,4}; `pip freeze > env/requirements-lock.txt` after env setup; compute and store SHA-256 checksums of all downloaded archives in `data/CHECKSUMS.txt` (record actual values after download — never copy hashes from anywhere); every results CSV includes columns: `git_commit`, `gpu_name`, `torch_version`, `timestamp`.
5. **Never fabricate, interpolate, or "estimate" any number.** Tables and figures are generated only from files in `results/`.
6. GPU-dependent steps (embedding extraction, latency/energy measurement) run on an NVIDIA GPU (Colab T4). If no CUDA device is available locally: implement + unit-test on CPU with tiny fixtures, and mark run scripts as Colab-only. Never report CPU timings as GPU results.
7. Everything must be runnable end-to-end via `bash scripts/run_all.sh` — idempotent, skips already-cached artifacts.
8. Respect dataset licenses (all are non-commercial research licenses; this is academic research). Cite the original papers in README.

## Datasets (first-party sources only)
| # | Dataset | First-party source | License | Size | Official splits |
|---|---------|--------------------|---------|------|-----------------|
| 1 | **ESC-50** (Piczak) | GitHub: `https://github.com/karolpiczak/ESC-50` — download `archive/master.zip` | CC BY-NC 3.0 (ESC-10 subset CC BY) | ~0.9 GB | 5 official folds (in `meta/esc50.csv`) |
| 2 | **UrbanSound8K** (NYU MARL — Salamon, Jacoby, Bello) | Zenodo DOI `10.5281/zenodo.1203745` | CC BY-NC 3.0 | ~5.6 GB | 10 official folds (in `metadata/UrbanSound8K.csv`) |
| 3 | **TAU Urban Acoustic Scenes 2020 Mobile, development** (Tampere University / DCASE) — *phase 2, optional* | Zenodo DOI `10.5281/zenodo.3670167` (multi-part) | Non-commercial (dataset-specific) | tens of GB | official train/test split in `evaluation_setup/` |

Download with `scripts/download_data.py` (use direct URLs or `zenodo_get`; retry with resume on failure). After download: verify archive integrity, unpack to `data/raw/<dataset>/`, write `data/CHECKSUMS.txt`, print per-dataset clip counts and total duration, and cross-check counts against the numbers in the dataset metadata files.

## Models
**Zero-shot (protocol P1):** LAION-CLAP (`laion/clap-htsat-unfused` via HuggingFace `transformers`/`laion_clap`), MS-CLAP 2023 (`msclap` package, Microsoft).
**Frozen embeddings (protocol P2):** PANNs CNN14 (official checkpoint from `qiuqiangkong/audioset_tagging_cnn` releases/Zenodo), AST (`MIT/ast-finetuned-audioset-10-10-0.4593` on HuggingFace), BEATs iter3+ (checkpoint from `microsoft/unilm` BEATs page), plus the audio encoders of both CLAP models.
**Baselines:** (a) log-mel (64 mels, mean-pooled over time) + logistic regression — the "classical floor"; (b) a small CNN (~1–3M params, e.g., MobileNetV2-width-slim on log-mels) trained per fold — the "efficient supervised floor".

Store checkpoint URLs and file sizes in `env/MODELS.md` as you fetch them.

## Protocols
**P1 — zero-shot (CLAP family only):** prompt = `"a sound of {class_name}"` (plus one alternative template as an ablation); cosine similarity between audio and text embeddings; metrics: accuracy, macro-F1 per dataset on official test folds. Tag all P1 results `weak_zero_shot=true` in the CSV (pretraining corpora of these models may overlap ESC-50/US8K sources — the paper will discuss this honestly).
**P2 — frozen linear probe (all models):** extract embeddings once, cache in `data/embeddings/<model>/<dataset>/` (npz + preprocessing/checkpoint metadata); probe = StandardScaler (fit on training folds only) + scikit-learn `LogisticRegression` (lbfgs, max_iter=2000) — deterministic, so exactly ONE result per official outer fold; C selected from {0.01, 0.1, 1, 10, 100} by leave-one-official-fold-out inner CV on the training folds (`PredefinedSplit` on original fold ids — never random KFold, never clip-level reshuffling; ties break toward smaller C); report mean ± std of accuracy and macro-F1 ACROSS OFFICIAL OUTER FOLDS (fold dispersion — never described as seed variance; lbfgs ignores random_state, so repeated-seed runs would be duplicates). Seeds {0..4} remain reserved for genuinely stochastic components (e.g., the small-CNN baseline when implemented).
**Efficiency (per model, on T4):**
- `params` (M), `GMACs` at fixed input (fvcore or thop; document the exact input shape per model);
- latency ms/clip: batch sizes 1 and 32, 20 warm-up iterations, 100 timed iterations, `torch.cuda.synchronize()` around timing, 3 independent repeats → report median and IQR;
- energy J/clip: background `pynvml` power-sampling thread at 10 Hz during the timed loop (energy = ∫power dt / clips), cross-checked with a `codecarbon` session total; log GPU name, driver version, and GPU temperature before/after;
- fp32 and fp16 (`model.half()`) variants where supported — this is the main efficiency ablation.
**Outputs:** `results/accuracy.csv`, `results/zeroshot.csv`, `results/efficiency.csv`; figures: Pareto front (x = J/clip log-scale, y = accuracy) per dataset, one combined figure; main table: model × dataset with accuracy, J/clip, ms/clip, params.

## Repository layout
```
greenaudiobench/
  CLAUDE.md  README.md  LICENSE
  env/            requirements.txt, requirements-lock.txt, MODELS.md
  scripts/        download_data.py, extract_embeddings.py, run_zeroshot.py,
                  run_probes.py, measure_efficiency.py, make_tables.py, run_all.sh
  src/gab/        datasets.py, folds.py, models/{panns,ast,beats,clap}.py,
                  probes.py, efficiency.py, utils.py
  data/           raw/  embeddings/  CHECKSUMS.txt        (gitignored except CHECKSUMS)
  results/        *.csv                                     (committed)
  figures/        *.pdf, *.png                               (committed)
  notebooks/      colab_runner.ipynb   (thin wrapper: clone repo → run scripts)
  tests/          test_folds.py, test_embeddings_shapes.py, test_efficiency_smoke.py
```

## Milestones (execute in order; stop after each and summarize)
- **M0** — repo skeleton, env files, unit tests for fold parsing (use tiny synthetic fixtures, no audio yet).
- **M1** — download ESC-50 + UrbanSound8K, checksums, loaders with official folds, dataset stats report (counts, durations, sample rates) cross-checked against metadata.
- **M2** — embedding extraction for all P2 models on ESC-50; cache; shape/NaN sanity tests.
- **M3** — linear probes on ESC-50 (5 seeds × 5 folds); `results/accuracy.csv`.
- **M4** — zero-shot CLAP on ESC-50; `results/zeroshot.csv`.
- **M5** — efficiency measurements on Colab T4 (fp32 + fp16); `results/efficiency.csv`.
- **M6** — repeat M2–M5 for UrbanSound8K (10 folds).
- **M7** — `make_tables.py`: LaTeX tables + Pareto figures; final consistency check (no NaNs, seeds complete, commit hashes present).
- **M8 (optional, if time before deadline)** — TAU 2020 Mobile phase.

**Definition of done per milestone:** script runs clean from scratch, outputs written, tests pass, README section updated, one-paragraph summary of findings (numbers quoted verbatim from CSVs) reported back to the human.

## What NOT to do
- No Kaggle downloads. No re-splitting. No fine-tuning. No paper prose. No invented or extrapolated numbers. No deletion of raw results (append-only; new runs get new timestamped rows). No paid services or APIs. Do not upload the datasets anywhere.

## Notes for Codex
If this repo is driven by OpenAI Codex instead of Claude Code, copy this file to `AGENTS.md` — the instructions are agent-agnostic.

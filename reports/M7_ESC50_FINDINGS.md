# M7 — ESC-50 factual findings (evidence report, not manuscript prose)

All values computed by `scripts/summarize_results.py` from `results/accuracy.csv`, `results/zeroshot.csv` and `results/efficiency.csv`. Hardware: NVIDIA Tesla T4 (Google Colab), driver 580.82.07, CUDA 12.8, torch 2.11.0+cu128.

## A. Frozen linear-probe ranking (protocol P2, fp32 embeddings)

Accuracy and macro-F1 are means across the five official ESC-50 folds; the +/- value is the standard deviation across those folds (fold dispersion, not seed variance — the probe is deterministic).

| rank | model | accuracy (%) | macro-F1 (%) | selected C per fold |
|---|---|---|---|---|
| 1 | MS-CLAP | 97.95 +/- 0.76 | 97.89 +/- 0.83 | [0.01, 0.01, 0.01, 0.01, 0.1] |
| 2 | LAION-CLAP | 97.20 +/- 0.78 | 97.17 +/- 0.79 | [0.01, 0.01, 0.01, 0.01, 0.1] |
| 3 | AST | 96.15 +/- 2.02 | 96.06 +/- 2.13 | [0.01, 0.1, 0.1, 0.01, 1.0] |
| 4 | BEATs | 96.15 +/- 1.45 | 96.08 +/- 1.50 | [1.0, 0.1, 0.01, 0.1, 1.0] |
| 5 | PANNs CNN14 | 92.25 +/- 1.33 | 92.15 +/- 1.45 | [1.0, 0.01, 1.0, 0.01, 0.1] |

Spread between best and worst model: **5.70 percentage points** (MS-CLAP 97.95% vs PANNs CNN14 92.25%).

## B. Zero-shot ranking (protocol P1, CLAP family only)

All rows carry `weak_zero_shot=True`: the CLAP pretraining corpora may overlap the ESC-50 source recordings, so these are not clean zero-shot numbers.

| model | template | prompt | accuracy (%) | macro-F1 (%) |
|---|---|---|---|---|
| MS-CLAP | primary | `a sound of {class_name}` | 95.00 +/- 1.43 | 94.79 +/- 1.64 |
| MS-CLAP | alternative | `this is the sound of {class_name}` | 93.85 +/- 1.61 | 93.51 +/- 1.89 |
| LAION-CLAP | alternative | `this is the sound of {class_name}` | 85.10 +/- 1.99 | 83.11 +/- 2.13 |
| LAION-CLAP | primary | `a sound of {class_name}` | 81.50 +/- 1.92 | 79.35 +/- 2.10 |

### Template sensitivity and probe gap

- **LAION-CLAP**: best template `alternative` at 85.10%, worst `primary` at 81.50% -> absolute template sensitivity **3.60 pp**. Frozen probe (97.20%) exceeds the best zero-shot template by **12.10 pp**.
- **MS-CLAP**: best template `primary` at 95.00%, worst `alternative` at 93.85% -> absolute template sensitivity **1.15 pp**. Frozen probe (97.95%) exceeds the best zero-shot template by **2.95 pp**.

## C. Efficiency ranking (fp32, measured waveform -> embedding)

### fp32, batch size 1

| rank by latency | model | median ms/clip | IQR (p25-p75) | dynamic J/clip | total J/clip | params (M) | GMACs (batch 1) |
|---|---|---|---|---|---|---|---|
| 1 | MS-CLAP | 12.34 | 11.80-14.92 | 0.7255 | 0.8564 | 33.2 | 6.980 |
| 2 | PANNs CNN14 | 12.67 | 12.59-12.85 | 0.7061 | 0.8356 | 81.8 | 10.512 |
| 3 | BEATs | 24.05 | 22.93-24.70 | 1.3651 | 1.6146 | 90.3 | 23.561 |
| 4 | LAION-CLAP | 44.51 | 43.07-48.28 | 1.4605 | 1.9485 | 27.5 | 5.933 |
| 5 | AST | 82.73 | 81.36-84.11 | 5.0134 | 5.8983 | 86.2 | 103.466 |

Ranked by dynamic energy: PANNs CNN14 (0.7061 J) < MS-CLAP (0.7255 J) < BEATs (1.3651 J) < LAION-CLAP (1.4605 J) < AST (5.0134 J)

### fp32, batch size 32

| rank by latency | model | median ms/clip | IQR (p25-p75) | dynamic J/clip | total J/clip | params (M) | GMACs (batch 1) |
|---|---|---|---|---|---|---|---|
| 1 | PANNs CNN14 | 4.89 | 4.73-5.04 | 0.2799 | 0.3305 | 81.8 | 10.512 |
| 2 | MS-CLAP | 9.44 | 9.19-9.54 | 0.5132 | 0.6066 | 33.2 | 6.980 |
| 3 | BEATs | 21.61 | 21.49-21.75 | 1.1916 | 1.4075 | 90.3 | 23.561 |
| 4 | LAION-CLAP | 36.31 | 35.23-38.43 | 1.4793 | 1.8598 | 27.5 | 5.933 |
| 5 | AST | 99.78 | 99.39-100.49 | 5.3375 | 6.3305 | 86.2 | 103.466 |

Ranked by dynamic energy: PANNs CNN14 (0.2799 J) < MS-CLAP (0.5132 J) < BEATs (1.1916 J) < LAION-CLAP (1.4793 J) < AST (5.3375 J)

Idle GPU baseline subtracted from every dynamic value: **9.976 W** (measured over a >=30 s idle window before the measurement loops).

## D. Pareto-optimal configurations (exact, computed programmatically)

- **A. accuracy vs latency, fp32 batch 1** -> frontier: MS-CLAP (accuracy 97.95%, cost 12.3359). Dominated: PANNs CNN14, BEATs, LAION-CLAP, AST.
- **B. accuracy vs dynamic J/clip, fp32 batch 1** -> frontier: PANNs CNN14 (accuracy 92.25%, cost 0.7061), MS-CLAP (accuracy 97.95%, cost 0.7255). Dominated: BEATs, LAION-CLAP, AST.
- **C. accuracy vs latency, fp32 batch 32** -> frontier: PANNs CNN14 (accuracy 92.25%, cost 4.8929), MS-CLAP (accuracy 97.95%, cost 9.4389). Dominated: BEATs, LAION-CLAP, AST.
- **D. accuracy vs dynamic J/clip, fp32 batch 32** -> frontier: PANNs CNN14 (accuracy 92.25%, cost 0.2799), MS-CLAP (accuracy 97.95%, cost 0.5132). Dominated: BEATs, LAION-CLAP, AST.

## E. Relative comparisons (all computed from the CSVs)

- At fp32 batch 1, MS-CLAP is **6.71x faster** than AST (12.34 vs 82.73 ms/clip) and uses **85.5% less dynamic energy per clip**, while scoring **1.80 pp higher** frozen-probe accuracy.
- PANNs CNN14 is the cheapest model at fp32 batch 1 (0.7061 J/clip), but it buys only 2.7% less dynamic energy than MS-CLAP while scoring **5.70 pp lower** accuracy — the two frontier points are close in cost and far apart in accuracy.
- Batching 1 -> 32 changes AST latency by **+20.6%** and dynamic energy by **+6.5%** (negative = cheaper at batch 32).
- Batching 1 -> 32 changes LAION-CLAP latency by **-18.4%** and dynamic energy by **+1.3%** (negative = cheaper at batch 32).
- Batching 1 -> 32 changes PANNs CNN14 latency by **-61.4%** and dynamic energy by **-60.4%** (negative = cheaper at batch 32).
- Batching 1 -> 32 changes BEATs latency by **-10.1%** and dynamic energy by **-12.7%** (negative = cheaper at batch 32).
- Batching 1 -> 32 changes MS-CLAP latency by **-23.5%** and dynamic energy by **-29.3%** (negative = cheaper at batch 32).

## F. FP16 compatibility

- **AST**: fp16 measured. At batch 1, latency -75.5% and dynamic energy -74.7% versus fp32.
- **LAION-CLAP**: fp16 measured. At batch 1, latency +1.0% and dynamic energy -8.6% versus fp32.
- **PANNs CNN14**: fp16 produced no valid measurement — `ValueError: panns_cnn14: NaN/Inf in embeddings` at both batch sizes. No fp16 number is reported for this model; the failure is kept explicitly in the CSV.
- **BEATs**: fp16 produced no valid measurement — `ValueError: beats: NaN/Inf in embeddings` at both batch sizes. No fp16 number is reported for this model; the failure is kept explicitly in the CSV.
- **MS-CLAP**: fp16 produced no valid measurement — `ValueError: ms_clap: NaN/Inf in embeddings` at both batch sizes. No fp16 number is reported for this model; the failure is kept explicitly in the CSV.

fp16 usability is model/implementation dependent: **2 of 5** models produced finite fp16 embeddings under `model.half()` with no other change.

## G. Limitations of the current evidence

- Single dataset (ESC-50, 2000 clips, 50 classes). UrbanSound8K and TAU are not yet run, so nothing here generalises across datasets.
- Single GPU type (one Tesla T4 instance). Latency and energy are hardware-specific and must not be presented as device-independent.
- Energy and latency come from one measurement session per configuration; the CSV records three latency repeats but a single energy window per configuration, so no run-to-run energy variance is available.
- fp16 failures are reported as measured failures of `model.half()` without further remediation; they are not evidence that these models cannot be quantised by other means.
- Zero-shot rows are `weak_zero_shot=True`: pretraining-corpus overlap with ESC-50 sources cannot be excluded.
- No statistical significance testing was performed. Fold dispersion is reported as a standard deviation over five folds and nothing more; no claim of significance is supported by these files.
- The probe is deterministic (lbfgs), so a single result exists per official fold; the dispersion is across folds, not across seeds.

## H. Strongest evidence-backed findings

1. On ESC-50 frozen linear probes, MS-CLAP leads at 97.95 +/- 0.76% accuracy, and the five models span only 5.70 pp (92.25-97.95%).
2. Deployment cost varies far more than accuracy: MS-CLAP and AST differ by only 1.80 pp in accuracy but by 6.71x in latency and 6.91x in dynamic energy per clip at fp32 batch 1 — accuracy-only comparisons hide this axis entirely.
3. The accuracy-energy Pareto frontier at fp32 batch 1 contains exactly 2 of five models: PANNs CNN14, MS-CLAP; the other three are strictly dominated.
4. Batching from 1 to 32 clips helps unevenly: PANNs CNN14 gains 61% latency and 60% dynamic energy per clip, whereas AST gets 21% SLOWER per clip at batch 32.
5. fp16 is not uniformly available: 2 of 5 models produced finite embeddings under `model.half()`; BEATs, MS-CLAP, PANNs CNN14 returned NaN/Inf and yield no fp16 measurement.
6. Where fp16 does work it is a large win: AST at batch 1 drops 76% in latency and 75% in dynamic energy per clip.
7. Frozen probes beat zero-shot for both CLAP models on ESC-50: LAION-CLAP 97.20% probe vs 85.10% best-template zero-shot (+12.10 pp); MS-CLAP 97.95% probe vs 95.00% best-template zero-shot (+2.95 pp).
8. Prompt wording matters unequally across CLAP implementations: LAION-CLAP shifts 3.60 pp between the two templates; MS-CLAP shifts 1.15 pp between the two templates (both rows flagged weak zero-shot).

---

Provenance: probes/zero-shot at commit `d20e86fe2617d8904ee95010904cdbdf8cfb12eb`, efficiency at commit `93eaacb9390a0eb42e3266021e0c905a04f2fc02` (the latter differs only by the NVML sampler thread-lifecycle fix; measurement methodology is identical). All rows recorded `git_dirty=False` on Tesla T4.


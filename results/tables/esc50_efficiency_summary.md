# ESC-50 efficiency on NVIDIA Tesla T4 (M5)

Measured region: canonical waveform batch -> embedding (model-specific feature extraction included). Latency: 20 warm-ups, 100 timed iterations, 3 repeats, median and IQR. Energy: NVML board power at 10 Hz over a >=30 s window, J/clip divided by actual clips processed; `j_per_clip_dynamic` subtracts the idle baseline. Failed configurations are kept with their error. Source: `results/efficiency.csv`.

| model | model_display | dtype | batch_size | status | note | params_millions | gmacs_fvcore_b1 | latency_ms_per_clip_median | latency_ms_per_clip_p25 | latency_ms_per_clip_p75 | j_per_clip_total | j_per_clip_dynamic | idle_power_w | clips_processed | energy_window_s | n_power_samples | gpu_temp_before_c | gpu_temp_after_c |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ast | AST | fp16 | 1 | ok |  | 86.1870 | 103.4660 | 20.2574 | 19.9262 | 21.0567 | 1.4731 | 1.2668 | 9.9759 | 1451.0000 | 30.0130 | 284.0000 | 78.0000 | 77.0000 |
| ast | AST | fp16 | 32 | ok |  | 86.1870 | 103.4660 | 25.1810 | 24.8661 | 25.7528 | 1.5574 | 1.3038 | 9.9759 | 1184.0000 | 30.0918 | 284.0000 | 77.0000 | 77.0000 |
| ast | AST | fp32 | 1 | ok |  | 86.1870 | 103.4660 | 82.7261 | 81.3642 | 84.1118 | 5.8983 | 5.0134 | 9.9759 | 339.0000 | 30.0703 | 284.0000 | 60.0000 | 69.0000 |
| ast | AST | fp32 | 32 | ok |  | 86.1870 | 103.4660 | 99.7801 | 99.3883 | 100.4904 | 6.3305 | 5.3375 | 9.9759 | 320.0000 | 31.8507 | 301.0000 | 77.0000 | 77.0000 |
| laion_clap | LAION-CLAP | fp16 | 1 | ok |  | 27.5340 | 5.9330 | 44.9476 | 43.7892 | 46.7152 | 1.8339 | 1.3352 | 9.9759 | 601.0000 | 30.0399 | 281.0000 | 74.0000 | 77.0000 |
| laion_clap | LAION-CLAP | fp16 | 32 | ok |  | 27.5340 | 5.9330 | 31.6776 | 30.8630 | 33.4851 | 1.2933 | 0.9567 | 9.9759 | 896.0000 | 30.2288 | 284.0000 | 79.0000 | 77.0000 |
| laion_clap | LAION-CLAP | fp32 | 1 | ok |  | 27.5340 | 5.9330 | 44.5115 | 43.0731 | 48.2792 | 1.9485 | 1.4605 | 9.9759 | 614.0000 | 30.0318 | 280.0000 | 73.0000 | 76.0000 |
| laion_clap | LAION-CLAP | fp32 | 32 | ok |  | 27.5340 | 5.9330 | 36.3108 | 35.2276 | 38.4263 | 1.8598 | 1.4793 | 9.9759 | 800.0000 | 30.5121 | 289.0000 | 78.0000 | 77.0000 |
| panns_cnn14 | PANNs CNN14 | fp16 | 1 | measure_error | ValueError: panns_cnn14: NaN/Inf in embeddings | 81.8370 | 10.5120 |  |  |  |  |  |  |  |  |  |  |  |
| panns_cnn14 | PANNs CNN14 | fp16 | 32 | measure_error | ValueError: panns_cnn14: NaN/Inf in embeddings | 81.8370 | 10.5120 |  |  |  |  |  |  |  |  |  |  |  |
| panns_cnn14 | PANNs CNN14 | fp32 | 1 | ok |  | 81.8370 | 10.5120 | 12.6655 | 12.5869 | 12.8498 | 0.8356 | 0.7061 | 9.9759 | 2311.0000 | 30.0058 | 284.0000 | 81.0000 | 76.0000 |
| panns_cnn14 | PANNs CNN14 | fp32 | 32 | ok |  | 81.8370 | 10.5120 | 4.8929 | 4.7332 | 5.0422 | 0.3305 | 0.2799 | 9.9759 | 5920.0000 | 30.0293 | 279.0000 | 78.0000 | 77.0000 |
| beats | BEATs | fp16 | 1 | measure_error | ValueError: beats: NaN/Inf in embeddings | 90.3120 |  |  |  |  |  |  |  |  |  |  |  |  |
| beats | BEATs | fp16 | 32 | measure_error | ValueError: beats: NaN/Inf in embeddings | 90.3120 |  |  |  |  |  |  |  |  |  |  |  |  |
| beats | BEATs | fp32 | 1 | ok |  | 90.3120 | 23.5610 | 24.0459 | 22.9305 | 24.7019 | 1.6146 | 1.3651 | 9.9759 | 1200.0000 | 30.0129 | 284.0000 | 76.0000 | 78.0000 |
| beats | BEATs | fp32 | 32 | ok |  | 90.3120 | 23.5610 | 21.6133 | 21.4895 | 21.7463 | 1.4075 | 1.1916 | 9.9759 | 1408.0000 | 30.4659 | 288.0000 | 77.0000 | 77.0000 |
| ms_clap | MS-CLAP | fp16 | 1 | measure_error | ValueError: ms_clap: NaN/Inf in embeddings | 33.1620 | 6.9800 |  |  |  |  |  |  |  |  |  |  |  |
| ms_clap | MS-CLAP | fp16 | 32 | measure_error | ValueError: ms_clap: NaN/Inf in embeddings | 33.1620 | 6.9800 |  |  |  |  |  |  |  |  |  |  |  |
| ms_clap | MS-CLAP | fp32 | 1 | ok |  | 33.1620 | 6.9800 | 12.3359 | 11.7953 | 14.9214 | 0.8564 | 0.7255 | 9.9759 | 2287.0000 | 30.0121 | 285.0000 | 76.0000 | 79.0000 |
| ms_clap | MS-CLAP | fp32 | 32 | ok |  | 33.1620 | 6.9800 | 9.4389 | 9.1946 | 9.5446 | 0.6066 | 0.5132 | 9.9759 | 3232.0000 | 30.2827 | 284.0000 | 77.0000 | 77.0000 |

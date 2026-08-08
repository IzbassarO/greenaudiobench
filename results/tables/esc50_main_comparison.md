# ESC-50 main comparison (accuracy x deployment efficiency, fp32)

fp32 is used as the universal fair comparison: all five models have valid fp32 measurements. Energy columns are labelled explicitly as dynamic (idle-subtracted) or total. Accuracy and macro-F1 are frozen-probe means across the five official folds.

| model | model_display | accuracy_pct | accuracy_std_pct | macro_f1_pct | macro_f1_std_pct | params_millions | fp32_bs1_latency_ms_per_clip_median | fp32_bs1_j_per_clip_dynamic | fp32_bs1_j_per_clip_total | fp32_bs32_latency_ms_per_clip_median | fp32_bs32_j_per_clip_dynamic | fp32_bs32_j_per_clip_total | fp16_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ms_clap | MS-CLAP | 97.9500 | 0.7583 | 97.8905 | 0.8341 | 33.1620 | 12.3359 | 0.7255 | 0.8564 | 9.4389 | 0.5132 | 0.6066 | failed: ValueError: ms_clap: NaN/Inf in embeddings |
| laion_clap | LAION-CLAP | 97.2000 | 0.7786 | 97.1716 | 0.7935 | 27.5340 | 44.5115 | 1.4605 | 1.9485 | 36.3108 | 1.4793 | 1.8598 | measured |
| ast | AST | 96.1500 | 2.0202 | 96.0596 | 2.1299 | 86.1870 | 82.7261 | 5.0134 | 5.8983 | 99.7801 | 5.3375 | 6.3305 | measured |
| beats | BEATs | 96.1500 | 1.4534 | 96.0826 | 1.5025 | 90.3120 | 24.0459 | 1.3651 | 1.6146 | 21.6133 | 1.1916 | 1.4075 | failed: ValueError: beats: NaN/Inf in embeddings |
| panns_cnn14 | PANNs CNN14 | 92.2500 | 1.3346 | 92.1466 | 1.4451 | 81.8370 | 12.6655 | 0.7061 | 0.8356 | 4.8929 | 0.2799 | 0.3305 | failed: ValueError: panns_cnn14: NaN/Inf in embeddings |

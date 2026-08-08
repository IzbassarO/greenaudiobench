# ESC-50 Pareto analysis (fp32)

A configuration is dominated iff another has >= accuracy AND <= cost with at least one strict inequality. Computed programmatically from the official CSVs.

| comparison | comparison_label | cost_metric | model | model_display | dtype | batch_size | accuracy_mean | cost_value | pareto_optimal |
|---|---|---|---|---|---|---|---|---|---|
| A | accuracy vs latency, fp32 batch 1 | latency_ms_per_clip_median | ms_clap | MS-CLAP | fp32 | 1 | 0.9795 | 12.3359 | True |
| A | accuracy vs latency, fp32 batch 1 | latency_ms_per_clip_median | panns_cnn14 | PANNs CNN14 | fp32 | 1 | 0.9225 | 12.6655 | False |
| A | accuracy vs latency, fp32 batch 1 | latency_ms_per_clip_median | beats | BEATs | fp32 | 1 | 0.9615 | 24.0459 | False |
| A | accuracy vs latency, fp32 batch 1 | latency_ms_per_clip_median | laion_clap | LAION-CLAP | fp32 | 1 | 0.9720 | 44.5115 | False |
| A | accuracy vs latency, fp32 batch 1 | latency_ms_per_clip_median | ast | AST | fp32 | 1 | 0.9615 | 82.7261 | False |
| B | accuracy vs dynamic J/clip, fp32 batch 1 | j_per_clip_dynamic | panns_cnn14 | PANNs CNN14 | fp32 | 1 | 0.9225 | 0.7061 | True |
| B | accuracy vs dynamic J/clip, fp32 batch 1 | j_per_clip_dynamic | ms_clap | MS-CLAP | fp32 | 1 | 0.9795 | 0.7255 | True |
| B | accuracy vs dynamic J/clip, fp32 batch 1 | j_per_clip_dynamic | beats | BEATs | fp32 | 1 | 0.9615 | 1.3651 | False |
| B | accuracy vs dynamic J/clip, fp32 batch 1 | j_per_clip_dynamic | laion_clap | LAION-CLAP | fp32 | 1 | 0.9720 | 1.4605 | False |
| B | accuracy vs dynamic J/clip, fp32 batch 1 | j_per_clip_dynamic | ast | AST | fp32 | 1 | 0.9615 | 5.0134 | False |
| C | accuracy vs latency, fp32 batch 32 | latency_ms_per_clip_median | panns_cnn14 | PANNs CNN14 | fp32 | 32 | 0.9225 | 4.8929 | True |
| C | accuracy vs latency, fp32 batch 32 | latency_ms_per_clip_median | ms_clap | MS-CLAP | fp32 | 32 | 0.9795 | 9.4389 | True |
| C | accuracy vs latency, fp32 batch 32 | latency_ms_per_clip_median | beats | BEATs | fp32 | 32 | 0.9615 | 21.6133 | False |
| C | accuracy vs latency, fp32 batch 32 | latency_ms_per_clip_median | laion_clap | LAION-CLAP | fp32 | 32 | 0.9720 | 36.3108 | False |
| C | accuracy vs latency, fp32 batch 32 | latency_ms_per_clip_median | ast | AST | fp32 | 32 | 0.9615 | 99.7801 | False |
| D | accuracy vs dynamic J/clip, fp32 batch 32 | j_per_clip_dynamic | panns_cnn14 | PANNs CNN14 | fp32 | 32 | 0.9225 | 0.2799 | True |
| D | accuracy vs dynamic J/clip, fp32 batch 32 | j_per_clip_dynamic | ms_clap | MS-CLAP | fp32 | 32 | 0.9795 | 0.5132 | True |
| D | accuracy vs dynamic J/clip, fp32 batch 32 | j_per_clip_dynamic | beats | BEATs | fp32 | 32 | 0.9615 | 1.1916 | False |
| D | accuracy vs dynamic J/clip, fp32 batch 32 | j_per_clip_dynamic | laion_clap | LAION-CLAP | fp32 | 32 | 0.9720 | 1.4793 | False |
| D | accuracy vs dynamic J/clip, fp32 batch 32 | j_per_clip_dynamic | ast | AST | fp32 | 32 | 0.9615 | 5.3375 | False |

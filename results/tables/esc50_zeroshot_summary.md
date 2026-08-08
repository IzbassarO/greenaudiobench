# ESC-50 zero-shot CLAP (protocol P1)

Mean and standard deviation across the five official folds. All rows are flagged `weak_zero_shot=True`: CLAP pretraining corpora may overlap the ESC-50 sources. Source: `results/zeroshot.csv`.

| model | template_id | prompt_template | n_folds | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std | model_display | weak_zero_shot |
|---|---|---|---|---|---|---|---|---|---|
| laion_clap | alternative | this is the sound of {class_name} | 5 | 0.8510 | 0.0199 | 0.8311 | 0.0213 | LAION-CLAP | True |
| laion_clap | primary | a sound of {class_name} | 5 | 0.8150 | 0.0192 | 0.7935 | 0.0210 | LAION-CLAP | True |
| ms_clap | alternative | this is the sound of {class_name} | 5 | 0.9385 | 0.0161 | 0.9351 | 0.0189 | MS-CLAP | True |
| ms_clap | primary | a sound of {class_name} | 5 | 0.9500 | 0.0143 | 0.9479 | 0.0164 | MS-CLAP | True |

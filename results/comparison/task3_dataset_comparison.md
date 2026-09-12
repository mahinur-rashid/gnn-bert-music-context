# Task 3 -- dataset comparison

| dataset | model | n_train | n_test | n_classes | test_macro_f1 | test_micro_f1 | test_auc_pr | test_mae_mean | test_r2_valence | test_graph_coherence | val_macro_f1 | val_micro_f1 | train_macro_f1 | train_micro_f1 | random_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fma_small | cross_attn | 6397 | 800 | 50 | 0.2054 | 0.2737 | 0.2994 | - | - | 0.9994 | 0.3584 | 0.4410 | 0.8829 | 0.9364 | 0.0393 |
| mtat | cross_attn | 15428 | 4392 | 50 | 0.3515 | 0.4406 | 0.3455 | - | - | 0.9992 | 0.4045 | 0.4776 | 0.4369 | 0.5094 | 0.0635 |
| deam | cross_attn | 1431 | 177 | 20 | 0.4141 | 0.3735 | 0.3978 | 0.1861 | 0.4964 | 1.0000 | 0.5675 | 0.5014 | 0.6128 | 0.7017 | 0.0845 |

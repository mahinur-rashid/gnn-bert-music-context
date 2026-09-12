# Task 3 -- dataset comparison

| dataset | model | n_train | n_test | n_classes | test_macro_f1 | test_micro_f1 | test_auc_pr | test_mae_mean | test_r2_valence | test_graph_coherence | val_macro_f1 | val_micro_f1 | train_macro_f1 | train_micro_f1 | random_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fma_small | cross_attn | 171 | 16 | 50 | 0.0331 | 0.2418 | 0.5242 | - | - | 1.0000 | 0.0556 | 0.5882 | 0.0335 | 0.3175 | 0.0050 |
| mtat | cross_attn | 136 | 48 | 50 | 0.0644 | 0.1669 | 0.2652 | - | - | 0.9990 | 0.1000 | 0.2792 | 0.0764 | 0.2081 | 0.0398 |
| deam | cross_attn | 145 | 19 | 20 | 0.0502 | 0.5116 | 0.6139 | 0.2665 | 0.2774 | 1.0000 | 0.0955 | 0.9589 | 0.0692 | 0.6948 | 0.0625 |

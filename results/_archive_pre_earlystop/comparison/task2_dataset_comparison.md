# Task 2 -- dataset comparison

| dataset | model | n_train | n_test | n_classes | test_accuracy | test_macro_f1 | test_micro_f1 | test_auc_pr | val_accuracy | val_macro_f1 | train_accuracy | train_macro_f1 | random_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gtzan | gat | 799 | 100 | 10 | 0.8400 | 0.8380 | 0.8400 | 0.8783 | 0.8800 | 0.8796 | 0.9987 | 0.9987 | 0.1124 |
| fma_small | cnn | 6397 | 800 | 8 | 0.4562 | 0.4486 | 0.4562 | 0.4818 | 0.5650 | 0.5506 | 0.8876 | 0.8844 | 0.1513 |
| deam | cnn | 1431 | 177 | 20 | - | 0.2003 | 0.3568 | 0.2691 | - | 0.2684 | - | 0.5514 | 0.0845 |
| mtat | sage | 15428 | 4392 | 50 | - | 0.4049 | 0.4754 | 0.3889 | - | 0.4321 | - | 0.5028 | 0.0635 |

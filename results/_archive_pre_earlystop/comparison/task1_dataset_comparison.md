# Task 1 -- dataset comparison

| dataset | model | n_train | n_test | n_classes | test_macro_f1 | test_micro_f1 | test_auc_pr | val_macro_f1 | val_micro_f1 | train_macro_f1 | train_micro_f1 | random_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| musiccaps | BERT(distilbert-base-uncased) | 3996 | 499 | 50 | 0.6305 | 0.6753 | 0.6706 | 0.6665 | 0.7201 | 0.6645 | 0.7106 | 0.0576 |
| mtat | BERT(distilbert-base-uncased) | 15431 | 4392 | 50 | 0.3156 | 0.3984 | 0.3026 | 0.3662 | 0.4509 | 0.4530 | 0.5051 | 0.0635 |
| fma_medium | BERT(distilbert-base-uncased) | 19521 | 2487 | 50 | 0.3174 | 0.4405 | 0.3441 | 0.4302 | 0.5104 | 0.8841 | 0.9201 | 0.0407 |
| deam | BERT(distilbert-base-uncased) | 1431 | 177 | 20 | 0.3213 | 0.3723 | 0.3784 | 0.3154 | 0.5097 | 0.4531 | 0.5193 | 0.0619 |

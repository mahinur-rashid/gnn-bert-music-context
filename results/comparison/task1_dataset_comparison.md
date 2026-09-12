# Task 1 -- dataset comparison

| dataset | model | n_train | n_test | n_classes | test_macro_f1 | test_micro_f1 | test_auc_pr | val_macro_f1 | val_micro_f1 | train_macro_f1 | train_micro_f1 | random_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fma_small | BERT(distilbert-base-uncased) | 6400 | 800 | 50 | 0.2671 | 0.3245 | 0.3264 | 0.4298 | 0.5031 | 0.9140 | 0.9477 | 0.0393 |
| mtat | BERT(distilbert-base-uncased) | 15431 | 4392 | 50 | 0.3041 | 0.3915 | 0.2869 | 0.3497 | 0.4365 | 0.5633 | 0.6008 | 0.0635 |
| gtzan | BERT(distilbert-base-uncased) | 800 | 100 | 10 | 0.4804 | 0.4512 | 0.5228 | 0.5543 | 0.4966 | 0.5362 | 0.4964 | 0.1479 |
| musiccaps | BERT(distilbert-base-uncased) | 3993 | 504 | 50 | 0.6954 | 0.7364 | 0.7308 | 0.7468 | 0.7707 | 0.8834 | 0.8963 | 0.0527 |

# Task 1 -- dataset comparison

| dataset | model | n_train | n_test | n_classes | test_macro_f1 | test_micro_f1 | test_auc_pr | val_macro_f1 | val_micro_f1 | train_macro_f1 | train_micro_f1 | random_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fma_small | BERT(distilbert-base-uncased) | 200 | 64 | 50 | 0.0295 | 0.1173 | 0.2414 | 0.0643 | 0.1609 | 0.0354 | 0.1174 | 0.0111 |
| mtat | BERT(distilbert-base-uncased) | 200 | 64 | 50 | 0.0758 | 0.1222 | 0.1815 | 0.1280 | 0.1903 | 0.1037 | 0.1562 | 0.0452 |
| gtzan | BERT(distilbert-base-uncased) | 200 | 64 | 10 | 0.1167 | 0.2445 | 0.3060 | 0.1605 | 0.2910 | 0.1567 | 0.3760 | 0.0789 |
| musiccaps | BERT(distilbert-base-uncased) | 200 | 64 | 50 | 0.1254 | 0.1204 | 0.1546 | 0.1472 | 0.1364 | 0.1155 | 0.1134 | 0.0525 |

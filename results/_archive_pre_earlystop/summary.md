| experiment | task | dataset | model | accuracy | macro_f1 | micro_f1 | auc_pr | mae_mean | r2_valence | caption_to_audio_R@5 | caption_to_audio_R@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task1_deam_distilbert-base-uncased | 1 | deam | BERT(distilbert-base-uncased) |  | 0.3213 | 0.3723 | 0.3784 |  |  |  |  |
| task1_fma_medium_distilbert-base-uncased | 1 | fma_medium | BERT(distilbert-base-uncased) |  | 0.3174 | 0.4405 | 0.3441 |  |  |  |  |
| task1_gtzan_distilbert-base-uncased | 1 | gtzan | BERT(distilbert-base-uncased) |  | 0.3025 | 0.2673 | 0.3130 |  |  |  |  |
| task1_mtat_distilbert-base-uncased | 1 | mtat | BERT(distilbert-base-uncased) |  | 0.3156 | 0.3984 | 0.3026 |  |  |  |  |
| task1_musiccaps_distilbert-base-uncased | 1 | musiccaps | BERT(distilbert-base-uncased) |  | 0.6305 | 0.6753 | 0.6706 |  |  |  |  |
| task2_deam_compare | 2 | deam | sage+gat+gcn+cnn |  | 0.2003 | 0.3568 | 0.2691 |  |  |  |  |
| task2_fma_small_compare | 2 | fma_small | sage+gat+gcn+cnn | 0.4562 | 0.4486 | 0.4562 | 0.4818 |  |  |  |  |
| task2_gtzan_compare | 2 | gtzan | sage+gat+gcn+cnn | 0.8000 | 0.8013 | 0.8000 | 0.8831 |  |  |  |  |
| task2_mtat_compare | 2 | mtat | sage+gat+gcn |  | 0.3952 | 0.4654 | 0.3850 |  |  |  |  |
| task3_deam_ablation | 3 | deam | cross_attn+concat+bert_only+gnn_only |  | 0.2899 | 0.3565 | 0.4122 | 0.1936 | 0.4729 |  |  |
| task3_fma_small_ablation | 3 | fma_small | cross_attn+concat+bert_only+gnn_only |  | 0.2615 | 0.3683 | 0.3382 |  |  |  |  |
| task4_deam_contrastive | 4 | deam | contrastive dual-encoder (sage + distilbert-base-uncased) |  |  |  |  |  |  | 0.2500 | 0.3750 |
| task4_deam_contrastive_zero_shot | None | None | None |  |  |  |  |  |  |  |  |
| task4_musiccaps_contrastive | 4 | musiccaps | contrastive dual-encoder (sage + distilbert-base-uncased) |  |  |  |  |  |  | 0.2778 | 0.5278 |
| task4_musiccaps_contrastive_zero_shot | None | None | None |  |  |  |  |  |  |  |  |

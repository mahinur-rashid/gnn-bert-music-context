| experiment | task | dataset | model | accuracy | macro_f1 | micro_f1 | auc_pr | mae_mean | r2_valence | caption_to_audio_R@5 | caption_to_audio_R@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task1_fma_small_distilbert-base-uncased | 1 | fma_small | BERT(distilbert-base-uncased) |  | 0.2671 | 0.3245 | 0.3264 |  |  |  |  |
| task1_gtzan_distilbert-base-uncased | 1 | gtzan | BERT(distilbert-base-uncased) |  | 0.4804 | 0.4512 | 0.5228 |  |  |  |  |
| task1_mtat_distilbert-base-uncased | 1 | mtat | BERT(distilbert-base-uncased) |  | 0.3041 | 0.3915 | 0.2869 |  |  |  |  |
| task1_musiccaps_distilbert-base-uncased | 1 | musiccaps | BERT(distilbert-base-uncased) |  | 0.6919 | 0.7352 | 0.7230 |  |  |  |  |
| task2_fma_small_compare | 2 | fma_small | sage+gat+gcn+cnn | 0.4062 | 0.4101 | 0.4062 | 0.4463 |  |  |  |  |
| task2_gtzan_compare | 2 | gtzan | sage+gat+gcn+cnn | 0.6100 | 0.5969 | 0.6100 | 0.7354 |  |  |  |  |
| task3_deam_ablation | 3 | deam | cross_attn+concat+bert_only+gnn_only |  | 0.4287 | 0.4166 | 0.4516 | 0.1837 | 0.5205 |  |  |
| task3_fma_medium_ablation | 3 | fma_medium | cross_attn+concat+bert_only+gnn_only |  | 0.3232 | 0.4456 | 0.3236 |  |  |  |  |
| task3_fma_small_ablation | 3 | fma_small | cross_attn+concat+bert_only+gnn_only |  | 0.2054 | 0.2737 | 0.2994 |  |  |  |  |
| task3_mtat_ablation | 3 | mtat | cross_attn+concat+bert_only+gnn_only |  | 0.3515 | 0.4406 | 0.3455 |  |  |  |  |
| task4_deam_contrastive | 4 | deam | contrastive dual-encoder (sage + distilbert-base-uncased) |  |  |  |  |  |  | 0.0508 | 0.1073 |
| task4_musiccaps_contrastive | 4 | musiccaps | contrastive dual-encoder (sage + distilbert-base-uncased) |  |  |  |  |  |  | 0.1930 | 0.3002 |

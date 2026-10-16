# Seed and fold variability

Final-epoch-80 results for seeds 0, 1 and 2 using the same five group-disjoint test folds.

- [Seed-specific means and fold standard deviations](seed_metrics_and_fold_std.csv)
- [Means and standard deviations across seeds](model_seed_summary.csv)

## Seed-specific performance

Each entry is the equal-group mean over all 93 test groups, followed by the sample SD of the five fold-level group means within that seed. The five folds contain 17, 19, 20, 19 and 18 groups. Consequently, the 93-group mean is not the unweighted mean of the five fold means.

| Model | Seed | SRCC | PLCC | Pair Acc |
|---|---:|---:|---:|---:|
| Baseline | 0 | 0.4138 ± 0.1058 | 0.4346 ± 0.1188 | 0.6830 ± 0.0474 |
| Baseline | 1 | 0.4257 ± 0.1186 | 0.4459 ± 0.1233 | 0.6896 ± 0.0558 |
| Baseline | 2 | 0.4316 ± 0.0959 | 0.4506 ± 0.1051 | 0.6913 ± 0.0435 |
| Layer 12 | 0 | 0.4107 ± 0.0765 | 0.4287 ± 0.0808 | 0.6781 ± 0.0329 |
| Layer 12 | 1 | 0.4108 ± 0.0608 | 0.4300 ± 0.0744 | 0.6823 ± 0.0310 |
| Layer 12 | 2 | 0.4398 ± 0.0936 | 0.4433 ± 0.1101 | 0.6947 ± 0.0414 |
| Layers 9, 12 | 0 | 0.4229 ± 0.0898 | 0.4304 ± 0.1021 | 0.6831 ± 0.0411 |
| Layers 9, 12 | 1 | 0.4576 ± 0.0942 | 0.4685 ± 0.1076 | 0.7004 ± 0.0438 |
| Layers 9, 12 | 2 | 0.4454 ± 0.0568 | 0.4544 ± 0.0793 | 0.6952 ± 0.0223 |
| MLVS-Net | 0 | 0.4561 ± 0.1058 | 0.4680 ± 0.1222 | 0.7005 ± 0.0486 |
| MLVS-Net | 1 | 0.4620 ± 0.0809 | 0.4690 ± 0.1035 | 0.7038 ± 0.0364 |
| MLVS-Net | 2 | 0.4506 ± 0.0903 | 0.4653 ± 0.1065 | 0.7006 ± 0.0439 |
| Layers 3, 6, 9, 12 | 0 | 0.4448 ± 0.0637 | 0.4648 ± 0.0898 | 0.6971 ± 0.0246 |
| Layers 3, 6, 9, 12 | 1 | 0.4398 ± 0.0780 | 0.4536 ± 0.0937 | 0.6918 ± 0.0344 |
| Layers 3, 6, 9, 12 | 2 | 0.4608 ± 0.0623 | 0.4709 ± 0.0852 | 0.7030 ± 0.0261 |

## Variability across seeds

Each entry is the mean and sample SD of the three seed-specific 93-group means. All SDs use ddof=1. They are descriptive statistics, not confidence intervals.

| Model | SRCC | PLCC | Pair Acc |
|---|---:|---:|---:|
| Baseline | 0.4237 ± 0.0091 | 0.4437 ± 0.0082 | 0.6880 ± 0.0044 |
| Layer 12 | 0.4204 ± 0.0168 | 0.4340 ± 0.0081 | 0.6850 ± 0.0086 |
| Layers 9, 12 | 0.4420 ± 0.0176 | 0.4511 ± 0.0193 | 0.6929 ± 0.0089 |
| MLVS-Net | 0.4562 ± 0.0057 | 0.4674 ± 0.0019 | 0.7016 ± 0.0019 |
| Layers 3, 6, 9, 12 | 0.4484 ± 0.0110 | 0.4631 ± 0.0088 | 0.6973 ± 0.0056 |

Layer numbers denote DINOv2 blocks supplying patch features; MLVS-Net uses blocks 6, 9 and 12. The Baseline instead uses the final-layer CLS feature.

The CSVs preserve the six-decimal precision of the completed run summaries. Aggregate means and SDs were calculated from the original unrounded group results; recomputing them from rounded seed summaries can change the last reported digit. These files contain seed-level summaries and fold SDs, not the individual fold means.

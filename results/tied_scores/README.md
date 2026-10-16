# Tied-score statistics

- [Statistics for all 93 examination groups](group_statistics.csv)
- [Category statistics and AED totals](category_statistics.csv)

Statistics use the original examination scores in `data/labels.csv` (the score column is named `gade`). Each artwork is counted once; these statistics do not depend on training seeds or model predictions.

For a group of N artworks, the total number of unordered pairs is N(N-1)/2. For each distinct score occurring n times, n(n-1)/2 pairs are tied. `tied_pairs_percent` is 100 times the tied-pair count divided by the total within-group pair count. `eligible_pairs` excludes these equal-score pairs and is the number of pairs eligible for Pair Acc.

Category and AED totals pool counts across examination groups before calculating percentages. They do not average group percentages or include pairs from different groups. `tied_images` counts artworks sharing their score with another artwork in the same group; it is distinct from the tied-pair count. `groups_with_ties` counts groups containing at least one tied pair. Quantiles use linear interpolation, and IQR is Q3 minus Q1.

Reproduce from the repository root:

```bash
python describe_labels.py --output results/tied_scores_recomputed
```

| Category | Groups | Groups with ties | Tied pairs | Total within-group pairs | Tied pairs (%) | Eligible pairs |
|---|---:|---:|---:|---:|---:|---:|
| Color Painting | 31 | 29 | 8425 | 35253 | 23.90 | 26828 |
| Quick Sketch | 29 | 28 | 3631 | 24120 | 15.05 | 20489 |
| Sketch | 33 | 30 | 4119 | 23065 | 17.86 | 18946 |
| All AED | 93 | 87 | 16175 | 82438 | 19.62 | 66263 |
